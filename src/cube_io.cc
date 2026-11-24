// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "cube_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>  // NOLINT(build/c++11): reader thread pool
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17): portable paths and mkdir -p
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>  // NOLINT(build/c++11): reader thread pool
#include <set>
#include <sstream>
#include <string>
#include <thread>  // NOLINT(build/c++11): reader thread pool
#include <utility>
#include <vector>

#include "logger.h"
#include "types.h"

namespace cd {
namespace {

namespace fs = std::filesystem;

constexpr char kMagicLine[] = "change_detect_cube 1";
constexpr char kDefaultRed[] = "red.u16";
constexpr char kDefaultNir[] = "nir.u16";
constexpr char kDefaultScl[] = "scl.u8";

// ---------------------------------------------------------------------------
// Small text helpers.

std::string Trim(const std::string& text) {
  const char* const kSpace = " \t\r\n";
  const size_t first = text.find_first_not_of(kSpace);
  if (first == std::string::npos) {
    return std::string();
  }
  const size_t last = text.find_last_not_of(kSpace);
  return text.substr(first, last - first + 1);
}

std::vector<std::string> SplitWhitespace(const std::string& text) {
  std::vector<std::string> fields;
  std::istringstream stream(text);
  std::string field;
  while (stream >> field) {
    fields.push_back(field);
  }
  return fields;
}

bool ParseInt(const std::string& text, int64_t* out) {
  if (text.empty()) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const int64_t value = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || *end != '\0') {
    return false;
  }
  *out = value;
  return true;
}

bool ParseDouble(const std::string& text, double* out) {
  if (text.empty()) {
    return false;
  }
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (*end != '\0' || !std::isfinite(value)) {
    return false;
  }
  *out = value;
  return true;
}

bool ParseFloat(const std::string& text, float* out) {
  if (text.empty()) {
    return false;
  }
  char* end = nullptr;
  const float value = std::strtof(text.c_str(), &end);
  if (*end != '\0' || !std::isfinite(value)) {
    return false;
  }
  *out = value;
  return true;
}

// Shortest decimal text that parses back to exactly `value`.
std::string FormatFloat(float value) {
  char buffer[40];
  for (int precision = 1; precision <= 9; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
    if (std::strtof(buffer, nullptr) == value) {
      break;
    }
  }
  return buffer;
}

std::string FormatDouble(double value) {
  char buffer[40];
  for (int precision = 1; precision <= 17; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
    if (std::strtod(buffer, nullptr) == value) {
      break;
    }
  }
  return buffer;
}

// "<manifest>:<line>: <message>" as an error status.
Status LineError(const std::string& path, int line,
                 const std::string& message) {
  return Status::Error(path + ":" + std::to_string(line) + ": " + message);
}

std::string ErrnoText() { return std::strerror(errno); }

// ---------------------------------------------------------------------------
// Files.

// Checks that `path` is a regular file of exactly `expected` bytes.
Status CheckFileSize(const std::string& path, const std::string& what,
                     uint64_t expected) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    return Status::Error(what + " '" + path +
                         "' cannot be read: " + ErrnoText());
  }
  if (!S_ISREG(info.st_mode)) {
    return Status::Error(what + " '" + path + "' is not a regular file");
  }
  if (static_cast<uint64_t>(info.st_size) != expected) {
    return Status::Error(what + " '" + path + "' has " +
                         std::to_string(info.st_size) + " bytes, expected " +
                         std::to_string(expected));
  }
  return Status::Ok();
}

// Checks the three band files and the optional reference of `info`.
Status CheckCubeFiles(const CubeInfo& info) {
  const uint64_t plane =
      static_cast<uint64_t>(info.height) * static_cast<uint64_t>(info.width);
  const uint64_t cells = plane * static_cast<uint64_t>(info.num_dates);
  Status status = CheckFileSize(info.red_file, "red file", cells * 2);
  if (!status.ok()) {
    return status;
  }
  status = CheckFileSize(info.nir_file, "nir file", cells * 2);
  if (!status.ok()) {
    return status;
  }
  status = CheckFileSize(info.scl_file, "scl file", cells);
  if (!status.ok()) {
    return status;
  }
  if (!info.reference_file.empty()) {
    return CheckFileSize(info.reference_file, "reference file", plane);
  }
  return Status::Ok();
}

// Reads `bytes` bytes at `offset`, retrying short reads and EINTR.
Status PreadFully(int fd, const std::string& path, void* buffer, size_t bytes,
                  off_t offset) {
  char* destination = static_cast<char*>(buffer);
  while (bytes > 0) {
    const ssize_t got = pread(fd, destination, bytes, offset);
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return Status::Error("read from '" + path + "' failed: " + ErrnoText());
    }
    if (got == 0) {
      return Status::Error("unexpected end of file in '" + path + "'");
    }
    destination += got;
    offset += got;
    bytes -= static_cast<size_t>(got);
  }
  return Status::Ok();
}

// ---------------------------------------------------------------------------
// Manifest parsing.

// Parses one "date <index> <iso> <day> <scale> <offset> [valid [scene]]" line
// (fields[0] is the keyword).
Status ParseDateLine(const std::string& path, int line,
                     const std::vector<std::string>& fields, CubeInfo* info) {
  if (fields.size() < 6) {
    return LineError(path, line,
                     "date line needs: date <index> <YYYY-MM-DD> <day> "
                     "<scale> <offset>");
  }
  int64_t index = 0;
  int64_t day = 0;
  float scale = 0;
  float offset = 0;
  const DateTable& dates = info->dates;
  if (!ParseInt(fields[1], &index) ||
      index != static_cast<int64_t>(dates.day.size())) {
    return LineError(path, line,
                     "date index '" + fields[1] + "' out of order, expected " +
                         std::to_string(dates.day.size()));
  }
  const std::string& iso = fields[2];
  const bool iso_ok = iso.size() == 10 && iso[4] == '-' && iso[7] == '-' &&
                      std::strspn(iso.c_str(), "0123456789-") == 10;
  if (!iso_ok) {
    return LineError(path, line,
                     "malformed date '" + iso + "' (expected YYYY-MM-DD)");
  }
  if (!ParseInt(fields[3], &day) || day < 0 ||
      day > std::numeric_limits<int32_t>::max()) {
    return LineError(path, line, "bad day number '" + fields[3] + "'");
  }
  if (!dates.day.empty() && day <= dates.day.back()) {
    return LineError(path, line,
                     "day " + fields[3] + " is not after the previous date (" +
                         std::to_string(dates.day.back()) +
                         "); dates must be strictly increasing");
  }
  if (!ParseFloat(fields[4], &scale) || !(scale > 0.0f)) {
    return LineError(path, line,
                     "scale '" + fields[4] + "' must be a positive number");
  }
  if (!ParseFloat(fields[5], &offset)) {
    return LineError(path, line, "bad offset '" + fields[5] + "'");
  }
  info->dates.iso.push_back(iso);
  info->dates.day.push_back(static_cast<int32_t>(day));
  info->dates.scale.push_back(scale);
  info->dates.offset.push_back(offset);
  return Status::Ok();
}

// Parses "geotransform a b c d e f".
Status ParseGeotransform(const std::string& path, int line,
                         const std::string& value, CubeInfo* info) {
  const std::vector<std::string> fields = SplitWhitespace(value);
  if (fields.size() != 6) {
    return LineError(path, line, "geotransform needs 6 numbers");
  }
  for (size_t i = 0; i < 6; ++i) {
    if (!ParseDouble(fields[i], &info->geotransform[i])) {
      return LineError(path, line,
                       "bad geotransform value '" + fields[i] + "'");
    }
  }
  info->has_geo = true;
  return Status::Ok();
}

// Parses a positive integer value that fits in an int.
Status ParsePositiveInt(const std::string& path, int line,
                        const std::string& key, const std::string& value,
                        int* out) {
  int64_t parsed = 0;
  if (!ParseInt(value, &parsed) || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return LineError(path, line,
                     key + " must be a positive integer, got '" + value + "'");
  }
  *out = static_cast<int>(parsed);
  return Status::Ok();
}

// Resolves a manifest file name against the manifest directory.
std::string ResolvePath(const fs::path& dir, const std::string& value) {
  fs::path file(value);
  if (file.is_relative()) {
    file = dir / file;
  }
  return file.lexically_normal().string();
}

// Applies one "key value" record to `info`. Returns true in *known when the
// key is part of the grammar.
Status ApplyKey(const std::string& path, int line, const std::string& key,
                const std::string& value, CubeInfo* info, bool* known) {
  *known = true;
  if (key == "name") {
    info->name = value;
  } else if (key == "source") {
    info->source = value;
  } else if (key == "license") {
    info->license = value;
  } else if (key == "crs") {
    info->crs = value;
  } else if (key == "reference_source") {
    info->reference_source = value;
  } else if (key == "geotransform") {
    return ParseGeotransform(path, line, value, info);
  } else if (key == "height") {
    return ParsePositiveInt(path, line, key, value, &info->height);
  } else if (key == "width") {
    return ParsePositiveInt(path, line, key, value, &info->width);
  } else if (key == "num_dates") {
    return ParsePositiveInt(path, line, key, value, &info->num_dates);
  } else if (key == "red_file") {
    info->red_file = value;
  } else if (key == "nir_file") {
    info->nir_file = value;
  } else if (key == "scl_file") {
    info->scl_file = value;
  } else if (key == "reference_file") {
    info->reference_file = value;
  } else if (key == "sha256") {
    // Checksums are verified by the Python tooling, not by this reader.
  } else {
    *known = false;
  }
  return Status::Ok();
}

// ---------------------------------------------------------------------------
// Tile requests shared by both CubeSource implementations.

// Validates a tile request. `dates` may be null, meaning all dates 0..T-1 (and
// then `n` must equal num_dates).
Status CheckTileRequest(const CubeInfo& info, int row0, int rows,
                        const int* dates, int n, const TileView* out) {
  if (out == nullptr || out->red == nullptr || out->nir == nullptr ||
      out->scl == nullptr) {
    return Status::Error("tile buffers are not set");
  }
  if (row0 < 0 || rows <= 0 || row0 > info.height - rows) {
    return Status::Error("tile rows [" + std::to_string(row0) + ", " +
                         std::to_string(static_cast<int64_t>(row0) + rows) +
                         ") are outside the cube (height " +
                         std::to_string(info.height) + ")");
  }
  if (n <= 0) {
    return Status::Error("no dates requested");
  }
  if (dates != nullptr) {
    for (int i = 0; i < n; ++i) {
      if (dates[i] < 0 || dates[i] >= info.num_dates) {
        return Status::Error("date index " + std::to_string(dates[i]) +
                             " is outside 0.." +
                             std::to_string(info.num_dates - 1));
      }
    }
  }
  return Status::Ok();
}

// ---------------------------------------------------------------------------
// Raw-file cube source.

class RawCube : public CubeSource {
 public:
  RawCube(const CubeInfo& info, int io_threads)
      : info_(info), io_threads_(io_threads < 1 ? 1 : io_threads) {}
  RawCube(const RawCube&) = delete;
  RawCube& operator=(const RawCube&) = delete;

  ~RawCube() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    work_ready_.notify_all();
    for (std::thread& worker : workers_) {
      worker.join();
    }
    for (const int fd : {red_fd_, nir_fd_, scl_fd_}) {
      if (fd >= 0) {
        close(fd);
      }
    }
  }

  // Validates sizes, opens the band files once and starts the reader pool
  // (only when more than one I/O thread is requested).
  Status Open() {
    Status status = CheckCubeFiles(info_);
    if (!status.ok()) {
      return status;
    }
    status = OpenFile(info_.red_file, &red_fd_);
    if (!status.ok()) {
      return status;
    }
    status = OpenFile(info_.nir_file, &nir_fd_);
    if (!status.ok()) {
      return status;
    }
    status = OpenFile(info_.scl_file, &scl_fd_);
    if (!status.ok()) {
      return status;
    }
    if (io_threads_ > 1) {
      workers_.reserve(static_cast<size_t>(io_threads_));
      for (int i = 0; i < io_threads_; ++i) {
        workers_.emplace_back([this] { WorkerLoop(); });
      }
    }
    return Status::Ok();
  }

  const CubeInfo& info() const override { return info_; }

  Status ReadTile(int row0, int rows, TileView* out) override {
    return ReadPlanes(row0, rows, nullptr, info_.num_dates, out);
  }

  Status ReadTileDates(int row0, int rows, const int* dates, int n,
                       TileView* out) override {
    if (dates == nullptr) {
      return Status::Error("date list is null");
    }
    return ReadPlanes(row0, rows, dates, n, out);
  }

 private:
  static Status OpenFile(const std::string& path, int* fd) {
    *fd = open(path.c_str(), O_RDONLY);
    if (*fd < 0) {
      return Status::Error("cannot open '" + path + "': " + ErrnoText());
    }
    // The cube is read front to back, tile after tile.
    posix_fadvise(*fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    return Status::Ok();
  }

  // One pread per date and band: rows * W contiguous elements starting at
  // (t * H * W + row0 * W), stored as plane i of the caller's buffer. The
  // reads are numbered job = 3 * plane + band (0 red, 1 nir, 2 scl).
  Status ReadPlanes(int row0, int rows, const int* dates, int n,
                    TileView* out) {
    Status status = CheckTileRequest(info_, row0, rows, dates, n, out);
    if (!status.ok()) {
      return status;
    }
    Batch batch;
    batch.row0 = row0;
    batch.rows = rows;
    batch.dates = dates;
    batch.out = out;
    batch.num_jobs = static_cast<size_t>(n) * 3;
    if (workers_.empty()) {
      for (size_t job = 0; job < batch.num_jobs && status.ok(); ++job) {
        status = RunJob(batch, job);
      }
    } else {
      status = RunParallel(&batch);
    }
    if (!status.ok()) {
      return status;
    }
    out->row0 = row0;
    out->rows = rows;
    return Status::Ok();
  }

  // The reads of one ReadTile/ReadTileDates call, shared with the workers.
  struct Batch {
    int row0 = 0;
    int rows = 0;
    const int* dates = nullptr;
    TileView* out = nullptr;
    size_t num_jobs = 0;
    std::atomic<size_t> next_job{0};
    size_t jobs_done = 0;               // guarded by mutex_
    Status first_error = Status::Ok();  // guarded by mutex_
  };

  // Performs read number `job` of `batch`; every job writes a disjoint part of
  // the caller's buffers.
  Status RunJob(const Batch& batch, size_t job) const {
    const size_t width = static_cast<size_t>(info_.width);
    const size_t plane_cells = static_cast<size_t>(info_.height) * width;
    const size_t tile_cells = static_cast<size_t>(batch.rows) * width;
    const size_t plane = job / 3;
    const size_t t = static_cast<size_t>(
        batch.dates != nullptr ? batch.dates[plane] : static_cast<int>(plane));
    const size_t first_cell =
        t * plane_cells + static_cast<size_t>(batch.row0) * width;
    const size_t destination = plane * tile_cells;
    switch (job % 3) {
      case 0:
        return PreadFully(red_fd_, info_.red_file, batch.out->red + destination,
                          tile_cells * 2, static_cast<off_t>(first_cell * 2));
      case 1:
        return PreadFully(nir_fd_, info_.nir_file, batch.out->nir + destination,
                          tile_cells * 2, static_cast<off_t>(first_cell * 2));
      default:
        return PreadFully(scl_fd_, info_.scl_file, batch.out->scl + destination,
                          tile_cells, static_cast<off_t>(first_cell));
    }
  }

  // Hands `batch` to the pool and waits until every job has finished or
  // failed. Returns the first error of any worker.
  Status RunParallel(Batch* batch) {
    std::unique_lock<std::mutex> lock(mutex_);
    batch_ = batch;
    lock.unlock();
    work_ready_.notify_all();
    lock.lock();
    batch_done_.wait(lock,
                     [batch] { return batch->jobs_done == batch->num_jobs; });
    batch_ = nullptr;
    return batch->first_error;
  }

  // Worker thread: takes jobs of the current batch until none is left. After
  // an error the remaining jobs are counted but skipped.
  void WorkerLoop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      work_ready_.wait(lock, [this] {
        return stop_ || (batch_ != nullptr &&
                         batch_->next_job.load() < batch_->num_jobs);
      });
      if (stop_) {
        return;
      }
      Batch* const batch = batch_;
      const size_t job = batch->next_job.fetch_add(1);
      if (job >= batch->num_jobs) {
        continue;
      }
      const bool skip = !batch->first_error.ok();
      lock.unlock();
      Status status = skip ? Status::Ok() : RunJob(*batch, job);
      lock.lock();
      if (!status.ok() && batch->first_error.ok()) {
        batch->first_error = status;
      }
      ++batch->jobs_done;
      if (batch->jobs_done == batch->num_jobs) {
        batch_done_.notify_all();
      }
    }
  }

  CubeInfo info_;
  const int io_threads_;
  int red_fd_ = -1;
  int nir_fd_ = -1;
  int scl_fd_ = -1;

  // Persistent reader pool, created in Open and joined in the destructor.
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_;  // a batch was posted or stop_ is set
  std::condition_variable batch_done_;  // the last job of a batch finished
  Batch* batch_ = nullptr;              // guarded by mutex_
  bool stop_ = false;                   // guarded by mutex_
};

// ---------------------------------------------------------------------------
// In-memory cube source.

class MemoryCube : public CubeSource {
 public:
  explicit MemoryCube(SyntheticCube cube) : cube_(std::move(cube)) {}

  const CubeInfo& info() const override { return cube_.info; }

  Status ReadTile(int row0, int rows, TileView* out) override {
    return CopyPlanes(row0, rows, nullptr, cube_.info.num_dates, out);
  }

  Status ReadTileDates(int row0, int rows, const int* dates, int n,
                       TileView* out) override {
    if (dates == nullptr) {
      return Status::Error("date list is null");
    }
    return CopyPlanes(row0, rows, dates, n, out);
  }

 private:
  // memcpy of the same slices the raw reader preads.
  Status CopyPlanes(int row0, int rows, const int* dates, int n,
                    TileView* out) {
    const CubeInfo& info = cube_.info;
    Status status = CheckTileRequest(info, row0, rows, dates, n, out);
    if (!status.ok()) {
      return status;
    }
    const size_t width = static_cast<size_t>(info.width);
    const size_t plane_cells = static_cast<size_t>(info.height) * width;
    const size_t cells = plane_cells * static_cast<size_t>(info.num_dates);
    if (cube_.red.size() != cells || cube_.nir.size() != cells ||
        cube_.scl.size() != cells) {
      return Status::Error("memory cube arrays do not have " +
                           std::to_string(cells) + " elements");
    }
    const size_t tile_cells = static_cast<size_t>(rows) * width;
    for (int i = 0; i < n; ++i) {
      const size_t t = static_cast<size_t>(dates != nullptr ? dates[i] : i);
      const size_t first_cell =
          t * plane_cells + static_cast<size_t>(row0) * width;
      const size_t destination = static_cast<size_t>(i) * tile_cells;
      std::memcpy(out->red + destination, cube_.red.data() + first_cell,
                  tile_cells * sizeof(uint16_t));
      std::memcpy(out->nir + destination, cube_.nir.data() + first_cell,
                  tile_cells * sizeof(uint16_t));
      std::memcpy(out->scl + destination, cube_.scl.data() + first_cell,
                  tile_cells);
    }
    out->row0 = row0;
    out->rows = rows;
    return Status::Ok();
  }

  SyntheticCube cube_;
};

// Path of `file` for writing into a manifest located in `dir`: relative to
// `dir` when it lies inside, otherwise unchanged; `fallback` when empty.
std::string ManifestFileName(const fs::path& dir, const std::string& file,
                             const char* fallback) {
  if (file.empty()) {
    return fallback;
  }
  const fs::path path(file);
  if (!path.is_absolute()) {
    return file;
  }
  const fs::path relative = path.lexically_relative(dir);
  if (relative.empty() || *relative.begin() == "..") {
    return file;
  }
  return relative.string();
}

}  // namespace

Status ReadManifest(const std::string& path, CubeInfo* info) {
  std::ifstream stream(path);
  if (!stream) {
    return Status::Error("cannot open manifest '" + path + "'");
  }
  CubeInfo parsed;
  std::string text;
  int line = 0;
  if (!std::getline(stream, text) || Trim(text) != kMagicLine) {
    return LineError(path, 1,
                     std::string("first line must be '") + kMagicLine + "'");
  }
  line = 1;
  std::set<std::string> seen;
  while (std::getline(stream, text)) {
    ++line;
    const std::string record = Trim(text);
    if (record.empty() || record[0] == '#') {
      continue;
    }
    const size_t split = record.find(' ');
    const std::string key = record.substr(0, split);
    const std::string value =
        split == std::string::npos ? std::string() : Trim(record.substr(split));
    if (key == "date") {
      const Status status =
          ParseDateLine(path, line, SplitWhitespace(record), &parsed);
      if (!status.ok()) {
        return status;
      }
      continue;
    }
    if (key != "sha256" && !seen.insert(key).second) {
      return LineError(path, line, "duplicate key '" + key + "'");
    }
    bool known = false;
    const Status status = ApplyKey(path, line, key, value, &parsed, &known);
    if (!status.ok()) {
      return status;
    }
    if (!known) {
      Log("warning: " + path + ":" + std::to_string(line) + ": unknown key '" +
          key + "' ignored");
    } else if (value.empty() && key != "source" && key != "license" &&
               key != "crs" && key != "reference_source" && key != "sha256") {
      return LineError(path, line, "key '" + key + "' has no value");
    }
  }

  for (const char* const key : {"name", "height", "width", "num_dates",
                                "red_file", "nir_file", "scl_file"}) {
    if (seen.count(key) == 0) {
      return Status::Error(path + ": missing required key '" + key + "'");
    }
  }
  if (parsed.num_dates > kMaxDates) {
    return Status::Error(
        path + ": num_dates " + std::to_string(parsed.num_dates) +
        " exceeds the supported maximum " + std::to_string(kMaxDates));
  }
  if (static_cast<int>(parsed.dates.day.size()) != parsed.num_dates) {
    return Status::Error(path + ": num_dates is " +
                         std::to_string(parsed.num_dates) + " but found " +
                         std::to_string(parsed.dates.day.size()) +
                         " date lines");
  }

  const fs::path dir = fs::absolute(fs::path(path)).parent_path();
  parsed.dir = dir.lexically_normal().string();
  parsed.red_file = ResolvePath(dir, parsed.red_file);
  parsed.nir_file = ResolvePath(dir, parsed.nir_file);
  parsed.scl_file = ResolvePath(dir, parsed.scl_file);
  if (!parsed.reference_file.empty()) {
    parsed.reference_file = ResolvePath(dir, parsed.reference_file);
  }
  const Status status = CheckCubeFiles(parsed);
  if (!status.ok()) {
    return Status::Error(path + ": " + status.message());
  }
  *info = std::move(parsed);
  return Status::Ok();
}

Status WriteManifest(const std::string& path, const CubeInfo& info) {
  return WriteManifestWithDateExtras(path, info, std::vector<double>(),
                                     std::string());
}

Status WriteManifestWithDateExtras(const std::string& path,
                                   const CubeInfo& info,
                                   const std::vector<double>& valid_fraction,
                                   const std::string& scene_id) {
  const size_t num_dates = static_cast<size_t>(info.num_dates);
  const DateTable& dates = info.dates;
  if (info.num_dates <= 0 || dates.iso.size() != num_dates ||
      dates.day.size() != num_dates || dates.scale.size() != num_dates ||
      dates.offset.size() != num_dates) {
    return Status::Error("cannot write manifest: date table does not have " +
                         std::to_string(info.num_dates) + " entries");
  }
  if (!valid_fraction.empty() && valid_fraction.size() != num_dates) {
    return Status::Error("cannot write manifest: valid_fraction needs " +
                         std::to_string(num_dates) + " entries");
  }
  if (scene_id.find_first_of(" \t\r\n") != std::string::npos) {
    return Status::Error("cannot write manifest: scene id contains spaces");
  }
  for (const std::string* text : {&info.name, &info.source, &info.license,
                                  &info.crs, &info.reference_source}) {
    if (text->find_first_of("\r\n") != std::string::npos) {
      return Status::Error("cannot write manifest: a value contains a newline");
    }
  }

  const fs::path dir = fs::absolute(fs::path(path)).parent_path();
  std::ostringstream out;
  out << kMagicLine << "\n";
  out << "name " << info.name << "\n";
  if (!info.source.empty()) out << "source " << info.source << "\n";
  if (!info.license.empty()) out << "license " << info.license << "\n";
  if (!info.crs.empty()) out << "crs " << info.crs << "\n";
  if (info.has_geo) {
    out << "geotransform";
    for (const double value : info.geotransform) {
      out << " " << FormatDouble(value);
    }
    out << "\n";
  }
  out << "height " << info.height << "\n";
  out << "width " << info.width << "\n";
  out << "num_dates " << info.num_dates << "\n";
  out << "red_file " << ManifestFileName(dir, info.red_file, kDefaultRed)
      << "\n";
  out << "nir_file " << ManifestFileName(dir, info.nir_file, kDefaultNir)
      << "\n";
  out << "scl_file " << ManifestFileName(dir, info.scl_file, kDefaultScl)
      << "\n";
  if (!info.reference_file.empty()) {
    out << "reference_file " << ManifestFileName(dir, info.reference_file, "")
        << "\n";
  }
  if (!info.reference_source.empty()) {
    out << "reference_source " << info.reference_source << "\n";
  }
  out << "# date <index> <YYYY-MM-DD> <days_since_1970> <scale> <offset> "
         "<valid_fraction> <scene_id>\n";
  for (size_t t = 0; t < num_dates; ++t) {
    out << "date " << t << " " << dates.iso[t] << " " << dates.day[t] << " "
        << FormatFloat(dates.scale[t]) << " " << FormatFloat(dates.offset[t])
        << " "
        << (valid_fraction.empty() ? std::string("-1")
                                   : FormatDouble(valid_fraction[t]))
        << " " << (scene_id.empty() ? std::string("-") : scene_id) << "\n";
  }
  const std::string text = out.str();
  return WriteRawFile(path, text.data(), text.size());
}

std::unique_ptr<CubeSource> OpenRawCube(const CubeInfo& info, Status* status) {
  return OpenRawCube(info, 1, status);
}

std::unique_ptr<CubeSource> OpenRawCube(const CubeInfo& info, int io_threads,
                                        Status* status) {
  Status local = Status::Ok();
  if (status == nullptr) {
    status = &local;
  }
  if (info.height <= 0 || info.width <= 0 || info.num_dates <= 0 ||
      info.num_dates > kMaxDates) {
    *status = Status::Error(
        "invalid cube dimensions " + std::to_string(info.num_dates) + " x " +
        std::to_string(info.height) + " x " + std::to_string(info.width));
    return nullptr;
  }
  auto cube = std::make_unique<RawCube>(info, io_threads);
  *status = cube->Open();
  if (!status->ok()) {
    return nullptr;
  }
  return cube;
}

std::unique_ptr<CubeSource> MakeMemoryCube(SyntheticCube cube) {
  return std::make_unique<MemoryCube>(std::move(cube));
}

Status ReadReferenceRaster(const CubeInfo& info,
                           std::vector<uint8_t>* reference) {
  if (info.reference_file.empty()) {
    return Status::Error("cube '" + info.name + "' has no reference raster");
  }
  return ReadRawArray(
      info.reference_file,
      static_cast<size_t>(info.height) * static_cast<size_t>(info.width),
      reference);
}

Status ReadRawBytes(const std::string& path, size_t bytes, void* out) {
  Status status = CheckFileSize(path, "file", bytes);
  if (!status.ok()) {
    return status;
  }
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return Status::Error("cannot open '" + path + "': " + ErrnoText());
  }
  status = PreadFully(fd, path, out, bytes, 0);
  close(fd);
  return status;
}

Status WriteRawFile(const std::string& path, const void* data, size_t bytes) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return Status::Error("cannot create '" + path + "': " + ErrnoText());
  }
  const bool written = bytes == 0 || std::fwrite(data, 1, bytes, file) == bytes;
  const bool closed = std::fclose(file) == 0;
  if (!written || !closed) {
    return Status::Error("write to '" + path + "' failed: " + ErrnoText());
  }
  return Status::Ok();
}

}  // namespace cd
