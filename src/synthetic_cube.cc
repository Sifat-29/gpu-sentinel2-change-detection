// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "synthetic_cube.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17): portable paths and mkdir -p
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cube_io.h"

namespace cd {
namespace {

namespace fs = std::filesystem;

constexpr double kPi = 3.14159265358979323846;

// 2019-01-01 as days since 1970-01-01.
constexpr int kFirstDay = 17897;
// Dates strictly after 2022-01-25 (day 19017) use the Collection-1 offset.
constexpr int kLastDayWithoutOffset = 19017;
constexpr float kScale = 0.0001f;
constexpr float kLaterOffset = -0.1f;
// Fraction of observations that are nodata (DN 0, SCL 0).
constexpr double kNodataFraction = 0.005;
// Valid observations are tagged with SCL class 4 (vegetation), clouds with 9.
constexpr uint8_t kSclVegetation = 4;
constexpr uint8_t kSclCloud = 9;
constexpr double kNdviMin = -0.1;
constexpr double kNdviMax = 0.95;
// Rows handed to a worker thread at a time.
constexpr int kRowBlock = 8;

// Hash channels: every random number is a pure function of
// (seed, pixel, date, channel).
enum Channel : uint32_t {
  kChBreakSelect = 1,
  kChBreakDate,
  kChBreakSign,
  kChBreakMagnitude,
  kChBase,
  kChSlope,
  kChAmplitude1,
  kChPhase1,
  kChAmplitude2,
  kChPhase2,
  kChEvent,
  kChNoiseA,
  kChNoiseB,
  kChCloudRed,
  kChCloudNir
};

// Date index used for per-pixel (date independent) draws.
constexpr uint32_t kPixelDraw = 0xFFFFFFFFu;

// SplitMix64 finaliser.
uint64_t Mix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Uniform number in (0, 1), a pure function of its arguments.
double Uniform(uint32_t seed, int64_t pixel, uint32_t date, uint32_t channel) {
  uint64_t h = Mix((static_cast<uint64_t>(seed) << 32) | channel);
  h = Mix(h + static_cast<uint64_t>(pixel));
  h = Mix(h + date);
  return (static_cast<double>(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// Converts days since 1970-01-01 to "YYYY-MM-DD" (proleptic Gregorian,
// Howard Hinnant's civil_from_days).
std::string IsoFromDays(int days) {
  const int z = days + 719468;
  const int era = (z >= 0 ? z : z - 146096) / 146097;
  const int day_of_era = z - era * 146097;
  const int year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36524 -
                           day_of_era / 146096) /
                          365;
  const int day_of_year =
      day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
  const int mp = (5 * day_of_year + 2) / 153;
  const int day = day_of_year - (153 * mp + 2) / 5 + 1;
  const int month = mp < 10 ? mp + 3 : mp - 9;
  const int year = year_of_era + era * 400 + (month <= 2 ? 1 : 0);
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d", year, month, day);
  return buffer;
}

int ResolveThreads(int threads, int work_items) {
  int count = threads > 0
                  ? threads
                  : static_cast<int>(std::thread::hardware_concurrency());
  count = std::max(count, 1);
  return std::max(1, std::min(count, work_items));
}

// Calls fn(row_begin, row_end) for blocks of rows on `threads` workers.
template <typename Fn>
void ParallelRows(int threads, int height, const Fn& fn) {
  const int blocks = (height + kRowBlock - 1) / kRowBlock;
  const int workers = ResolveThreads(threads, blocks);
  std::atomic<int> next_block(0);
  const auto work = [&]() {
    for (int block = next_block.fetch_add(1); block < blocks;
         block = next_block.fetch_add(1)) {
      fn(block * kRowBlock, std::min(height, (block + 1) * kRowBlock));
    }
  };
  if (workers == 1) {
    work();
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers));
  for (int i = 0; i < workers; ++i) {
    pool.emplace_back(work);
  }
  for (std::thread& worker : pool) {
    worker.join();
  }
}

// Validates the options and fills the metadata of a synthetic cube.
Status BuildInfo(const SyntheticOptions& options, CubeInfo* info) {
  if (options.height <= 0 || options.width <= 0) {
    return Status::Error("synthetic cube: height and width must be positive");
  }
  if (options.num_dates <= 0 || options.num_dates > kMaxDates) {
    return Status::Error("synthetic cube: num_dates must be in 1.." +
                         std::to_string(kMaxDates));
  }
  if (!(options.years > 0.0f) || !(options.train_years > 0.0f)) {
    return Status::Error("synthetic cube: years and train_years must be > 0");
  }
  if (options.noise < 0.0f || !(options.cloud_fraction >= 0.0f) ||
      options.cloud_fraction > 1.0f || !(options.break_fraction >= 0.0f) ||
      options.break_fraction > 1.0f) {
    return Status::Error(
        "synthetic cube: noise must be >= 0 and the cloud and break "
        "fractions must be in [0, 1]");
  }
  if (options.consecutive < 1) {
    return Status::Error("synthetic cube: consecutive must be >= 1");
  }
  CubeInfo built;
  built.name = "synthetic_" + std::to_string(options.height) + "x" +
               std::to_string(options.width) + "x" +
               std::to_string(options.num_dates) + "_seed" +
               std::to_string(options.seed);
  built.source = "synthetic";
  built.license = "none (synthetic data generated by change_detect)";
  built.height = options.height;
  built.width = options.width;
  built.num_dates = options.num_dates;
  DateTable& dates = built.dates;
  for (int i = 0; i < options.num_dates; ++i) {
    const int day =
        kFirstDay + static_cast<int>(std::llround(
                        i * 365.25 * static_cast<double>(options.years) /
                        options.num_dates));
    if (!dates.day.empty() && day <= dates.day.back()) {
      return Status::Error(
          "synthetic cube: too many dates for the time span (dates must be "
          "at least one day apart)");
    }
    dates.day.push_back(day);
    dates.iso.push_back(IsoFromDays(day));
    dates.scale.push_back(kScale);
    dates.offset.push_back(day > kLastDayWithoutOffset ? kLaterOffset : 0.0f);
  }
  ModelParams params;
  params.train_years = options.train_years;
  const Status status = BuildDateTable(params, &dates);
  if (!status.ok()) {
    return Status::Error("synthetic cube: " + status.message());
  }
  if (dates.num_train_dates > options.num_dates - options.consecutive) {
    return Status::Error("synthetic cube: no room for a break with " +
                         std::to_string(options.consecutive) +
                         " dates after it inside the monitoring period");
  }
  *info = std::move(built);
  return Status::Ok();
}

// Per-date terms of the seasonal model.
struct DateTerms {
  double tau = 0;
  double cos1 = 0;
  double sin1 = 0;
  double cos2 = 0;
  double sin2 = 0;
  double offset = 0;
  double scale = 0;
};

// Per-pixel parameters of the NDVI model.
struct PixelModel {
  int truth = -1;
  double base = 0;
  double slope = 0;
  double a1 = 0;
  double b1 = 0;
  double a2 = 0;
  double b2 = 0;
  double step = 0;
};

// Generates observations of the synthetic cube.
//
// NDVI model of pixel p at date t (tau = years since the first date):
//   ndvi = base + slope * tau + a1 cos(2 pi tau) + b1 sin(2 pi tau)
//          + a2 cos(4 pi tau) + b2 sin(4 pi tau)
//          [+ step for t >= truth_break_index]            (injected break)
//          [+ N(0, noise)]                                (observation noise)
// clamped to [-0.1, 0.95]. base is 0.55 + U(-0.04, 0.04) (0.25 lower for
// pixels that get a positive step, so that the gain fits below 0.95), the
// annual amplitude is U(0.05, 0.20), the semi-annual one U(0, 0.05) and the
// total trend over the run at most 0.05, so the clean model stays inside
// [0.17, 0.89] (-0.08 .. 0.64 for gain pixels, before the step).
//
// NDVI -> reflectance (the "(red, nir) mapping"): nir = 0.30 + 0.1 * ndvi and
// red = nir * (1 - ndvi) / (1 + ndvi), so (nir - red) / (nir + red) equals
// ndvi exactly before quantisation. DN = round((reflectance - offset) /
// scale), at least 1 because DN 0 means nodata.
class Generator {
 public:
  Generator(const SyntheticOptions& options, const DateTable& dates)
      : options_(options), num_train_(dates.num_train_dates) {
    terms_.resize(dates.day.size());
    for (size_t t = 0; t < dates.day.size(); ++t) {
      DateTerms& d = terms_[t];
      d.tau = static_cast<double>(dates.day[t] - dates.day[0]) / 365.25;
      d.cos1 = std::cos(2.0 * kPi * d.tau);
      d.sin1 = std::sin(2.0 * kPi * d.tau);
      d.cos2 = std::cos(4.0 * kPi * d.tau);
      d.sin2 = std::sin(4.0 * kPi * d.tau);
      d.offset = static_cast<double>(dates.offset[t]);
      d.scale = static_cast<double>(dates.scale[t]);
    }
  }

  // Injected break index of `pixel`, or -1. A break lies in the monitoring
  // period with at least `consecutive` dates (itself included) remaining.
  int TruthIndex(int64_t pixel) const {
    const uint32_t seed = options_.seed;
    if (Uniform(seed, pixel, kPixelDraw, kChBreakSelect) >=
        static_cast<double>(options_.break_fraction)) {
      return -1;
    }
    const int count =
        static_cast<int>(terms_.size()) - options_.consecutive - num_train_ + 1;
    const int offset = static_cast<int>(
        Uniform(seed, pixel, kPixelDraw, kChBreakDate) * count);
    return num_train_ + std::min(offset, count - 1);
  }

  PixelModel Model(int64_t pixel) const {
    const uint32_t seed = options_.seed;
    const auto draw = [&](Channel channel) {
      return Uniform(seed, pixel, kPixelDraw, channel);
    };
    PixelModel m;
    m.truth = TruthIndex(pixel);
    const bool gain = m.truth >= 0 && draw(kChBreakSign) >= 0.8;
    m.base = 0.55 + 0.04 * (2.0 * draw(kChBase) - 1.0) - (gain ? 0.25 : 0.0);
    m.slope = (2.0 * draw(kChSlope) - 1.0) * 0.05 /
              static_cast<double>(options_.years);
    const double amplitude1 = 0.05 + 0.15 * draw(kChAmplitude1);
    const double phase1 = 2.0 * kPi * draw(kChPhase1);
    const double amplitude2 = 0.05 * draw(kChAmplitude2);
    const double phase2 = 2.0 * kPi * draw(kChPhase2);
    m.a1 = amplitude1 * std::cos(phase1);
    m.b1 = amplitude1 * std::sin(phase1);
    m.a2 = amplitude2 * std::cos(phase2);
    m.b2 = amplitude2 * std::sin(phase2);
    if (m.truth >= 0) {
      const double magnitude = 0.25 + 0.20 * draw(kChBreakMagnitude);
      m.step = gain ? magnitude : -magnitude;
    }
    return m;
  }

  // Noise-free NDVI of a pixel with model `m` at date `t`.
  double Clean(const PixelModel& m, int t) const {
    const DateTerms& d = terms_[static_cast<size_t>(t)];
    double ndvi = m.base + m.slope * d.tau + m.a1 * d.cos1 + m.b1 * d.sin1 +
                  m.a2 * d.cos2 + m.b2 * d.sin2;
    if (m.truth >= 0 && t >= m.truth) {
      ndvi += m.step;
    }
    return std::min(std::max(ndvi, kNdviMin), kNdviMax);
  }

  // Fills rows [row_begin, row_end) of one date plane; `red`, `nir` and `scl`
  // point at the start of the plane (pixel index = row * width + col).
  void FillRows(int t, int row_begin, int row_end, uint16_t* red, uint16_t* nir,
                uint8_t* scl) const {
    const int64_t width = options_.width;
    for (int64_t pixel = row_begin * width; pixel < row_end * width; ++pixel) {
      Observe(pixel, Model(pixel), t, &red[pixel], &nir[pixel], &scl[pixel]);
    }
  }

 private:
  uint16_t ToDn(double reflectance, const DateTerms& d) const {
    const double dn = std::llround((reflectance - d.offset) / d.scale);
    return static_cast<uint16_t>(std::min(std::max(dn, 1.0), 65535.0));
  }

  // One observation: nodata, cloud or a valid noisy NDVI value.
  void Observe(int64_t pixel, const PixelModel& m, int t, uint16_t* red,
               uint16_t* nir, uint8_t* scl) const {
    const uint32_t seed = options_.seed;
    const uint32_t date = static_cast<uint32_t>(t);
    const DateTerms& d = terms_[static_cast<size_t>(t)];
    const double event = Uniform(seed, pixel, date, kChEvent);
    if (event < kNodataFraction) {
      *red = 0;
      *nir = 0;
      *scl = 0;
      return;
    }
    if (event <
        kNodataFraction + static_cast<double>(options_.cloud_fraction)) {
      // Bright cloud: both bands 0.4 - 0.6.
      *red = ToDn(0.4 + 0.2 * Uniform(seed, pixel, date, kChCloudRed), d);
      *nir = ToDn(0.4 + 0.2 * Uniform(seed, pixel, date, kChCloudNir), d);
      *scl = kSclCloud;
      return;
    }
    double ndvi = Clean(m, t);
    if (options_.noise > 0.0f) {
      // Box-Muller: one Gaussian from two uniforms.
      const double u1 = Uniform(seed, pixel, date, kChNoiseA);
      const double u2 = Uniform(seed, pixel, date, kChNoiseB);
      const double gauss =
          std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
      ndvi =
          std::min(std::max(ndvi + static_cast<double>(options_.noise) * gauss,
                            kNdviMin),
                   kNdviMax);
    }
    const double nir_reflectance = 0.30 + 0.1 * ndvi;
    const double red_reflectance =
        nir_reflectance * (1.0 - ndvi) / (1.0 + ndvi);
    *red = ToDn(red_reflectance, d);
    *nir = ToDn(nir_reflectance, d);
    *scl = kSclVegetation;
  }

  SyntheticOptions options_;
  int num_train_;
  std::vector<DateTerms> terms_;
};

// Counts the valid (SCL 4) pixels of a plane: the manifest's valid_fraction.
double ValidFraction(const uint8_t* scl, size_t count) {
  size_t valid = 0;
  for (size_t i = 0; i < count; ++i) {
    valid += scl[i] == kSclVegetation ? 1 : 0;
  }
  return static_cast<double>(valid) / static_cast<double>(count);
}

void ComputeTruth(const Generator& generator, const SyntheticOptions& options,
                  std::vector<int16_t>* truth) {
  const int64_t width = options.width;
  truth->assign(static_cast<size_t>(width) * options.height, -1);
  ParallelRows(options.threads, options.height, [&](int begin, int end) {
    for (int64_t pixel = begin * width; pixel < end * width; ++pixel) {
      (*truth)[static_cast<size_t>(pixel)] =
          static_cast<int16_t>(generator.TruthIndex(pixel));
    }
  });
}

// RAII wrapper so error paths cannot leak a FILE*.
struct FileCloser {
  void operator()(std::FILE* file) const {
    if (file != nullptr) {
      std::fclose(file);
    }
  }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

Status WritePlane(std::FILE* file, const std::string& path, const void* data,
                  size_t bytes) {
  if (std::fwrite(data, 1, bytes, file) != bytes) {
    return Status::Error("write to '" + path +
                         "' failed: " + std::strerror(errno));
  }
  return Status::Ok();
}

// Closes a file explicitly so that a failing flush is reported.
Status CloseFile(FilePtr* file, const std::string& path) {
  std::FILE* raw = file->release();
  if (std::fclose(raw) != 0) {
    return Status::Error("closing '" + path +
                         "' failed: " + std::strerror(errno));
  }
  return Status::Ok();
}

// Absolute directory path as text, creating the directory when needed.
Status MakeDirectory(const std::string& dir, std::string* absolute) {
  std::error_code error;
  fs::create_directories(dir, error);
  if (error) {
    return Status::Error("cannot create directory '" + dir +
                         "': " + error.message());
  }
  *absolute = fs::absolute(fs::path(dir)).lexically_normal().string();
  return Status::Ok();
}

// Points the file names of `info` at the standard files inside `dir`.
void SetCubeFiles(const std::string& dir, CubeInfo* info) {
  info->dir = dir;
  info->red_file = (fs::path(dir) / "red.u16").string();
  info->nir_file = (fs::path(dir) / "nir.u16").string();
  info->scl_file = (fs::path(dir) / "scl.u8").string();
}

}  // namespace

Status GenerateSyntheticCube(const SyntheticOptions& options,
                             SyntheticCube* out) {
  SyntheticCube cube;
  Status status = BuildInfo(options, &cube.info);
  if (!status.ok()) {
    return status;
  }
  const Generator generator(options, cube.info.dates);
  const size_t plane =
      static_cast<size_t>(options.height) * static_cast<size_t>(options.width);
  const size_t cells = plane * static_cast<size_t>(options.num_dates);
  cube.red.assign(cells, 0);
  cube.nir.assign(cells, 0);
  cube.scl.assign(cells, 0);
  ComputeTruth(generator, options, &cube.truth_break_index);
  for (int t = 0; t < options.num_dates; ++t) {
    const size_t first = static_cast<size_t>(t) * plane;
    ParallelRows(options.threads, options.height, [&](int begin, int end) {
      generator.FillRows(t, begin, end, cube.red.data() + first,
                         cube.nir.data() + first, cube.scl.data() + first);
    });
  }
  *out = std::move(cube);
  return Status::Ok();
}

Status WriteSyntheticCube(const SyntheticCube& cube, const std::string& dir) {
  const CubeInfo& info = cube.info;
  const size_t plane =
      static_cast<size_t>(info.height) * static_cast<size_t>(info.width);
  const size_t cells = plane * static_cast<size_t>(info.num_dates);
  if (cube.red.size() != cells || cube.nir.size() != cells ||
      cube.scl.size() != cells || cube.truth_break_index.size() != plane) {
    return Status::Error("synthetic cube arrays do not match its dimensions");
  }
  std::string absolute;
  Status status = MakeDirectory(dir, &absolute);
  if (!status.ok()) {
    return status;
  }
  CubeInfo written = info;
  SetCubeFiles(absolute, &written);
  status = WriteRawFile(written.red_file, cube.red.data(), cells * 2);
  if (!status.ok()) {
    return status;
  }
  status = WriteRawFile(written.nir_file, cube.nir.data(), cells * 2);
  if (!status.ok()) {
    return status;
  }
  status = WriteRawFile(written.scl_file, cube.scl.data(), cells);
  if (!status.ok()) {
    return status;
  }
  status = WriteRawFile((fs::path(absolute) / "truth_break_index.i16").string(),
                        cube.truth_break_index.data(), plane * 2);
  if (!status.ok()) {
    return status;
  }
  std::vector<double> valid(static_cast<size_t>(info.num_dates));
  for (size_t t = 0; t < valid.size(); ++t) {
    valid[t] = ValidFraction(cube.scl.data() + t * plane, plane);
  }
  return WriteManifestWithDateExtras(
      (fs::path(absolute) / "manifest.txt").string(), written, valid,
      "synthetic");
}

Status GenerateSyntheticCubeToDisk(const SyntheticOptions& options,
                                   const std::string& dir) {
  CubeInfo info;
  Status status = BuildInfo(options, &info);
  if (!status.ok()) {
    return status;
  }
  std::string absolute;
  status = MakeDirectory(dir, &absolute);
  if (!status.ok()) {
    return status;
  }
  SetCubeFiles(absolute, &info);
  const Generator generator(options, info.dates);
  const size_t plane =
      static_cast<size_t>(options.height) * static_cast<size_t>(options.width);

  FilePtr red_file(std::fopen(info.red_file.c_str(), "wb"));
  FilePtr nir_file(std::fopen(info.nir_file.c_str(), "wb"));
  FilePtr scl_file(std::fopen(info.scl_file.c_str(), "wb"));
  if (!red_file || !nir_file || !scl_file) {
    return Status::Error("cannot create cube files in '" + absolute +
                         "': " + std::strerror(errno));
  }

  // One date plane of each band at a time: O(H * W) memory.
  std::vector<uint16_t> red(plane);
  std::vector<uint16_t> nir(plane);
  std::vector<uint8_t> scl(plane);
  std::vector<double> valid(static_cast<size_t>(options.num_dates));
  for (int t = 0; t < options.num_dates; ++t) {
    ParallelRows(options.threads, options.height, [&](int begin, int end) {
      generator.FillRows(t, begin, end, red.data(), nir.data(), scl.data());
    });
    valid[static_cast<size_t>(t)] = ValidFraction(scl.data(), plane);
    status = WritePlane(red_file.get(), info.red_file, red.data(), plane * 2);
    if (status.ok()) {
      status = WritePlane(nir_file.get(), info.nir_file, nir.data(), plane * 2);
    }
    if (status.ok()) {
      status = WritePlane(scl_file.get(), info.scl_file, scl.data(), plane);
    }
    if (!status.ok()) {
      return status;
    }
  }
  for (const auto& entry : {std::make_pair(&red_file, &info.red_file),
                            std::make_pair(&nir_file, &info.nir_file),
                            std::make_pair(&scl_file, &info.scl_file)}) {
    status = CloseFile(entry.first, *entry.second);
    if (!status.ok()) {
      return status;
    }
  }

  std::vector<int16_t> truth;
  ComputeTruth(generator, options, &truth);
  status = WriteRawFile((fs::path(absolute) / "truth_break_index.i16").string(),
                        truth.data(), plane * 2);
  if (!status.ok()) {
    return status;
  }
  return WriteManifestWithDateExtras(
      (fs::path(absolute) / "manifest.txt").string(), info, valid, "synthetic");
}

double SyntheticCleanNdvi(const SyntheticOptions& options, int64_t pixel,
                          int date) {
  CubeInfo info;
  if (!BuildInfo(options, &info).ok() || date < 0 ||
      date >= options.num_dates) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const Generator generator(options, info.dates);
  return generator.Clean(generator.Model(pixel), date);
}

Status ComputeRecovery(CubeSource* source, const std::vector<int16_t>& truth,
                       const DetectionResult& result, const ModelParams& params,
                       RecoveryStats* out) {
  const CubeInfo& info = source->info();
  const size_t height = static_cast<size_t>(info.height);
  const size_t width = static_cast<size_t>(info.width);
  const size_t num_dates = static_cast<size_t>(info.num_dates);
  const size_t pixels = height * width;
  if (truth.size() != pixels || result.status.size() != pixels ||
      result.break_index.size() != pixels) {
    return Status::Error("truth and result maps must have " +
                         std::to_string(pixels) +
                         " elements (height x width of the cube)");
  }
  if (info.dates.day.size() != num_dates) {
    return Status::Error("cube date table does not match num_dates");
  }
  const int min_train_obs = params.min_train_obs > 0
                                ? params.min_train_obs
                                : 3 * NumParams(params.harmonics);
  // Training dates: the leading dates with tau < train_years.
  size_t num_train = 0;
  while (num_train < num_dates &&
         static_cast<double>(info.dates.day[num_train] - info.dates.day[0]) /
                 365.25 <
             static_cast<double>(params.train_years)) {
    ++num_train;
  }

  // Tiles of about 64 MiB of input keep the memory use small.
  const size_t row_bytes = width * num_dates * 5;
  const size_t tile_rows = std::max<size_t>(
      1,
      std::min<size_t>(height, (64ull << 20) / std::max<size_t>(row_bytes, 1)));
  std::vector<uint16_t> red(tile_rows * width * num_dates);
  std::vector<uint16_t> nir(red.size());
  std::vector<uint8_t> scl(red.size());
  std::vector<uint8_t> valid(num_dates);

  RecoveryStats stats;
  for (size_t row0 = 0; row0 < height; row0 += tile_rows) {
    const size_t rows = std::min(tile_rows, height - row0);
    // Only tiles with an injected break need their observations.
    bool needs_cube = false;
    for (size_t i = row0 * width; i < (row0 + rows) * width; ++i) {
      needs_cube = needs_cube || truth[i] >= 0;
    }
    if (needs_cube) {
      TileView view;
      view.red = red.data();
      view.nir = nir.data();
      view.scl = scl.data();
      const Status status = source->ReadTile(static_cast<int>(row0),
                                             static_cast<int>(rows), &view);
      if (!status.ok()) {
        return status;
      }
    }
    const size_t tile_pixels = rows * width;
    for (size_t q = 0; q < tile_pixels; ++q) {
      const size_t pixel = row0 * width + q;
      const int injected = truth[pixel];
      if (injected < 0) {
        ++stats.stable_truth;
        if (result.status[pixel] == static_cast<uint8_t>(PixelStatus::kBreak)) {
          ++stats.false_breaks;
        }
        continue;
      }
      ++stats.injected;
      if (static_cast<size_t>(injected) >= num_dates) {
        return Status::Error("truth index " + std::to_string(injected) +
                             " is outside the cube's dates");
      }
      // Validity of every observation of this pixel.
      size_t train_valid = 0;
      for (size_t t = 0; t < num_dates; ++t) {
        const size_t cell = t * tile_pixels + q;
        const uint8_t code = scl[cell];
        const bool usable = code < 32 &&
                            ((params.valid_class_mask >> code) & 1u) != 0 &&
                            red[cell] != 0 && nir[cell] != 0;
        valid[t] = usable ? 1 : 0;
        train_valid += (usable && t < num_train) ? 1 : 0;
      }
      if (train_valid < static_cast<size_t>(min_train_obs)) {
        ++stats.insufficient_training;
        continue;
      }
      // First valid observation at/after the injected index and the next one.
      const size_t first = static_cast<size_t>(injected);
      size_t valid_after = 0;
      size_t second = num_dates;
      size_t first_valid = num_dates;
      for (size_t t = first; t < num_dates; ++t) {
        if (valid[t] == 0) {
          continue;
        }
        ++valid_after;
        if (first_valid == num_dates) {
          first_valid = t;
        } else if (second == num_dates) {
          second = t;
        }
      }
      if (valid_after < static_cast<size_t>(params.consecutive)) {
        continue;
      }
      ++stats.eligible;
      if (second == num_dates) {
        second = first_valid;
      }
      const int detected = result.break_index[pixel];
      if (result.status[pixel] == static_cast<uint8_t>(PixelStatus::kBreak) &&
          detected >= static_cast<int>(first_valid) &&
          detected <= static_cast<int>(second)) {
        ++stats.recovered;
      }
    }
  }
  *out = stats;
  return Status::Ok();
}

}  // namespace cd
