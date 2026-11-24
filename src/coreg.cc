// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "coreg.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>  // NOLINT(build/c++11): parallel window cropping
#include <vector>

#include "app_common.h"
#include "cube_source.h"
#include "cuda/gpu_backend.h"
#include "logger.h"
#include "outputs.h"
#include "timing.h"
#include "types.h"

namespace cd {
namespace {

// Dates read per ReadTileDates call (the reads of a call run in parallel).
constexpr int kReadGroup = 8;

// The cropped window of every date, date-major.
struct WindowData {
  std::vector<uint16_t> band;          // DN of the correlated band
  std::vector<uint8_t> valid;          // 1 = usable observation
  std::vector<double> valid_fraction;  // per date, over the window
};

// Number of valid pixels counted while cropping one date.
using ValidCount = uint64_t;

// Crops the window out of the rows read for one date (plane `plane` of the
// tile buffers, rows x width pixels) and counts the valid observations: SCL
// class selected in `class_ok`, and a non-zero DN in both bands.
ValidCount CropDate(const TileView& tile, int plane, int width,
                    const CoregWindow& window, bool use_nir,
                    const bool* class_ok, uint16_t* band_out,
                    uint8_t* valid_out) {
  const size_t plane_cells =
      static_cast<size_t>(window.size) * static_cast<size_t>(width);
  const uint16_t* red = tile.red + plane * plane_cells;
  const uint16_t* nir = tile.nir + plane * plane_cells;
  const uint8_t* scl = tile.scl + plane * plane_cells;
  ValidCount count = 0;
  for (int r = 0; r < window.size; ++r) {
    const size_t row_start = static_cast<size_t>(r) * width + window.col0;
    const uint16_t* chosen = use_nir ? nir : red;
    std::memcpy(band_out + static_cast<size_t>(r) * window.size,
                chosen + row_start,
                static_cast<size_t>(window.size) * sizeof(uint16_t));
    uint8_t* valid_row = valid_out + static_cast<size_t>(r) * window.size;
    for (int c = 0; c < window.size; ++c) {
      const size_t cell = row_start + c;
      const bool ok = class_ok[scl[cell]] && red[cell] != 0 && nir[cell] != 0;
      valid_row[c] = ok ? 1 : 0;
      count += ok ? 1 : 0;
    }
  }
  return count;
}

// Reads the window rows of every date (a few dates per call, full-width row
// bands as the cube is stored by rows) and crops them into `data`.
Status ReadWindow(CubeSource* source, const CoregWindow& window, bool use_nir,
                  uint32_t valid_class_mask, WindowData* data) {
  const CubeInfo& info = source->info();
  const int num_dates = info.num_dates;
  const size_t pixels =
      static_cast<size_t>(window.size) * static_cast<size_t>(window.size);
  data->band.assign(pixels * num_dates, 0);
  data->valid.assign(pixels * num_dates, 0);
  data->valid_fraction.assign(num_dates, 0.0);
  bool class_ok[256];
  for (int c = 0; c < 256; ++c) {
    class_ok[c] = c < 32 && ((valid_class_mask >> c) & 1u) != 0;
  }
  const size_t tile_cells =
      static_cast<size_t>(window.size) * static_cast<size_t>(info.width);
  const int group = std::min(kReadGroup, num_dates);
  std::vector<uint16_t> red(tile_cells * group);
  std::vector<uint16_t> nir(tile_cells * group);
  std::vector<uint8_t> scl(tile_cells * group);
  for (int first = 0; first < num_dates; first += group) {
    const int count = std::min(group, num_dates - first);
    std::vector<int> dates(count);
    for (int i = 0; i < count; ++i) {
      dates[i] = first + i;
    }
    TileView tile;
    tile.red = red.data();
    tile.nir = nir.data();
    tile.scl = scl.data();
    const Status status = source->ReadTileDates(window.row0, window.size,
                                                dates.data(), count, &tile);
    if (!status.ok()) {
      return status;
    }
    std::vector<ValidCount> valid_counts(count, 0);
    std::vector<std::thread> workers;
    for (int i = 0; i < count; ++i) {
      workers.emplace_back([&, i] {
        valid_counts[i] =
            CropDate(tile, i, info.width, window, use_nir, class_ok,
                     data->band.data() + (first + i) * pixels,
                     data->valid.data() + (first + i) * pixels);
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
    for (int i = 0; i < count; ++i) {
      data->valid_fraction[first + i] =
          static_cast<double>(valid_counts[i]) / static_cast<double>(pixels);
    }
  }
  return Status::Ok();
}

// Index of the date with the highest valid fraction (the first one on ties).
int BestReferenceDate(const std::vector<double>& valid_fraction) {
  return static_cast<int>(
      std::max_element(valid_fraction.begin(), valid_fraction.end()) -
      valid_fraction.begin());
}

// "%.<digits>f" text of a number.
std::string Fixed(double value, int digits) {
  char text[64];
  std::snprintf(text, sizeof(text), "%.*f", digits, value);
  return text;
}

// One row of coreg_shifts.csv and of the results summary.
struct DateShift {
  double valid_fraction = 0;
  float dy = 0;
  float dx = 0;
  float shift = 0;
  float peak = 0;
  const char* flag = "ok";
};

// Writes coreg_shifts.csv (columns: docs/FORMATS.md).
Status WriteShiftsCsv(const std::string& path, const DateTable& dates,
                      const std::vector<DateShift>& shifts) {
  std::FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) {
    return Status::Error("cannot write '" + path + "'");
  }
  std::fprintf(file,
               "date_index,date,valid_fraction,dy_px,dx_px,shift_px,peak,"
               "flag\n");
  for (size_t t = 0; t < shifts.size(); ++t) {
    const DateShift& s = shifts[t];
    std::fprintf(file, "%zu,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%s\n", t,
                 dates.iso[t].c_str(), s.valid_fraction, s.dy, s.dx, s.shift,
                 s.peak, s.flag);
  }
  const bool failed = std::ferror(file) != 0;
  std::fclose(file);
  return failed ? Status::Error("error while writing '" + path + "'")
                : Status::Ok();
}

// Logs the "--- parameters ---" section of the mode.
void LogCoregParameters(const Flags& flags, const ModelParams& params,
                        const CoregWindow& window) {
  std::string classes;
  for (int c = 0; c < 16; ++c) {
    if ((params.valid_class_mask >> c) & 1u) {
      classes += (classes.empty() ? "" : ",") + std::to_string(c);
    }
  }
  LogSection("parameters");
  Log("coreg_window: " + std::to_string(window.size) + " | origin: " +
      std::to_string(window.row0) + "," + std::to_string(window.col0) +
      (flags.coreg_origin_row < 0 ? " (centred)" : "") +
      " | band: " + flags.coreg_band + " | reference_date: " +
      (flags.reference_date < 0 ? "auto"
                                : std::to_string(flags.reference_date)) +
      " | valid_classes: " + classes +
      " | low_peak_below: " + Fixed(LowPeakThreshold(window.size), 3) +
      " | large_shift_above_px: " + Fixed(kLargeShiftPx, 2));
}

}  // namespace

Status ResolveCoregWindow(int size, int origin_row, int origin_col, int height,
                          int width, CoregWindow* window) {
  if (size < kMinCoregWindow || size > kMaxCoregWindow ||
      (size & (size - 1)) != 0) {
    return Status::Error("coreg window must be a power of two in " +
                         std::to_string(kMinCoregWindow) + ".." +
                         std::to_string(kMaxCoregWindow) + ", got " +
                         std::to_string(size));
  }
  if (size > height || size > width) {
    return Status::Error("coreg window " + std::to_string(size) +
                         " does not fit the " + std::to_string(height) + " x " +
                         std::to_string(width) +
                         " cube; use a smaller --coreg-window");
  }
  CoregWindow resolved;
  resolved.size = size;
  resolved.row0 = origin_row < 0 ? (height - size) / 2 : origin_row;
  resolved.col0 = origin_col < 0 ? (width - size) / 2 : origin_col;
  if (resolved.row0 + size > height || resolved.col0 + size > width) {
    return Status::Error("coreg window " + std::to_string(size) +
                         " at origin " + std::to_string(resolved.row0) + "," +
                         std::to_string(resolved.col0) + " exceeds the " +
                         std::to_string(height) + " x " +
                         std::to_string(width) + " cube");
  }
  *window = resolved;
  return Status::Ok();
}

Status CheckCoregBand(const std::string& band) {
  if (band != "nir" && band != "red") {
    return Status::Error("coreg band must be nir or red, got '" + band + "'");
  }
  return Status::Ok();
}

float LowPeakThreshold(int window) {
  return std::max(kLowPeakFloor,
                  kLowPeakNoiseFactor / static_cast<float>(window));
}

const char* CoregFlag(float peak, float shift_px, int window) {
  if (peak < LowPeakThreshold(window)) {
    return "low_peak";
  }
  return shift_px > kLargeShiftPx ? "large_shift" : "ok";
}

int RunCoregMode(const Flags& flags) {
  Backend backend;
  Status status = ParseBackend(flags.backends.front(), &backend);
  if (status.ok() && !IsGpuBackend(backend)) {
    status = Status::Error(
        "--mode coreg needs a GPU backend (gpu or gpu-unfused), got '" +
        BackendName(backend) + "'");
  }
  ModelParams params;
  if (status.ok()) {
    status = CheckBackendAvailable(backend);
  }
  if (status.ok()) {
    status = FlagsToModelParams(flags, &params);
  }
  if (status.ok()) {
    status = CheckCoregBand(flags.coreg_band);
  }
  AppInput input;
  if (status.ok()) {
    status = OpenInput(flags, params, &input);
  }
  CoregWindow window;
  if (status.ok()) {
    status = ResolveCoregWindow(flags.coreg_window, flags.coreg_origin_row,
                                flags.coreg_origin_col, input.info.height,
                                input.info.width, &window);
  }
  const int num_dates = input.info.num_dates;
  if (status.ok() && flags.reference_date >= num_dates) {
    status = Status::Error("--reference-date must be in 0.." +
                           std::to_string(num_dates - 1) + ", got " +
                           std::to_string(flags.reference_date));
  }
  if (status.ok()) {
    status = EnsureDirectory(flags.output_dir);
  }
  if (!status.ok()) {
    return FailRun(status.message());
  }
  LogInputSection(input);
  LogCoregParameters(flags, params, window);

  const WallTimer total_timer;
  const bool use_nir = flags.coreg_band == "nir";
  WindowData data;
  status = ReadWindow(input.source.get(), window, use_nir,
                      params.valid_class_mask, &data);
  const double read_ms = total_timer.ElapsedMs();
  if (!status.ok()) {
    return FailRun(status.message());
  }
  const int reference = flags.reference_date >= 0
                            ? flags.reference_date
                            : BestReferenceDate(data.valid_fraction);

  CoregGpuInput gpu_input;
  gpu_input.window = window.size;
  gpu_input.num_dates = num_dates;
  gpu_input.reference_date = reference;
  gpu_input.device = flags.device;
  gpu_input.band = data.band.data();
  gpu_input.valid = data.valid.data();
  gpu_input.scale = input.dates.scale.data();
  gpu_input.offset = input.dates.offset.data();
  CoregGpuOutput output;
  status = GpuPhaseCorrelation(gpu_input, &output);
  if (!status.ok()) {
    return FailRun(status.message());
  }
  const double total_ms = total_timer.ElapsedMs();

  std::vector<DateShift> shifts(num_dates);
  std::vector<double> magnitudes;
  int num_low_peak = 0;
  int num_large = 0;
  int max_date = 0;
  for (int t = 0; t < num_dates; ++t) {
    DateShift& s = shifts[t];
    s.valid_fraction = data.valid_fraction[t];
    s.dy = output.dy[t];
    s.dx = output.dx[t];
    s.shift = std::hypot(s.dy, s.dx);
    s.peak = output.peak[t];
    s.flag = CoregFlag(s.peak, s.shift, window.size);
    num_low_peak += std::strcmp(s.flag, "low_peak") == 0 ? 1 : 0;
    num_large += std::strcmp(s.flag, "large_shift") == 0 ? 1 : 0;
    magnitudes.push_back(s.shift);
    if (s.shift > shifts[max_date].shift) {
      max_date = t;
    }
  }
  std::vector<double> sorted = magnitudes;
  std::sort(sorted.begin(), sorted.end());
  const size_t mid = sorted.size() / 2;
  const double median = sorted.size() % 2 == 1
                            ? sorted[mid]
                            : 0.5 * (sorted[mid - 1] + sorted[mid]);

  const std::string csv_path = flags.output_dir + "/coreg_shifts.csv";
  status = WriteShiftsCsv(csv_path, input.dates, shifts);
  if (!status.ok()) {
    return FailRun(status.message());
  }

  const CoregGpuTimings& t = output.timings;
  LogSection("execution");
  Log("backend: gpu (cuFFT phase correlation) | window: " +
      std::to_string(window.size) + " x " + std::to_string(window.size) +
      " at row " + std::to_string(window.row0) + ", col " +
      std::to_string(window.col0) + " | band: " + flags.coreg_band +
      " | reference_date: " + std::to_string(reference) + " (" +
      input.dates.iso[reference] + ", valid " +
      Fixed(data.valid_fraction[reference], 4) +
      ") | dates_per_batch: " + std::to_string(t.chunk_dates));
  LogSection("timings (ms)");
  Log("read_input: " + FormatMs(read_ms) + " | setup: " + FormatMs(t.setup_ms) +
      " | h2d: " + FormatMs(t.h2d_ms) + " | fft: " + FormatMs(t.fft_ms) +
      " | spectrum: " + FormatMs(t.spectrum_ms) +
      " | peak: " + FormatMs(t.peak_ms) + " | d2h: " + FormatMs(t.d2h_ms) +
      " | total: " + FormatMs(total_ms));
  LogSection("results");
  Log("median_shift_px: " + Fixed(median, 4) +
      " | max_shift_px: " + Fixed(shifts[max_date].shift, 4) + " (date " +
      std::to_string(max_date) + ", " + input.dates.iso[max_date] +
      ") | flagged: " + std::to_string(num_low_peak + num_large) +
      " (low_peak " + std::to_string(num_low_peak) + ", large_shift " +
      std::to_string(num_large) + ")");
  for (int d = 0; d < num_dates; ++d) {
    if (std::strcmp(shifts[d].flag, "ok") != 0) {
      Log("flagged_date: " + std::to_string(d) + " " + input.dates.iso[d] +
          " | " + shifts[d].flag + " | shift_px " + Fixed(shifts[d].shift, 4) +
          " | peak " + Fixed(shifts[d].peak, 4) + " | valid_fraction " +
          Fixed(shifts[d].valid_fraction, 4));
    }
  }
  LogKv("outputs", DisplayPath(csv_path, csv_path));
  const uint64_t samples = static_cast<uint64_t>(num_dates) *
                           static_cast<uint64_t>(window.size) *
                           static_cast<uint64_t>(window.size);
  Log("SUMMARY: processed " + std::to_string(num_dates) + " scenes of " +
      std::to_string(window.size) + " x " + std::to_string(window.size) +
      " pixels (" + FormatCount(samples) + " samples) in " +
      Fixed(total_ms / 1e3, 3) + " seconds (backend gpu, cuFFT)");
  return 0;
}

}  // namespace cd
