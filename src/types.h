// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Plain types shared by every module: model parameters, date table, detection
// results, timings and small value types. Keeping them in one header lets
// modules be implemented in parallel against fixed interfaces.
#ifndef SRC_TYPES_H_
#define SRC_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "status.h"

namespace cd {

inline constexpr int kMaxHarmonics = 3;
inline constexpr int kMaxParams = 2 + 2 * kMaxHarmonics;  // 8
// GPU constant-memory table size: the largest supported number of dates.
inline constexpr int kMaxDates = 1024;
// Sentinel for an invalid observation; a valid NDVI is in [-1, 1].
inline constexpr float kInvalidNdvi = -2.0f;

// Outcome of the per-pixel analysis.
enum class PixelStatus : uint8_t {
  kStable = 0,
  kBreak = 1,
  kInsufficientData = 2
};

// Execution backends selectable with --backend.
enum class Backend { kGpuFused, kGpuUnfused, kCpuFp64, kCpuFp32, kCpuMt };

// Returns "gpu", "gpu-unfused", "cpu", "cpu-fp32" or "cpu-mt".
std::string BackendName(Backend b);
// Parses a backend name; error if `s` is not one of the five names.
Status ParseBackend(const std::string& s, Backend* out);
// True for the two GPU backends.
bool IsGpuBackend(Backend b);

// Model and detection parameters. Plain data, passed by value to kernels.
struct ModelParams {
  int harmonics = 2;
  float train_years = 2.0f;
  // True: the trend column of the design matrix stays at its end-of-history
  // value for monitoring dates (the seasonal terms still vary). False: the
  // fitted linear trend is extrapolated. Holding avoids slope errors growing
  // into large prediction drift over the monitoring years.
  bool hold_trend = true;
  int min_train_obs = 0;  // 0 -> 3 * (2 + 2K), resolved by ResolveModelParams
  // Break rule defaults, tuned on the real development cube (see README.md,
  // "Parameters"): |z| > 3.5 on 2 consecutive same-sign observations.
  float threshold = 3.5f;
  int consecutive = 2;
  float rmse_floor = 0.02f;
  float ridge = 0.0f;
  bool losses_only = false;
  uint32_t valid_class_mask = (1u << 4) | (1u << 5) | (1u << 6);
};

// Fills min_train_obs when it is 0. Errors if harmonics is outside 1..3 or the
// resolved min_train_obs is smaller than the parameter count plus one.
Status ResolveModelParams(ModelParams* p);
// Number of regression coefficients: 2 + 2K.
int NumParams(int harmonics);

// Per-date calendar information, size T (host side).
struct DateTable {
  std::vector<std::string> iso;  // "YYYY-MM-DD"
  std::vector<int32_t> day;      // days since 1970-01-01
  std::vector<float> scale;      // reflectance = DN * scale + offset
  std::vector<float> offset;
  std::vector<double> tau;    // years since day[0]: (day - day[0]) / 365.25
  std::vector<int16_t> year;  // calendar year of each date
  int num_train_dates = 0;    // leading dates with tau < train_years
};

// Fills tau, year (parsed from iso) and num_train_dates from iso/day and
// params.train_years. Errors if num_train_dates is 0 or equals T (nothing to
// train on or nothing to monitor), or if iso/day are inconsistent.
Status BuildDateTable(const ModelParams& params, DateTable* dates);

// Stage timings in milliseconds; -1 means not applicable for the backend.
struct StageTimings {
  double read_ms = -1;       // time in CubeSource reads (may overlap compute)
  double read_wait_ms = -1;  // CPU: compute waiting for prefetcher; GPU: -1
  // Time inside the backend before the first tile is read: GPU = design table
  // upload, tile planning and all device and pinned allocations; CPU = result
  // allocation, design table and tile buffers. Included in total_ms.
  double setup_ms = -1;
  double h2d_ms = -1;
  double ndvi_ms = -1;
  double fit_ms = -1;
  double detect_ms = -1;
  double fused_ms = -1;
  double d2h_ms = -1;
  double kernel_ms = -1;      // GPU: sum of kernel events; CPU: compute loop
  double post_ms = -1;        // cleanup + colorize + validation (detect mode)
  double write_ms = -1;       // writing output files (detect mode only)
  double total_ms = -1;       // backend wall: first read .. results in host
  double wall_total_ms = -1;  // detect mode: whole run including post/write
};

// Execution configuration of one backend run.
struct RunConfig {
  Backend backend = Backend::kGpuFused;
  int tile_rows = 0;
  int streams = 3;
  int block_size = 256;
  int threads = 0;
  int io_threads = 8;  // parallel disk reads per tile; 1 = sequential
  int device = 0;
  bool save_coefficients = false;
  bool quiet = false;
  // GPU detection only: when non-empty, the per-tile stage timeline is written
  // to this CSV file (docs/FORMATS.md, "Pipeline trace").
  std::string trace_csv;
};

// What a backend run actually did.
struct RunStats {
  int num_tiles = 0;
  int tile_rows_used = 0;
  int threads_used = 0;
  int streams_used = 0;
  double device_mem_peak_mb = -1;   // sum of this run's device allocations
  double device_mem_alloc_mb = -1;  // same value as device_mem_peak_mb
  std::string device_name;          // GPU name or CPU model
  std::vector<std::string> notes;   // e.g. "retried with 128 rows"
};

// Detection maps, H x W, row-major.
struct DetectionResult {
  int height = 0;
  int width = 0;
  int num_params = 0;
  std::vector<int16_t> break_index;  // -1 = no break
  std::vector<float> magnitude;      // signed mean |residual| over the run
  std::vector<uint8_t> status;       // PixelStatus
  std::vector<float> rmse;           // floored RMSE; 0 if insufficient
  std::vector<uint16_t> train_obs;   // valid training observations
  std::vector<float> coefficients;   // optional, coefficient-major
};

// Sizes every map for height x width pixels. break_index is filled with -1,
// everything else with 0. Coefficients (num_params * height * width) are
// allocated only when with_coefficients is true, otherwise cleared.
void AllocateResult(int height, int width, int num_params,
                    bool with_coefficients, DetectionResult* result);

// Quick two-date NDVI difference result, H x W.
struct QuickDiffResult {
  int height = 0;
  int width = 0;
  std::vector<float> diff;    // ndvi_b - ndvi_a; 0 when flag == 0
  std::vector<uint8_t> flag;  // 0 no data, 1 no change, 2 loss, 3 gain
};

// Monitoring years used for validation, inclusive on both ends.
struct ValidationWindow {
  int first_year = 0;
  int last_year = 0;
};

// Confusion counts against the reference raster.
struct ConfusionCounts {
  uint64_t tp = 0;
  uint64_t fp = 0;
  uint64_t fn = 0;
  uint64_t tn = 0;
  uint64_t excluded = 0;
  uint64_t year_exact = 0;
  uint64_t year_within1 = 0;
};

// Properties of one CUDA device.
struct GpuInfo {
  std::string name;
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  size_t total_mem = 0;
  size_t free_mem = 0;
  int driver_version = 0;
  int runtime_version = 0;
};

// FP64 recomputation of one sample pixel (trajectories.csv).
struct PixelTrajectory {
  int row = 0;
  int col = 0;
  std::vector<double> ndvi;  // kInvalidNdvi when invalid
  std::vector<double> predicted;
  std::vector<double> z;
  std::vector<uint8_t> in_break_run;
  double coef[kMaxParams] = {0};
  double rmse = 0;
  int n_train = 0;
  int16_t break_index = -1;
  double magnitude = 0;
  uint8_t status = 0;
};

}  // namespace cd

#endif  // SRC_TYPES_H_
