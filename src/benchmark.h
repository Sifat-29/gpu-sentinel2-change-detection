// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Benchmark support: agreement between two detection results, the benchmark
// CSV (columns: docs/FORMATS.md, "Benchmark CSV") and the end-of-run median
// table.
#ifndef SRC_BENCHMARK_H_
#define SRC_BENCHMARK_H_

#include <cstdint>
#include <string>
#include <vector>

#include "cube_source.h"
#include "status.h"
#include "types.h"

namespace cd {

// How closely a result matches a reference result; -1 = not available.
struct Agreement {
  double break_frac = -1;     // fraction of pixels with equal break_index
  double status_frac = -1;    // fraction of pixels with equal status
  double max_coef_diff = -1;  // largest |coefficient difference|, if both saved
};

// Compares `result` with `reference` (same size). Coefficients are compared
// only when both results hold them. Errors if the sizes differ.
Status ComputeAgreement(const DetectionResult& result,
                        const DetectionResult& reference, Agreement* out);

// The system columns of the benchmark CSV, identical for every row of a run.
struct SystemColumns {
  std::string gpu_name = "none";
  double gpu_mem_mb = -1;
  std::string cpu_model;
  int cpu_threads = 0;
  std::string cuda_runtime = "none";
  std::string driver_version = "none";
  std::string git_commit;
};

// Reads the CPU/GPU description of this machine (GPU fields keep their "none"
// defaults when no GPU is usable).
SystemColumns CollectSystemColumns(int device);

// One CSV row (one timed repetition). Numeric not-applicable values are -1.
struct BenchmarkRow {
  std::string run_id;
  std::string timestamp;
  std::string mode = "benchmark";
  std::string backend;
  std::string precision;  // fp32 or fp64
  std::string kernels;    // fused, unfused or n/a
  std::string input_name;
  int height = 0;
  int width = 0;
  int dates = 0;
  int64_t pixels = 0;
  int64_t samples = 0;
  int tile_rows = -1;
  int num_tiles = -1;
  int streams = -1;
  int block_size = -1;
  int threads = -1;
  int io_threads = -1;
  int repeat_index = 0;
  int64_t input_bytes = 0;
  StageTimings timings;  // read_ms .. total_ms are written
  double device_mem_peak_mb = -1;
  int64_t changed_pixels = 0;
  std::string compare_to = "none";
  Agreement agreement;
  SystemColumns system;
};

// The exact header line of the benchmark CSV (without a newline).
const std::string& BenchmarkCsvHeader();

// Builds the row for one timed run of `config.backend` on `info`; the
// throughput columns are derived by FormatBenchmarkRow from the timings.
BenchmarkRow MakeBenchmarkRow(const std::string& run_id, const CubeInfo& info,
                              const RunConfig& config,
                              const StageTimings& timings,
                              const RunStats& stats, int repeat_index,
                              int64_t changed_pixels,
                              const SystemColumns& system);

// One CSV line (without newline): milliseconds with 3 decimals, -1 for
// not-applicable values, strings containing commas or quotes quoted.
std::string FormatBenchmarkRow(const BenchmarkRow& row);

// Appends `rows` to the CSV at `path`. A new or empty file gets the header
// first; an existing file must start with exactly BenchmarkCsvHeader(), else
// nothing is written and an error is returned. Creates missing directories.
Status AppendBenchmarkRows(const std::string& path,
                           const std::vector<BenchmarkRow>& rows);

// Median of `values` (mean of the two middle values for an even count);
// 0 for an empty vector.
double Median(std::vector<double> values);

// Timed repetitions of one backend, input of the median table.
struct BackendSamples {
  std::string backend;
  std::vector<double> total_ms;
  std::vector<double> kernel_ms;  // -1 entries when not applicable
};

// The median table printed at the end of benchmark mode: a header line
// "backend | median backend_total ms | min | max | median kernel ms | speedup
// vs cpu-mt" and one line per backend. The speedup is the cpu-mt median
// divided by the backend median ("-" when cpu-mt was not benchmarked).
std::vector<std::string> FormatMedianTable(
    const std::vector<BackendSamples>& samples);

}  // namespace cd

#endif  // SRC_BENCHMARK_H_
