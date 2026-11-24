// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "app_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): relative display paths
#include <string>
#include <utility>
#include <vector>

#include "cpu/cpu_backend.h"
#include "cube_io.h"
#include "cuda/gpu_backend.h"
#include "logger.h"
#include "system_info.h"
#include "timing.h"

namespace cd {

std::string DisplayPath(const std::string& path, const std::string& as_given) {
  namespace fs = std::filesystem;
  std::error_code error;
  const fs::path absolute = fs::weakly_canonical(fs::absolute(path), error);
  const fs::path cwd = fs::weakly_canonical(fs::current_path(error), error);
  if (error || absolute.empty() || cwd.empty()) {
    return as_given;
  }
  const fs::path relative = absolute.lexically_relative(cwd);
  // lexically_relative gives ".." components (or an empty path) when the
  // path is not below the working directory.
  if (relative.empty() || *relative.begin() == "..") {
    return as_given;
  }
  return relative.string();
}

namespace {

// Number with `%g` formatting; `decimal` forces a ".0" on whole numbers.
std::string FormatNumber(double value, bool decimal) {
  char text[64];
  std::snprintf(text, sizeof(text), "%g", value);
  std::string result = text;
  if (decimal && result.find_first_of(".en") == std::string::npos) {
    result += ".0";
  }
  return result;
}

// "6.00 GB" (decimal units) or "12.50 MB" below one gigabyte.
std::string FormatBytes(uint64_t bytes) {
  char text[64];
  if (bytes >= 1000000000ULL) {
    std::snprintf(text, sizeof(text), "%.2f GB",
                  static_cast<double>(bytes) / 1e9);
  } else {
    std::snprintf(text, sizeof(text), "%.2f MB",
                  static_cast<double>(bytes) / 1e6);
  }
  return text;
}

// Description of a backend for the execution line.
std::string BackendDescription(const RunConfig& config, const RunStats& stats) {
  switch (config.backend) {
    case Backend::kGpuFused:
      return "gpu (fused kernel)";
    case Backend::kGpuUnfused:
      return "gpu-unfused (separate NDVI, fit and detect kernels)";
    case Backend::kCpuFp64:
      return "cpu (FP64, 1 thread)";
    case Backend::kCpuFp32:
      return "cpu-fp32 (FP32, 1 thread)";
    case Backend::kCpuMt:
      return "cpu-mt (FP32, " + std::to_string(stats.threads_used) +
             " threads)";
  }
  return BackendName(config.backend);
}

}  // namespace

void LogHeader(const Flags& flags, int argc, char** argv) {
  Log("=== change_detect 1.0 ===");
  LogKv("command", CommandLine(argc, argv));
  LogKv("start_time", IsoTimestamp());
  LogKv("git_commit", GitShortCommit());
  LogSection("device");
  Log(GpuSummaryLine(flags.device));
  Log(CpuSummaryLine());
}

int FailRun(const std::string& message) {
  Log("error: " + message);
  return 1;
}

SyntheticOptions SyntheticOptionsFromFlags(const Flags& flags) {
  SyntheticOptions options;
  options.height = flags.synthetic_height;
  options.width = flags.synthetic_width;
  options.num_dates = flags.synthetic_dates;
  options.years = flags.synthetic_years;
  options.seed = flags.seed;
  options.noise = flags.noise;
  options.cloud_fraction = flags.cloud_fraction;
  options.break_fraction = flags.break_fraction;
  options.train_years = flags.train_years;
  options.consecutive = flags.consecutive;
  options.threads = flags.threads;
  return options;
}

Status CheckBackendAvailable(Backend backend) {
  if (IsGpuBackend(backend) && !GpuAvailable()) {
    return Status::Error("backend '" + BackendName(backend) +
                         "' requires a CUDA build and an NVIDIA GPU (see "
                         "INSTALL)");
  }
  return Status::Ok();
}

Status OpenInput(const Flags& flags, const ModelParams& params,
                 AppInput* input) {
  if (flags.input == "synthetic") {
    const WallTimer timer;
    SyntheticCube cube;
    const Status generated =
        GenerateSyntheticCube(SyntheticOptionsFromFlags(flags), &cube);
    if (!generated.ok()) {
      return generated;
    }
    input->truth = cube.truth_break_index;  // the cube is moved below
    input->source = MakeMemoryCube(std::move(cube));
    input->generation_ms = timer.ElapsedMs();
    input->manifest_path = "synthetic";
    input->synthetic = true;
  } else {
    CubeInfo info;
    const Status read = ReadManifest(flags.input, &info);
    if (!read.ok()) {
      return read;
    }
    Status opened = Status::Ok();
    input->source = OpenRawCube(info, flags.io_threads, &opened);
    if (!opened.ok() || input->source == nullptr) {
      return opened.ok()
                 ? Status::Error("cannot open cube '" + flags.input + "'")
                 : opened;
    }
    input->manifest_path = DisplayPath(flags.input, flags.input);
    input->synthetic = false;
  }
  input->info = input->source->info();
  input->dates = input->info.dates;
  return BuildDateTable(params, &input->dates);
}

void LogInputSection(const AppInput& input) {
  const CubeInfo& info = input.info;
  const uint64_t pixels =
      static_cast<uint64_t>(info.height) * static_cast<uint64_t>(info.width);
  const uint64_t samples = pixels * static_cast<uint64_t>(info.num_dates);
  LogSection("input");
  Log("cube: " + info.name + " (" + input.manifest_path + ")");
  Log("scenes (dates) T: " + std::to_string(info.num_dates) + " [" +
      input.dates.iso.front() + " .. " + input.dates.iso.back() + "]");
  Log("image size H x W: " + std::to_string(info.height) + " x " +
      std::to_string(info.width));
  Log("pixels: " + FormatCount(pixels));
  Log("pixel time series: " + FormatCount(pixels));
  Log("samples (pixel-dates): " + FormatCount(samples));
  Log("input bytes: " + FormatBytes(samples * 5));
  if (input.generation_ms >= 0) {
    Log("synthetic_generation: " + FormatMs(input.generation_ms) +
        " ms (not included in any timing below)");
  }
}

void LogParametersSection(const ModelParams& params, const DateTable& dates,
                          bool cleanup) {
  std::string classes;
  for (int c = 0; c < 16; ++c) {
    if ((params.valid_class_mask >> c) & 1u) {
      classes += (classes.empty() ? "" : ",") + std::to_string(c);
    }
  }
  LogSection("parameters");
  Log("harmonics: " + std::to_string(params.harmonics) +
      " (p = " + std::to_string(NumParams(params.harmonics)) +
      ") | trend: " + (params.hold_trend ? "hold" : "extrapolate") +
      " | train_years: " + FormatNumber(params.train_years, true) +
      " (train dates: " + std::to_string(dates.num_train_dates) +
      ") | min_train_obs: " + std::to_string(params.min_train_obs) +
      " | threshold: " + FormatNumber(params.threshold, true) +
      " | consecutive: " + std::to_string(params.consecutive) +
      " | rmse_floor: " + FormatNumber(params.rmse_floor, false) +
      " | ridge: " + FormatNumber(params.ridge, false) + " | losses_only: " +
      (params.losses_only ? "yes" : "no") + " | valid_classes: " + classes +
      " | cleanup: " + (cleanup ? "yes" : "no"));
}

Status RunBackend(CubeSource* source, const ModelParams& params,
                  const RunConfig& config, DetectionResult* result,
                  StageTimings* timings, RunStats* stats) {
  const Status available = CheckBackendAvailable(config.backend);
  if (!available.ok()) {
    return available;
  }
  if (IsGpuBackend(config.backend)) {
    return RunGpuDetection(source, params, config, result, timings, stats);
  }
  return RunCpuDetection(source, params, config, result, timings, stats);
}

void LogTimingsSection(const StageTimings& t, uint64_t samples,
                       const RunStats* stats) {
  LogSection("timings (ms)");
  // Input bytes read per second of read time (1 GB = 1e9 bytes).
  std::string read_gbps = "-";
  if (t.read_ms > 0) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.2f",
                  static_cast<double>(samples) * 5.0 / (t.read_ms * 1e6));
    read_gbps = text;
  }
  Log("read_input: " + FormatMs(t.read_ms) + " | read_gbps: " + read_gbps +
      " | read_wait: " + FormatMs(t.read_wait_ms) +
      " | setup: " + FormatMs(t.setup_ms) + " | h2d: " + FormatMs(t.h2d_ms) +
      " | ndvi: " + FormatMs(t.ndvi_ms) + " | fit: " + FormatMs(t.fit_ms) +
      " | detect: " + FormatMs(t.detect_ms) +
      " | fused: " + FormatMs(t.fused_ms) + " | d2h: " + FormatMs(t.d2h_ms) +
      " | kernel_total: " + FormatMs(t.kernel_ms) + " | backend_total: " +
      FormatMs(t.total_ms) + " | post: " + FormatMs(t.post_ms) +
      " | write_outputs: " + FormatMs(t.write_ms) +
      " | wall_total: " + FormatMs(t.wall_total_ms));
  Log("throughput: " + FormatThroughput(samples, t.total_ms) +
      " Msamples/s | kernel_throughput: " +
      FormatThroughput(samples, t.kernel_ms) + " Msamples/s");
  if (stats == nullptr) {
    return;
  }
  const auto mib = [](double value) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.0f MiB", value);
    return value < 0 ? std::string("-") : std::string(text);
  };
  Log("device_memory_peak: " + mib(stats->device_mem_peak_mb) +
      (stats->device_mem_peak_mb < 0
           ? ""
           : " (pipeline buffers; CUDA context not included)"));
}

void LogExecutionSection(const RunConfig& config, const RunStats& stats,
                         const CubeInfo& info) {
  const bool gpu = IsGpuBackend(config.backend);
  const uint64_t tile_bytes = static_cast<uint64_t>(stats.tile_rows_used) *
                              static_cast<uint64_t>(info.width) *
                              static_cast<uint64_t>(info.num_dates) * 5;
  LogSection("execution");
  Log("backend: " + BackendDescription(config, stats) + " | block_size: " +
      (gpu ? std::to_string(config.block_size) : "-") + " | streams: " +
      (gpu ? std::to_string(stats.streams_used > 0 ? stats.streams_used
                                                   : config.streams)
           : "-") +
      " | io_threads: " + std::to_string(config.io_threads) +
      " | tile_rows: " + std::to_string(stats.tile_rows_used) +
      " | tiles: " + std::to_string(stats.num_tiles) +
      " | tile_input_bytes: " + FormatCount(tile_bytes));
  for (const std::string& note : stats.notes) {
    // The memory line is a "key: value" line of its own, not a note.
    const bool is_memory_line = note.rfind("gpu_memory: ", 0) == 0;
    Log(is_memory_line ? note : "note: " + note);
  }
  if (config.quiet || stats.tile_rows_used <= 0) {
    return;
  }
  for (int k = 0; k < stats.num_tiles; ++k) {
    const int row0 = k * stats.tile_rows_used;
    const int row1 = std::min(info.height, row0 + stats.tile_rows_used) - 1;
    Log("[tile " + std::to_string(k + 1) + "/" +
        std::to_string(stats.num_tiles) + "] rows " + std::to_string(row0) +
        ".." + std::to_string(row1));
  }
}

std::string FormatMs(double ms) {
  if (ms < 0) {
    return "-";
  }
  char text[64];
  std::snprintf(text, sizeof(text), "%.3f", ms);
  return text;
}

std::string FormatCount(uint64_t value) { return std::to_string(value); }

std::string FormatThroughput(uint64_t samples, double ms) {
  if (ms <= 0) {
    return "-";
  }
  char text[64];
  std::snprintf(text, sizeof(text), "%.1f",
                static_cast<double>(samples) / (ms * 1e3));
  return text;
}

std::string SummaryLine(uint64_t pixels, uint64_t samples, double seconds,
                        const std::string& backend) {
  char text[64];
  std::snprintf(text, sizeof(text), "%.3f", seconds);
  return "SUMMARY: processed " + FormatCount(pixels) + " pixel time series (" +
         FormatCount(samples) + " samples) in " + text + " seconds (backend " +
         backend + ")";
}

}  // namespace cd
