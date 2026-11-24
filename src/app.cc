// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Application layer: mode dispatch plus the quick-diff and generate modes.
// The detect and benchmark modes live in app_detect.cc and app_benchmark.cc;
// shared log and input helpers in app_common.cc.
#include "app.h"

#include <cstdint>
#include <cstdio>
#include <string>

#include "app_common.h"
#include "app_modes.h"
#include "coreg.h"
#include "cuda/gpu_backend.h"
#include "logger.h"
#include "outputs.h"
#include "postprocess.h"
#include "synthetic_cube.h"
#include "timing.h"

namespace cd {
namespace {

// Number of pixels with each quick-diff flag (0 no data .. 3 gain).
struct FlagCounts {
  uint64_t count[4] = {0, 0, 0, 0};
};

FlagCounts CountFlags(const QuickDiffResult& result) {
  FlagCounts counts;
  for (const uint8_t flag : result.flag) {
    ++counts.count[flag < 4 ? flag : 0];
  }
  return counts;
}

std::string SecondsText(double seconds) {
  char text[64];
  std::snprintf(text, sizeof(text), "%.3f", seconds);
  return text;
}

}  // namespace

int RunQuickDiffMode(const Flags& flags) {
  Backend backend;
  Status status = ParseBackend(flags.backends.front(), &backend);
  ModelParams params;
  if (status.ok()) {
    status = FlagsToModelParams(flags, &params);
  }
  if (status.ok()) {
    status = CheckBackendAvailable(backend);
  }
  if (!status.ok()) {
    return FailRun(status.message());
  }
  AppInput input;
  status = OpenInput(flags, params, &input);
  const int num_dates = input.info.num_dates;
  const int date_a = flags.date_a;
  const int date_b = flags.date_b < 0 ? num_dates - 1 : flags.date_b;
  if (status.ok() &&
      (date_a >= num_dates || date_b >= num_dates || date_a == date_b)) {
    status = Status::Error(
        "--date-a and --date-b must be two different dates in 0.." +
        std::to_string(num_dates - 1) + ", got " + std::to_string(date_a) +
        " and " + std::to_string(date_b));
  }
  if (status.ok()) {
    status = EnsureDirectory(flags.output_dir);
  }
  if (!status.ok()) {
    return FailRun(status.message());
  }
  LogInputSection(input);
  LogParametersSection(params, input.dates, false);
  LogSection("quick-diff");
  Log("backend: " + BackendName(backend) +
      " | date_a: " + std::to_string(date_a) + " (" + input.dates.iso[date_a] +
      ") | date_b: " + std::to_string(date_b) + " (" + input.dates.iso[date_b] +
      ") | diff_threshold: " +
      std::to_string(flags.diff_threshold).substr(0, 5));

  const WallTimer wall_timer;
  RunConfig config;
  status = FlagsToRunConfig(flags, backend, &config);
  QuickDiffResult result;
  StageTimings timings;
  if (status.ok()) {
    status = IsGpuBackend(backend)
                 ? GpuQuickDiff(input.source.get(), date_a, date_b,
                                flags.diff_threshold, params, config, &result,
                                &timings)
                 : CpuQuickDiff(input.source.get(), date_a, date_b,
                                flags.diff_threshold, params, config, &result,
                                &timings);
  }
  if (!status.ok()) {
    return FailRun(status.message());
  }
  const WallTimer write_timer;
  status = WriteQuickDiffOutputs(flags.output_dir, result, flags.png_max_dim,
                                 nullptr);
  if (!status.ok()) {
    return FailRun(status.message());
  }
  timings.write_ms = write_timer.ElapsedMs();
  timings.wall_total_ms = wall_timer.ElapsedMs();

  const uint64_t pixels = static_cast<uint64_t>(input.info.height) *
                          static_cast<uint64_t>(input.info.width);
  LogTimingsSection(timings, 2 * pixels, nullptr);
  const FlagCounts counts = CountFlags(result);
  LogSection("results");
  Log("no_data: " + FormatCount(counts.count[0]) +
      " | no_change: " + FormatCount(counts.count[1]) +
      " | loss: " + FormatCount(counts.count[2]) +
      " | gain: " + FormatCount(counts.count[3]));
  LogKv("outputs", DisplayPath(flags.output_dir, flags.output_dir));
  Log("SUMMARY: processed " + FormatCount(pixels) + " pixels (2 dates, " +
      FormatCount(2 * pixels) + " samples) in " +
      SecondsText(timings.wall_total_ms / 1e3) + " seconds (backend " +
      BackendName(backend) + ")");
  return 0;
}

int RunGenerateMode(const Flags& flags) {
  const SyntheticOptions options = SyntheticOptionsFromFlags(flags);
  const uint64_t pixels = static_cast<uint64_t>(options.height) *
                          static_cast<uint64_t>(options.width);
  const uint64_t samples = pixels * static_cast<uint64_t>(options.num_dates);
  const std::string size = std::to_string(options.height) + " x " +
                           std::to_string(options.width) + " x " +
                           std::to_string(options.num_dates);
  LogSection("generate");
  Log("size H x W x T: " + size);
  Log("samples (pixel-dates): " + FormatCount(samples));
  Log("raw bytes: " + FormatCount(samples * 5 + pixels * 2) +
      " (red, nir, scl, truth)");
  Log("seed: " + std::to_string(options.seed) + " | noise: " +
      std::to_string(options.noise).substr(0, 5) + " | cloud_fraction: " +
      std::to_string(options.cloud_fraction).substr(0, 5) +
      " | break_fraction: " +
      std::to_string(options.break_fraction).substr(0, 5));
  const WallTimer timer;
  const Status status = GenerateSyntheticCubeToDisk(options, flags.output_dir);
  if (!status.ok()) {
    return FailRun(status.message());
  }
  const double seconds = timer.ElapsedMs() / 1e3;
  LogKv("outputs", DisplayPath(flags.output_dir + "/manifest.txt",
                               flags.output_dir + "/manifest.txt"));
  Log("SUMMARY: generated " + std::to_string(options.height) + " x " +
      std::to_string(options.width) + " x " +
      std::to_string(options.num_dates) + " synthetic cube (" +
      FormatCount(samples) + " samples) in " + SecondsText(seconds) +
      " seconds");
  return 0;
}

int RunApp(const Flags& flags, int argc, char** argv) {
  // The log file may live in a directory that does not exist yet.
  const size_t slash = flags.log_file.find_last_of('/');
  if (slash != std::string::npos && slash > 0) {
    static_cast<void>(EnsureDirectory(flags.log_file.substr(0, slash)));
  }
  const Status log_status = InitLog(flags.log_file);
  if (!log_status.ok()) {
    Log("warning: " + log_status.message());
  }
  LogHeader(flags, argc, argv);
  int code = 1;
  if (flags.mode == "detect") {
    code = RunDetectMode(flags);
  } else if (flags.mode == "quick-diff") {
    code = RunQuickDiffMode(flags);
  } else if (flags.mode == "benchmark") {
    code = RunBenchmarkMode(flags);
  } else if (flags.mode == "generate") {
    code = RunGenerateMode(flags);
  } else if (flags.mode == "coreg") {
    code = RunCoregMode(flags);
  } else {
    code = FailRun("unknown mode '" + flags.mode + "'");
  }
  CloseLog();
  return code;
}

}  // namespace cd
