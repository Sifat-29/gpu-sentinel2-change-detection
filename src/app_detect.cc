// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// The detect mode: run one backend, optionally compare it with a second one,
// post-process, validate, write every output file and print the log.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "app_common.h"
#include "app_modes.h"
#include "benchmark.h"
#include "cpu/cpu_backend.h"
#include "cube_io.h"
#include "cuda/gpu_backend.h"
#include "logger.h"
#include "outputs.h"
#include "postprocess.h"
#include "synthetic_cube.h"
#include "system_info.h"
#include "timing.h"
#include "validation.h"

namespace cd {
namespace {

// The validation reference raster, when there is one.
struct ReferenceData {
  bool present = false;
  std::string path = "none";
  std::vector<uint8_t> raster;
};

// Output of the post-processing stage.
struct PostResult {
  Image png;  // colorized change map at PNG resolution
  bool has_validation = false;
  ConfusionCounts counts;
  ValidationMetrics metrics;
};

// Resolves --reference: "none" disables, "auto" uses the manifest's
// reference_file when it names one, anything else is a path to a raster.
Status LoadReference(const Flags& flags, const CubeInfo& info,
                     ReferenceData* reference) {
  CubeInfo named = info;
  if (flags.reference == "none") {
    return Status::Ok();
  }
  if (flags.reference != "auto") {
    named.reference_file = flags.reference;
  }
  if (named.reference_file.empty()) {
    return Status::Ok();
  }
  const Status status = ReadReferenceRaster(named, &reference->raster);
  if (!status.ok()) {
    return status;
  }
  reference->present = true;
  reference->path = DisplayPath(named.reference_file, named.reference_file);
  return Status::Ok();
}

// Cleanup, colorize and validation on the device class of the backend. The
// time of the whole stage is stored in timings->post_ms. The change map is
// first reduced to PNG resolution (DownsampleDetection) and only that small
// map is coloured. When `validated_in_pipeline` is set, post->counts already
// holds the confusion counts of the backend run and only the metrics are
// derived here.
Status PostProcess(const Flags& flags, const AppInput& input,
                   const ReferenceData& reference,
                   const ValidationWindow& window, Backend backend,
                   bool validated_in_pipeline, DetectionResult* result,
                   StageTimings* timings, PostResult* post) {
  const WallTimer timer;
  const bool gpu = IsGpuBackend(backend);
  double ms = 0;
  if (flags.cleanup) {
    const Status status =
        gpu ? GpuCleanup(result, flags.device, &ms) : CpuCleanup(result, &ms);
    if (!status.ok()) {
      return status;
    }
  }
  // The single-thread CPU baselines stay single-threaded; the other backends
  // use the cpu-mt worker count for this host-side step.
  const bool single_thread =
      backend == Backend::kCpuFp64 || backend == Backend::kCpuFp32;
  const int threads = single_thread       ? 1
                      : flags.threads > 0 ? flags.threads
                                          : HardwareThreads();
  const DetectionResult small =
      DownsampleDetection(*result, flags.png_max_dim, threads);
  post->png.width = small.width;
  post->png.height = small.height;
  const Status colorized =
      gpu ? GpuColorizeChangeMap(small, input.dates, flags.device,
                                 &post->png.rgba, &ms)
          : CpuColorizeChangeMap(small, input.dates, &post->png.rgba, &ms);
  if (!colorized.ok()) {
    return colorized;
  }
  if (reference.present) {
    if (!validated_in_pipeline) {
      const Status status =
          gpu ? GpuConfusion(*result, reference.raster, input.dates, window,
                             flags.device, &post->counts, &ms)
              : ComputeConfusionCpu(*result, reference.raster, input.dates,
                                    window, &post->counts, &ms);
      if (!status.ok()) {
        return status;
      }
    }
    post->metrics = MetricsFromCounts(post->counts);
    post->has_validation = true;
  }
  timings->post_ms = timer.ElapsedMs();
  return Status::Ok();
}

// Picks the sample pixels and recomputes their trajectories in FP64.
Status ComputeSamples(const Flags& flags, const AppInput& input,
                      const ModelParams& params, const DetectionResult& result,
                      const ReferenceData& reference,
                      const ValidationWindow& window,
                      std::vector<PixelTrajectory>* trajectories) {
  std::vector<std::pair<int, int>> pixels = flags.sample_pixel_list;
  if (pixels.empty()) {
    pixels = SelectSamplePixels(result,
                                reference.present ? &reference.raster : nullptr,
                                window, flags.sample_count);
  }
  for (const std::pair<int, int>& pixel : pixels) {
    PixelTrajectory trajectory;
    const Status status = ComputeTrajectory(
        input.source.get(), params, pixel.first, pixel.second, &trajectory);
    if (!status.ok()) {
      return status;
    }
    trajectories->push_back(std::move(trajectory));
  }
  return Status::Ok();
}

// Writes every output file (docs/FORMATS.md, "Detect outputs") and
// result_manifest.txt.
Status WriteOutputs(const Flags& flags, const AppInput& input,
                    const ModelParams& params, const DetectionResult& result,
                    const PostResult& post, const ReferenceData& reference,
                    const ValidationWindow& window) {
  const std::string& dir = flags.output_dir;
  std::vector<OutputFile> files;
  Status status = WriteDetectionMaps(dir, result, &files);
  if (!status.ok()) {
    return status;
  }
  status = WritePng(dir + "/change_map.png", post.png);
  if (!status.ok()) {
    return status;
  }
  files.push_back({"change_map.png", "png",
                   "colorized change map, colour = break date, block "
                   "priority downsampling"});

  std::vector<PixelTrajectory> trajectories;
  status = ComputeSamples(flags, input, params, result, reference, window,
                          &trajectories);
  if (!status.ok()) {
    return status;
  }
  const std::string backend = flags.backends.front();
  status = WriteTextFile(
      dir + "/pixel_stats.csv",
      BuildPixelStatsCsv(result, backend, trajectories, input.dates));
  if (!status.ok()) {
    return status;
  }
  files.push_back({"pixel_stats.csv", "csv",
                   "per-pixel fit statistics of the sample pixels"});
  status = WriteTextFile(dir + "/trajectories.csv",
                         BuildTrajectoriesCsv(trajectories, input.dates));
  if (!status.ok()) {
    return status;
  }
  files.push_back({"trajectories.csv", "csv",
                   "FP64 NDVI series, fit and z-scores of the sample pixels"});
  if (post.has_validation) {
    status =
        WriteTextFile(dir + "/validation.csv",
                      FormatValidationCsv(post.counts, post.metrics, window));
    if (!status.ok()) {
      return status;
    }
    files.push_back({"validation.csv", "csv",
                     "confusion counts and metrics against the reference"});
  }

  ResultManifestInput manifest;
  manifest.cube_manifest = input.manifest_path;
  manifest.name = input.info.name;
  manifest.height = input.info.height;
  manifest.width = input.info.width;
  manifest.num_dates = input.info.num_dates;
  manifest.backend = backend;
  manifest.params = params;
  manifest.cleanup = flags.cleanup;
  manifest.reference_file = reference.path;
  manifest.dates = input.dates;
  manifest.files = files;
  return WriteResultManifest(dir + "/result_manifest.txt", manifest);
}

// Counts of the status map: stable, break (loss, gain), insufficient.
struct StatusCounts {
  uint64_t stable = 0;
  uint64_t breaks = 0;
  uint64_t loss = 0;
  uint64_t gain = 0;
  uint64_t insufficient = 0;
};

StatusCounts CountStatus(const DetectionResult& result) {
  StatusCounts counts;
  for (size_t i = 0; i < result.status.size(); ++i) {
    if (result.status[i] == static_cast<uint8_t>(PixelStatus::kStable)) {
      ++counts.stable;
    } else if (result.status[i] == static_cast<uint8_t>(PixelStatus::kBreak)) {
      ++counts.breaks;
      if (result.magnitude[i] < 0.0f) {
        ++counts.loss;
      } else {
        ++counts.gain;
      }
    } else {
      ++counts.insufficient;
    }
  }
  return counts;
}

// "N (x.xx%)" of `part` in `total`.
std::string WithPercent(uint64_t part, uint64_t total) {
  char text[64];
  std::snprintf(text, sizeof(text), " (%.2f%%)",
                total == 0 ? 0.0
                           : 100.0 * static_cast<double>(part) /
                                 static_cast<double>(total));
  return FormatCount(part) + text;
}

std::string Fixed(double value, const char* format) {
  char text[64];
  std::snprintf(text, sizeof(text), format, value);
  return text;
}

}  // namespace

int RunDetectMode(const Flags& flags) {
  Backend backend;
  Status status = ParseBackend(flags.backends.front(), &backend);
  ModelParams params;
  if (status.ok()) {
    status = FlagsToModelParams(flags, &params);
  }
  Backend compare_backend = backend;
  const bool compare = flags.compare_to != "none";
  if (status.ok() && compare) {
    status = ParseBackend(flags.compare_to, &compare_backend);
  }
  if (status.ok()) {
    status = CheckBackendAvailable(backend);
  }
  if (status.ok() && compare) {
    status = CheckBackendAvailable(compare_backend);
  }
  if (!status.ok()) {
    return FailRun(status.message());
  }

  AppInput input;
  status = OpenInput(flags, params, &input);
  if (status.ok()) {
    status = ValidateSamplePixels(flags.sample_pixel_list, input.info.height,
                                  input.info.width);
  }
  if (status.ok()) {
    status = EnsureDirectory(flags.output_dir);
  }
  const size_t trace_slash = flags.trace_csv.find_last_of('/');
  if (status.ok() && trace_slash != std::string::npos && trace_slash > 0) {
    // The trace may be requested in a directory that does not exist yet.
    status = EnsureDirectory(flags.trace_csv.substr(0, trace_slash));
  }
  if (!status.ok()) {
    return FailRun(status.message());
  }
  LogInputSection(input);
  LogParametersSection(params, input.dates, flags.cleanup);

  const WallTimer wall_timer;
  double excluded_ms = 0;  // untimed work: comparison run, synthetic scoring
  // The reference is loaded before the run: GPU backends count the
  // validation inside the tile pipeline, while the results are still on the
  // device. With --cleanup the results change after the pipeline, so the
  // counts are made afterwards instead.
  ReferenceData reference;
  status = LoadReference(flags, input.info, &reference);
  if (!status.ok()) {
    return FailRun(status.message());
  }
  const ValidationWindow window = MakeValidationWindow(input.dates);
  const bool gpu = IsGpuBackend(backend);
  const bool validate_in_pipeline = gpu && reference.present && !flags.cleanup;
  PostResult post;
  GpuValidationRequest validation;
  validation.reference = &reference.raster;
  validation.window = window;
  validation.year = &input.dates.year;

  RunConfig config;
  status = FlagsToRunConfig(flags, backend, &config);
  DetectionResult result;
  StageTimings timings;
  RunStats stats;
  if (status.ok() && validate_in_pipeline) {
    // Availability of the GPU was checked above.
    status = RunGpuDetection(input.source.get(), params, config, &result,
                             &timings, &stats, &validation, &post.counts);
  } else if (status.ok()) {
    status = RunBackend(input.source.get(), params, config, &result, &timings,
                        &stats);
  }
  if (!status.ok()) {
    return FailRun(status.message());
  }
  LogExecutionSection(config, stats, input.info);
  if (!config.trace_csv.empty()) {
    LogKv("trace_csv", config.trace_csv);
  }

  Agreement agreement;
  if (compare) {
    const WallTimer compare_timer;
    RunConfig compare_config;
    DetectionResult reference_result;
    status = FlagsToRunConfig(flags, compare_backend, &compare_config);
    compare_config.trace_csv.clear();  // only the primary run is traced
    if (status.ok()) {
      status = RunBackend(input.source.get(), params, compare_config,
                          &reference_result, nullptr, nullptr);
    }
    if (status.ok()) {
      status = ComputeAgreement(result, reference_result, &agreement);
    }
    if (!status.ok()) {
      return FailRun("--compare-to " + flags.compare_to + ": " +
                     status.message());
    }
    excluded_ms += compare_timer.ElapsedMs();
  }

  status = PostProcess(flags, input, reference, window, backend,
                       validate_in_pipeline, &result, &timings, &post);
  if (!status.ok()) {
    return FailRun(status.message());
  }

  RecoveryStats recovery;
  if (input.synthetic) {
    const WallTimer recovery_timer;
    status = ComputeRecovery(input.source.get(), input.truth, result, params,
                             &recovery);
    if (!status.ok()) {
      return FailRun(status.message());
    }
    excluded_ms += recovery_timer.ElapsedMs();
  }

  const WallTimer write_timer;
  status = WriteOutputs(flags, input, params, result, post, reference, window);
  if (!status.ok()) {
    return FailRun(status.message());
  }
  timings.write_ms = write_timer.ElapsedMs();
  timings.wall_total_ms = wall_timer.ElapsedMs() - excluded_ms;

  const uint64_t pixels = static_cast<uint64_t>(input.info.height) *
                          static_cast<uint64_t>(input.info.width);
  const uint64_t samples = pixels * static_cast<uint64_t>(input.info.num_dates);
  LogTimingsSection(timings, samples, &stats);

  const StatusCounts counts = CountStatus(result);
  LogSection("results");
  Log("stable: " + WithPercent(counts.stable, pixels) +
      " | break: " + WithPercent(counts.breaks, pixels) + " [loss " +
      FormatCount(counts.loss) + ", gain " + FormatCount(counts.gain) +
      "] | insufficient_data: " + WithPercent(counts.insufficient, pixels));
  Log("reference: " + reference.path);
  if (post.has_validation) {
    LogKv("validation", FormatValidationLine(post.counts, post.metrics));
  }
  if (compare) {
    LogKv("agreement_vs_" + BackendName(compare_backend),
          "identical_break_index " + Fixed(agreement.break_frac, "%.4f") +
              " | identical_status " + Fixed(agreement.status_frac, "%.4f") +
              " | max_coef_diff " +
              (agreement.max_coef_diff < 0
                   ? std::string("-")
                   : Fixed(agreement.max_coef_diff, "%.3g")));
  }
  if (input.synthetic) {
    const auto percent = [](uint64_t part, uint64_t total) {
      return Fixed(total == 0 ? 0.0
                              : 100.0 * static_cast<double>(part) /
                                    static_cast<double>(total),
                   "%.1f");
    };
    LogKv("synthetic_recovery",
          "recovered " + FormatCount(recovery.recovered) + " of " +
              FormatCount(recovery.eligible) + " eligible (" +
              percent(recovery.recovered, recovery.eligible) +
              "%) | false_breaks " + FormatCount(recovery.false_breaks) +
              " of " + FormatCount(recovery.stable_truth) + " (" +
              percent(recovery.false_breaks, recovery.stable_truth) + "%)");
  }
  LogKv("outputs", DisplayPath(flags.output_dir, flags.output_dir));
  Log(SummaryLine(pixels, samples, timings.wall_total_ms / 1e3,
                  BackendName(backend)));
  return 0;
}

}  // namespace cd
