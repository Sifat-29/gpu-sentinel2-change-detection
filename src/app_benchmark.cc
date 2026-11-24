// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// The benchmark mode: warm-up and timed repetitions of one or more backends,
// one CSV row per timed repetition, and a median table at the end.
#include <cstdint>
#include <string>
#include <vector>

#include "app_common.h"
#include "app_modes.h"
#include "benchmark.h"
#include "logger.h"

namespace cd {
namespace {

// Number of pixels with status kBreak.
int64_t CountBreaks(const DetectionResult& result) {
  int64_t count = 0;
  for (const uint8_t status : result.status) {
    count += status == static_cast<uint8_t>(PixelStatus::kBreak) ? 1 : 0;
  }
  return count;
}

// Warm-up and timed runs of one backend. Appends the timed rows (agreement
// columns from the last repetition against `reference`, when given) and the
// timings to `samples`.
Status BenchmarkBackend(const Flags& flags, const ModelParams& params,
                        const AppInput& input, Backend backend,
                        const DetectionResult* reference,
                        const SystemColumns& system, BackendSamples* samples) {
  RunConfig config;
  Status status = FlagsToRunConfig(flags, backend, &config);
  if (!status.ok()) {
    return status;
  }
  // The only result kept for this backend; the reference is the second one.
  DetectionResult result;
  for (int i = 0; i < flags.warmup; ++i) {
    status = RunBackend(input.source.get(), params, config, &result, nullptr,
                        nullptr);
    if (!status.ok()) {
      return status;
    }
  }
  std::vector<BenchmarkRow> rows;
  samples->backend = BackendName(backend);
  for (int i = 1; i <= flags.repeat; ++i) {
    StageTimings timings;
    RunStats stats;
    status = RunBackend(input.source.get(), params, config, &result, &timings,
                        &stats);
    if (!status.ok()) {
      return status;
    }
    BenchmarkRow row =
        MakeBenchmarkRow(flags.run_id, input.info, config, timings, stats, i,
                         CountBreaks(result), system);
    rows.push_back(row);
    samples->total_ms.push_back(timings.total_ms);
    samples->kernel_ms.push_back(timings.kernel_ms);
    Log("bench: " + BackendName(backend) + " repeat " + std::to_string(i) +
        "/" + std::to_string(flags.repeat) + " backend_total " +
        FormatMs(timings.total_ms) + " ms kernel_total " +
        FormatMs(timings.kernel_ms) + " ms");
  }
  if (reference != nullptr) {
    Agreement agreement;
    status = ComputeAgreement(result, *reference, &agreement);
    if (!status.ok()) {
      return status;
    }
    for (BenchmarkRow& row : rows) {
      row.compare_to = flags.compare_to;
      row.agreement = agreement;
    }
  }
  return AppendBenchmarkRows(flags.benchmark_csv, rows);
}

}  // namespace

int RunBenchmarkMode(const Flags& flags) {
  std::vector<Backend> backends;
  Status status = Status::Ok();
  for (const std::string& name : flags.backends) {
    Backend backend;
    status = ParseBackend(name, &backend);
    if (!status.ok()) {
      return FailRun(status.message());
    }
    backends.push_back(backend);
  }
  ModelParams params;
  status = FlagsToModelParams(flags, &params);
  Backend compare_backend = backends.front();
  const bool compare = flags.compare_to != "none";
  if (status.ok() && compare) {
    status = ParseBackend(flags.compare_to, &compare_backend);
  }
  for (size_t i = 0; status.ok() && i < backends.size(); ++i) {
    status = CheckBackendAvailable(backends[i]);
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
    // Creates the CSV with its header, or fails now (not after the runs) when
    // an existing file has a different header.
    status = AppendBenchmarkRows(flags.benchmark_csv, {});
  }
  if (!status.ok()) {
    return FailRun(status.message());
  }
  LogInputSection(input);
  LogParametersSection(params, input.dates, false);
  LogKv("benchmark", "repeat " + std::to_string(flags.repeat) + " | warmup " +
                         std::to_string(flags.warmup) + " | csv " +
                         flags.benchmark_csv);

  // The reference result is computed once, untimed.
  DetectionResult reference;
  if (compare) {
    RunConfig config;
    status = FlagsToRunConfig(flags, compare_backend, &config);
    if (status.ok()) {
      status = RunBackend(input.source.get(), params, config, &reference,
                          nullptr, nullptr);
    }
    if (!status.ok()) {
      return FailRun("--compare-to " + flags.compare_to + ": " +
                     status.message());
    }
  }

  const SystemColumns system = CollectSystemColumns(flags.device);
  std::vector<BackendSamples> all_samples;
  for (const Backend backend : backends) {
    BackendSamples samples;
    status = BenchmarkBackend(flags, params, input, backend,
                              compare ? &reference : nullptr, system, &samples);
    if (!status.ok()) {
      return FailRun(BackendName(backend) + ": " + status.message());
    }
    all_samples.push_back(samples);
  }

  LogSection("results");
  for (const std::string& line : FormatMedianTable(all_samples)) {
    Log(line);
  }
  LogKv("benchmark_csv", DisplayPath(flags.benchmark_csv, flags.benchmark_csv));
  const uint64_t pixels = static_cast<uint64_t>(input.info.height) *
                          static_cast<uint64_t>(input.info.width);
  const uint64_t samples = pixels * static_cast<uint64_t>(input.info.num_dates);
  const double median_ms = Median(all_samples.front().total_ms);
  Log(SummaryLine(pixels, samples, median_ms / 1e3,
                  all_samples.front().backend));
  return 0;
}

}  // namespace cd
