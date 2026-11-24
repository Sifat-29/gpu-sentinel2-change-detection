// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Helpers shared by the run modes (app.cc, app_detect.cc, app_benchmark.cc):
// the log header and sections (format: docs/FORMATS.md, "Log format"), input
// opening, backend dispatch and number formatting.
#ifndef SRC_APP_COMMON_H_
#define SRC_APP_COMMON_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cube_source.h"
#include "flags.h"
#include "status.h"
#include "synthetic_cube.h"
#include "types.h"

namespace cd {

// An opened input cube with everything the modes need to know about it.
struct AppInput {
  std::unique_ptr<CubeSource> source;
  CubeInfo info;              // copy of source->info()
  DateTable dates;            // info.dates completed by BuildDateTable
  std::string manifest_path;  // DisplayPath of the manifest or "synthetic"
  bool synthetic = false;
  std::vector<int16_t> truth;  // injected break index (synthetic only)
  double generation_ms = -1;   // synthetic generation time, -1 if not used
};

// Text of `path` for logs and result files: relative to the current working
// directory when the path lies under it, otherwise `as_given` (the text the
// user typed, or the stored path). Keeps personal absolute paths out of logs
// that get committed.
std::string DisplayPath(const std::string& path, const std::string& as_given);

// Logs the banner, command, start time, git commit and the device section
// (always a "gpu:" and a "cpu:" line).
void LogHeader(const Flags& flags, int argc, char** argv);

// Logs "error: <message>" and returns 1, the exit code of a failed run.
int FailRun(const std::string& message);

// Generator options for `--input synthetic` and `--mode generate`.
SyntheticOptions SyntheticOptionsFromFlags(const Flags& flags);

// Error "backend 'gpu' requires a CUDA build and an NVIDIA GPU (see INSTALL)"
// when `backend` is a GPU backend and no GPU is usable; Ok otherwise.
Status CheckBackendAvailable(Backend backend);

// Opens the cube named by flags.input (manifest or synthetic). Synthetic
// generation is timed into input->generation_ms. `params` must be resolved.
Status OpenInput(const Flags& flags, const ModelParams& params,
                 AppInput* input);

// Logs the "--- input ---" section.
void LogInputSection(const AppInput& input);

// Logs the "--- parameters ---" section.
void LogParametersSection(const ModelParams& params, const DateTable& dates,
                          bool cleanup);

// Runs `config.backend` (GPU or CPU) on `source`.
Status RunBackend(CubeSource* source, const ModelParams& params,
                  const RunConfig& config, DetectionResult* result,
                  StageTimings* timings, RunStats* stats);

// Logs the "--- execution ---" section: backend line, notes and, unless
// config.quiet, one line per tile.
void LogExecutionSection(const RunConfig& config, const RunStats& stats,
                         const CubeInfo& info);

// Logs the "--- timings (ms) ---" section: the stage line, the throughput
// line (over `samples` samples) and, when `stats` is given, the device memory
// line.
void LogTimingsSection(const StageTimings& timings, uint64_t samples,
                       const RunStats* stats);

// "12.345", or "-" for a negative (not applicable) value.
std::string FormatMs(double ms);

// Plain-integer text of a count.
std::string FormatCount(uint64_t value);

// Throughput in million samples per second, "-" when ms <= 0.
std::string FormatThroughput(uint64_t samples, double ms);

// "SUMMARY: processed P pixel time series (S samples) in X seconds (backend
// B)".
std::string SummaryLine(uint64_t pixels, uint64_t samples, double seconds,
                        const std::string& backend);

}  // namespace cd

#endif  // SRC_APP_COMMON_H_
