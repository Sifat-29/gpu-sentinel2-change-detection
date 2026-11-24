// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// CPU detection backends (declarations; implemented in cpu_backend.cc).
#ifndef SRC_CPU_CPU_BACKEND_H_
#define SRC_CPU_CPU_BACKEND_H_

#include "cube_source.h"
#include "status.h"
#include "types.h"

namespace cd {

// Runs change detection on the CPU backend selected by config.backend:
// kCpuFp64 (one thread, FP64 reference), kCpuFp32 (one thread, FP32) or
// kCpuMt (FP32 on config.threads workers; 0 = all hardware threads). The cube
// is processed in row-band tiles of config.tile_rows rows (0 = 256); a
// prefetch thread reads tile k+1 while tile k is computed. `params` must
// already be resolved (ResolveModelParams) and the date table built. Fills
// `result` (allocated by the callee), and, when non-null, `timings` and
// `stats`.
Status RunCpuDetection(CubeSource* source, const ModelParams& params,
                       const RunConfig& config, DetectionResult* result,
                       StageTimings* timings, RunStats* stats);

// Recomputes the series of pixel (row, col) in FP64: NDVI, fit, predicted
// values, z-scores and the break run. Used for trajectories.csv and
// pixel_stats.csv.
Status ComputeTrajectory(CubeSource* source, const ModelParams& params, int row,
                         int col, PixelTrajectory* out);

}  // namespace cd

#endif  // SRC_CPU_CPU_BACKEND_H_
