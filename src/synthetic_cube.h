// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Synthetic cube generator with injected step changes at known dates, and the
// recovery metric that scores a detection result against the injected truth
// (file format: docs/FORMATS.md, "Synthetic cubes").
#ifndef SRC_SYNTHETIC_CUBE_H_
#define SRC_SYNTHETIC_CUBE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "cube_source.h"
#include "status.h"
#include "types.h"

namespace cd {

// Generator parameters. `threads` = 0 uses all hardware threads; the output
// does not depend on it.
struct SyntheticOptions {
  int height = 1024;
  int width = 1024;
  int num_dates = 48;
  float years = 4.0f;
  uint32_t seed = 42;
  float noise = 0.03f;
  float cloud_fraction = 0.2f;
  float break_fraction = 0.15f;
  float train_years = 2.0f;
  int consecutive = 3;
  int threads = 0;
};

// A whole cube in memory (date-major arrays, as on disk).
struct SyntheticCube {
  CubeInfo info;
  std::vector<uint16_t> red;
  std::vector<uint16_t> nir;
  std::vector<uint8_t> scl;
  std::vector<int16_t> truth_break_index;  // H x W, -1 = no injected break
};

// Generates the cube in memory. Every value depends only on (seed, pixel,
// date), so the result is identical for any `threads`.
Status GenerateSyntheticCube(const SyntheticOptions& options,
                             SyntheticCube* out);

// Writes manifest.txt, red.u16, nir.u16, scl.u8 and truth_break_index.i16 into
// `dir` (created if needed).
Status WriteSyntheticCube(const SyntheticCube& cube, const std::string& dir);

// Generates straight to disk, one date plane at a time, so memory use is
// O(H * W) whatever the number of dates. The files are byte-identical to
// WriteSyntheticCube(GenerateSyntheticCube(options)).
Status GenerateSyntheticCubeToDisk(const SyntheticOptions& options,
                                   const std::string& dir);

// Noise-free NDVI of one pixel at one date (including the injected step and
// the final clamp to [-0.1, 0.95]); `pixel` is row * width + col. Returns
// NaN if the options are invalid. Slow (rebuilds the date terms on every
// call); meant for tests.
double SyntheticCleanNdvi(const SyntheticOptions& options, int64_t pixel,
                          int date);

// Counts produced by ComputeRecovery.
struct RecoveryStats {
  uint64_t injected = 0;      // pixels with an injected break
  uint64_t eligible = 0;      // injected and enough valid observations
  uint64_t recovered = 0;     // eligible and detected within one observation
  uint64_t stable_truth = 0;  // pixels without an injected break
  uint64_t false_breaks = 0;  // stable_truth pixels that report kBreak
  // Injected pixels left out of `eligible` because they have fewer than
  // min_train_obs valid training observations (status "insufficient data" is
  // the designed outcome there, not a missed detection).
  uint64_t insufficient_training = 0;
};

// Scores `result` against `truth`. A pixel is "eligible" when it has at least
// min_train_obs valid observations in the training window (dates with
// tau < train_years; min_train_obs = 0 means 3 * (2 + 2K)) and at least
// params.consecutive valid observations at or after the injected index.
// "Recovered" means status kBreak and break_index between the first valid
// observation at/after the injected index and the next valid observation after
// it. An observation is valid when its SCL class is in params.valid_class_mask
// and both DNs are non-zero. Reads the cube once, tile by tile.
Status ComputeRecovery(CubeSource* source, const std::vector<int16_t>& truth,
                       const DetectionResult& result, const ModelParams& params,
                       RecoveryStats* out);

}  // namespace cd

#endif  // SRC_SYNTHETIC_CUBE_H_
