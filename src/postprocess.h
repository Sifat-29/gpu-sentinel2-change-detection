// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// CPU post-processing twins of the GPU functions in cuda/gpu_backend.h
// (declarations; implemented in postprocess.cc). They must give results that
// are byte-identical to the GPU versions for colorize and cleanup, and equal
// within 1e-6 for quick-diff.
#ifndef SRC_POSTPROCESS_H_
#define SRC_POSTPROCESS_H_

#include <cstdint>
#include <vector>

#include "cube_source.h"
#include "status.h"
#include "types.h"

namespace cd {

// Colours the change map into `rgba` (4 bytes per pixel, rule in colormap.h).
// `ms` (optional) receives the elapsed time.
Status CpuColorizeChangeMap(const DetectionResult& r, const DateTable& d,
                            std::vector<uint8_t>* rgba, double* ms);

// Removes isolated break pixels in place: a break pixel with fewer than 2
// break pixels among its 8 neighbours (evaluated on the unmodified input)
// becomes stable with break_index -1 and magnitude 0.
Status CpuCleanup(DetectionResult* r, double* ms);

// Quick two-date NDVI difference (see GpuQuickDiff).
Status CpuQuickDiff(CubeSource* source, int date_a, int date_b,
                    float diff_threshold, const ModelParams& p,
                    const RunConfig& c, QuickDiffResult* out, StageTimings* t);

}  // namespace cd

#endif  // SRC_POSTPROCESS_H_
