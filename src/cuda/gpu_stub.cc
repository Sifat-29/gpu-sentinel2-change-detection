// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <vector>

#include "cuda/gpu_backend.h"

// Stand-ins for the GPU API in builds that cannot run CUDA code:
//  - CPU-only builds (no CD_WITH_CUDA) get stubs for every function.
//  - GPU builds link this file only while cuda/gpu_backend.cu does not exist
//    yet, and then GpuAvailable/GpuDeviceInfo come from device_info.cu.

namespace cd {
namespace {

Status NoCuda() {
  return Status::Error("built without CUDA; build with 'make gpu'");
}

}  // namespace

#ifndef CD_WITH_CUDA
bool GpuAvailable() { return false; }

Status GpuDeviceInfo(int /*device*/, GpuInfo* /*info*/) { return NoCuda(); }
#endif  // CD_WITH_CUDA

Status RunGpuDetection(CubeSource* /*source*/, const ModelParams& /*params*/,
                       const RunConfig& /*config*/, DetectionResult* /*result*/,
                       StageTimings* /*timings*/, RunStats* /*stats*/,
                       const GpuValidationRequest* /*validation*/,
                       ConfusionCounts* /*counts*/) {
  return NoCuda();
}

Status GpuColorizeChangeMap(const DetectionResult& /*r*/,
                            const DateTable& /*d*/, int /*device*/,
                            std::vector<uint8_t>* /*rgba*/, double* /*ms*/) {
  return NoCuda();
}

Status GpuCleanup(DetectionResult* /*r*/, int /*device*/, double* /*ms*/) {
  return NoCuda();
}

Status GpuConfusion(const DetectionResult& /*r*/,
                    const std::vector<uint8_t>& /*reference*/,
                    const DateTable& /*d*/, const ValidationWindow& /*w*/,
                    int /*device*/, ConfusionCounts* /*counts*/,
                    double* /*ms*/) {
  return NoCuda();
}

Status GpuQuickDiff(CubeSource* /*source*/, int /*date_a*/, int /*date_b*/,
                    float /*diff_threshold*/, const ModelParams& /*p*/,
                    const RunConfig& /*c*/, QuickDiffResult* /*out*/,
                    StageTimings* /*t*/) {
  return NoCuda();
}

Status GpuPhaseCorrelation(const CoregGpuInput& /*input*/,
                           CoregGpuOutput* /*output*/) {
  return NoCuda();
}

}  // namespace cd
