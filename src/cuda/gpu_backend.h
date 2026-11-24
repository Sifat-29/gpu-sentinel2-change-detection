// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// GPU backend API. Pure C++ declarations (no CUDA types) so that ordinary
// .cc files can call into the CUDA code. In a CPU-only build every function
// except GpuAvailable() returns an error (see gpu_stub.cc).
#ifndef SRC_CUDA_GPU_BACKEND_H_
#define SRC_CUDA_GPU_BACKEND_H_

#include <cstdint>
#include <vector>

#include "cube_source.h"
#include "status.h"
#include "types.h"

namespace cd {

// True when this build includes CUDA and at least one device is present.
bool GpuAvailable();

// Fills `info` for CUDA device `device` (properties and current free memory).
Status GpuDeviceInfo(int device, GpuInfo* info);

// Validation computed inside the detection pipeline: every tile's reference
// rows travel to the device with the tile and the confusion kernel runs on
// the tile's results before they leave the device.
struct GpuValidationRequest {
  // Reference raster (Hansen lossyear), height x width bytes, row-major.
  const std::vector<uint8_t>* reference = nullptr;
  ValidationWindow window;
  // Calendar year of each date (DateTable::year), one entry per cube date.
  const std::vector<int16_t>* year = nullptr;
};

// Runs change detection on the GPU with config.backend kGpuFused (one fused
// kernel) or kGpuUnfused (NDVI, fit and detect kernels). Tiles are streamed
// through pinned buffers with config.streams (1 or 3) CUDA streams. `params`
// must be resolved. Fills `result`, and when non-null `timings` and `stats`.
// When `validation` is given (then `counts` is required), the confusion
// counts of the results against the reference are accumulated on the device
// during the run and returned in `counts`, identical to ComputeConfusionCpu
// on `result`; the kernel time is part of timings->kernel_ms.
Status RunGpuDetection(CubeSource* source, const ModelParams& params,
                       const RunConfig& config, DetectionResult* result,
                       StageTimings* timings, RunStats* stats,
                       const GpuValidationRequest* validation = nullptr,
                       ConfusionCounts* counts = nullptr);

// Colours the change map: `rgba` receives 4 bytes per pixel (see colormap.h).
// `ms` (optional) receives the elapsed GPU time.
Status GpuColorizeChangeMap(const DetectionResult& r, const DateTable& d,
                            int device, std::vector<uint8_t>* rgba, double* ms);

// Removes isolated break pixels in place (fewer than 2 break neighbours among
// the 8 neighbours become stable).
Status GpuCleanup(DetectionResult* r, int device, double* ms);

// Counts true/false positives and negatives against `reference` (H x W bytes,
// Hansen lossyear) inside the monitoring window `w` (see validation_rules.h).
Status GpuConfusion(const DetectionResult& r,
                    const std::vector<uint8_t>& reference, const DateTable& d,
                    const ValidationWindow& w, int device,
                    ConfusionCounts* counts, double* ms);

// Quick two-date NDVI difference between dates `date_a` and `date_b`, tiled
// and streamed like detection. Flags 0 no data, 1 no change, 2 loss, 3 gain.
Status GpuQuickDiff(CubeSource* source, int date_a, int date_b,
                    float diff_threshold, const ModelParams& p,
                    const RunConfig& c, QuickDiffResult* out, StageTimings* t);

// Input of the phase-correlation co-registration check: the same square
// window of `num_dates` scenes, already read and cropped on the host. Arrays
// are date-major: date t starts at element t * window * window.
struct CoregGpuInput {
  int window = 0;  // side n of the window, a power of two in 8..4096
  int num_dates = 0;
  int reference_date = 0;  // index of the reference scene
  int device = 0;
  const uint16_t* band = nullptr;  // digital numbers of the correlated band
  const uint8_t* valid = nullptr;  // 1 = usable observation, 0 = replaced
  const float* scale = nullptr;    // per date, reflectance = DN*scale+offset
  const float* offset = nullptr;
};

// GPU stage times of one co-registration run in milliseconds (CUDA events,
// summed over the date chunks). `fft_ms` includes the window preparation
// (valid mean, mean removal, Hann window) and both transform directions.
struct CoregGpuTimings {
  double setup_ms = 0;  // device selection, allocations, cuFFT plans
  double h2d_ms = 0;
  double fft_ms = 0;
  double spectrum_ms = 0;  // normalised cross-power spectrum
  double peak_ms = 0;      // argmax and sub-pixel refinement
  double d2h_ms = 0;
  int chunk_dates = 0;  // dates transformed per batch
};

// Per-date shift estimate. dy and dx are the displacement of the date's scene
// relative to the reference scene in pixels (positive = content moved down /
// to the right); `peak` is the correlation peak height, 1 for a perfect
// translation, near 0 without a match.
struct CoregGpuOutput {
  std::vector<float> dy;
  std::vector<float> dx;
  std::vector<float> peak;
  CoregGpuTimings timings;
};

// Estimates the translation of every date relative to the reference date by
// FFT phase correlation with cuFFT (details: README.md, "Co-registration
// check"). Dates are processed in batches sized from the free device memory.
Status GpuPhaseCorrelation(const CoregGpuInput& input, CoregGpuOutput* output);

}  // namespace cd

#endif  // SRC_CUDA_GPU_BACKEND_H_
