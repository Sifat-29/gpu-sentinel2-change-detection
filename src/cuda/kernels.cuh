// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Host-callable launchers of the CUDA kernels. The kernels
// themselves live in detection_kernels.cu (NDVI, fit, detect, fused; they
// share the __constant__ date tables) and postprocess_kernels.cu (colorize,
// cleanup, confusion, quick-diff). Every launcher validates its arguments,
// launches asynchronously on `config.stream` and checks cudaGetLastError().
#ifndef SRC_CUDA_KERNELS_CUH_
#define SRC_CUDA_KERNELS_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "status.h"
#include "types.h"

namespace cd {

// How a kernel is launched: threads per block and the stream.
struct LaunchConfig {
  int block_size = 256;
  cudaStream_t stream = nullptr;
};

// Raw bands of one tile on the device, date-major: element (t, pixel) is at
// t * pixels + pixel in each array.
struct RawBands {
  const uint16_t* red = nullptr;
  const uint16_t* nir = nullptr;
  const uint8_t* scl = nullptr;
};

// Per-pixel detection outputs on the device. All arrays have `pixels`
// elements except `coef`, which is coefficient-major (c * pixels + pixel) with
// 2 + 2K rows. `coef` may be null for the fused kernel (coefficients are then
// not written).
struct DetectionBuffers {
  float* coef = nullptr;
  float* rmse = nullptr;
  uint16_t* n_train = nullptr;
  int16_t* break_index = nullptr;
  float* magnitude = nullptr;
  uint8_t* status = nullptr;
};

// ---- Detection kernels (detection_kernels.cu) ------------------------------

// Uploads the per-date tables to __constant__ memory: the design table (built
// in FP64 on the host with BuildDesignTable, then cast to float) and the
// reflectance scale and offset of each date. `dates` must come from
// BuildDateTable. Fails when there are more than kMaxDates dates. Call once
// per run before any detection launcher; the tables are process-global.
Status UploadDateTables(const DateTable& dates, const ModelParams& params);

// NDVI of every (date, pixel) of a tile into `ndvi` (num_dates * pixels
// floats, date-major); invalid observations get kInvalidNdvi.
Status LaunchComputeNdvi(const RawBands& bands, int num_dates, int64_t pixels,
                         uint32_t valid_mask, float* ndvi,
                         const LaunchConfig& config);

// Fits the harmonic model of every pixel from the NDVI cube. Writes
// out.coef, out.rmse, out.n_train and out.status (kStable or
// kInsufficientData); out.coef is required here.
Status LaunchFitHarmonicModel(const float* ndvi, int64_t pixels,
                              int num_train_dates, const ModelParams& params,
                              const DetectionBuffers& out,
                              const LaunchConfig& config);

// Monitors every pixel against its fitted model. Reads io.coef, io.rmse and
// io.status (as written by the fit); writes io.break_index, io.magnitude and
// the final io.status.
Status LaunchDetectBreak(const float* ndvi, int64_t pixels, int num_dates,
                         int num_train_dates, const ModelParams& params,
                         const DetectionBuffers& io,
                         const LaunchConfig& config);

// NDVI, fit and detection in one kernel straight from the raw bands. Writes
// out.rmse, out.n_train, out.break_index, out.magnitude, out.status and, when
// non-null, out.coef.
Status LaunchFusedChangeDetection(const RawBands& bands, int64_t pixels,
                                  int num_dates, int num_train_dates,
                                  const ModelParams& params,
                                  const DetectionBuffers& out,
                                  const LaunchConfig& config);

// ---- Post-processing kernels (postprocess_kernels.cu) ----------------------

// Colours `pixels` pixels (4 bytes each into `rgba`, rule in colormap.h).
// `day` is the device array of day numbers of the `num_dates` dates.
Status LaunchColorizeChangeMap(const int16_t* break_index,
                               const float* magnitude, const uint8_t* status,
                               const int32_t* day, int num_dates,
                               int first_monitor_day, int last_day,
                               int64_t pixels, uint8_t* rgba,
                               const LaunchConfig& config);

// Cleanup of `rows` image rows of width `width`. `status_halo` holds rows + 2
// rows of status codes: the row above the chunk, the chunk, the row below
// (rows outside the image must be zero). The outputs `status_out`,
// `break_index` and `magnitude` hold the `rows` chunk rows; break_index and
// magnitude are updated in place.
Status LaunchCleanup(const uint8_t* status_halo, int rows, int width,
                     uint8_t* status_out, int16_t* break_index,
                     float* magnitude, cudaStream_t stream);

// Number of 64-bit counters written by LaunchConfusion, in this order: tp,
// fp, fn, tn, excluded, year_exact, year_within1.
inline constexpr int kNumConfusionCounters = 7;

// Adds the confusion counts of `pixels` pixels to `counters` (device array of
// kNumConfusionCounters uint64_t values, zeroed by the caller).
// `year` is the device array of calendar years of the `num_dates` dates.
Status LaunchConfusion(const uint8_t* status, const int16_t* break_index,
                       const float* magnitude, const uint8_t* reference,
                       const int16_t* year, int num_dates,
                       const ValidationWindow& window, int64_t pixels,
                       uint64_t* counters, cudaStream_t stream);

// Quick-diff of one tile: planes 0 and 1 of the bands are the two dates.
Status LaunchQuickDiff(const RawBands& bands, int64_t pixels,
                       const float scale[2], const float offset[2],
                       uint32_t valid_mask, float threshold, bool losses_only,
                       float* diff, uint8_t* flag, const LaunchConfig& config);

// ---- Test hooks (gpu_backend.cu) -------------------------------------------

// Pretend that the device has only `bytes` bytes for the detection and
// quick-diff slot buffers (0 = no limit), so that tests can exercise the
// "tile allocation failed; retrying" path deterministically.
void SetSlotAllocationLimitForTesting(size_t bytes);

// Limit post-processing chunks to `rows` image rows (0 = size from free
// memory), so that tests can exercise multi-chunk processing on small images.
void SetPostprocessChunkRowsForTesting(int rows);

}  // namespace cd

#endif  // SRC_CUDA_KERNELS_CUH_
