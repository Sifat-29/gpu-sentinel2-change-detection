// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Post-processing kernels: change-map colouring, isolated-pixel cleanup,
// validation counts and the quick two-date NDVI difference. They apply the
// rules of colormap.h, validation_rules.h and pixel_model.h, so that the GPU
// and CPU versions give identical results.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "colormap.h"
#include "cuda/cuda_check.cuh"
#include "cuda/kernels.cuh"
#include "pixel_model.h"
#include "types.h"
#include "validation_rules.h"

namespace cd {
namespace {

constexpr uint8_t kBreakCode = static_cast<uint8_t>(PixelStatus::kBreak);
constexpr uint8_t kStableCode = static_cast<uint8_t>(PixelStatus::kStable);

// Block shape of the cleanup stencil: 32 columns x 8 rows (one warp per row).
constexpr int kCleanupBlockX = 32;
constexpr int kCleanupBlockY = 8;

// 64-bit counter type of the confusion kernel: CUDA's atomicAdd is overloaded
// for unsigned long long only, not for uint64_t.
using AtomicCounter = unsigned long long;  // NOLINT(runtime/int) see above

// Threads per block of the colorize, confusion and quick-diff kernels.
constexpr int kPostBlockSize = 256;

// Largest grid of the grid-stride confusion kernel.
constexpr int64_t kMaxConfusionBlocks = 4096;

// ---------------------------------------------------------------------------
// ColorizeChangeMapKernel
// ---------------------------------------------------------------------------

// Purpose: RGBA colour of every pixel of the change map. Thread mapping: one
// thread per pixel. Memory pattern: coalesced reads of break index, magnitude
// and status, a tiny gather from the day table (cached), and one coalesced
// 4-byte store per thread (uchar4).
__global__ void ColorizeChangeMapKernel(const int16_t* __restrict__ break_index,
                                        const float* __restrict__ magnitude,
                                        const uint8_t* __restrict__ status,
                                        const int32_t* __restrict__ day,
                                        int num_dates, int first_monitor_day,
                                        int last_day, int64_t pixels,
                                        uchar4* __restrict__ rgba) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pixels) {
    return;
  }
  const int16_t index = break_index[i];
  // Pixels without a valid break date use the first monitoring day; the
  // colour rule ignores it for them.
  int break_day = first_monitor_day;
  if (index >= 0 && index < num_dates) {
    break_day = day[index];
  }
  uint8_t color[4];
  ChangeColor(index, magnitude[i], status[i], first_monitor_day, last_day,
              break_day, color);
  rgba[i] = make_uchar4(color[0], color[1], color[2], color[3]);
}

// ---------------------------------------------------------------------------
// CleanupKernel
// ---------------------------------------------------------------------------

// Purpose: turns a break pixel with fewer than 2 break pixels among its 8
// neighbours into a stable pixel (3x3 stencil, evaluated on the unmodified
// input). Thread mapping: a 32x8 block covers a 32x8 patch of the chunk, one
// thread per pixel. Memory pattern: the block first loads its patch plus a
// 1-pixel halo (10 rows of 34 break flags) into shared memory with coalesced
// row reads, so each status byte is read from global memory about 1.4 times
// instead of 9. `status_halo` has rows + 2 rows (the row above and below the
// chunk are included); `status_out` receives the new status of every chunk
// pixel.
__global__ void CleanupKernel(const uint8_t* __restrict__ status_halo, int rows,
                              int width, uint8_t* __restrict__ status_out,
                              int16_t* __restrict__ break_index,
                              float* __restrict__ magnitude) {
  constexpr int kTileX = kCleanupBlockX + 2;
  constexpr int kTileY = kCleanupBlockY + 2;
  __shared__ uint8_t is_break[kTileY][kTileX];

  const int x0 = blockIdx.x * kCleanupBlockX;
  const int y0 = blockIdx.y * kCleanupBlockY;
  const int thread = threadIdx.y * kCleanupBlockX + threadIdx.x;
  for (int i = thread; i < kTileX * kTileY;
       i += kCleanupBlockX * kCleanupBlockY) {
    const int ty = i / kTileX;
    const int tx = i % kTileX;
    const int gx = x0 + tx - 1;  // image column
    const int gy = y0 + ty - 1;  // chunk row; -1 and `rows` are halo rows
    uint8_t flag = 0;
    if (gx >= 0 && gx < width && gy >= -1 && gy <= rows) {
      flag =
          status_halo[static_cast<int64_t>(gy + 1) * width + gx] == kBreakCode
              ? 1
              : 0;
    }
    is_break[ty][tx] = flag;
  }
  __syncthreads();

  const int x = x0 + threadIdx.x;
  const int y = y0 + threadIdx.y;
  if (x >= width || y >= rows) {
    return;
  }
  const int64_t index = static_cast<int64_t>(y) * width + x;
  const uint8_t own = status_halo[static_cast<int64_t>(y + 1) * width + x];
  uint8_t result = own;
  if (own == kBreakCode) {
    const int ly = threadIdx.y + 1;
    const int lx = threadIdx.x + 1;
    const int neighbours = is_break[ly - 1][lx - 1] + is_break[ly - 1][lx] +
                           is_break[ly - 1][lx + 1] + is_break[ly][lx - 1] +
                           is_break[ly][lx + 1] + is_break[ly + 1][lx - 1] +
                           is_break[ly + 1][lx] + is_break[ly + 1][lx + 1];
    if (neighbours < 2) {
      result = kStableCode;
      break_index[index] = -1;
      magnitude[index] = 0.0f;
    }
  }
  status_out[index] = result;
}

// ---------------------------------------------------------------------------
// ConfusionKernel
// ---------------------------------------------------------------------------

// Purpose: confusion counts of the detection against the reference raster
// (rule in validation_rules.h). Thread mapping: grid-stride loop, one pixel per
// iteration. Memory pattern: coalesced reads of status and reference; break
// index, magnitude and the year table are read only for break pixels. Each
// thread keeps seven private counters; they are reduced with warp shuffles,
// then across warps in shared memory, and thread 0 adds the block's totals to
// the global counters with one atomicAdd per counter.
__global__ void ConfusionKernel(const uint8_t* __restrict__ status,
                                const int16_t* __restrict__ break_index,
                                const float* __restrict__ magnitude,
                                const uint8_t* __restrict__ reference,
                                const int16_t* __restrict__ year, int num_dates,
                                ValidationWindow window, int64_t pixels,
                                AtomicCounter* __restrict__ counters) {
  AtomicCounter local[kNumConfusionCounters] = {0, 0, 0, 0, 0, 0, 0};
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < pixels; i += step) {
    const uint8_t pixel_status = status[i];
    int break_year = 0;
    int16_t index = -1;
    float change = 0.0f;
    if (pixel_status == kBreakCode) {
      index = break_index[i];
      change = magnitude[i];
      if (index >= 0 && index < num_dates) {
        break_year = year[index];
      }
    }
    const PixelValidation v =
        ClassifyPixel(pixel_status, break_year, change, reference[i], window);
    switch (v.cls) {
      case ValidationClass::kTruePositive:
        ++local[0];
        break;
      case ValidationClass::kFalsePositive:
        ++local[1];
        break;
      case ValidationClass::kFalseNegative:
        ++local[2];
        break;
      case ValidationClass::kTrueNegative:
        ++local[3];
        break;
      default:
        ++local[4];
        break;
    }
    local[5] += v.year_exact ? 1 : 0;
    local[6] += v.year_within1 ? 1 : 0;
  }

  // Warp-level reduction of every counter.
  for (int c = 0; c < kNumConfusionCounters; ++c) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      local[c] += __shfl_down_sync(0xffffffffu, local[c], offset);
    }
  }
  __shared__ AtomicCounter warp_totals[32][kNumConfusionCounters];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    for (int c = 0; c < kNumConfusionCounters; ++c) {
      warp_totals[warp][c] = local[c];
    }
  }
  __syncthreads();
  // Thread 0 adds the (at most 32) warp totals and publishes the block total.
  if (threadIdx.x == 0) {
    const int num_warps = blockDim.x >> 5;
    for (int c = 0; c < kNumConfusionCounters; ++c) {
      AtomicCounter sum = 0;
      for (int w = 0; w < num_warps; ++w) {
        sum += warp_totals[w][c];
      }
      atomicAdd(&counters[c], sum);
    }
  }
}

// ---------------------------------------------------------------------------
// QuickDiffKernel
// ---------------------------------------------------------------------------

// Purpose: NDVI difference between two dates of a tile and its loss/gain flag
// (QuickDiffPixel). Thread mapping: one thread per pixel. Memory pattern: six
// coalesced reads (red, nir, scl of the two date planes) and two coalesced
// writes (diff, flag); purely memory bound.
__global__ void QuickDiffKernel(const uint16_t* __restrict__ red,
                                const uint16_t* __restrict__ nir,
                                const uint8_t* __restrict__ scl, int64_t pixels,
                                float scale_a, float offset_a, float scale_b,
                                float offset_b, uint32_t valid_mask,
                                float threshold, bool losses_only,
                                float* __restrict__ diff,
                                uint8_t* __restrict__ flag) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pixels) {
    return;
  }
  const float ndvi_a =
      ComputeNdvi<float>(red[i], nir[i], scl[i], scale_a, offset_a, valid_mask);
  const int64_t j = pixels + i;
  const float ndvi_b =
      ComputeNdvi<float>(red[j], nir[j], scl[j], scale_b, offset_b, valid_mask);
  QuickDiffPixel(ndvi_a, ndvi_b, threshold, losses_only, &diff[i], &flag[i]);
}

// 1D grid of blocks covering `count` threads.
unsigned BlocksFor(int64_t count, int block_size) {
  return static_cast<unsigned>((count + block_size - 1) / block_size);
}

}  // namespace

Status LaunchColorizeChangeMap(const int16_t* break_index,
                               const float* magnitude, const uint8_t* status,
                               const int32_t* day, int num_dates,
                               int first_monitor_day, int last_day,
                               int64_t pixels, uint8_t* rgba,
                               const LaunchConfig& config) {
  if (pixels < 1) {
    return Status::Error("colorize launch: no pixels");
  }
  ColorizeChangeMapKernel<<<BlocksFor(pixels, kPostBlockSize), kPostBlockSize,
                            0, config.stream>>>(
      break_index, magnitude, status, day, num_dates, first_monitor_day,
      last_day, pixels, reinterpret_cast<uchar4*>(rgba));
  CD_CUDA_CHECK(cudaGetLastError());
  return Status::Ok();
}

Status LaunchCleanup(const uint8_t* status_halo, int rows, int width,
                     uint8_t* status_out, int16_t* break_index,
                     float* magnitude, cudaStream_t stream) {
  if (rows < 1 || width < 1) {
    return Status::Error("cleanup launch: empty chunk");
  }
  const dim3 block(kCleanupBlockX, kCleanupBlockY);
  const dim3 grid(
      static_cast<unsigned>((width + kCleanupBlockX - 1) / kCleanupBlockX),
      static_cast<unsigned>((rows + kCleanupBlockY - 1) / kCleanupBlockY));
  CleanupKernel<<<grid, block, 0, stream>>>(status_halo, rows, width,
                                            status_out, break_index, magnitude);
  CD_CUDA_CHECK(cudaGetLastError());
  return Status::Ok();
}

Status LaunchConfusion(const uint8_t* status, const int16_t* break_index,
                       const float* magnitude, const uint8_t* reference,
                       const int16_t* year, int num_dates,
                       const ValidationWindow& window, int64_t pixels,
                       uint64_t* counters, cudaStream_t stream) {
  static_assert(sizeof(uint64_t) == sizeof(AtomicCounter),
                "atomicAdd needs a 64-bit counter");
  if (pixels < 1) {
    return Status::Error("confusion launch: no pixels");
  }
  const int64_t blocks = std::min<int64_t>(
      (pixels + kPostBlockSize - 1) / kPostBlockSize, kMaxConfusionBlocks);
  ConfusionKernel<<<static_cast<unsigned>(blocks), kPostBlockSize, 0, stream>>>(
      status, break_index, magnitude, reference, year, num_dates, window,
      pixels, reinterpret_cast<AtomicCounter*>(counters));
  CD_CUDA_CHECK(cudaGetLastError());
  return Status::Ok();
}

Status LaunchQuickDiff(const RawBands& bands, int64_t pixels,
                       const float scale[2], const float offset[2],
                       uint32_t valid_mask, float threshold, bool losses_only,
                       float* diff, uint8_t* flag, const LaunchConfig& config) {
  if (pixels < 1) {
    return Status::Error("quick-diff launch: no pixels");
  }
  CD_RETURN_IF_ERROR(
      config.block_size >= 32 && config.block_size <= 1024 &&
              config.block_size % 32 == 0
          ? Status::Ok()
          : Status::Error("block size must be a multiple of 32 in 32..1024"));
  QuickDiffKernel<<<BlocksFor(pixels, config.block_size), config.block_size, 0,
                    config.stream>>>(
      bands.red, bands.nir, bands.scl, pixels, scale[0], offset[0], scale[1],
      offset[1], valid_mask, threshold, losses_only, diff, flag);
  CD_CUDA_CHECK(cudaGetLastError());
  return Status::Ok();
}

}  // namespace cd
