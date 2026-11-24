// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// The four detection kernels (NDVI, harmonic fit, break detection and the
// fused kernel) and the __constant__ per-date tables they share. The tables
// are per translation unit (the build does not use relocatable device code),
// which is why all four kernels and the upload function live in this file.
//
// The per-pixel mathematics is not written here: every kernel calls the
// CD_HOST_DEVICE templates of pixel_model.h instantiated with float, the same
// source the CPU baselines run.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda/cuda_check.cuh"
#include "cuda/kernels.cuh"
#include "pixel_model.h"
#include "types.h"

namespace cd {
namespace {

// Per-date tables, uploaded once per run. Every thread of a warp reads the
// same date at the same time, so the constant cache broadcasts each value.
// 1024 * 8 * 4 B (design) + 2 * 1024 * 4 B (scale, offset) = 40 KiB, inside
// the 64 KiB constant bank.
__constant__ float kDesignTable[kMaxDates * kMaxParams];
__constant__ float kScaleTable[kMaxDates];
__constant__ float kOffsetTable[kMaxDates];

// Largest grid of the grid-stride NDVI kernel: enough blocks to fill any
// current GPU many times over without launching millions of blocks.
constexpr int64_t kMaxNdviBlocks = 1 << 16;

// ---------------------------------------------------------------------------
// ComputeNdviKernel
// ---------------------------------------------------------------------------

// Purpose: NDVI of every observation of a tile (the unfused baseline's first
// pass). Thread mapping: one thread per (date, pixel) pair, grid-stride loop
// over the date-major index i = t * pixels + pixel. Memory pattern: adjacent
// threads handle adjacent pixels of the same date, so the loads of red, nir
// and scl and the store of the float NDVI are coalesced; scale and offset are
// read from constant memory (one date per warp in practice).
__global__ void ComputeNdviKernel(const uint16_t* __restrict__ red,
                                  const uint16_t* __restrict__ nir,
                                  const uint8_t* __restrict__ scl,
                                  float* __restrict__ ndvi, int num_dates,
                                  int64_t pixels, uint32_t valid_mask) {
  const int64_t total = static_cast<int64_t>(num_dates) * pixels;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += step) {
    const int t = static_cast<int>(i / pixels);
    ndvi[i] = ComputeNdvi<float>(red[i], nir[i], scl[i], kScaleTable[t],
                                 kOffsetTable[t], valid_mask);
  }
}

// ---------------------------------------------------------------------------
// FitHarmonicModelKernel
// ---------------------------------------------------------------------------

// Purpose: least-squares fit of the trend + K harmonics model of every pixel
// (normal equations in registers, Cholesky solve, RMSE), from the NDVI cube.
// Thread mapping: one thread per pixel. Memory pattern: the thread walks the
// training dates of its pixel; at each date adjacent threads read adjacent
// floats (coalesced, stride `pixels` between dates). Outputs are
// coefficient-major (c * pixels + pixel) so that stores are coalesced too.
template <int kHarmonics>
__global__ void FitHarmonicModelKernel(
    const float* __restrict__ ndvi, int64_t pixels, int num_train_dates,
    ModelParams params, float* __restrict__ coef, float* __restrict__ rmse,
    uint16_t* __restrict__ n_train, uint8_t* __restrict__ status) {
  constexpr int kParams = 2 + 2 * kHarmonics;
  const int64_t pixel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixels) {
    return;
  }
  const NdviArraySeries<float> series{ndvi + pixel, pixels};
  const FitOutput<float> fit = FitModel<float, kHarmonics>(
      series, kDesignTable, num_train_dates, params);
  for (int c = 0; c < kParams; ++c) {
    coef[c * pixels + pixel] = fit.coef[c];
  }
  rmse[pixel] = fit.rmse;
  n_train[pixel] = static_cast<uint16_t>(min(fit.n_train, 65535));
  status[pixel] = static_cast<uint8_t>(fit.ok ? PixelStatus::kStable
                                              : PixelStatus::kInsufficientData);
}

// ---------------------------------------------------------------------------
// DetectBreakKernel
// ---------------------------------------------------------------------------

// Purpose: monitoring pass of every pixel against its fitted model (run of
// consecutive same-sign exceedances). Thread mapping: one thread per pixel.
// Memory pattern: coefficients, RMSE and status are read once (coalesced,
// coefficient-major); the monitoring dates of the NDVI cube are read with a
// stride of `pixels` between dates, coalesced across the warp. `status` comes
// in as the fit result and leaves as the final status.
template <int kHarmonics>
__global__ void DetectBreakKernel(
    const float* __restrict__ ndvi, int64_t pixels, int num_dates,
    int num_train_dates, ModelParams params, const float* __restrict__ coef,
    const float* __restrict__ rmse, uint8_t* status,
    int16_t* __restrict__ break_index, float* __restrict__ magnitude) {
  constexpr int kParams = 2 + 2 * kHarmonics;
  const int64_t pixel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixels) {
    return;
  }
  FitOutput<float> fit;
  for (int c = 0; c < kMaxParams; ++c) {
    fit.coef[c] = c < kParams ? coef[c * pixels + pixel] : 0.0f;
  }
  fit.rmse = rmse[pixel];
  fit.n_train = 0;  // not used by DetectBreak
  fit.ok =
      status[pixel] != static_cast<uint8_t>(PixelStatus::kInsufficientData);
  const NdviArraySeries<float> series{ndvi + pixel, pixels};
  const DetectOutput<float> detect = DetectBreak<float, kHarmonics>(
      fit, series, kDesignTable, num_dates, num_train_dates, params);
  break_index[pixel] = detect.break_index;
  magnitude[pixel] = detect.magnitude;
  status[pixel] = detect.status;
}

// ---------------------------------------------------------------------------
// FusedChangeDetectionKernel
// ---------------------------------------------------------------------------

// Purpose: NDVI, fit and detection in a single pass, straight from the raw
// bands, so that the NDVI cube is never stored. Thread mapping: one thread per
// pixel. Memory pattern: NDVI is recomputed from red, nir and scl in each of
// the three passes over the dates (training sums, residual sum, monitoring);
// adjacent threads read adjacent pixels of a date (coalesced). `coef` may be
// null, in which case the coefficients are not written.
template <int kHarmonics>
__global__ void FusedChangeDetectionKernel(
    const uint16_t* __restrict__ red, const uint16_t* __restrict__ nir,
    const uint8_t* __restrict__ scl, int64_t pixels, int num_dates,
    int num_train_dates, ModelParams params, float* __restrict__ coef,
    float* __restrict__ rmse, uint16_t* __restrict__ n_train,
    int16_t* __restrict__ break_index, float* __restrict__ magnitude,
    uint8_t* __restrict__ status) {
  constexpr int kParams = 2 + 2 * kHarmonics;
  const int64_t pixel =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixels) {
    return;
  }
  const RawSeries<float> series{red + pixel,
                                nir + pixel,
                                scl + pixel,
                                pixels,
                                kScaleTable,
                                kOffsetTable,
                                params.valid_class_mask};
  const PixelOutput<float> out = FitAndDetect<float, kHarmonics>(
      series, kDesignTable, num_dates, num_train_dates, params);
  if (coef != nullptr) {
    for (int c = 0; c < kParams; ++c) {
      coef[c * pixels + pixel] = out.fit.coef[c];
    }
  }
  rmse[pixel] = out.fit.rmse;
  n_train[pixel] = static_cast<uint16_t>(min(out.fit.n_train, 65535));
  break_index[pixel] = out.detect.break_index;
  magnitude[pixel] = out.detect.magnitude;
  status[pixel] = out.detect.status;
}

// ---------------------------------------------------------------------------
// Host-side launch helpers
// ---------------------------------------------------------------------------

Status CheckBlockSize(int block_size) {
  if (block_size < 32 || block_size > 1024 || block_size % 32 != 0) {
    return Status::Error(
        "block size must be a multiple of 32 in 32..1024, got " +
        std::to_string(block_size));
  }
  return Status::Ok();
}

// Fails with a clear message when `block_size` exceeds the number of threads
// per block that the register use of `kernel` allows (K=3 needs about 70
// registers per thread, which limits a block to fewer than 1024 threads).
// `what` names the kernel in the message, e.g. "K=3 fused kernel".
template <typename Kernel>
Status CheckBlockFitsKernel(Kernel kernel, int block_size,
                            const std::string& what) {
  cudaFuncAttributes attributes;
  CD_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  const int limit = attributes.maxThreadsPerBlock;
  if (block_size <= limit) {
    return Status::Ok();
  }
  return Status::Error("--block-size " + std::to_string(block_size) +
                       " exceeds the " + std::to_string(limit) +
                       "-thread limit of the " + what +
                       "; use <= " + std::to_string(limit / 32 * 32));
}

// Number of blocks that cover `count` threads of `block_size`.
int64_t BlocksFor(int64_t count, int block_size) {
  return (count + block_size - 1) / block_size;
}

Status CheckDetectionShape(int64_t pixels, int num_dates, int num_train_dates,
                           const ModelParams& params, int block_size) {
  CD_RETURN_IF_ERROR(CheckBlockSize(block_size));
  if (pixels < 1 || num_dates < 1 || num_dates > kMaxDates) {
    return Status::Error("detection launch: " + std::to_string(pixels) +
                         " pixels and " + std::to_string(num_dates) +
                         " dates are not valid (dates must be 1.." +
                         std::to_string(kMaxDates) + ")");
  }
  if (num_train_dates < 1 || num_train_dates >= num_dates) {
    return Status::Error(
        "detection launch: " + std::to_string(num_train_dates) +
        " training dates out of " + std::to_string(num_dates));
  }
  if (params.harmonics < 1 || params.harmonics > kMaxHarmonics) {
    return Status::Error("harmonics must be in 1..3, got " +
                         std::to_string(params.harmonics));
  }
  return Status::Ok();
}

template <int kHarmonics>
Status FitLaunch(const float* ndvi, int64_t pixels, int num_train_dates,
                 const ModelParams& params, const DetectionBuffers& out,
                 const LaunchConfig& config) {
  CD_RETURN_IF_ERROR(CheckBlockFitsKernel(
      FitHarmonicModelKernel<kHarmonics>, config.block_size,
      "K=" + std::to_string(kHarmonics) + " fit kernel"));
  FitHarmonicModelKernel<kHarmonics>
      <<<static_cast<unsigned>(BlocksFor(pixels, config.block_size)),
         config.block_size, 0, config.stream>>>(ndvi, pixels, num_train_dates,
                                                params, out.coef, out.rmse,
                                                out.n_train, out.status);
  CD_CUDA_CHECK(cudaGetLastError());
  return Status::Ok();
}

template <int kHarmonics>
Status DetectLaunch(const float* ndvi, int64_t pixels, int num_dates,
                    int num_train_dates, const ModelParams& params,
                    const DetectionBuffers& io, const LaunchConfig& config) {
  CD_RETURN_IF_ERROR(CheckBlockFitsKernel(
      DetectBreakKernel<kHarmonics>, config.block_size,
      "K=" + std::to_string(kHarmonics) + " detect kernel"));
  DetectBreakKernel<kHarmonics>
      <<<static_cast<unsigned>(BlocksFor(pixels, config.block_size)),
         config.block_size, 0, config.stream>>>(
          ndvi, pixels, num_dates, num_train_dates, params, io.coef, io.rmse,
          io.status, io.break_index, io.magnitude);
  CD_CUDA_CHECK(cudaGetLastError());
  return Status::Ok();
}

template <int kHarmonics>
Status FusedLaunch(const RawBands& bands, int64_t pixels, int num_dates,
                   int num_train_dates, const ModelParams& params,
                   const DetectionBuffers& out, const LaunchConfig& config) {
  CD_RETURN_IF_ERROR(CheckBlockFitsKernel(
      FusedChangeDetectionKernel<kHarmonics>, config.block_size,
      "K=" + std::to_string(kHarmonics) + " fused kernel"));
  FusedChangeDetectionKernel<kHarmonics>
      <<<static_cast<unsigned>(BlocksFor(pixels, config.block_size)),
         config.block_size, 0, config.stream>>>(
          bands.red, bands.nir, bands.scl, pixels, num_dates, num_train_dates,
          params, out.coef, out.rmse, out.n_train, out.break_index,
          out.magnitude, out.status);
  CD_CUDA_CHECK(cudaGetLastError());
  return Status::Ok();
}

}  // namespace

Status UploadDateTables(const DateTable& dates, const ModelParams& params) {
  const size_t num_dates = dates.day.size();
  if (num_dates == 0 || num_dates > static_cast<size_t>(kMaxDates)) {
    return Status::Error("the GPU backend supports 1.." +
                         std::to_string(kMaxDates) + " dates, got " +
                         std::to_string(num_dates));
  }
  if (dates.tau.size() != num_dates || dates.scale.size() != num_dates ||
      dates.offset.size() != num_dates) {
    return Status::Error(
        "date table is incomplete (tau, scale and offset need one entry per "
        "date; run BuildDateTable)");
  }
  if (params.harmonics < 1 || params.harmonics > kMaxHarmonics) {
    return Status::Error("harmonics must be in 1..3, got " +
                         std::to_string(params.harmonics));
  }
  // The design rows depend only on the date: evaluate them once in FP64 on
  // the host (no device libm differences) and cast to float.
  std::vector<double> table;
  BuildDesignTable(dates, params.harmonics, params.train_years,
                   params.hold_trend, &table);
  const std::vector<float> design(table.begin(), table.end());
  CD_CUDA_CHECK(cudaMemcpyToSymbol(kDesignTable, design.data(),
                                   design.size() * sizeof(float)));
  CD_CUDA_CHECK(cudaMemcpyToSymbol(kScaleTable, dates.scale.data(),
                                   num_dates * sizeof(float)));
  CD_CUDA_CHECK(cudaMemcpyToSymbol(kOffsetTable, dates.offset.data(),
                                   num_dates * sizeof(float)));
  return Status::Ok();
}

Status LaunchComputeNdvi(const RawBands& bands, int num_dates, int64_t pixels,
                         uint32_t valid_mask, float* ndvi,
                         const LaunchConfig& config) {
  CD_RETURN_IF_ERROR(CheckBlockSize(config.block_size));
  if (pixels < 1 || num_dates < 1 || num_dates > kMaxDates) {
    return Status::Error("NDVI launch: " + std::to_string(pixels) +
                         " pixels and " + std::to_string(num_dates) +
                         " dates are not valid");
  }
  const int64_t total = static_cast<int64_t>(num_dates) * pixels;
  const int64_t blocks =
      std::min(BlocksFor(total, config.block_size), kMaxNdviBlocks);
  ComputeNdviKernel<<<static_cast<unsigned>(blocks), config.block_size, 0,
                      config.stream>>>(bands.red, bands.nir, bands.scl, ndvi,
                                       num_dates, pixels, valid_mask);
  CD_CUDA_CHECK(cudaGetLastError());
  return Status::Ok();
}

Status LaunchFitHarmonicModel(const float* ndvi, int64_t pixels,
                              int num_train_dates, const ModelParams& params,
                              const DetectionBuffers& out,
                              const LaunchConfig& config) {
  CD_RETURN_IF_ERROR(CheckDetectionShape(
      pixels, num_train_dates + 1, num_train_dates, params, config.block_size));
  if (out.coef == nullptr) {
    return Status::Error("fit launch: the coefficient buffer is required");
  }
  switch (params.harmonics) {
    case 1:
      return FitLaunch<1>(ndvi, pixels, num_train_dates, params, out, config);
    case 2:
      return FitLaunch<2>(ndvi, pixels, num_train_dates, params, out, config);
    default:
      return FitLaunch<3>(ndvi, pixels, num_train_dates, params, out, config);
  }
}

Status LaunchDetectBreak(const float* ndvi, int64_t pixels, int num_dates,
                         int num_train_dates, const ModelParams& params,
                         const DetectionBuffers& io,
                         const LaunchConfig& config) {
  CD_RETURN_IF_ERROR(CheckDetectionShape(pixels, num_dates, num_train_dates,
                                         params, config.block_size));
  switch (params.harmonics) {
    case 1:
      return DetectLaunch<1>(ndvi, pixels, num_dates, num_train_dates, params,
                             io, config);
    case 2:
      return DetectLaunch<2>(ndvi, pixels, num_dates, num_train_dates, params,
                             io, config);
    default:
      return DetectLaunch<3>(ndvi, pixels, num_dates, num_train_dates, params,
                             io, config);
  }
}

Status LaunchFusedChangeDetection(const RawBands& bands, int64_t pixels,
                                  int num_dates, int num_train_dates,
                                  const ModelParams& params,
                                  const DetectionBuffers& out,
                                  const LaunchConfig& config) {
  CD_RETURN_IF_ERROR(CheckDetectionShape(pixels, num_dates, num_train_dates,
                                         params, config.block_size));
  switch (params.harmonics) {
    case 1:
      return FusedLaunch<1>(bands, pixels, num_dates, num_train_dates, params,
                            out, config);
    case 2:
      return FusedLaunch<2>(bands, pixels, num_dates, num_train_dates, params,
                            out, config);
    default:
      return FusedLaunch<3>(bands, pixels, num_dates, num_train_dates, params,
                            out, config);
  }
}

}  // namespace cd
