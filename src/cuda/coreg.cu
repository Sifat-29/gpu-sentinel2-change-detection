// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Co-registration check on the GPU: FFT phase correlation of every date of a
// square window against a reference date, with cuFFT for the transforms and
// small kernels for the window preparation, the normalised cross-power
// spectrum and the peak search.
//
// Data flow for a batch of dates (all arrays date-major, n = window side):
//   raw DN + valid flags  --PrepareKernel-->  real image   (n x n floats)
//   real image            --cuFFT R2C------>  spectrum     (n x (n/2+1)
//   complex) spectrum, ref spectrum --CrossPowerKernel-> unit-magnitude
//   spectrum spectrum              --cuFFT C2R------>  correlation surface
//   (real) correlation surface   --PeakKernel----->  (dy, dx, peak) per date
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda/cuda_check.cuh"
#include "cuda/gpu_backend.h"
#include "timing.h"

// Evaluates a cuFFT call and returns its error as a cd::Status.
#define CD_CUFFT_CHECK(expr)                                              \
  do {                                                                    \
    const cufftResult cd_cufft_result = (expr);                           \
    if (cd_cufft_result != CUFFT_SUCCESS) {                               \
      return ::cd::Status::Error(                                         \
          std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + \
          #expr + " failed with cuFFT error " +                           \
          std::to_string(static_cast<int>(cd_cufft_result)));             \
    }                                                                     \
  } while (0)

namespace cd {
namespace {

constexpr int kBlockThreads = 256;   // threads of the element-wise kernels
constexpr int kPeakThreads = 1024;   // threads of the one-block-per-date peak
constexpr int kMaxMeanBlocks = 128;  // blocks per date in the mean kernel
// Cross-power bins weaker than this are set to zero instead of normalised
// (guards the division for empty dates and exactly zero frequencies).
constexpr float kCrossPowerEpsilon = 1e-10f;
// Fraction of the free device memory the batch buffers may use.
constexpr double kMemoryFraction = 0.6;
constexpr int kNumEvents = 8;

// Counter type of the valid-pixel tally: CUDA's atomicAdd has an overload for
// unsigned long long only (not for uint64_t, which is unsigned long).
using CountType = unsigned long long;  // NOLINT(runtime/int): see above

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Sums the reflectance of the valid pixels of every date of a batch.
// Purpose: the mean of the valid observations, used to fill and centre the
// image. Mapping: blockIdx.y = date within the batch, blocks along x and
// threads stride over the n*n pixels of that date. Memory: coalesced reads of
// the u16 and u8 planes; one double and one counter atomic per block.
__global__ void ValidMeanKernel(const uint16_t* __restrict__ band,
                                const uint8_t* __restrict__ valid,
                                const float* __restrict__ scale,
                                const float* __restrict__ offset,
                                int first_date, int num_pixels, double* sums,
                                CountType* counts) {
  const int date = blockIdx.y;
  const size_t base = static_cast<size_t>(date) * num_pixels;
  const float s = scale[first_date + date];
  const float o = offset[first_date + date];
  double sum = 0.0;
  unsigned int count = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += gridDim.x * blockDim.x) {
    if (valid[base + i] != 0) {
      sum += static_cast<double>(static_cast<float>(band[base + i]) * s + o);
      ++count;
    }
  }
  // Warp reduction, then one value per warp through shared memory.
  for (int step = 16; step > 0; step >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, step);
    count += __shfl_down_sync(0xffffffffu, count, step);
  }
  __shared__ double warp_sum[kBlockThreads / 32];
  __shared__ unsigned int warp_count[kBlockThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sum[warp] = sum;
    warp_count[warp] = count;
  }
  __syncthreads();
  if (warp == 0) {
    sum = lane < kBlockThreads / 32 ? warp_sum[lane] : 0.0;
    count = lane < kBlockThreads / 32 ? warp_count[lane] : 0u;
    for (int step = 16; step > 0; step >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, step);
      count += __shfl_down_sync(0xffffffffu, count, step);
    }
    if (lane == 0) {
      atomicAdd(&sums[date], sum);
      atomicAdd(&counts[date], static_cast<CountType>(count));
    }
  }
}

// Hann weight of sample i of n (0 at both ends, 1 in the middle).
__device__ float HannWeight(int i, int n) {
  return 0.5f - 0.5f * cospif(2.0f * static_cast<float>(i) /
                              static_cast<float>(n - 1));
}

// Builds the real input image of the FFT: valid reflectance minus the valid
// mean (invalid pixels become 0, which is the mean after centring), times a
// separable 2-D Hann window that suppresses the edge discontinuity.
// Mapping: one thread per pixel, blockIdx.y = date within the batch.
// Memory: coalesced reads of u16/u8, coalesced float writes.
__global__ void PrepareKernel(const uint16_t* __restrict__ band,
                              const uint8_t* __restrict__ valid,
                              const float* __restrict__ scale,
                              const float* __restrict__ offset, int first_date,
                              const double* __restrict__ sums,
                              const CountType* __restrict__ counts, int n,
                              float* __restrict__ image) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel >= n * n) {
    return;
  }
  const int date = blockIdx.y;
  const size_t index = static_cast<size_t>(date) * n * n + pixel;
  const float mean =
      counts[date] > 0 ? static_cast<float>(sums[date] / counts[date]) : 0.0f;
  float value = 0.0f;
  if (valid[index] != 0) {
    value = static_cast<float>(band[index]) * scale[first_date + date] +
            offset[first_date + date] - mean;
  }
  image[index] = value * HannWeight(pixel / n, n) * HannWeight(pixel % n, n);
}

// Replaces the spectrum of each date by the unit-magnitude cross-power
// spectrum R = F_ref conj(F_t) / |F_ref conj(F_t)| (0 where the magnitude is
// below kCrossPowerEpsilon). Mapping: one thread per frequency bin,
// blockIdx.y = date within the batch. Memory: coalesced reads of the batch
// spectrum and of the (cached, shared by all dates) reference spectrum;
// in-place coalesced writes.
__global__ void CrossPowerKernel(const cufftComplex* __restrict__ reference,
                                 int num_bins, cufftComplex* spectrum) {
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= num_bins) {
    return;
  }
  const size_t index = static_cast<size_t>(blockIdx.y) * num_bins + bin;
  const cufftComplex a = reference[bin];
  const cufftComplex b = spectrum[index];
  // a * conj(b)
  float re = a.x * b.x + a.y * b.y;
  float im = a.y * b.x - a.x * b.y;
  const float magnitude = sqrtf(re * re + im * im);
  if (magnitude > kCrossPowerEpsilon) {
    re /= magnitude;
    im /= magnitude;
  } else {
    re = 0.0f;
    im = 0.0f;
  }
  spectrum[index] = make_float2(re, im);
}

// Value at (y, x) of an n x n surface with wrap-around indices.
__device__ float WrappedAt(const float* surface, int n, int y, int x) {
  return surface[static_cast<size_t>((y + n) % n) * n + (x + n) % n];
}

// Offset of the maximum of the parabola through three equally spaced samples
// (minus, centre, plus) in -0.5..0.5; 0 when they do not form a maximum.
__device__ float ParabolaOffset(float minus, float centre, float plus) {
  const float curvature = minus - 2.0f * centre + plus;
  if (curvature >= -1e-12f) {
    return 0.0f;
  }
  return fminf(0.5f, fmaxf(-0.5f, 0.5f * (minus - plus) / curvature));
}

// Finds the maximum of each date's correlation surface and refines it.
// Purpose: peak location -> signed shift. Mapping: one block of
// kPeakThreads per date; threads stride over the n*n surface keeping a
// private (value, index) maximum (ties go to the lower index), then a warp
// shuffle and a shared-memory step reduce them. Thread 0 reads the four
// neighbours of the maximum, applies a 3-point parabola per axis, wraps the
// index to a signed shift and writes the result. Memory: one coalesced pass
// over the surface plus four scattered reads per date.
// The correlation of F_ref conj(F_t) peaks at minus the displacement of the
// date relative to the reference, so the signs are flipped on output; the
// peak is divided by n*n, the value of a perfect unit-magnitude spectrum.
__global__ void PeakKernel(const float* __restrict__ surface_base, int n,
                           int first_date, float* dy, float* dx, float* peak) {
  const int date = blockIdx.x;
  const float* surface = surface_base + static_cast<size_t>(date) * n * n;
  float best = -INFINITY;
  int best_index = 0x7fffffff;
  for (int i = threadIdx.x; i < n * n; i += blockDim.x) {
    const float v = surface[i];
    if (v > best) {  // strides ascend, so the first maximum keeps its index
      best = v;
      best_index = i;
    }
  }
  const auto better = [](float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
  };
  for (int step = 16; step > 0; step >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu, best, step);
    const int other_index = __shfl_down_sync(0xffffffffu, best_index, step);
    if (better(other, other_index, best, best_index)) {
      best = other;
      best_index = other_index;
    }
  }
  __shared__ float warp_best[kPeakThreads / 32];
  __shared__ int warp_index[kPeakThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_best[warp] = best;
    warp_index[warp] = best_index;
  }
  __syncthreads();
  if (warp == 0) {
    best = lane < kPeakThreads / 32 ? warp_best[lane] : -INFINITY;
    best_index = lane < kPeakThreads / 32 ? warp_index[lane] : 0x7fffffff;
    for (int step = 16; step > 0; step >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, best, step);
      const int other_index = __shfl_down_sync(0xffffffffu, best_index, step);
      if (better(other, other_index, best, best_index)) {
        best = other;
        best_index = other_index;
      }
    }
  }
  if (threadIdx.x != 0) {
    return;
  }
  const int py = best_index / n;
  const int px = best_index % n;
  const float sub_y = ParabolaOffset(WrappedAt(surface, n, py - 1, px), best,
                                     WrappedAt(surface, n, py + 1, px));
  const float sub_x = ParabolaOffset(WrappedAt(surface, n, py, px - 1), best,
                                     WrappedAt(surface, n, py, px + 1));
  // Indices above n/2 are negative shifts (wrap-around).
  const float signed_y = static_cast<float>(py > n / 2 ? py - n : py) + sub_y;
  const float signed_x = static_cast<float>(px > n / 2 ? px - n : px) + sub_x;
  dy[first_date + date] = -signed_y;
  dx[first_date + date] = -signed_x;
  peak[first_date + date] = best / (static_cast<float>(n) * n);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// RAII cuFFT plan for batched 2-D transforms of n x n images.
class FftPlan {
 public:
  FftPlan() = default;
  ~FftPlan() { Destroy(); }
  FftPlan(const FftPlan&) = delete;
  FftPlan& operator=(const FftPlan&) = delete;

  // Creates a plan of `type` (CUFFT_R2C or CUFFT_C2R) for `batch` images laid
  // out contiguously (real n*n, complex n*(n/2+1) per image) on `stream`.
  Status Create(int n, int batch, cufftType type, cudaStream_t stream) {
    Destroy();
    const int dims[2] = {n, n};
    CD_CUFFT_CHECK(cufftPlanMany(&handle_, 2, const_cast<int*>(dims), nullptr,
                                 1, 0, nullptr, 1, 0, type, batch));
    created_ = true;
    CD_CUFFT_CHECK(cufftSetStream(handle_, stream));
    return Status::Ok();
  }

  cufftHandle get() const { return handle_; }

 private:
  void Destroy() {
    if (created_) {
      cufftDestroy(handle_);
    }
    created_ = false;
  }

  cufftHandle handle_ = 0;
  bool created_ = false;
};

// Forward and inverse plans for one batch size.
struct PlanPair {
  FftPlan forward;
  FftPlan inverse;
  int batch = 0;

  Status Create(int n, int new_batch, cudaStream_t stream) {
    batch = new_batch;
    CD_RETURN_IF_ERROR(forward.Create(n, new_batch, CUFFT_R2C, stream));
    return inverse.Create(n, new_batch, CUFFT_C2R, stream);
  }
};

bool IsPowerOfTwo(int value) { return value > 0 && (value & (value - 1)) == 0; }

// Device buffers and plans of one run.
struct Workspace {
  DeviceBuffer<uint16_t> band;
  DeviceBuffer<uint8_t> valid;
  DeviceBuffer<float> image;  // real images, then correlation surfaces
  DeviceBuffer<cufftComplex> spectrum;
  DeviceBuffer<cufftComplex> reference;
  DeviceBuffer<float> scale;
  DeviceBuffer<float> offset;
  DeviceBuffer<double> sums;
  DeviceBuffer<CountType> counts;
  DeviceBuffer<float> dy;
  DeviceBuffer<float> dx;
  DeviceBuffer<float> peak;
};

// Milliseconds between two recorded events (the later one is synchronised).
Status ElapsedMs(const CudaEvent& from, const CudaEvent& to, double* ms) {
  CD_CUDA_CHECK(cudaEventSynchronize(to.get()));
  float elapsed = 0;
  CD_CUDA_CHECK(cudaEventElapsedTime(&elapsed, from.get(), to.get()));
  *ms = elapsed;
  return Status::Ok();
}

// Uploads dates [first, first + count) and transforms them into `out`.
// Records events[0..3]: start, after the copies, after the window
// preparation, after the forward FFT.
Status UploadAndTransform(const CoregGpuInput& in, Workspace* ws,
                          const PlanPair& plans, cudaStream_t stream, int first,
                          int count, cufftComplex* out, CudaEvent* events) {
  const int n = in.window;
  const size_t pixels = static_cast<size_t>(n) * n;
  const size_t offset_pixels = static_cast<size_t>(first) * pixels;
  CD_CUDA_CHECK(cudaEventRecord(events[0].get(), stream));
  CD_CUDA_CHECK(cudaMemcpyAsync(ws->band.get(), in.band + offset_pixels,
                                count * pixels * sizeof(uint16_t),
                                cudaMemcpyHostToDevice, stream));
  CD_CUDA_CHECK(cudaMemcpyAsync(ws->valid.get(), in.valid + offset_pixels,
                                count * pixels, cudaMemcpyHostToDevice,
                                stream));
  CD_CUDA_CHECK(cudaEventRecord(events[1].get(), stream));
  CD_CUDA_CHECK(
      cudaMemsetAsync(ws->sums.get(), 0, count * sizeof(double), stream));
  CD_CUDA_CHECK(
      cudaMemsetAsync(ws->counts.get(), 0, count * sizeof(CountType), stream));
  const int mean_blocks = static_cast<int>(std::min<size_t>(
      kMaxMeanBlocks, (pixels + kBlockThreads - 1) / kBlockThreads));
  ValidMeanKernel<<<dim3(mean_blocks, count), kBlockThreads, 0, stream>>>(
      ws->band.get(), ws->valid.get(), ws->scale.get(), ws->offset.get(), first,
      static_cast<int>(pixels), ws->sums.get(), ws->counts.get());
  const int pixel_blocks =
      static_cast<int>((pixels + kBlockThreads - 1) / kBlockThreads);
  PrepareKernel<<<dim3(pixel_blocks, count), kBlockThreads, 0, stream>>>(
      ws->band.get(), ws->valid.get(), ws->scale.get(), ws->offset.get(), first,
      ws->sums.get(), ws->counts.get(), n, ws->image.get());
  CD_CUDA_CHECK(cudaGetLastError());
  CD_CUDA_CHECK(cudaEventRecord(events[2].get(), stream));
  CD_CUFFT_CHECK(cufftExecR2C(plans.forward.get(), ws->image.get(), out));
  CD_CUDA_CHECK(cudaEventRecord(events[3].get(), stream));
  return Status::Ok();
}

// Checks the input description.
Status ValidateInput(const CoregGpuInput& in) {
  if (!IsPowerOfTwo(in.window) || in.window < 8 || in.window > 4096) {
    return Status::Error(
        "co-registration window must be a power of two in "
        "8..4096, got " +
        std::to_string(in.window));
  }
  if (in.num_dates < 1 || in.reference_date < 0 ||
      in.reference_date >= in.num_dates) {
    return Status::Error("reference date " + std::to_string(in.reference_date) +
                         " is outside 0.." + std::to_string(in.num_dates - 1));
  }
  if (in.band == nullptr || in.valid == nullptr || in.scale == nullptr ||
      in.offset == nullptr) {
    return Status::Error("co-registration input arrays are null");
  }
  return Status::Ok();
}

}  // namespace

Status GpuPhaseCorrelation(const CoregGpuInput& in, CoregGpuOutput* out) {
  CD_RETURN_IF_ERROR(ValidateInput(in));
  const WallTimer setup_timer;
  CD_CUDA_CHECK(cudaSetDevice(in.device));
  const int n = in.window;
  const int num_dates = in.num_dates;
  const size_t pixels = static_cast<size_t>(n) * n;
  const size_t bins = static_cast<size_t>(n) * (n / 2 + 1);

  // Batch size: per date a raw plane (u16 + u8), a real image and a spectrum.
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CD_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const double per_date = static_cast<double>(pixels) * 3.0 +
                          static_cast<double>(pixels) * sizeof(float) +
                          static_cast<double>(bins) * sizeof(cufftComplex);
  const double reserved = static_cast<double>(bins) * sizeof(cufftComplex);
  const double budget = kMemoryFraction * static_cast<double>(free_bytes);
  const int chunk = static_cast<int>(std::max(
      1.0, std::min<double>(num_dates, (budget - reserved) / per_date)));

  CudaStream stream;
  CD_RETURN_IF_ERROR(stream.Create());
  Workspace ws;
  CD_CUDA_CHECK(ws.band.Allocate(chunk * pixels));
  CD_CUDA_CHECK(ws.valid.Allocate(chunk * pixels));
  CD_CUDA_CHECK(ws.image.Allocate(chunk * pixels));
  CD_CUDA_CHECK(ws.spectrum.Allocate(chunk * bins));
  CD_CUDA_CHECK(ws.reference.Allocate(bins));
  CD_CUDA_CHECK(ws.scale.Allocate(num_dates));
  CD_CUDA_CHECK(ws.offset.Allocate(num_dates));
  CD_CUDA_CHECK(ws.sums.Allocate(chunk));
  CD_CUDA_CHECK(ws.counts.Allocate(chunk));
  CD_CUDA_CHECK(ws.dy.Allocate(num_dates));
  CD_CUDA_CHECK(ws.dx.Allocate(num_dates));
  CD_CUDA_CHECK(ws.peak.Allocate(num_dates));
  CD_CUDA_CHECK(cudaMemcpyAsync(ws.scale.get(), in.scale,
                                num_dates * sizeof(float),
                                cudaMemcpyHostToDevice, stream.get()));
  CD_CUDA_CHECK(cudaMemcpyAsync(ws.offset.get(), in.offset,
                                num_dates * sizeof(float),
                                cudaMemcpyHostToDevice, stream.get()));

  // Plans: one image (reference), a full batch and, when needed, the tail.
  PlanPair single;
  PlanPair full;
  PlanPair tail;
  CD_RETURN_IF_ERROR(single.Create(n, 1, stream.get()));
  const PlanPair* full_plans = &single;
  if (chunk > 1) {
    CD_RETURN_IF_ERROR(full.Create(n, chunk, stream.get()));
    full_plans = &full;
  }
  const int tail_dates = num_dates % chunk;
  if (tail_dates > 1) {
    CD_RETURN_IF_ERROR(tail.Create(n, tail_dates, stream.get()));
  }
  CudaEvent events[kNumEvents];
  for (CudaEvent& event : events) {
    CD_RETURN_IF_ERROR(event.Create());
  }

  CoregGpuTimings timings;
  timings.chunk_dates = chunk;
  timings.setup_ms = setup_timer.ElapsedMs();
  double ms = 0;

  // The reference spectrum is computed once and kept for all batches.
  CD_RETURN_IF_ERROR(UploadAndTransform(in, &ws, single, stream.get(),
                                        in.reference_date, 1,
                                        ws.reference.get(), events));
  CD_RETURN_IF_ERROR(ElapsedMs(events[0], events[1], &ms));
  timings.h2d_ms += ms;
  CD_RETURN_IF_ERROR(ElapsedMs(events[1], events[3], &ms));
  timings.fft_ms += ms;

  const int bin_blocks =
      static_cast<int>((bins + kBlockThreads - 1) / kBlockThreads);
  for (int first = 0; first < num_dates; first += chunk) {
    const int count = std::min(chunk, num_dates - first);
    const PlanPair& plans =
        count == chunk ? *full_plans : (count == 1 ? single : tail);
    CD_RETURN_IF_ERROR(UploadAndTransform(in, &ws, plans, stream.get(), first,
                                          count, ws.spectrum.get(), events));
    CrossPowerKernel<<<dim3(bin_blocks, count), kBlockThreads, 0,
                       stream.get()>>>(
        ws.reference.get(), static_cast<int>(bins), ws.spectrum.get());
    CD_CUDA_CHECK(cudaGetLastError());
    CD_CUDA_CHECK(cudaEventRecord(events[4].get(), stream.get()));
    // The inverse transform overwrites the real image buffer with the
    // correlation surface (its input spectrum is destroyed).
    CD_CUFFT_CHECK(
        cufftExecC2R(plans.inverse.get(), ws.spectrum.get(), ws.image.get()));
    CD_CUDA_CHECK(cudaEventRecord(events[5].get(), stream.get()));
    PeakKernel<<<count, kPeakThreads, 0, stream.get()>>>(
        ws.image.get(), n, first, ws.dy.get(), ws.dx.get(), ws.peak.get());
    CD_CUDA_CHECK(cudaGetLastError());
    CD_CUDA_CHECK(cudaEventRecord(events[6].get(), stream.get()));
    CD_RETURN_IF_ERROR(ElapsedMs(events[0], events[1], &ms));
    timings.h2d_ms += ms;
    CD_RETURN_IF_ERROR(ElapsedMs(events[1], events[3], &ms));
    timings.fft_ms += ms;
    CD_RETURN_IF_ERROR(ElapsedMs(events[3], events[4], &ms));
    timings.spectrum_ms += ms;
    CD_RETURN_IF_ERROR(ElapsedMs(events[4], events[5], &ms));
    timings.fft_ms += ms;
    CD_RETURN_IF_ERROR(ElapsedMs(events[5], events[6], &ms));
    timings.peak_ms += ms;
  }

  out->dy.assign(num_dates, 0.0f);
  out->dx.assign(num_dates, 0.0f);
  out->peak.assign(num_dates, 0.0f);
  CD_CUDA_CHECK(cudaEventRecord(events[0].get(), stream.get()));
  CD_CUDA_CHECK(cudaMemcpyAsync(out->dy.data(), ws.dy.get(),
                                num_dates * sizeof(float),
                                cudaMemcpyDeviceToHost, stream.get()));
  CD_CUDA_CHECK(cudaMemcpyAsync(out->dx.data(), ws.dx.get(),
                                num_dates * sizeof(float),
                                cudaMemcpyDeviceToHost, stream.get()));
  CD_CUDA_CHECK(cudaMemcpyAsync(out->peak.data(), ws.peak.get(),
                                num_dates * sizeof(float),
                                cudaMemcpyDeviceToHost, stream.get()));
  CD_CUDA_CHECK(cudaEventRecord(events[1].get(), stream.get()));
  CD_RETURN_IF_ERROR(ElapsedMs(events[0], events[1], &ms));
  timings.d2h_ms = ms;
  out->timings = timings;
  return Status::Ok();
}

}  // namespace cd
