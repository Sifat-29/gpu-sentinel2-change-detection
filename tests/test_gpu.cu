// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// GPU tests: kernels against their host twins, the streamed pipeline against
// the CPU backends, the post-processing kernels against the CPU twins, and
// (LargeGpu, only with --filter) a cube larger than the device memory.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17) removing the large test cube
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cpu/cpu_backend.h"
#include "cpu_test_cube.h"
#include "cube_io.h"
#include "cuda/cuda_check.cuh"
#include "cuda/gpu_backend.h"
#include "cuda/kernels.cuh"
#include "logger.h"
#include "outputs.h"
#include "pixel_model.h"
#include "postprocess.h"
#include "synthetic_cube.h"
#include "test_framework.h"
#include "timing.h"
#include "types.h"
#include "validation.h"
#include "validation_rules.h"

namespace cd {
namespace {

constexpr uint8_t Code(PixelStatus s) { return static_cast<uint8_t>(s); }

constexpr uint8_t kBreakCode = Code(PixelStatus::kBreak);
constexpr uint8_t kNoDataCode = Code(PixelStatus::kInsufficientData);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

ModelParams DefaultParams() {
  ModelParams params;
  const Status status = ResolveModelParams(&params);
  EXPECT_TRUE(status.ok());
  return params;
}

// A synthetic cube served from memory together with its injected truth.
struct TestCube {
  std::unique_ptr<CubeSource> source;
  std::vector<int16_t> truth;
};

TestCube MakeTestCube(int height, int width, int num_dates, uint32_t seed) {
  SyntheticOptions options;
  options.height = height;
  options.width = width;
  options.num_dates = num_dates;
  options.noise = 0.03f;
  options.seed = seed;
  SyntheticCube cube;
  const Status status = GenerateSyntheticCube(options, &cube);
  EXPECT_TRUE(status.ok());
  TestCube out;
  out.truth = cube.truth_break_index;
  out.source = MakeMemoryCube(std::move(cube));
  return out;
}

RunConfig GpuConfig(Backend backend, int streams, int tile_rows,
                    int block_size) {
  RunConfig config;
  config.backend = backend;
  config.streams = streams;
  config.tile_rows = tile_rows;
  config.block_size = block_size;
  config.quiet = true;
  return config;
}

// Fraction of entries that are equal in two vectors of the same size.
template <typename T>
double EqualFraction(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size() || a.empty()) {
    return 0.0;
  }
  size_t same = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    same += a[i] == b[i] ? 1 : 0;
  }
  return static_cast<double>(same) / static_cast<double>(a.size());
}

// Largest |coefficient difference| over pixels where both results fitted.
double MaxCoefDiff(const DetectionResult& a, const DetectionResult& b) {
  const size_t pixels = a.status.size();
  double worst = 0.0;
  for (size_t i = 0; i < pixels; ++i) {
    if (a.status[i] == kNoDataCode || b.status[i] == kNoDataCode) {
      continue;
    }
    for (int c = 0; c < a.num_params; ++c) {
      const size_t k = static_cast<size_t>(c) * pixels + i;
      worst =
          std::max(worst, std::fabs(static_cast<double>(a.coefficients[k]) -
                                    static_cast<double>(b.coefficients[k])));
    }
  }
  return worst;
}

// True when every map of the two results is byte-identical.
bool Identical(const DetectionResult& a, const DetectionResult& b) {
  return a.break_index == b.break_index && a.magnitude == b.magnitude &&
         a.status == b.status && a.rmse == b.rmse &&
         a.train_obs == b.train_obs && a.coefficients == b.coefficients;
}

// Runs the GPU backend and checks the status.
DetectionResult RunGpu(CubeSource* source, const ModelParams& params,
                       const RunConfig& config, StageTimings* timings,
                       RunStats* stats) {
  DetectionResult result;
  const Status status =
      RunGpuDetection(source, params, config, &result, timings, stats);
  EXPECT_TRUE(status.ok());
  if (!status.ok()) {
    std::printf("RunGpuDetection: %s\n", status.message().c_str());
  }
  return result;
}

DetectionResult RunCpu(CubeSource* source, const ModelParams& params,
                       Backend backend, bool coefficients) {
  RunConfig config;
  config.backend = backend;
  config.threads = 4;
  config.save_coefficients = coefficients;
  config.quiet = true;
  DetectionResult result;
  const Status status =
      RunCpuDetection(source, params, config, &result, nullptr, nullptr);
  EXPECT_TRUE(status.ok());
  return result;
}

// Restores the testing hooks when a test ends, even after a failed ASSERT.
struct HookGuard {
  ~HookGuard() {
    SetSlotAllocationLimitForTesting(0);
    SetPostprocessChunkRowsForTesting(0);
  }
};

// The 256 x 256 x 48 cube and the CPU results shared by the pipeline tests.
struct PipelineFixture {
  TestCube cube;
  ModelParams params;
  DetectionResult cpu64;
  DetectionResult cpu32;
};

PipelineFixture& Fixture() {
  static PipelineFixture* fixture = [] {
    auto* f = new PipelineFixture();
    f->cube = MakeTestCube(256, 256, 48, 7);
    f->params = DefaultParams();
    f->cpu64 = RunCpu(f->cube.source.get(), f->params, Backend::kCpuFp64, true);
    f->cpu32 = RunCpu(f->cube.source.get(), f->params, Backend::kCpuFp32, true);
    return f;
  }();
  return *fixture;
}

// Distance in units in the last place between two finite floats.
int64_t UlpDistance(float a, float b) {
  int32_t ia = 0;
  int32_t ib = 0;
  std::memcpy(&ia, &a, sizeof(a));
  std::memcpy(&ib, &b, sizeof(b));
  if (ia < 0) {
    ia = INT32_MIN - ia;
  }
  if (ib < 0) {
    ib = INT32_MIN - ib;
  }
  return std::llabs(static_cast<int64_t>(ia) - static_cast<int64_t>(ib));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Device copies of one small tile read from a cube.
struct DeviceTile {
  DeviceBuffer<uint16_t> red;
  DeviceBuffer<uint16_t> nir;
  DeviceBuffer<uint8_t> scl;
  int64_t pixels = 0;
  int num_dates = 0;
  std::vector<uint16_t> host_red;
  std::vector<uint16_t> host_nir;
  std::vector<uint8_t> host_scl;

  RawBands Bands() const { return RawBands{red.get(), nir.get(), scl.get()}; }
};

bool LoadTile(CubeSource* source, DeviceTile* tile) {
  const CubeInfo& info = source->info();
  tile->pixels = static_cast<int64_t>(info.height) * info.width;
  tile->num_dates = info.num_dates;
  const size_t n = static_cast<size_t>(tile->pixels) * info.num_dates;
  tile->host_red.resize(n);
  tile->host_nir.resize(n);
  tile->host_scl.resize(n);
  TileView view;
  view.red = tile->host_red.data();
  view.nir = tile->host_nir.data();
  view.scl = tile->host_scl.data();
  if (!source->ReadTile(0, info.height, &view).ok()) {
    return false;
  }
  return tile->red.Allocate(n) == cudaSuccess &&
         tile->nir.Allocate(n) == cudaSuccess &&
         tile->scl.Allocate(n) == cudaSuccess &&
         cudaMemcpy(tile->red.get(), tile->host_red.data(), n * 2,
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(tile->nir.get(), tile->host_nir.data(), n * 2,
                    cudaMemcpyHostToDevice) == cudaSuccess &&
         cudaMemcpy(tile->scl.get(), tile->host_scl.data(), n,
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

// Device output arrays of the detection kernels.
struct DeviceOutputs {
  DeviceBuffer<float> coef;
  DeviceBuffer<float> rmse;
  DeviceBuffer<uint16_t> n_train;
  DeviceBuffer<int16_t> break_index;
  DeviceBuffer<float> magnitude;
  DeviceBuffer<uint8_t> status;

  bool Allocate(size_t pixels, int num_params, bool with_coef) {
    return coef.Allocate(with_coef ? pixels * num_params : 0) == cudaSuccess &&
           rmse.Allocate(pixels) == cudaSuccess &&
           n_train.Allocate(pixels) == cudaSuccess &&
           break_index.Allocate(pixels) == cudaSuccess &&
           magnitude.Allocate(pixels) == cudaSuccess &&
           status.Allocate(pixels) == cudaSuccess;
  }

  DetectionBuffers Buffers() {
    return DetectionBuffers{coef.get(),        rmse.get(),      n_train.get(),
                            break_index.get(), magnitude.get(), status.get()};
  }

  // Copies the maps to the host.
  void Download(size_t pixels, int num_params, bool with_coef,
                DetectionResult* host) {
    AllocateResult(1, static_cast<int>(pixels), num_params, with_coef, host);
    cudaMemcpy(host->break_index.data(), break_index.get(), pixels * 2,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host->magnitude.data(), magnitude.get(), pixels * 4,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host->status.data(), status.get(), pixels,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host->rmse.data(), rmse.get(), pixels * 4,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host->train_obs.data(), n_train.get(), pixels * 2,
               cudaMemcpyDeviceToHost);
    if (with_coef) {
      cudaMemcpy(host->coefficients.data(), coef.get(), pixels * num_params * 4,
                 cudaMemcpyDeviceToHost);
    }
  }
};

TEST(GpuKernels, NdviMatchesHost) {
  TestCube cube = MakeTestCube(24, 20, 48, 3);
  const ModelParams params = DefaultParams();
  DateTable dates = cube.source->info().dates;
  ASSERT_TRUE(BuildDateTable(params, &dates).ok());
  DeviceTile tile;
  ASSERT_TRUE(LoadTile(cube.source.get(), &tile));
  ASSERT_TRUE(UploadDateTables(dates, params).ok());

  DeviceBuffer<float> ndvi;
  const size_t n = static_cast<size_t>(tile.pixels) * tile.num_dates;
  ASSERT_EQ(ndvi.Allocate(n), cudaSuccess);
  ASSERT_TRUE(LaunchComputeNdvi(tile.Bands(), tile.num_dates, tile.pixels,
                                params.valid_class_mask, ndvi.get(),
                                LaunchConfig())
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<float> gpu(n);
  ASSERT_EQ(cudaMemcpy(gpu.data(), ndvi.get(), n * 4, cudaMemcpyDeviceToHost),
            cudaSuccess);

  int64_t worst_ulp = 0;
  size_t invalid_mismatch = 0;
  for (int t = 0; t < tile.num_dates; ++t) {
    for (int64_t p = 0; p < tile.pixels; ++p) {
      const size_t i = static_cast<size_t>(t * tile.pixels + p);
      const float host = ComputeNdvi<float>(
          tile.host_red[i], tile.host_nir[i], tile.host_scl[i],
          dates.scale[static_cast<size_t>(t)],
          dates.offset[static_cast<size_t>(t)], params.valid_class_mask);
      if ((host == kInvalidNdvi) != (gpu[i] == kInvalidNdvi)) {
        ++invalid_mismatch;
      } else {
        worst_ulp = std::max(worst_ulp, UlpDistance(host, gpu[i]));
      }
    }
  }
  std::printf("ndvi kernel vs host: max %d ulp, validity mismatches %zu\n",
              static_cast<int>(worst_ulp), invalid_mismatch);
  EXPECT_EQ(invalid_mismatch, 0u);
  EXPECT_LE(worst_ulp, 1);
}

TEST(GpuKernels, EveryKernelAfterOneUpload) {
  TestCube cube = MakeTestCube(24, 20, 48, 5);
  DeviceTile tile;
  ASSERT_TRUE(LoadTile(cube.source.get(), &tile));
  for (int harmonics = 1; harmonics <= 3; ++harmonics) {
    ModelParams params;
    params.harmonics = harmonics;
    ASSERT_TRUE(ResolveModelParams(&params).ok());
    DateTable dates = cube.source->info().dates;
    ASSERT_TRUE(BuildDateTable(params, &dates).ok());
    const int num_params = NumParams(harmonics);
    const size_t pixels = static_cast<size_t>(tile.pixels);
    ASSERT_TRUE(UploadDateTables(dates, params).ok());  // the only upload

    DeviceBuffer<float> ndvi;
    ASSERT_EQ(ndvi.Allocate(pixels * tile.num_dates), cudaSuccess);
    DeviceOutputs unfused;
    DeviceOutputs fused;
    ASSERT_TRUE(unfused.Allocate(pixels, num_params, true));
    ASSERT_TRUE(fused.Allocate(pixels, num_params, true));
    const int train = dates.num_train_dates;
    LaunchConfig launch;
    launch.block_size = 128;
    ASSERT_TRUE(LaunchComputeNdvi(tile.Bands(), tile.num_dates, tile.pixels,
                                  params.valid_class_mask, ndvi.get(), launch)
                    .ok());
    ASSERT_TRUE(LaunchFitHarmonicModel(ndvi.get(), tile.pixels, train, params,
                                       unfused.Buffers(), launch)
                    .ok());
    ASSERT_TRUE(LaunchDetectBreak(ndvi.get(), tile.pixels, tile.num_dates,
                                  train, params, unfused.Buffers(), launch)
                    .ok());
    ASSERT_TRUE(LaunchFusedChangeDetection(tile.Bands(), tile.pixels,
                                           tile.num_dates, train, params,
                                           fused.Buffers(), launch)
                    .ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    DetectionResult a;
    DetectionResult b;
    unfused.Download(pixels, num_params, true, &a);
    fused.Download(pixels, num_params, true, &b);
    const double same_break = EqualFraction(a.break_index, b.break_index);
    const double same_status = EqualFraction(a.status, b.status);
    std::printf(
        "K=%d fused vs unfused: break %.4f status %.4f coef diff %.2e\n",
        harmonics, same_break, same_status, MaxCoefDiff(a, b));
    EXPECT_GE(same_break, 0.995);
    EXPECT_GE(same_status, 0.995);
    EXPECT_NEAR(MaxCoefDiff(a, b), 0.0, 1e-4);

    // The fused kernel also runs without a coefficient output.
    DeviceOutputs no_coef;
    ASSERT_TRUE(no_coef.Allocate(pixels, num_params, false));
    ASSERT_TRUE(LaunchFusedChangeDetection(tile.Bands(), tile.pixels,
                                           tile.num_dates, train, params,
                                           no_coef.Buffers(), launch)
                    .ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    DetectionResult c;
    no_coef.Download(pixels, num_params, false, &c);
    EXPECT_TRUE(c.break_index == b.break_index && c.status == b.status &&
                c.magnitude == b.magnitude);
  }
}

TEST(GpuKernels, RejectsInvalidArguments) {
  TestCube cube = MakeTestCube(8, 8, 48, 3);
  const ModelParams params = DefaultParams();
  DateTable dates = cube.source->info().dates;
  ASSERT_TRUE(BuildDateTable(params, &dates).ok());
  DeviceTile tile;
  ASSERT_TRUE(LoadTile(cube.source.get(), &tile));
  ASSERT_TRUE(UploadDateTables(dates, params).ok());
  DeviceOutputs out;
  ASSERT_TRUE(out.Allocate(64, 6, true));

  LaunchConfig bad_block;
  bad_block.block_size = 100;
  EXPECT_FALSE(LaunchFusedChangeDetection(tile.Bands(), tile.pixels,
                                          tile.num_dates, dates.num_train_dates,
                                          params, out.Buffers(), bad_block)
                   .ok());
  // The unfused fit needs the coefficient buffer.
  DetectionBuffers no_coef = out.Buffers();
  no_coef.coef = nullptr;
  DeviceBuffer<float> ndvi;
  ASSERT_EQ(ndvi.Allocate(64 * 48), cudaSuccess);
  EXPECT_FALSE(LaunchFitHarmonicModel(ndvi.get(), 64, dates.num_train_dates,
                                      params, no_coef, LaunchConfig())
                   .ok());
  ModelParams bad_k = params;
  bad_k.harmonics = 4;
  EXPECT_FALSE(LaunchFusedChangeDetection(tile.Bands(), tile.pixels,
                                          tile.num_dates, dates.num_train_dates,
                                          bad_k, out.Buffers(), LaunchConfig())
                   .ok());
  // More than kMaxDates dates cannot be uploaded.
  DateTable many;
  many.day.assign(static_cast<size_t>(kMaxDates) + 1, 18000);
  many.iso.assign(many.day.size(), "2019-01-01");
  many.scale.assign(many.day.size(), 1e-4f);
  many.offset.assign(many.day.size(), 0.0f);
  many.tau.assign(many.day.size(), 0.0);
  const Status too_many = UploadDateTables(many, params);
  EXPECT_FALSE(too_many.ok());
  EXPECT_TRUE(too_many.message().find("1024") != std::string::npos);
}

// A device array of `count` elements surrounded by guard bytes filled with a
// known pattern. After a kernel ran, GuardsIntact() shows whether it wrote
// outside the array: a poor man's memcheck for machines where
// compute-sanitizer cannot run (it can detect out-of-bounds writes only).
template <typename T>
class GuardedBuffer {
 public:
  explicit GuardedBuffer(size_t count) : count_(count) {
    const size_t bytes = kGuard + count * sizeof(T) + kGuard;
    ok_ = raw_.Allocate(bytes) == cudaSuccess &&
          cudaMemset(raw_.get(), kPattern, bytes) == cudaSuccess;
  }

  bool ok() const { return ok_; }
  T* get() const { return reinterpret_cast<T*>(raw_.get() + kGuard); }

  bool GuardsIntact() const {
    std::vector<uint8_t> guard(kGuard);
    const size_t tail = kGuard + count_ * sizeof(T);
    for (const size_t offset : {size_t{0}, tail}) {
      if (cudaMemcpy(guard.data(), raw_.get() + offset, kGuard,
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
      for (const uint8_t byte : guard) {
        if (byte != kPattern) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  static constexpr size_t kGuard = 256;
  static constexpr int kPattern = 0xAB;
  DeviceBuffer<uint8_t> raw_;
  size_t count_;
  bool ok_ = false;
};

TEST(GpuKernels, NoWritesOutsideTheOutputBuffers) {
  // 23 x 21 = 483 pixels: not a multiple of any block size, so the last block
  // has idle threads that must not write.
  TestCube cube = MakeTestCube(23, 21, 48, 9);
  const ModelParams params = DefaultParams();
  DateTable dates = cube.source->info().dates;
  ASSERT_TRUE(BuildDateTable(params, &dates).ok());
  DeviceTile tile;
  ASSERT_TRUE(LoadTile(cube.source.get(), &tile));
  ASSERT_TRUE(UploadDateTables(dates, params).ok());
  const size_t pixels = static_cast<size_t>(tile.pixels);
  const int num_params = NumParams(params.harmonics);
  const int train = dates.num_train_dates;
  LaunchConfig launch;
  launch.block_size = 128;

  GuardedBuffer<float> ndvi(pixels * tile.num_dates);
  GuardedBuffer<float> coef(pixels * num_params);
  GuardedBuffer<float> rmse(pixels);
  GuardedBuffer<uint16_t> n_train(pixels);
  GuardedBuffer<int16_t> break_index(pixels);
  GuardedBuffer<float> magnitude(pixels);
  GuardedBuffer<uint8_t> status(pixels);
  GuardedBuffer<float> fused_coef(pixels * num_params);
  GuardedBuffer<float> fused_rmse(pixels);
  GuardedBuffer<uint16_t> fused_n_train(pixels);
  GuardedBuffer<int16_t> fused_break(pixels);
  GuardedBuffer<float> fused_magnitude(pixels);
  GuardedBuffer<uint8_t> fused_status(pixels);
  ASSERT_TRUE(ndvi.ok() && coef.ok() && rmse.ok() && n_train.ok() &&
              break_index.ok() && magnitude.ok() && status.ok() &&
              fused_coef.ok() && fused_rmse.ok() && fused_n_train.ok() &&
              fused_break.ok() && fused_magnitude.ok() && fused_status.ok());
  const DetectionBuffers unfused{coef.get(),      rmse.get(),
                                 n_train.get(),   break_index.get(),
                                 magnitude.get(), status.get()};
  const DetectionBuffers fused{fused_coef.get(),      fused_rmse.get(),
                               fused_n_train.get(),   fused_break.get(),
                               fused_magnitude.get(), fused_status.get()};
  ASSERT_TRUE(LaunchComputeNdvi(tile.Bands(), tile.num_dates, tile.pixels,
                                params.valid_class_mask, ndvi.get(), launch)
                  .ok());
  ASSERT_TRUE(LaunchFitHarmonicModel(ndvi.get(), tile.pixels, train, params,
                                     unfused, launch)
                  .ok());
  ASSERT_TRUE(LaunchDetectBreak(ndvi.get(), tile.pixels, tile.num_dates, train,
                                params, unfused, launch)
                  .ok());
  ASSERT_TRUE(LaunchFusedChangeDetection(tile.Bands(), tile.pixels,
                                         tile.num_dates, train, params, fused,
                                         launch)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_TRUE(ndvi.GuardsIntact());
  EXPECT_TRUE(coef.GuardsIntact());
  EXPECT_TRUE(rmse.GuardsIntact());
  EXPECT_TRUE(n_train.GuardsIntact());
  EXPECT_TRUE(break_index.GuardsIntact());
  EXPECT_TRUE(magnitude.GuardsIntact());
  EXPECT_TRUE(status.GuardsIntact());
  EXPECT_TRUE(fused_coef.GuardsIntact());
  EXPECT_TRUE(fused_rmse.GuardsIntact());
  EXPECT_TRUE(fused_n_train.GuardsIntact());
  EXPECT_TRUE(fused_break.GuardsIntact());
  EXPECT_TRUE(fused_magnitude.GuardsIntact());
  EXPECT_TRUE(fused_status.GuardsIntact());

  // Post-processing kernels on the same odd-sized image (23 rows x 21).
  const int rows = 23;
  const int width = 21;
  DetectionResult r;
  AllocateResult(rows, width, 6, false, &r);
  for (size_t i = 0; i < pixels; ++i) {
    r.status[i] = (i % 3 == 0) ? kBreakCode : 0;
    r.break_index[i] = r.status[i] == kBreakCode ? 30 : -1;
    r.magnitude[i] = r.status[i] == kBreakCode ? -0.2f : 0.0f;
  }
  DeviceBuffer<int16_t> d_break;
  DeviceBuffer<float> d_magnitude;
  DeviceBuffer<uint8_t> d_status;
  DeviceBuffer<uint8_t> d_halo;
  DeviceBuffer<uint8_t> d_reference;
  DeviceBuffer<int32_t> d_day;
  DeviceBuffer<int16_t> d_year;
  ASSERT_EQ(d_break.Allocate(pixels), cudaSuccess);
  ASSERT_EQ(d_magnitude.Allocate(pixels), cudaSuccess);
  ASSERT_EQ(d_status.Allocate(pixels), cudaSuccess);
  ASSERT_EQ(d_halo.Allocate(pixels + 2 * width), cudaSuccess);
  ASSERT_EQ(d_reference.Allocate(pixels), cudaSuccess);
  ASSERT_EQ(d_day.Allocate(dates.day.size()), cudaSuccess);
  ASSERT_EQ(d_year.Allocate(dates.year.size()), cudaSuccess);
  cudaMemcpy(d_break.get(), r.break_index.data(), pixels * 2,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_magnitude.get(), r.magnitude.data(), pixels * 4,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_status.get(), r.status.data(), pixels, cudaMemcpyHostToDevice);
  cudaMemset(d_halo.get(), 0, pixels + 2 * width);
  cudaMemcpy(d_halo.get() + width, r.status.data(), pixels,
             cudaMemcpyHostToDevice);
  cudaMemset(d_reference.get(), 22, pixels);
  cudaMemcpy(d_day.get(), dates.day.data(), dates.day.size() * 4,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_year.get(), dates.year.data(), dates.year.size() * 2,
             cudaMemcpyHostToDevice);

  GuardedBuffer<uint8_t> rgba(pixels * 4);
  GuardedBuffer<uint8_t> cleaned(pixels);
  GuardedBuffer<int16_t> cleaned_break(pixels);
  GuardedBuffer<float> cleaned_magnitude(pixels);
  GuardedBuffer<uint64_t> counters(kNumConfusionCounters);
  GuardedBuffer<float> diff(pixels);
  GuardedBuffer<uint8_t> flag(pixels);
  ASSERT_TRUE(rgba.ok() && cleaned.ok() && cleaned_break.ok() &&
              cleaned_magnitude.ok() && counters.ok() && diff.ok() &&
              flag.ok());
  cudaMemcpy(cleaned_break.get(), r.break_index.data(), pixels * 2,
             cudaMemcpyHostToDevice);
  cudaMemcpy(cleaned_magnitude.get(), r.magnitude.data(), pixels * 4,
             cudaMemcpyHostToDevice);
  cudaMemset(counters.get(), 0, kNumConfusionCounters * sizeof(uint64_t));
  ASSERT_TRUE(LaunchColorizeChangeMap(
                  d_break.get(), d_magnitude.get(), d_status.get(), d_day.get(),
                  static_cast<int>(dates.day.size()), dates.day[train],
                  dates.day.back(), tile.pixels, rgba.get(), launch)
                  .ok());
  ASSERT_TRUE(LaunchCleanup(d_halo.get(), rows, width, cleaned.get(),
                            cleaned_break.get(), cleaned_magnitude.get(),
                            nullptr)
                  .ok());
  ValidationWindow window;
  window.first_year = 2019;
  window.last_year = 2024;
  ASSERT_TRUE(LaunchConfusion(d_status.get(), d_break.get(), d_magnitude.get(),
                              d_reference.get(), d_year.get(),
                              static_cast<int>(dates.year.size()), window,
                              tile.pixels, counters.get(), nullptr)
                  .ok());
  const float scale[2] = {1e-4f, 1e-4f};
  const float offset[2] = {0.0f, 0.0f};
  RawBands two_dates = tile.Bands();
  ASSERT_TRUE(LaunchQuickDiff(two_dates, tile.pixels, scale, offset,
                              params.valid_class_mask, 0.1f, false, diff.get(),
                              flag.get(), launch)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_TRUE(rgba.GuardsIntact());
  EXPECT_TRUE(cleaned.GuardsIntact());
  EXPECT_TRUE(cleaned_break.GuardsIntact());
  EXPECT_TRUE(cleaned_magnitude.GuardsIntact());
  EXPECT_TRUE(counters.GuardsIntact());
  EXPECT_TRUE(diff.GuardsIntact());
  EXPECT_TRUE(flag.GuardsIntact());
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

TEST(GpuPipeline, AgreementWithCpuAndRecovery) {
  PipelineFixture& f = Fixture();
  ASSERT_EQ(f.cpu64.status.size(), 256u * 256u);
  struct Variant {
    Backend backend;
    const char* name;
  };
  const Variant variants[] = {{Backend::kGpuFused, "gpu"},
                              {Backend::kGpuUnfused, "gpu-unfused"}};
  DetectionResult gpu_results[2];
  for (int v = 0; v < 2; ++v) {
    RunConfig config = GpuConfig(variants[v].backend, 3, 64, 256);
    config.save_coefficients = true;
    gpu_results[v] =
        RunGpu(f.cube.source.get(), f.params, config, nullptr, nullptr);
    const DetectionResult& g = gpu_results[v];
    const double b64 = EqualFraction(g.break_index, f.cpu64.break_index);
    const double s64 = EqualFraction(g.status, f.cpu64.status);
    const double b32 = EqualFraction(g.break_index, f.cpu32.break_index);
    const double coef = MaxCoefDiff(g, f.cpu64);
    RecoveryStats recovery;
    ASSERT_TRUE(ComputeRecovery(f.cube.source.get(), f.cube.truth, g, f.params,
                                &recovery)
                    .ok());
    const double recovered = static_cast<double>(recovery.recovered) /
                             static_cast<double>(recovery.eligible);
    const double false_breaks = static_cast<double>(recovery.false_breaks) /
                                static_cast<double>(recovery.stable_truth);
    std::printf(
        "%s vs cpu(FP64): break %.4f status %.4f | vs cpu-fp32: break %.4f | "
        "max coef diff %.2e | recovered %.4f false breaks %.4f\n",
        variants[v].name, b64, s64, b32, coef, recovered, false_breaks);
    EXPECT_GE(b64, 0.995);
    EXPECT_GE(s64, 0.995);
    EXPECT_GE(b32, 0.999);
    EXPECT_LT(coef, 1e-3);
    // Default model: the trend is held at its end-of-history value and the
    // break rule is |z| > 3.5 on 2 observations (tuned on real data). On this
    // cube that gives about 98.8 % recovery and 1.2 % false breaks; the
    // stricter 3 x 3 rule gives about 0.6 % but misses more real clearings.
    EXPECT_TRUE(f.params.hold_trend);
    EXPECT_GE(recovered, 0.97);
    EXPECT_LE(false_breaks, 0.02);
  }
  const double fused_vs_unfused =
      EqualFraction(gpu_results[0].break_index, gpu_results[1].break_index);
  std::printf("fused vs unfused: break %.5f status %.5f\n", fused_vs_unfused,
              EqualFraction(gpu_results[0].status, gpu_results[1].status));
  EXPECT_GE(fused_vs_unfused, 0.999);
}

TEST(GpuPipeline, HarmonicsOneAndThreeMatchCpuFp32) {
  PipelineFixture& f = Fixture();
  for (const int harmonics : {1, 3}) {
    ModelParams params;
    params.harmonics = harmonics;
    ASSERT_TRUE(ResolveModelParams(&params).ok());
    const DetectionResult cpu =
        RunCpu(f.cube.source.get(), params, Backend::kCpuFp32, false);
    for (const Backend backend : {Backend::kGpuFused, Backend::kGpuUnfused}) {
      const DetectionResult gpu =
          RunGpu(f.cube.source.get(), params, GpuConfig(backend, 3, 0, 256),
                 nullptr, nullptr);
      const double same = EqualFraction(gpu.break_index, cpu.break_index);
      std::printf("K=%d %s vs cpu-fp32: break %.4f\n", harmonics,
                  BackendName(backend).c_str(), same);
      EXPECT_GE(same, 0.999);
    }
  }
}

TEST(GpuPipeline, StreamsTilesAndBlocksGiveIdenticalResults) {
  PipelineFixture& f = Fixture();
  for (const Backend backend : {Backend::kGpuFused, Backend::kGpuUnfused}) {
    RunConfig base_config = GpuConfig(backend, 3, 64, 256);
    base_config.save_coefficients = true;
    const DetectionResult base =
        RunGpu(f.cube.source.get(), f.params, base_config, nullptr, nullptr);
    struct Variant {
      int streams;
      int tile_rows;
      int block;
    };
    // 100 rows leaves a short last tile (256 = 100 + 100 + 56).
    const Variant variants[] = {{1, 64, 256},  {3, 8, 256},  {3, 100, 256},
                                {3, 0, 256},   {3, 64, 128}, {1, 1, 32},
                                {3, 256, 1024}};
    for (const Variant& v : variants) {
      RunConfig config = GpuConfig(backend, v.streams, v.tile_rows, v.block);
      config.save_coefficients = true;
      const DetectionResult other =
          RunGpu(f.cube.source.get(), f.params, config, nullptr, nullptr);
      EXPECT_TRUE(Identical(base, other));
      if (!Identical(base, other)) {
        std::printf("%s differs for streams %d tile_rows %d block %d\n",
                    BackendName(backend).c_str(), v.streams, v.tile_rows,
                    v.block);
      }
    }
  }
}

TEST(GpuPipeline, TimingsAndStatsAreFilled) {
  PipelineFixture& f = Fixture();
  StageTimings fused;
  StageTimings unfused;
  RunStats stats;
  RunGpu(f.cube.source.get(), f.params,
         GpuConfig(Backend::kGpuFused, 3, 64, 256), &fused, &stats);
  EXPECT_EQ(stats.num_tiles, 4);
  EXPECT_EQ(stats.tile_rows_used, 64);
  EXPECT_EQ(stats.streams_used, 3);
  EXPECT_FALSE(stats.device_name.empty());
  EXPECT_GT(stats.device_mem_peak_mb, 0.0);
  EXPECT_GT(stats.device_mem_alloc_mb, 0.0);
  EXPECT_EQ(stats.device_mem_peak_mb, stats.device_mem_alloc_mb);
  // Setup is part of the total and ends before the first tile is read.
  EXPECT_GT(fused.setup_ms, 0.0);
  EXPECT_LT(fused.setup_ms, fused.total_ms);
  EXPECT_GT(fused.h2d_ms, 0.0);
  EXPECT_GT(fused.d2h_ms, 0.0);
  EXPECT_GT(fused.fused_ms, 0.0);
  EXPECT_GT(fused.read_ms, 0.0);
  EXPECT_GE(fused.total_ms, fused.kernel_ms);
  EXPECT_EQ(fused.kernel_ms, fused.fused_ms);
  EXPECT_EQ(fused.ndvi_ms, -1.0);
  EXPECT_EQ(fused.read_wait_ms, -1.0);

  RunGpu(f.cube.source.get(), f.params,
         GpuConfig(Backend::kGpuUnfused, 1, 64, 256), &unfused, &stats);
  EXPECT_EQ(stats.streams_used, 1);
  EXPECT_GT(unfused.ndvi_ms, 0.0);
  EXPECT_GT(unfused.fit_ms, 0.0);
  EXPECT_GT(unfused.detect_ms, 0.0);
  EXPECT_EQ(unfused.fused_ms, -1.0);
  EXPECT_NEAR(unfused.kernel_ms,
              unfused.ndvi_ms + unfused.fit_ms + unfused.detect_ms, 1e-9);
  std::printf(
      "fused: h2d %.2f kernel %.2f d2h %.2f read %.2f total %.2f ms | "
      "unfused: ndvi %.2f fit %.2f detect %.2f ms\n",
      fused.h2d_ms, fused.kernel_ms, fused.d2h_ms, fused.read_ms,
      fused.total_ms, unfused.ndvi_ms, unfused.fit_ms, unfused.detect_ms);
}

TEST(GpuPipeline, RetriesWithFewerRowsWhenAllocationFails) {
  PipelineFixture& f = Fixture();
  const HookGuard guard;
  const RunConfig config = GpuConfig(Backend::kGpuFused, 3, 128, 256);
  const DetectionResult base =
      RunGpu(f.cube.source.get(), f.params, config, nullptr, nullptr);
  // 3 slots of 128 rows need about 25 MB; 3 MB forces three halvings.
  SetSlotAllocationLimitForTesting(3u << 20);
  RunStats stats;
  const DetectionResult limited =
      RunGpu(f.cube.source.get(), f.params, config, nullptr, &stats);
  EXPECT_TRUE(Identical(base, limited));
  EXPECT_EQ(stats.tile_rows_used, 16);
  ASSERT_EQ(stats.notes.size(), 3u);
  EXPECT_EQ(stats.notes[0], "tile allocation failed; retrying with 64 rows");
  EXPECT_EQ(stats.notes[2], "tile allocation failed; retrying with 16 rows");
  // A limit below one row's buffers is an error, not an endless retry.
  SetSlotAllocationLimitForTesting(1000);
  DetectionResult failed;
  const Status status = RunGpuDetection(f.cube.source.get(), f.params, config,
                                        &failed, nullptr, nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.message().find("out of device memory") !=
              std::string::npos);
}

TEST(GpuPipeline, BlockSizeAboveRegisterLimitIsReported) {
  PipelineFixture& f = Fixture();
  ModelParams params;
  params.harmonics = 3;
  ASSERT_TRUE(ResolveModelParams(&params).ok());
  // K = 3 kernels use more than 64 registers per thread, so a block of 1024
  // threads cannot launch; the error names the limit instead of reporting a
  // failed launch.
  for (const Backend backend : {Backend::kGpuFused, Backend::kGpuUnfused}) {
    DetectionResult result;
    const Status status = RunGpuDetection(f.cube.source.get(), params,
                                          GpuConfig(backend, 3, 64, 1024),
                                          &result, nullptr, nullptr);
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.message().find("--block-size 1024 exceeds the") !=
                std::string::npos);
    EXPECT_TRUE(status.message().find("-thread limit of the K=3") !=
                std::string::npos);
    EXPECT_TRUE(status.message().find("; use <= ") != std::string::npos);
    // The message does not describe a launch failure, and 512 works.
    EXPECT_TRUE(status.message().find("cudaGetLastError") == std::string::npos);
    const DetectionResult ok =
        RunGpu(f.cube.source.get(), params, GpuConfig(backend, 3, 64, 512),
               nullptr, nullptr);
    EXPECT_EQ(ok.status.size(), 256u * 256u);
  }
  // K = 2 still allows 1024 threads per block.
  const DetectionResult k2 =
      RunGpu(f.cube.source.get(), f.params,
             GpuConfig(Backend::kGpuFused, 3, 64, 1024), nullptr, nullptr);
  EXPECT_EQ(k2.status.size(), 256u * 256u);
}

TEST(GpuPipeline, NeverAllocatesMoreSlotsThanTiles) {
  PipelineFixture& f = Fixture();
  RunStats three_one_tile;
  RunStats one_one_tile;
  RunGpu(f.cube.source.get(), f.params,
         GpuConfig(Backend::kGpuFused, 3, 256, 256), nullptr, &three_one_tile);
  RunGpu(f.cube.source.get(), f.params,
         GpuConfig(Backend::kGpuFused, 1, 256, 256), nullptr, &one_one_tile);
  // A single tile needs a single slot, whatever the stream count.
  EXPECT_EQ(three_one_tile.num_tiles, 1);
  EXPECT_EQ(three_one_tile.device_mem_alloc_mb,
            one_one_tile.device_mem_alloc_mb);
  // Two tiles use two slots, not three.
  RunStats three_two_tiles;
  RunStats one_two_tiles;
  RunGpu(f.cube.source.get(), f.params,
         GpuConfig(Backend::kGpuFused, 3, 128, 256), nullptr, &three_two_tiles);
  RunGpu(f.cube.source.get(), f.params,
         GpuConfig(Backend::kGpuFused, 1, 128, 256), nullptr, &one_two_tiles);
  EXPECT_EQ(three_two_tiles.num_tiles, 2);
  EXPECT_NEAR(three_two_tiles.device_mem_alloc_mb,
              2.0 * one_two_tiles.device_mem_alloc_mb, 1e-9);
}

// One parsed row of a --trace-csv file.
struct TraceLine {
  int tile = 0;
  int slot = 0;
  std::string stage;
  int stream = 0;
  double start_ms = 0.0;
  double end_ms = 0.0;
};

// Reads a trace file; `header_ok` tells whether the header line was right.
std::vector<TraceLine> ReadTrace(const std::string& path, bool* header_ok) {
  std::vector<TraceLine> lines;
  std::FILE* file = std::fopen(path.c_str(), "r");
  *header_ok = false;
  if (file == nullptr) {
    return lines;
  }
  char text[256];
  *header_ok = std::fgets(text, sizeof(text), file) != nullptr &&
               std::string(text) == "tile,slot,stage,stream,start_ms,end_ms\n";
  while (std::fgets(text, sizeof(text), file) != nullptr) {
    TraceLine line;
    char stage[32];
    if (std::sscanf(text, "%d,%d,%31[a-z0-9],%d,%lf,%lf", &line.tile,
                    &line.slot, stage, &line.stream, &line.start_ms,
                    &line.end_ms) == 6) {
      line.stage = stage;
      lines.push_back(line);
    }
  }
  std::fclose(file);
  return lines;
}

// Checks one trace of `num_tiles` tiles run with `streams` streams on a
// backend whose compute stages are `compute` (in order).
void ExpectValidTrace(const std::vector<TraceLine>& lines, int num_tiles,
                      int streams, const std::vector<std::string>& compute) {
  // read, h2d, the compute stages and d2h of every tile, tile by tile.
  const size_t per_tile = compute.size() + 3;
  ASSERT_EQ(lines.size(), per_tile * static_cast<size_t>(num_tiles));
  const int slots = streams == 3 ? std::min(3, num_tiles) : 1;
  // A single slot (one tile, or --streams 1) runs on one stream.
  const int lanes = slots == 1 ? 1 : 3;
  // Device events share one timeline (tiny tolerance for the CSV rounding);
  // the host read is aligned to it only to within the wake-up latency of the
  // wait on the reference event.
  constexpr double kDeviceTol = 1e-3;
  constexpr double kHostTol = 0.5;
  for (int k = 0; k < num_tiles; ++k) {
    const TraceLine* t = &lines[per_tile * static_cast<size_t>(k)];
    EXPECT_EQ(t[0].stage, std::string("read"));
    EXPECT_EQ(t[1].stage, std::string("h2d"));
    for (size_t i = 0; i < compute.size(); ++i) {
      EXPECT_EQ(t[2 + i].stage, compute[i]);
    }
    EXPECT_EQ(t[per_tile - 1].stage, std::string("d2h"));
    for (size_t i = 0; i < per_tile; ++i) {
      EXPECT_EQ(t[i].tile, k);
      EXPECT_EQ(t[i].slot, k % slots);
      EXPECT_GE(t[i].start_ms, 0.0);
      EXPECT_GE(t[i].end_ms, t[i].start_ms);
    }
    // Stream lanes: the host read is -1; with one stream everything is 0.
    EXPECT_EQ(t[0].stream, -1);
    EXPECT_EQ(t[1].stream, 0);
    EXPECT_EQ(t[2].stream, lanes == 3 ? 1 : 0);
    EXPECT_EQ(t[per_tile - 1].stream, lanes == 3 ? 2 : 0);
    // Order inside a tile: read <= h2d <= kernels <= d2h.
    EXPECT_GE(t[1].start_ms, t[0].end_ms - kHostTol);
    EXPECT_GE(t[2].start_ms, t[1].end_ms - kDeviceTol);
    for (size_t i = 3; i < per_tile - 1; ++i) {
      EXPECT_GE(t[i].start_ms, t[i - 1].end_ms - kDeviceTol);
    }
    EXPECT_GE(t[per_tile - 1].start_ms, t[per_tile - 2].end_ms - kDeviceTol);
    if (streams == 1 && k + 1 < num_tiles) {
      // The baseline finishes a tile before it reads the next one.
      EXPECT_GE(t[per_tile].start_ms, t[per_tile - 1].end_ms - kHostTol);
    }
  }
}

TEST(GpuPipeline, TraceCsvHasOneRowPerTileAndStage) {
  PipelineFixture& f = Fixture();
  const std::string path =
      (std::filesystem::temp_directory_path() / "cd_trace_test.csv").string();
  struct Variant {
    Backend backend;
    int streams;
    int tile_rows;
    int num_tiles;
    std::vector<std::string> compute;
  };
  const std::vector<Variant> variants = {
      {Backend::kGpuFused, 3, 32, 8, {"fused"}},
      {Backend::kGpuFused, 1, 32, 8, {"fused"}},
      {Backend::kGpuUnfused, 3, 100, 3, {"ndvi", "fit", "detect"}},
      {Backend::kGpuFused, 3, 256, 1, {"fused"}},
      {Backend::kGpuFused, 3, 128, 2, {"fused"}}};
  for (const Variant& v : variants) {
    RunConfig config = GpuConfig(v.backend, v.streams, v.tile_rows, 256);
    config.trace_csv = path;
    std::remove(path.c_str());
    RunGpu(f.cube.source.get(), f.params, config, nullptr, nullptr);
    bool header_ok = false;
    const std::vector<TraceLine> lines = ReadTrace(path, &header_ok);
    EXPECT_TRUE(header_ok);
    ExpectValidTrace(lines, v.num_tiles, v.streams, v.compute);
  }
  std::remove(path.c_str());
}

TEST(GpuPipeline, TraceDoesNotChangeResults) {
  PipelineFixture& f = Fixture();
  const std::string path =
      (std::filesystem::temp_directory_path() / "cd_trace_same.csv").string();
  for (const Backend backend : {Backend::kGpuFused, Backend::kGpuUnfused}) {
    for (const int streams : {1, 3}) {
      RunConfig plain = GpuConfig(backend, streams, 48, 256);
      plain.save_coefficients = true;
      RunConfig traced = plain;
      traced.trace_csv = path;
      const DetectionResult a =
          RunGpu(f.cube.source.get(), f.params, plain, nullptr, nullptr);
      const DetectionResult b =
          RunGpu(f.cube.source.get(), f.params, traced, nullptr, nullptr);
      EXPECT_TRUE(Identical(a, b));
    }
  }
  std::remove(path.c_str());
}

TEST(GpuPipeline, TraceToAnUnwritablePathIsAnError) {
  PipelineFixture& f = Fixture();
  RunConfig config = GpuConfig(Backend::kGpuFused, 3, 64, 256);
  config.trace_csv = "/nonexistent_cd_dir/trace.csv";
  DetectionResult result;
  const Status status = RunGpuDetection(f.cube.source.get(), f.params, config,
                                        &result, nullptr, nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.message().find("trace file") != std::string::npos);
}

TEST(GpuPipeline, RejectsInvalidConfigurations) {
  PipelineFixture& f = Fixture();
  DetectionResult result;
  CubeSource* source = f.cube.source.get();
  EXPECT_FALSE(RunGpuDetection(source, f.params,
                               GpuConfig(Backend::kGpuFused, 2, 0, 256),
                               &result, nullptr, nullptr)
                   .ok());
  EXPECT_FALSE(RunGpuDetection(source, f.params,
                               GpuConfig(Backend::kGpuFused, 3, 0, 100),
                               &result, nullptr, nullptr)
                   .ok());
  EXPECT_FALSE(RunGpuDetection(source, f.params,
                               GpuConfig(Backend::kCpuFp32, 3, 0, 256), &result,
                               nullptr, nullptr)
                   .ok());
  EXPECT_FALSE(RunGpuDetection(nullptr, f.params,
                               GpuConfig(Backend::kGpuFused, 3, 0, 256),
                               &result, nullptr, nullptr)
                   .ok());
  ModelParams unresolved;
  EXPECT_FALSE(RunGpuDetection(source, unresolved,
                               GpuConfig(Backend::kGpuFused, 3, 0, 256),
                               &result, nullptr, nullptr)
                   .ok());
  // More dates than the constant-memory tables hold.
  cputest::MemoryCubeSource many(2, 2,
                                 cputest::MakeRegularDates(kMaxDates + 1, 2));
  const Status status =
      RunGpuDetection(&many, f.params, GpuConfig(Backend::kGpuFused, 3, 0, 256),
                      &result, nullptr, nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(status.message().find("1024") != std::string::npos);
}

TEST(GpuPipeline, ReadErrorIsReportedAndResourcesAreFreed) {
  const ModelParams params = DefaultParams();
  for (const int streams : {1, 3}) {
    cputest::MemoryCubeSource cube(40, 8, cputest::MakeRegularDates(30, 40));
    for (int t = 0; t < 30; ++t) {
      for (int y = 0; y < 40; ++y) {
        for (int x = 0; x < 8; ++x) {
          cube.SetNdvi(t, y, x, 0.5 + 0.001 * t);
        }
      }
    }
    cube.FailReadsAfter(2);  // the third tile fails
    DetectionResult result;
    const Status status = RunGpuDetection(
        &cube, params, GpuConfig(Backend::kGpuFused, streams, 8, 256), &result,
        nullptr, nullptr);
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(status.message().find("injected read failure") !=
                std::string::npos);
  }
  // The device is still usable afterwards.
  PipelineFixture& f = Fixture();
  const DetectionResult again =
      RunGpu(f.cube.source.get(), f.params,
             GpuConfig(Backend::kGpuFused, 3, 64, 256), nullptr, nullptr);
  EXPECT_EQ(again.status.size(), 256u * 256u);
}

TEST(GpuPipeline, QuickDiffMatchesCpu) {
  TestCube cube = MakeTestCube(100, 70, 48, 11);
  const ModelParams params = DefaultParams();
  QuickDiffResult cpu;
  RunConfig cpu_config;
  ASSERT_TRUE(CpuQuickDiff(cube.source.get(), 3, 40, 0.15f, params, cpu_config,
                           &cpu, nullptr)
                  .ok());
  for (const int streams : {1, 3}) {
    for (const int tile_rows : {0, 16, 33}) {
      QuickDiffResult gpu;
      StageTimings timings;
      ASSERT_TRUE(
          GpuQuickDiff(cube.source.get(), 3, 40, 0.15f, params,
                       GpuConfig(Backend::kGpuFused, streams, tile_rows, 128),
                       &gpu, &timings)
              .ok());
      ASSERT_EQ(gpu.diff.size(), cpu.diff.size());
      double worst = 0.0;
      for (size_t i = 0; i < gpu.diff.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(gpu.diff[i]) -
                                          static_cast<double>(cpu.diff[i])));
      }
      EXPECT_LE(worst, 1e-6);
      EXPECT_TRUE(gpu.flag == cpu.flag);
      EXPECT_GT(timings.total_ms, 0.0);
      EXPECT_GT(timings.kernel_ms, 0.0);
    }
  }
  QuickDiffResult unused;
  EXPECT_FALSE(GpuQuickDiff(cube.source.get(), 3, 48, 0.15f, params,
                            GpuConfig(Backend::kGpuFused, 3, 0, 256), &unused,
                            nullptr)
                   .ok());
}

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------

// 48 dates 30 days apart from 2019-01-01 with calendar years filled; the
// first 24 are training dates.
DateTable PostprocessDates() {
  DateTable dates = cputest::MakeRegularDates(48, 30);
  ModelParams params;
  const Status status = BuildDateTable(params, &dates);
  EXPECT_TRUE(status.ok());
  return dates;
}

// A random result map: about 35 % break pixels (so that the cleanup rule has
// neighbours to count), 10 % insufficient data, and a few break pixels with an
// out-of-range break index.
DetectionResult RandomResult(int height, int width, uint32_t seed) {
  DetectionResult r;
  AllocateResult(height, width, 6, false, &r);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (size_t i = 0; i < r.status.size(); ++i) {
    const float draw = uniform(rng);
    if (draw < 0.35f) {
      r.status[i] = kBreakCode;
      r.break_index[i] = static_cast<int16_t>(24 + rng() % 24);
      const float size = 0.1f + 0.4f * uniform(rng);
      r.magnitude[i] = uniform(rng) < 0.8f ? -size : size;
      if (uniform(rng) < 0.01f) {
        r.break_index[i] = 99;  // outside the date table
      }
    } else if (draw < 0.45f) {
      r.status[i] = kNoDataCode;
    }
  }
  return r;
}

TEST(GpuPostprocess, ColorizeMatchesCpuByteForByte) {
  const HookGuard guard;
  const DateTable dates = PostprocessDates();
  const DetectionResult r = RandomResult(37, 41, 1);
  std::vector<uint8_t> cpu;
  ASSERT_TRUE(CpuColorizeChangeMap(r, dates, &cpu, nullptr).ok());
  for (const int chunk : {0, 1, 5, 36}) {
    SetPostprocessChunkRowsForTesting(chunk);
    std::vector<uint8_t> gpu;
    double ms = -1.0;
    ASSERT_TRUE(GpuColorizeChangeMap(r, dates, 0, &gpu, &ms).ok());
    EXPECT_TRUE(gpu == cpu);
    EXPECT_GT(ms, 0.0);
  }
  DetectionResult bad = r;
  bad.status.pop_back();
  std::vector<uint8_t> unused;
  EXPECT_FALSE(GpuColorizeChangeMap(bad, dates, 0, &unused, nullptr).ok());
  DateTable no_monitoring = dates;
  no_monitoring.num_train_dates = 48;
  EXPECT_FALSE(
      GpuColorizeChangeMap(r, no_monitoring, 0, &unused, nullptr).ok());
}

TEST(GpuPostprocess, CleanupMatchesCpu) {
  const HookGuard guard;
  // Odd sizes and chunk heights exercise the halo handling at chunk borders,
  // partial blocks and the image edges (including a one-column image).
  const int sizes[][2] = {{37, 41}, {70, 33}, {16, 1}, {1, 20}, {64, 64}};
  for (const auto& size : sizes) {
    const DetectionResult input = RandomResult(size[0], size[1], 2);
    DetectionResult cpu = input;
    ASSERT_TRUE(CpuCleanup(&cpu, nullptr).ok());
    for (const int chunk : {0, 1, 3, 8, 11}) {
      SetPostprocessChunkRowsForTesting(chunk);
      DetectionResult gpu = input;
      double ms = -1.0;
      ASSERT_TRUE(GpuCleanup(&gpu, 0, &ms).ok());
      EXPECT_TRUE(gpu.status == cpu.status);
      EXPECT_TRUE(gpu.break_index == cpu.break_index);
      EXPECT_TRUE(gpu.magnitude == cpu.magnitude);
      EXPECT_GT(ms, 0.0);
    }
  }
  // Something was actually cleaned.
  const DetectionResult input = RandomResult(37, 41, 2);
  DetectionResult cleaned = input;
  ASSERT_TRUE(GpuCleanup(&cleaned, 0, nullptr).ok());
  EXPECT_TRUE(input.status != cleaned.status);
}

// Confusion counts of the host, with the same rule as the kernel.
ConfusionCounts HostConfusion(const DetectionResult& r,
                              const std::vector<uint8_t>& reference,
                              const DateTable& d, const ValidationWindow& w) {
  ConfusionCounts counts;
  for (size_t i = 0; i < r.status.size(); ++i) {
    int break_year = 0;
    if (r.status[i] == kBreakCode && r.break_index[i] >= 0 &&
        static_cast<size_t>(r.break_index[i]) < d.year.size()) {
      break_year = d.year[static_cast<size_t>(r.break_index[i])];
    }
    const PixelValidation v =
        ClassifyPixel(r.status[i], break_year, r.magnitude[i], reference[i], w);
    switch (v.cls) {
      case ValidationClass::kTruePositive:
        ++counts.tp;
        break;
      case ValidationClass::kFalsePositive:
        ++counts.fp;
        break;
      case ValidationClass::kFalseNegative:
        ++counts.fn;
        break;
      case ValidationClass::kTrueNegative:
        ++counts.tn;
        break;
      default:
        ++counts.excluded;
        break;
    }
    counts.year_exact += v.year_exact ? 1 : 0;
    counts.year_within1 += v.year_within1 ? 1 : 0;
  }
  return counts;
}

TEST(GpuPostprocess, ConfusionMatchesHostCounts) {
  const HookGuard guard;
  const DateTable dates = PostprocessDates();
  ValidationWindow window;
  window.first_year = 2021;
  window.last_year = 2022;
  // 1200 x 1000 pixels is more than the grid-stride loop covers in one pass.
  const int sizes[][2] = {{37, 41}, {1200, 1000}};
  for (const auto& size : sizes) {
    const DetectionResult r = RandomResult(size[0], size[1], 3);
    std::mt19937 rng(9);
    std::vector<uint8_t> reference(r.status.size());
    for (uint8_t& value : reference) {
      const uint32_t draw = rng() % 10;
      // 255 excluded, 0 no loss, 21/22 loss inside the window, 24 outside.
      value = draw == 0 ? 255
                        : (draw < 5 ? 0
                                    : static_cast<uint8_t>(
                                          draw == 9 ? 24 : 21 + draw % 2));
    }
    const ConfusionCounts host = HostConfusion(r, reference, dates, window);
    for (const int chunk : {0, 7}) {
      SetPostprocessChunkRowsForTesting(chunk);
      ConfusionCounts gpu;
      double ms = -1.0;
      ASSERT_TRUE(GpuConfusion(r, reference, dates, window, 0, &gpu, &ms).ok());
      EXPECT_EQ(gpu.tp, host.tp);
      EXPECT_EQ(gpu.fp, host.fp);
      EXPECT_EQ(gpu.fn, host.fn);
      EXPECT_EQ(gpu.tn, host.tn);
      EXPECT_EQ(gpu.excluded, host.excluded);
      EXPECT_EQ(gpu.year_exact, host.year_exact);
      EXPECT_EQ(gpu.year_within1, host.year_within1);
      EXPECT_EQ(gpu.tp + gpu.fp + gpu.fn + gpu.tn + gpu.excluded,
                static_cast<uint64_t>(r.status.size()));
    }
    EXPECT_GT(host.tp, 0u);
    EXPECT_GT(host.fp, 0u);
    EXPECT_GT(host.fn, 0u);
    EXPECT_GT(host.year_exact, 0u);
  }
  // Mismatched reference size is an error.
  const DetectionResult r = RandomResult(10, 10, 4);
  ConfusionCounts unused;
  EXPECT_FALSE(GpuConfusion(r, std::vector<uint8_t>(5), dates, window, 0,
                            &unused, nullptr)
                   .ok());
}

// A reference raster for `cube`: about 10 % excluded (255), 40 % without
// loss and the rest with a loss year from one year before the monitoring
// window to one year after it.
std::vector<uint8_t> MakeReference(size_t pixels, const ValidationWindow& w,
                                   uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> reference(pixels);
  const uint32_t years = static_cast<uint32_t>(w.last_year - w.first_year + 3);
  for (uint8_t& value : reference) {
    const uint32_t draw = rng() % 10;
    value = draw == 0
                ? 255
                : (draw < 5
                       ? 0
                       : static_cast<uint8_t>(w.first_year - 1 - 2000 +
                                              static_cast<int>(rng() % years)));
  }
  return reference;
}

TEST(GpuPipeline, InPipelineValidationMatchesCpuCounts) {
  PipelineFixture& f = Fixture();
  DateTable dates = f.cube.source->info().dates;
  ASSERT_TRUE(BuildDateTable(f.params, &dates).ok());
  const ValidationWindow window = MakeValidationWindow(dates);
  const std::vector<uint8_t> reference = MakeReference(256 * 256, window, 5);
  GpuValidationRequest request;
  request.reference = &reference;
  request.window = window;
  request.year = &dates.year;
  struct Variant {
    Backend backend;
    int streams;
    int tile_rows;
  };
  const std::vector<Variant> variants = {
      {Backend::kGpuFused, 3, 32},   {Backend::kGpuFused, 1, 32},
      {Backend::kGpuFused, 3, 100},  {Backend::kGpuFused, 3, 256},
      {Backend::kGpuUnfused, 3, 48}, {Backend::kGpuUnfused, 1, 100}};
  for (const Variant& v : variants) {
    const RunConfig config = GpuConfig(v.backend, v.streams, v.tile_rows, 256);
    const DetectionResult plain =
        RunGpu(f.cube.source.get(), f.params, config, nullptr, nullptr);
    DetectionResult validated;
    ConfusionCounts gpu;
    StageTimings timings;
    ASSERT_TRUE(RunGpuDetection(f.cube.source.get(), f.params, config,
                                &validated, &timings, nullptr, &request, &gpu)
                    .ok());
    // The validation does not change the detection results...
    EXPECT_TRUE(Identical(plain, validated));
    // ...and its counts equal the host counts on those results.
    ConfusionCounts cpu;
    ASSERT_TRUE(
        ComputeConfusionCpu(validated, reference, dates, window, &cpu, nullptr)
            .ok());
    EXPECT_EQ(gpu.tp, cpu.tp);
    EXPECT_EQ(gpu.fp, cpu.fp);
    EXPECT_EQ(gpu.fn, cpu.fn);
    EXPECT_EQ(gpu.tn, cpu.tn);
    EXPECT_EQ(gpu.excluded, cpu.excluded);
    EXPECT_EQ(gpu.year_exact, cpu.year_exact);
    EXPECT_EQ(gpu.year_within1, cpu.year_within1);
    EXPECT_EQ(gpu.tp + gpu.fp + gpu.fn + gpu.tn + gpu.excluded, 256u * 256u);
    EXPECT_GT(cpu.tp, 0u);
    EXPECT_GT(cpu.fp, 0u);
    EXPECT_GT(timings.kernel_ms, 0.0);
  }
}

TEST(GpuPipeline, InPipelineValidationIsTracedAndChecked) {
  PipelineFixture& f = Fixture();
  DateTable dates = f.cube.source->info().dates;
  ASSERT_TRUE(BuildDateTable(f.params, &dates).ok());
  const ValidationWindow window = MakeValidationWindow(dates);
  const std::vector<uint8_t> reference = MakeReference(256 * 256, window, 6);
  GpuValidationRequest request;
  request.reference = &reference;
  request.window = window;
  request.year = &dates.year;
  // The validation kernel is one more compute stage of the trace.
  const std::string path =
      (std::filesystem::temp_directory_path() / "cd_trace_validate.csv")
          .string();
  RunConfig config = GpuConfig(Backend::kGpuFused, 3, 64, 256);
  config.trace_csv = path;
  DetectionResult result;
  ConfusionCounts counts;
  ASSERT_TRUE(RunGpuDetection(f.cube.source.get(), f.params, config, &result,
                              nullptr, nullptr, &request, &counts)
                  .ok());
  bool header_ok = false;
  const std::vector<TraceLine> lines = ReadTrace(path, &header_ok);
  EXPECT_TRUE(header_ok);
  ExpectValidTrace(lines, 4, 3, {"fused", "validate"});
  config.backend = Backend::kGpuUnfused;
  ASSERT_TRUE(RunGpuDetection(f.cube.source.get(), f.params, config, &result,
                              nullptr, nullptr, &request, &counts)
                  .ok());
  ExpectValidTrace(ReadTrace(path, &header_ok), 4, 3,
                   {"ndvi", "fit", "detect", "validate"});
  std::remove(path.c_str());

  // A request that does not fit the cube is refused.
  config.trace_csv.clear();
  const std::vector<uint8_t> short_reference(100, 0);
  GpuValidationRequest bad = request;
  bad.reference = &short_reference;
  EXPECT_FALSE(RunGpuDetection(f.cube.source.get(), f.params, config, &result,
                               nullptr, nullptr, &bad, &counts)
                   .ok());
  const std::vector<int16_t> short_years(3, 2020);
  bad = request;
  bad.year = &short_years;
  EXPECT_FALSE(RunGpuDetection(f.cube.source.get(), f.params, config, &result,
                               nullptr, nullptr, &bad, &counts)
                   .ok());
  EXPECT_FALSE(RunGpuDetection(f.cube.source.get(), f.params, config, &result,
                               nullptr, nullptr, &request, nullptr)
                   .ok());
}

TEST(GpuPostprocess, ColorizeOfDownsampledMapMatchesCpu) {
  const DateTable dates = PostprocessDates();
  const DetectionResult r = RandomResult(100, 130, 5);
  const DetectionResult small = DownsampleDetection(r, 32);
  EXPECT_EQ(small.width, 26);  // factor 5
  EXPECT_EQ(small.height, 20);
  std::vector<uint8_t> cpu;
  std::vector<uint8_t> gpu;
  ASSERT_TRUE(CpuColorizeChangeMap(small, dates, &cpu, nullptr).ok());
  ASSERT_TRUE(GpuColorizeChangeMap(small, dates, 0, &gpu, nullptr).ok());
  EXPECT_TRUE(cpu == gpu);
}

// ---------------------------------------------------------------------------
// Pipeline against the other pipeline entry point and the device queries
// ---------------------------------------------------------------------------

TEST(GpuPipeline, DeviceIsAvailable) {
  EXPECT_TRUE(GpuAvailable());
  GpuInfo info;
  ASSERT_TRUE(GpuDeviceInfo(0, &info).ok());
  EXPECT_FALSE(info.name.empty());
  EXPECT_GT(info.total_mem, 0u);
}

// ---------------------------------------------------------------------------
// Large: a cube bigger than the device memory (run with --filter LargeGpu)
// ---------------------------------------------------------------------------

// Directory of the large cube: $CD_LARGE_TEST_DIR or a default under /tmp.
std::string LargeCubeDir() {
  const char* dir = std::getenv("CD_LARGE_TEST_DIR");
  return dir != nullptr ? dir : "/tmp/cd_large_gpu_cube";
}

// True when $CD_LARGE_TEST_KEEP is set: the cube is then kept after the test
// and reused (not generated again) when it already exists.
bool KeepLargeCube() { return std::getenv("CD_LARGE_TEST_KEEP") != nullptr; }

// Removes the large cube when the test ends, however it ends (unless the
// cube is to be kept).
struct CubeDirCleaner {
  explicit CubeDirCleaner(std::string d) : dir(std::move(d)) {}
  ~CubeDirCleaner() {
    if (!KeepLargeCube()) {
      std::error_code ignored;
      std::filesystem::remove_all(dir, ignored);
    }
  }
  std::string dir;
};

TEST(LargeGpu, StreamsCubeLargerThanDeviceMemory) {
  constexpr int kSide = 7168;
  constexpr int kDates = 48;
  const std::string dir = LargeCubeDir();
  const CubeDirCleaner cleaner(dir);

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
  const uint64_t input_bytes =
      static_cast<uint64_t>(kSide) * kSide * kDates * 5;
  std::printf("cube input %.2f GB, device free %.2f GB of %.2f GB\n",
              static_cast<double>(input_bytes) / 1e9,
              static_cast<double>(free_bytes) / 1e9,
              static_cast<double>(total_bytes) / 1e9);
  EXPECT_GT(input_bytes, static_cast<uint64_t>(free_bytes));

  SyntheticOptions options;
  options.height = kSide;
  options.width = kSide;
  options.num_dates = kDates;
  options.noise = 0.03f;
  const std::string manifest_path = dir + "/manifest.txt";
  const std::string truth_path = dir + "/truth_break_index.i16";
  if (KeepLargeCube() && std::filesystem::exists(manifest_path) &&
      std::filesystem::exists(truth_path)) {
    std::printf("reusing the cube in %s\n", dir.c_str());
  } else {
    const WallTimer generate_timer;
    ASSERT_TRUE(GenerateSyntheticCubeToDisk(options, dir).ok());
    std::printf("generated the cube in %.1f s\n",
                generate_timer.ElapsedMs() / 1000.0);
  }

  CubeInfo info;
  ASSERT_TRUE(ReadManifest(manifest_path, &info).ok());
  Status open_status = Status::Ok();
  const std::unique_ptr<CubeSource> source = OpenRawCube(info, &open_status);
  ASSERT_TRUE(open_status.ok());
  std::vector<int16_t> truth;
  ASSERT_TRUE(ReadRawArray<int16_t>(truth_path,
                                    static_cast<size_t>(kSide) * kSide, &truth)
                  .ok());
  const ModelParams params = DefaultParams();

  StageTimings timings3;
  RunStats stats3;
  DetectionResult gpu3;
  {
    RunConfig config = GpuConfig(Backend::kGpuFused, 3, 0, 256);
    const Status status = RunGpuDetection(source.get(), params, config, &gpu3,
                                          &timings3, &stats3);
    ASSERT_TRUE(status.ok());
  }
  std::printf(
      "gpu streams 3: tiles %d x %d rows | read %.0f h2d %.0f kernel %.0f "
      "d2h %.0f total %.0f ms | device %.0f MiB\n",
      stats3.num_tiles, stats3.tile_rows_used, timings3.read_ms,
      timings3.h2d_ms, timings3.kernel_ms, timings3.d2h_ms, timings3.total_ms,
      stats3.device_mem_alloc_mb);
  EXPECT_GT(stats3.num_tiles, 1);

  RecoveryStats recovery;
  ASSERT_TRUE(
      ComputeRecovery(source.get(), truth, gpu3, params, &recovery).ok());
  const double recovered = static_cast<double>(recovery.recovered) /
                           static_cast<double>(recovery.eligible);
  const double false_breaks = static_cast<double>(recovery.false_breaks) /
                              static_cast<double>(recovery.stable_truth);
  std::printf("recovery: %.4f of %.0f eligible, false breaks %.4f\n", recovered,
              static_cast<double>(recovery.eligible), false_breaks);
  EXPECT_GE(recovered, 0.95);
  EXPECT_LE(false_breaks, 0.05);

  {
    RunConfig config;
    config.backend = Backend::kCpuMt;
    config.quiet = true;
    DetectionResult cpu;
    StageTimings cpu_timings;
    ASSERT_TRUE(RunCpuDetection(source.get(), params, config, &cpu,
                                &cpu_timings, nullptr)
                    .ok());
    const double break_agreement =
        EqualFraction(gpu3.break_index, cpu.break_index);
    const double status_agreement = EqualFraction(gpu3.status, cpu.status);
    std::printf(
        "gpu vs cpu-mt: break %.5f status %.5f | cpu-mt total %.0f ms\n",
        break_agreement, status_agreement, cpu_timings.total_ms);
    EXPECT_GE(break_agreement, 0.995);
    EXPECT_GE(status_agreement, 0.995);
  }

  {
    StageTimings timings1;
    RunStats stats1;
    DetectionResult gpu1;
    RunConfig config = GpuConfig(Backend::kGpuFused, 1, 0, 256);
    ASSERT_TRUE(
        RunGpuDetection(source.get(), params, config, &gpu1, &timings1, &stats1)
            .ok());
    std::printf(
        "gpu streams 1: tiles %d x %d rows | read %.0f h2d %.0f kernel %.0f "
        "d2h %.0f total %.0f ms\n",
        stats1.num_tiles, stats1.tile_rows_used, timings1.read_ms,
        timings1.h2d_ms, timings1.kernel_ms, timings1.d2h_ms,
        timings1.total_ms);
    EXPECT_TRUE(Identical(gpu1, gpu3));
  }
}

}  // namespace
}  // namespace cd
