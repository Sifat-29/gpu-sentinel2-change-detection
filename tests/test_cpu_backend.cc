// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "cpu/cpu_backend.h"
#include "cpu_test_cube.h"
#include "pixel_model.h"
#include "system_info.h"
#include "test_framework.h"
#include "types.h"

namespace cd {
namespace {

// Model parameters used by every test: 1.5 training years on a 20-day cadence
// (28 training dates), everything else default.
ModelParams TestParams(int harmonics) {
  ModelParams params;
  params.harmonics = harmonics;
  params.train_years = 1.5f;
  ResolveModelParams(&params);
  return params;
}

// A random cube with known step changes, plus the parameters to analyse it.
struct Fixture {
  Fixture(int height, int width, int num_dates, int harmonics = 2,
          uint32_t seed = 7)
      : params(TestParams(harmonics)),
        cube(height, width, cputest::MakeRegularDates(num_dates, 20)) {
    DateTable dates = cube.info().dates;
    BuildDateTable(params, &dates);
    num_train = dates.num_train_dates;
    cputest::FillRandomCube(&cube, seed, num_train + 2, &step_date);
  }

  ModelParams params;
  cputest::MemoryCubeSource cube;
  std::vector<int> step_date;
  int num_train = 0;
};

struct RunOutput {
  Status status = Status::Ok();
  DetectionResult result;
  StageTimings timings;
  RunStats stats;
};

RunOutput Run(Fixture* fixture, Backend backend, int threads, int tile_rows,
              bool coefficients) {
  RunConfig config;
  config.backend = backend;
  config.threads = threads;
  config.tile_rows = tile_rows;
  config.save_coefficients = coefficients;
  RunOutput out;
  out.status = RunCpuDetection(&fixture->cube, fixture->params, config,
                               &out.result, &out.timings, &out.stats);
  return out;
}

// True when every map (including coefficients) is bit-identical.
bool Identical(const DetectionResult& a, const DetectionResult& b) {
  return a.height == b.height && a.width == b.width &&
         a.num_params == b.num_params && a.break_index == b.break_index &&
         a.magnitude == b.magnitude && a.status == b.status &&
         a.rmse == b.rmse && a.train_obs == b.train_obs &&
         a.coefficients == b.coefficients;
}

TEST(CpuBackend, MultithreadedMatchesSingleThreadFp32Exactly) {
  Fixture fixture(64, 96, 40);
  const RunOutput reference =
      Run(&fixture, Backend::kCpuFp32, 1, 0, /*coefficients=*/true);
  ASSERT_TRUE(reference.status.ok());
  const int thread_counts[] = {1, 3, 8};
  const int tile_rows[] = {7, 64, 0};
  for (const int threads : thread_counts) {
    for (const int rows : tile_rows) {
      const RunOutput mt = Run(&fixture, Backend::kCpuMt, threads, rows, true);
      ASSERT_TRUE(mt.status.ok());
      EXPECT_TRUE(Identical(reference.result, mt.result));
      EXPECT_EQ(mt.stats.threads_used, threads);
    }
  }
}

TEST(CpuBackend, Fp32MatchesFp64OnMostPixels) {
  Fixture fixture(64, 96, 40);
  const RunOutput fp64 = Run(&fixture, Backend::kCpuFp64, 1, 0, true);
  const RunOutput fp32 = Run(&fixture, Backend::kCpuFp32, 1, 0, true);
  ASSERT_TRUE(fp64.status.ok());
  ASSERT_TRUE(fp32.status.ok());
  const size_t pixels = fp64.result.break_index.size();
  size_t same_break = 0;
  size_t same_status = 0;
  double max_coef_diff = 0.0;
  for (size_t i = 0; i < pixels; ++i) {
    same_break += fp64.result.break_index[i] == fp32.result.break_index[i];
    same_status += fp64.result.status[i] == fp32.result.status[i];
    const uint8_t insufficient =
        static_cast<uint8_t>(PixelStatus::kInsufficientData);
    if (fp64.result.status[i] == insufficient ||
        fp32.result.status[i] == insufficient) {
      continue;
    }
    for (int j = 0; j < fp64.result.num_params; ++j) {
      const size_t k = static_cast<size_t>(j) * pixels + i;
      max_coef_diff =
          std::max(max_coef_diff,
                   std::fabs(static_cast<double>(fp64.result.coefficients[k]) -
                             fp32.result.coefficients[k]));
    }
  }
  const double agreement = static_cast<double>(same_break) / pixels;
  std::printf(
      "  fp32 vs fp64: identical break index %.3f %%, status %.3f %%,"
      " max |coef diff| %.2e\n",
      100.0 * agreement, 100.0 * same_status / pixels, max_coef_diff);
  EXPECT_GE(agreement, 0.99);
  EXPECT_LT(max_coef_diff, 1e-3);
}

TEST(CpuBackend, ResultsIndependentOfTileRows) {
  Fixture fixture(37, 29, 40);
  const RunOutput reference = Run(&fixture, Backend::kCpuFp64, 1, 0, true);
  ASSERT_TRUE(reference.status.ok());
  const int tile_rows[] = {1, 5, 36, 37, 1000};
  for (const int rows : tile_rows) {
    const RunOutput other = Run(&fixture, Backend::kCpuFp64, 1, rows, true);
    ASSERT_TRUE(other.status.ok());
    EXPECT_TRUE(Identical(reference.result, other.result));
  }
}

TEST(CpuBackend, RecoversInjectedSteps) {
  Fixture fixture(64, 96, 40);
  const RunOutput run = Run(&fixture, Backend::kCpuFp64, 1, 0, false);
  ASSERT_TRUE(run.status.ok());
  int steps = 0;
  int found = 0;
  int unchanged = 0;
  int false_alarms = 0;
  for (size_t i = 0; i < fixture.step_date.size(); ++i) {
    const int step = fixture.step_date[i];
    const uint8_t status = run.result.status[i];
    if (status == static_cast<uint8_t>(PixelStatus::kInsufficientData)) {
      continue;
    }
    // A run needs 3 valid observations: steps in the last 4 dates (or hidden
    // by clouds) cannot be found, so they are not counted.
    if (step >= 0 && step > fixture.cube.info().num_dates - 6) {
      continue;
    }
    if (step >= 0) {
      ++steps;
      // Clouds can hide the first dates after the step, so the run may start
      // up to two observations late.
      const int detected = run.result.break_index[i];
      if (status == static_cast<uint8_t>(PixelStatus::kBreak) &&
          detected >= step && detected <= step + 3 &&
          run.result.magnitude[i] < -0.2f) {
        ++found;
      }
    } else {
      ++unchanged;
      false_alarms += status == static_cast<uint8_t>(PixelStatus::kBreak);
    }
  }
  std::printf("  steps found %d of %d, false alarms %d of %d\n", found, steps,
              false_alarms, unchanged);
  EXPECT_GT(steps, 500);
  EXPECT_GE(found, static_cast<int>(0.95 * steps));
  EXPECT_LE(false_alarms, static_cast<int>(0.02 * unchanged));
}

TEST(CpuBackend, ReportsTimingsAndStats) {
  Fixture fixture(64, 96, 40);
  const RunOutput run = Run(&fixture, Backend::kCpuMt, 3, 7, false);
  ASSERT_TRUE(run.status.ok());
  const StageTimings& t = run.timings;
  EXPECT_GE(t.read_ms, 0.0);
  EXPECT_GE(t.read_wait_ms, 0.0);
  EXPECT_GT(t.kernel_ms, 0.0);
  EXPECT_GE(t.total_ms, t.kernel_ms);
  EXPECT_GE(t.setup_ms, 0.0);
  EXPECT_LT(t.setup_ms, t.total_ms);
  // GPU-only stages are not applicable on the CPU.
  EXPECT_EQ(t.h2d_ms, -1.0);
  EXPECT_EQ(t.ndvi_ms, -1.0);
  EXPECT_EQ(t.fit_ms, -1.0);
  EXPECT_EQ(t.detect_ms, -1.0);
  EXPECT_EQ(t.fused_ms, -1.0);
  EXPECT_EQ(t.d2h_ms, -1.0);
  EXPECT_EQ(run.stats.num_tiles, 10);  // ceil(64 / 7)
  EXPECT_EQ(run.stats.tile_rows_used, 7);
  EXPECT_EQ(run.stats.threads_used, 3);
  EXPECT_FALSE(run.stats.device_name.empty());
  // Defaults: 0 rows -> min(256, H), 0 threads -> all hardware threads for
  // cpu-mt and exactly one for the single-thread baselines.
  const RunOutput defaults = Run(&fixture, Backend::kCpuMt, 0, 0, false);
  EXPECT_EQ(defaults.stats.tile_rows_used, 64);
  EXPECT_EQ(defaults.stats.num_tiles, 1);
  EXPECT_EQ(defaults.stats.threads_used, HardwareThreads());
  const RunOutput single = Run(&fixture, Backend::kCpuFp64, 8, 0, false);
  EXPECT_EQ(single.stats.threads_used, 1);
}

TEST(CpuBackend, AutomaticTileRowsUseTheSharedInputCap) {
  // One row holds 4096 * 40 * 5 = 819200 bytes, so the 64 MiB input cap
  // allows floor(67108864 / 819200) = 81 rows -> 80 (multiple of 8), the
  // same size the GPU backend plans for this cube.
  Fixture fixture(100, 4096, 40);
  const RunOutput run =
      Run(&fixture, Backend::kCpuMt, 0, 0, /*coefficients=*/false);
  ASSERT_TRUE(run.status.ok());
  EXPECT_EQ(run.stats.tile_rows_used, 80);
  EXPECT_EQ(run.stats.num_tiles, 2);
}

TEST(CpuBackend, NullTimingsAndStatsAreAllowed) {
  Fixture fixture(8, 8, 40);
  RunConfig config;
  config.backend = Backend::kCpuFp32;
  DetectionResult result;
  EXPECT_TRUE(RunCpuDetection(&fixture.cube, fixture.params, config, &result,
                              nullptr, nullptr)
                  .ok());
  EXPECT_EQ(result.break_index.size(), 64u);
}

TEST(CpuBackend, CoefficientsOnlyWhenRequested) {
  Fixture fixture(16, 20, 40);
  const RunOutput without = Run(&fixture, Backend::kCpuFp32, 1, 0, false);
  ASSERT_TRUE(without.status.ok());
  EXPECT_TRUE(without.result.coefficients.empty());
  EXPECT_EQ(without.result.num_params, 6);
  const RunOutput with = Run(&fixture, Backend::kCpuFp32, 1, 0, true);
  ASSERT_TRUE(with.status.ok());
  EXPECT_EQ(with.result.coefficients.size(), 6u * 16u * 20u);
  // The other maps do not depend on the coefficient switch.
  EXPECT_TRUE(without.result.break_index == with.result.break_index);
  EXPECT_TRUE(without.result.rmse == with.result.rmse);
}

TEST(CpuBackend, OtherHarmonicCounts) {
  for (const int harmonics : {1, 3}) {
    Fixture fixture(20, 24, 40, harmonics);
    const RunOutput fp32 = Run(&fixture, Backend::kCpuFp32, 1, 0, true);
    const RunOutput mt = Run(&fixture, Backend::kCpuMt, 3, 6, true);
    ASSERT_TRUE(fp32.status.ok());
    ASSERT_TRUE(mt.status.ok());
    EXPECT_EQ(fp32.result.num_params, 2 + 2 * harmonics);
    EXPECT_EQ(fp32.result.coefficients.size(),
              static_cast<size_t>(2 + 2 * harmonics) * 20u * 24u);
    EXPECT_TRUE(Identical(fp32.result, mt.result));
    // The FP64 oracle agrees on the large majority of pixels.
    const RunOutput fp64 = Run(&fixture, Backend::kCpuFp64, 1, 0, false);
    size_t same = 0;
    for (size_t i = 0; i < fp64.result.break_index.size(); ++i) {
      same += fp64.result.break_index[i] == fp32.result.break_index[i];
    }
    EXPECT_GE(same, fp64.result.break_index.size() * 97 / 100);
  }
}

TEST(CpuBackend, RejectsInvalidRequests) {
  Fixture fixture(8, 8, 40);
  RunConfig config;
  DetectionResult result;
  config.backend = Backend::kGpuFused;
  EXPECT_FALSE(RunCpuDetection(&fixture.cube, fixture.params, config, &result,
                               nullptr, nullptr)
                   .ok());
  config.backend = Backend::kCpuFp32;
  EXPECT_FALSE(RunCpuDetection(nullptr, fixture.params, config, &result,
                               nullptr, nullptr)
                   .ok());
  EXPECT_FALSE(RunCpuDetection(&fixture.cube, fixture.params, config, nullptr,
                               nullptr, nullptr)
                   .ok());
  config.tile_rows = -1;
  EXPECT_FALSE(RunCpuDetection(&fixture.cube, fixture.params, config, &result,
                               nullptr, nullptr)
                   .ok());
  config.tile_rows = 0;
  // Unresolved min_train_obs (0) and an all-training date range are errors.
  ModelParams unresolved = fixture.params;
  unresolved.min_train_obs = 0;
  EXPECT_FALSE(RunCpuDetection(&fixture.cube, unresolved, config, &result,
                               nullptr, nullptr)
                   .ok());
  ModelParams no_monitoring = fixture.params;
  no_monitoring.train_years = 100.0f;
  const Status status = RunCpuDetection(&fixture.cube, no_monitoring, config,
                                        &result, nullptr, nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(status.message().empty());
}

TEST(CpuBackend, ReadFailureIsReported) {
  Fixture fixture(32, 8, 40);
  fixture.cube.FailReadsAfter(2);
  RunConfig config;
  config.backend = Backend::kCpuMt;
  config.tile_rows = 8;
  DetectionResult result;
  const Status status = RunCpuDetection(&fixture.cube, fixture.params, config,
                                        &result, nullptr, nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("injected read failure"), std::string::npos);
}

TEST(CpuBackend, TrajectoryMatchesDetection) {
  Fixture fixture(24, 30, 40);
  const RunOutput run = Run(&fixture, Backend::kCpuFp64, 1, 0, true);
  ASSERT_TRUE(run.status.ok());
  const size_t pixels = run.result.break_index.size();
  const std::vector<std::pair<int, int>> samples = {
      {0, 0},   {0, 29}, {23, 0},  {23, 29}, {5, 7}, {11, 13},
      {12, 14}, {17, 3}, {20, 25}, {2, 28},  {8, 8}, {15, 22}};
  int breaks_seen = 0;
  for (const auto& [row, col] : samples) {
    PixelTrajectory trajectory;
    ASSERT_TRUE(
        ComputeTrajectory(&fixture.cube, fixture.params, row, col, &trajectory)
            .ok());
    const size_t i = static_cast<size_t>(row) * 30 + col;
    EXPECT_EQ(trajectory.row, row);
    EXPECT_EQ(trajectory.col, col);
    EXPECT_EQ(trajectory.break_index, run.result.break_index[i]);
    EXPECT_EQ(trajectory.status, run.result.status[i]);
    EXPECT_EQ(trajectory.n_train, run.result.train_obs[i]);
    EXPECT_EQ(static_cast<float>(trajectory.rmse), run.result.rmse[i]);
    EXPECT_EQ(static_cast<float>(trajectory.magnitude),
              run.result.magnitude[i]);
    for (int j = 0; j < 6; ++j) {
      EXPECT_EQ(static_cast<float>(trajectory.coef[j]),
                run.result.coefficients[static_cast<size_t>(j) * pixels + i]);
    }
    ASSERT_EQ(trajectory.ndvi.size(), 40u);
    ASSERT_EQ(trajectory.predicted.size(), 40u);
    ASSERT_EQ(trajectory.z.size(), 40u);
    ASSERT_EQ(trajectory.in_break_run.size(), 40u);
    int in_run = 0;
    for (int t = 0; t < 40; ++t) {
      in_run += trajectory.in_break_run[static_cast<size_t>(t)];
    }
    if (trajectory.status == static_cast<uint8_t>(PixelStatus::kBreak)) {
      ++breaks_seen;
      EXPECT_EQ(in_run, fixture.params.consecutive);
      EXPECT_EQ(
          trajectory.in_break_run[static_cast<size_t>(trajectory.break_index)],
          1);
      EXPECT_GT(
          std::fabs(trajectory.z[static_cast<size_t>(trajectory.break_index)]),
          fixture.params.threshold);
    } else {
      EXPECT_EQ(in_run, 0);
    }
    // z = (ndvi - predicted) / rmse wherever the observation is valid.
    for (int t = 0; t < 40; ++t) {
      const double y = trajectory.ndvi[static_cast<size_t>(t)];
      if (y < -1.5) {
        EXPECT_EQ(trajectory.z[static_cast<size_t>(t)], 0.0);
      } else if (trajectory.rmse > 0.0) {
        EXPECT_NEAR(trajectory.z[static_cast<size_t>(t)],
                    (y - trajectory.predicted[static_cast<size_t>(t)]) /
                        trajectory.rmse,
                    1e-12);
      }
    }
  }
  EXPECT_GE(breaks_seen, 1);
}

TEST(CpuBackend, TrajectoryRejectsOutOfRangePixel) {
  Fixture fixture(8, 8, 40);
  PixelTrajectory trajectory;
  EXPECT_FALSE(
      ComputeTrajectory(&fixture.cube, fixture.params, 8, 0, &trajectory).ok());
  EXPECT_FALSE(
      ComputeTrajectory(&fixture.cube, fixture.params, 0, -1, &trajectory)
          .ok());
  EXPECT_FALSE(
      ComputeTrajectory(&fixture.cube, fixture.params, 0, 0, nullptr).ok());
}

// Performance sanity check; prints numbers, asserts nothing about speed. The
// suite name starts with "Large", so it only runs with --filter LargePerf.
double MedianOfThree(Fixture* fixture, Backend backend, int threads) {
  double times[3];
  for (double& time : times) {
    const RunOutput run = Run(fixture, backend, threads, 0, false);
    time = run.timings.total_ms;
  }
  std::sort(times, times + 3);
  return times[1];
}

TEST(LargePerf, Backends) {
  Fixture fixture(512, 512, 48);
  const double pixels = 512.0 * 512.0;
  const double fp64 = MedianOfThree(&fixture, Backend::kCpuFp64, 1);
  const double fp32 = MedianOfThree(&fixture, Backend::kCpuFp32, 1);
  const double mt = MedianOfThree(&fixture, Backend::kCpuMt, 0);
  std::printf("  512x512x48, median of 3, total_ms (us/pixel):\n");
  std::printf("    cpu      %9.1f ms  %.3f us/px\n", fp64, 1e3 * fp64 / pixels);
  std::printf("    cpu-fp32 %9.1f ms  %.3f us/px\n", fp32, 1e3 * fp32 / pixels);
  std::printf("    cpu-mt   %9.1f ms  %.3f us/px (%d threads)\n", mt,
              1e3 * mt / pixels, HardwareThreads());
  std::printf("    speed-up cpu-mt vs cpu-fp32: %.2fx, vs cpu: %.2fx\n",
              fp32 / mt, fp64 / mt);
  EXPECT_GT(mt, 0.0);
}

}  // namespace
}  // namespace cd
