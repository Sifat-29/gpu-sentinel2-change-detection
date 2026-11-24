// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "colormap.h"
#include "cpu/cpu_backend.h"
#include "cube_source.h"
#include "cuda/gpu_backend.h"
#include "logger.h"
#include "postprocess.h"
#include "status.h"
#include "system_info.h"
#include "test_framework.h"
#include "types.h"
#include "validation_rules.h"

namespace cd {
namespace {

// 48 dates spread evenly over four years starting at 2019-01-01 (day 17897).
DateTable MakeDates(int count, double years) {
  DateTable dates;
  for (int i = 0; i < count; ++i) {
    const int day =
        17897 + static_cast<int>(std::llround(i * 365.25 * years / count));
    dates.day.push_back(day);
    const int year = 2019 + static_cast<int>((day - 17897) / 365.25);
    dates.iso.push_back(std::to_string(year) + "-06-15");
  }
  return dates;
}

TEST(Status, OkAndError) {
  const Status ok = Status::Ok();
  EXPECT_TRUE(ok.ok());
  EXPECT_TRUE(ok.message().empty());
  const Status error = Status::Error("boom");
  EXPECT_FALSE(error.ok());
  EXPECT_EQ(error.message(), "boom");
  EXPECT_FALSE(Status::Error("").ok());
}

TEST(Types, BackendNamesRoundTrip) {
  const Backend all[] = {Backend::kGpuFused, Backend::kGpuUnfused,
                         Backend::kCpuFp64, Backend::kCpuFp32, Backend::kCpuMt};
  const char* const names[] = {"gpu", "gpu-unfused", "cpu", "cpu-fp32",
                               "cpu-mt"};
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(BackendName(all[i]), names[i]);
    Backend parsed = Backend::kCpuMt;
    ASSERT_TRUE(ParseBackend(names[i], &parsed).ok());
    EXPECT_EQ(static_cast<int>(parsed), static_cast<int>(all[i]));
  }
  Backend unused;
  const Status status = ParseBackend("fpga", &unused);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("fpga"), std::string::npos);
}

TEST(Types, IsGpuBackend) {
  EXPECT_TRUE(IsGpuBackend(Backend::kGpuFused));
  EXPECT_TRUE(IsGpuBackend(Backend::kGpuUnfused));
  EXPECT_FALSE(IsGpuBackend(Backend::kCpuFp64));
  EXPECT_FALSE(IsGpuBackend(Backend::kCpuFp32));
  EXPECT_FALSE(IsGpuBackend(Backend::kCpuMt));
}

TEST(Types, NumParams) {
  EXPECT_EQ(NumParams(1), 4);
  EXPECT_EQ(NumParams(2), 6);
  EXPECT_EQ(NumParams(3), 8);
  EXPECT_EQ(NumParams(kMaxHarmonics), kMaxParams);
}

TEST(Types, ResolveModelParamsAutoMinObs) {
  for (int k = 1; k <= 3; ++k) {
    ModelParams p;
    p.harmonics = k;
    ASSERT_TRUE(ResolveModelParams(&p).ok());
    EXPECT_EQ(p.min_train_obs, 3 * (2 + 2 * k));
  }
}

TEST(Types, ResolveModelParamsErrors) {
  ModelParams bad_k;
  bad_k.harmonics = 0;
  EXPECT_FALSE(ResolveModelParams(&bad_k).ok());
  bad_k.harmonics = 4;
  EXPECT_FALSE(ResolveModelParams(&bad_k).ok());

  ModelParams too_few;  // K = 2, p = 6: need at least 7
  too_few.min_train_obs = 6;
  EXPECT_FALSE(ResolveModelParams(&too_few).ok());
  too_few.min_train_obs = 7;
  EXPECT_TRUE(ResolveModelParams(&too_few).ok());
  EXPECT_EQ(too_few.min_train_obs, 7);

  ModelParams negative;
  negative.min_train_obs = -1;
  EXPECT_FALSE(ResolveModelParams(&negative).ok());
}

TEST(Types, AllocateResult) {
  DetectionResult r;
  AllocateResult(3, 5, 6, false, &r);
  EXPECT_EQ(r.height, 3);
  EXPECT_EQ(r.width, 5);
  EXPECT_EQ(r.num_params, 6);
  EXPECT_EQ(r.break_index.size(), 15u);
  EXPECT_EQ(r.magnitude.size(), 15u);
  EXPECT_EQ(r.status.size(), 15u);
  EXPECT_EQ(r.rmse.size(), 15u);
  EXPECT_EQ(r.train_obs.size(), 15u);
  EXPECT_TRUE(r.coefficients.empty());
  EXPECT_EQ(r.break_index[7], -1);
  EXPECT_EQ(r.status[7], static_cast<uint8_t>(PixelStatus::kStable));

  AllocateResult(2, 2, 4, true, &r);
  EXPECT_EQ(r.break_index.size(), 4u);
  EXPECT_EQ(r.coefficients.size(), 16u);
  AllocateResult(2, 2, 4, false, &r);
  EXPECT_TRUE(r.coefficients.empty());
}

TEST(Types, BuildDateTableMonthly) {
  DateTable dates = MakeDates(48, 4.0);
  ModelParams p;
  p.train_years = 2.0f;
  ASSERT_TRUE(BuildDateTable(p, &dates).ok());
  ASSERT_EQ(dates.tau.size(), 48u);
  ASSERT_EQ(dates.year.size(), 48u);
  EXPECT_NEAR(dates.tau[0], 0.0, 1e-12);
  EXPECT_NEAR(dates.tau[1], (dates.day[1] - dates.day[0]) / 365.25, 1e-12);
  EXPECT_EQ(dates.year[0], 2019);
  EXPECT_EQ(dates.year[47], 2022);
  EXPECT_EQ(dates.num_train_dates, 24);
  EXPECT_LT(dates.tau[23], 2.0);
  EXPECT_GE(dates.tau[24], 2.0);
}

TEST(Types, BuildDateTableErrors) {
  ModelParams p;
  DateTable empty;
  EXPECT_FALSE(BuildDateTable(p, &empty).ok());

  DateTable all_training = MakeDates(10, 1.0);  // spans one year only
  p.train_years = 2.0f;
  EXPECT_FALSE(BuildDateTable(p, &all_training).ok());

  DateTable none_training = MakeDates(10, 4.0);
  p.train_years = 1e-9f;  // tau[0] = 0 is below, so one training date remains
  EXPECT_TRUE(BuildDateTable(p, &none_training).ok());
  EXPECT_EQ(none_training.num_train_dates, 1);

  DateTable mismatch = MakeDates(10, 4.0);
  mismatch.iso.pop_back();
  p.train_years = 2.0f;
  EXPECT_FALSE(BuildDateTable(p, &mismatch).ok());

  DateTable bad_iso = MakeDates(10, 4.0);
  bad_iso.iso[3] = "garbage";
  EXPECT_FALSE(BuildDateTable(p, &bad_iso).ok());
}

TEST(Types, StageTimingsDefaultsAreNotApplicable) {
  const StageTimings t;
  EXPECT_EQ(t.read_ms, -1.0);
  EXPECT_EQ(t.wall_total_ms, -1.0);
  const RunConfig c;
  EXPECT_EQ(c.streams, 3);
  EXPECT_EQ(c.block_size, 256);
  const PixelTrajectory trajectory;
  EXPECT_EQ(trajectory.break_index, -1);
  EXPECT_EQ(trajectory.coef[kMaxParams - 1], 0.0);
}

// Colour tests: first monitoring day 1000, last day 1800 (span 800).
void Color(int16_t index, float magnitude, PixelStatus status, int break_day,
           uint8_t rgba[4]) {
  ChangeColor(index, magnitude, static_cast<uint8_t>(status), 1000, 1800,
              break_day, rgba);
}

TEST(Colormap, StableAndInsufficient) {
  uint8_t rgba[4];
  Color(-1, 0.0f, PixelStatus::kStable, 0, rgba);
  EXPECT_EQ(rgba[0], 40);
  EXPECT_EQ(rgba[1], 40);
  EXPECT_EQ(rgba[2], 40);
  EXPECT_EQ(rgba[3], 255);
  Color(-1, 0.0f, PixelStatus::kInsufficientData, 0, rgba);
  EXPECT_EQ(rgba[0], 0);
  EXPECT_EQ(rgba[1], 0);
  EXPECT_EQ(rgba[2], 0);
  EXPECT_EQ(rgba[3], 0);
}

TEST(Colormap, BreakRampEndpointsAndAlpha) {
  uint8_t rgba[4];
  Color(30, -0.3f, PixelStatus::kBreak, 1000, rgba);  // first stop
  EXPECT_EQ(rgba[0], 68);
  EXPECT_EQ(rgba[1], 1);
  EXPECT_EQ(rgba[2], 84);
  EXPECT_EQ(rgba[3], 255);
  Color(47, -0.3f, PixelStatus::kBreak, 1800, rgba);  // last stop
  EXPECT_EQ(rgba[0], 253);
  EXPECT_EQ(rgba[1], 231);
  EXPECT_EQ(rgba[2], 37);
  EXPECT_EQ(rgba[3], 255);
  Color(47, 0.3f, PixelStatus::kBreak, 1800, rgba);  // gain
  EXPECT_EQ(rgba[3], 160);
}

TEST(Colormap, StopBoundariesUseIntegerArithmetic) {
  // stop = min(7, 8 * (day - 1000) / 800): the stop changes every 100 days.
  const uint8_t expected_r[8] = {68, 70, 54, 39, 31, 74, 159, 253};
  for (int stop = 0; stop < 8; ++stop) {
    uint8_t rgba[4];
    Color(1, -0.2f, PixelStatus::kBreak, 1000 + 100 * stop, rgba);
    EXPECT_EQ(rgba[0], expected_r[stop]);
    if (stop > 0) {
      Color(1, -0.2f, PixelStatus::kBreak, 1000 + 100 * stop - 1, rgba);
      EXPECT_EQ(rgba[0], expected_r[stop - 1]);
    }
  }
  uint8_t rgba[4];
  Color(1, -0.2f, PixelStatus::kBreak, 5000, rgba);  // clamped to the last
  EXPECT_EQ(rgba[0], 253);
  Color(1, -0.2f, PixelStatus::kBreak, -50, rgba);  // clamped to the first
  EXPECT_EQ(rgba[0], 68);
}

TEST(Colormap, DegenerateSpanDoesNotDivideByZero) {
  uint8_t rgba[4];
  ChangeColor(1, -0.2f, static_cast<uint8_t>(PixelStatus::kBreak), 500, 500,
              500, rgba);
  EXPECT_EQ(rgba[0], 68);
  EXPECT_EQ(rgba[3], 255);
}

constexpr uint8_t kBreakStatus = static_cast<uint8_t>(PixelStatus::kBreak);
constexpr uint8_t kStableStatus = static_cast<uint8_t>(PixelStatus::kStable);

PixelValidation Classify(uint8_t status, int year, float magnitude,
                         uint8_t reference) {
  ValidationWindow window;
  window.first_year = 2021;
  window.last_year = 2024;
  return ClassifyPixel(status, year, magnitude, reference, window);
}

TEST(ValidationRules, Excluded) {
  EXPECT_EQ(Classify(kBreakStatus, 2022, -0.3f, 255).cls,
            ValidationClass::kExcluded);
  EXPECT_EQ(Classify(kStableStatus, -1, 0.0f, 255).cls,
            ValidationClass::kExcluded);
}

TEST(ValidationRules, TruePositiveWithYearAgreement) {
  PixelValidation v = Classify(kBreakStatus, 2022, -0.3f, 22);
  EXPECT_EQ(v.cls, ValidationClass::kTruePositive);
  EXPECT_TRUE(v.year_exact);
  EXPECT_TRUE(v.year_within1);

  v = Classify(kBreakStatus, 2023, -0.3f, 22);
  EXPECT_EQ(v.cls, ValidationClass::kTruePositive);
  EXPECT_FALSE(v.year_exact);
  EXPECT_TRUE(v.year_within1);

  v = Classify(kBreakStatus, 2021, -0.3f, 22);
  EXPECT_TRUE(v.year_within1);

  v = Classify(kBreakStatus, 2024, -0.3f, 22);
  EXPECT_EQ(v.cls, ValidationClass::kTruePositive);
  EXPECT_FALSE(v.year_exact);
  EXPECT_FALSE(v.year_within1);
}

TEST(ValidationRules, FalsePositiveAndNegatives) {
  EXPECT_EQ(Classify(kBreakStatus, 2022, -0.3f, 0).cls,
            ValidationClass::kFalsePositive);
  EXPECT_EQ(Classify(kStableStatus, -1, 0.0f, 22).cls,
            ValidationClass::kFalseNegative);
  EXPECT_EQ(Classify(kStableStatus, -1, 0.0f, 0).cls,
            ValidationClass::kTrueNegative);
}

TEST(ValidationRules, GainsDoNotCountAsDetections) {
  // A positive magnitude is a greenness gain: not a positive detection.
  EXPECT_EQ(Classify(kBreakStatus, 2022, 0.3f, 22).cls,
            ValidationClass::kFalseNegative);
  EXPECT_EQ(Classify(kBreakStatus, 2022, 0.3f, 0).cls,
            ValidationClass::kTrueNegative);
}

TEST(ValidationRules, WindowLimitsBothSides) {
  // Reference years outside 2021..2024 are not positive.
  EXPECT_EQ(Classify(kStableStatus, -1, 0.0f, 20).cls,
            ValidationClass::kTrueNegative);
  EXPECT_EQ(Classify(kStableStatus, -1, 0.0f, 25).cls,
            ValidationClass::kTrueNegative);
  EXPECT_EQ(Classify(kBreakStatus, 2022, -0.3f, 20).cls,
            ValidationClass::kFalsePositive);
  // A detection dated outside the window is not positive.
  EXPECT_EQ(Classify(kBreakStatus, 2020, -0.3f, 22).cls,
            ValidationClass::kFalseNegative);
  EXPECT_EQ(Classify(kBreakStatus, 2025, -0.3f, 0).cls,
            ValidationClass::kTrueNegative);
}

TEST(Logger, WritesToFile) {
  const std::string path = "/tmp/change_detect_test_logger.txt";
  ASSERT_TRUE(InitLog(path).ok());
  Log("hello");
  LogKv("key", "value");
  LogSection("section");
  CloseLog();
  std::FILE* file = std::fopen(path.c_str(), "r");
  ASSERT_TRUE(file != nullptr);
  char buffer[256] = {0};
  const size_t n = std::fread(buffer, 1, sizeof(buffer) - 1, file);
  std::fclose(file);
  std::remove(path.c_str());
  EXPECT_GT(n, 0u);
  EXPECT_EQ(std::string(buffer), "hello\nkey: value\n--- section ---\n");
  EXPECT_FALSE(InitLog("/nonexistent_dir/x/y.log").ok());
}

TEST(SystemInfo, BasicFields) {
  EXPECT_GE(HardwareThreads(), 1);
  EXPECT_FALSE(CpuModel().empty());
  const std::string stamp = IsoTimestamp();
  ASSERT_EQ(stamp.size(), 19u);
  EXPECT_EQ(stamp[4], '-');
  EXPECT_EQ(stamp[10], 'T');
  EXPECT_FALSE(GitShortCommit().empty());
  EXPECT_EQ(CpuSummaryLine().compare(0, 5, "cpu: "), 0);
  EXPECT_EQ(GpuSummaryLine(0).compare(0, 5, "gpu: "), 0);
  char arg0[] = "prog";
  char arg1[] = "--x";
  char* argv[] = {arg0, arg1};
  EXPECT_EQ(CommandLine(2, argv), "prog --x");
}

#ifndef CD_WITH_CUDA
TEST(GpuStub, ReportsMissingCuda) {
  EXPECT_FALSE(GpuAvailable());
  GpuInfo info;
  const Status status = GpuDeviceInfo(0, &info);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(), "built without CUDA; build with 'make gpu'");
  DetectionResult result;
  EXPECT_FALSE(RunGpuDetection(nullptr, ModelParams(), RunConfig(), &result,
                               nullptr, nullptr)
                   .ok());
  EXPECT_FALSE(GpuCleanup(&result, 0, nullptr).ok());
  EXPECT_NE(GpuSummaryLine(0).find("CPU-only build"), std::string::npos);
}
#endif  // CD_WITH_CUDA

}  // namespace
}  // namespace cd
