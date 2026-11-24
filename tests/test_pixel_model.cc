// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "cpu_test_cube.h"
#include "pixel_model.h"
#include "test_framework.h"
#include "types.h"

namespace cd {
namespace {

// One pixel problem: dates, model parameters, the design table in precision
// Real and a series y that starts as the noise-free model for `coef`.
template <typename Real, int kHarmonics>
class Scenario {
 public:
  static constexpr int kParams = 2 + 2 * kHarmonics;

  // `num_dates` dates `spacing` days apart; the first dates with tau below
  // train_years (2 years) are the training window.
  explicit Scenario(int num_dates = 60, int spacing = 15)
      : Scenario(cputest::MakeRegularDates(num_dates, spacing)) {}

  explicit Scenario(DateTable raw_dates) : dates_(std::move(raw_dates)) {
    params_.harmonics = kHarmonics;
    // The rule tests below are written for |z| > 3 on 3 consecutive
    // observations; pin them so the tests do not depend on the defaults.
    params_.threshold = 3.0f;
    params_.consecutive = 3;
    ResolveModelParams(&params_);
    BuildDateTable(params_, &dates_);
    BuildDesignTable(dates_, kHarmonics, params_.train_years,
                     params_.hold_trend, &design_double_);
    for (const double v : design_double_) {
      design_.push_back(static_cast<Real>(v));
    }
    // Plausible NDVI coefficients: level, trend, then shrinking harmonics.
    const double kCoef[kMaxParams] = {0.5,   0.1,  0.15, 0.05,
                                      -0.04, 0.02, 0.01, -0.01};
    for (int j = 0; j < kParams; ++j) {
      coef_.push_back(kCoef[j]);
    }
    const int num_dates_total = static_cast<int>(dates_.day.size());
    for (int t = 0; t < num_dates_total; ++t) {
      double y = 0.0;
      for (int j = 0; j < kParams; ++j) {
        y += coef_[static_cast<size_t>(j)] * design_double_[Row(t) + j];
      }
      truth_.push_back(y);
      y_.push_back(static_cast<Real>(y));
    }
  }

  int num_dates() const { return static_cast<int>(dates_.day.size()); }
  int num_train() const { return dates_.num_train_dates; }
  ModelParams& params() { return params_; }
  const std::vector<double>& coef() const { return coef_; }
  std::vector<Real>& y() { return y_; }
  const std::vector<double>& truth() const { return truth_; }
  const std::vector<Real>& design() const { return design_; }

  // Adds `delta` to observations first .. last (inclusive).
  void AddStep(int first, int last, double delta) {
    for (int t = first; t <= last; ++t) {
      y_[static_cast<size_t>(t)] += static_cast<Real>(delta);
    }
  }

  void Invalidate(int t) { y_[static_cast<size_t>(t)] = Real(kInvalidNdvi); }

  PixelOutput<Real> Run() const {
    const NdviArraySeries<Real, Real> series{y_.data(), 1};
    return FitAndDetect<Real, kHarmonics>(series, design_.data(), num_dates(),
                                          num_train(), params_);
  }

 private:
  static size_t Row(int t) { return static_cast<size_t>(t) * kParams; }

  DateTable dates_;
  ModelParams params_;
  std::vector<double> design_double_;
  std::vector<Real> design_;
  std::vector<double> coef_;
  std::vector<double> truth_;
  std::vector<Real> y_;
};

uint8_t Code(PixelStatus s) { return static_cast<uint8_t>(s); }

TEST(PixelModel, DesignRowValues) {
  double row[8];  // large enough for K = 3
  // tau = 0 is the start of a 2-year training window: s = -1.
  DesignRow(0.0, 2, 2.0, row);
  EXPECT_NEAR(row[0], 1.0, 1e-15);
  EXPECT_NEAR(row[1], -1.0, 1e-15);
  EXPECT_NEAR(row[2], 1.0, 1e-15);
  EXPECT_NEAR(row[3], 0.0, 1e-15);
  EXPECT_NEAR(row[4], 1.0, 1e-15);
  EXPECT_NEAR(row[5], 0.0, 1e-15);
  // tau = 0.25: quarter year, s = -0.75, cos/sin of 90 and 180 degrees.
  DesignRow(0.25, 2, 2.0, row);
  EXPECT_NEAR(row[1], -0.75, 1e-15);
  EXPECT_NEAR(row[2], 0.0, 1e-15);
  EXPECT_NEAR(row[3], 1.0, 1e-15);
  EXPECT_NEAR(row[4], -1.0, 1e-15);
  EXPECT_NEAR(row[5], 0.0, 1e-15);
  // tau = 1 is the middle of the window: s = 0; tau = 2 gives s = +1.
  DesignRow(1.0, 1, 2.0, row);
  EXPECT_NEAR(row[1], 0.0, 1e-15);
  DesignRow(2.0, 3, 2.0, row);
  EXPECT_NEAR(row[1], 1.0, 1e-15);
}

TEST(PixelModel, DesignTableMatchesRows) {
  DateTable dates = cputest::MakeRegularDates(30, 20);
  ModelParams params;
  params.train_years = 1.0f;
  // Extrapolation: every row equals DesignRow of its own tau.
  params.hold_trend = false;
  ASSERT_TRUE(BuildDateTable(params, &dates).ok());
  std::vector<double> table;
  BuildDesignTable(dates, 3, params.train_years, params.hold_trend, &table);
  ASSERT_EQ(table.size(), 30u * 8u);
  double row[8];
  for (int t = 0; t < 30; ++t) {
    DesignRow(dates.tau[static_cast<size_t>(t)], 3, 1.0, row);
    for (int j = 0; j < 8; ++j) {
      EXPECT_EQ(table[static_cast<size_t>(t) * 8 + j], row[j]);
    }
  }
}

// With hold_trend the trend column is frozen at 1.0 (its value at
// tau = train_years) for monitoring dates, the seasonal columns still follow
// tau, and training rows are untouched. With extrapolation it exceeds 1.
TEST(PixelModel, DesignTableHoldsOrExtrapolatesTrend) {
  DateTable dates = cputest::MakeRegularDates(60, 20);
  ModelParams params;
  params.train_years = 1.0f;
  ASSERT_TRUE(BuildDateTable(params, &dates).ok());
  ASSERT_GT(dates.num_train_dates, 0);
  ASSERT_LT(dates.num_train_dates, 60);
  std::vector<double> held;
  std::vector<double> extrapolated;
  BuildDesignTable(dates, 2, params.train_years, true, &held);
  BuildDesignTable(dates, 2, params.train_years, false, &extrapolated);
  ASSERT_EQ(held.size(), extrapolated.size());
  const size_t num_params = 6;
  int monitored = 0;
  for (size_t t = 0; t < dates.tau.size(); ++t) {
    const double* h = held.data() + t * num_params;
    const double* x = extrapolated.data() + t * num_params;
    if (dates.tau[t] >= static_cast<double>(params.train_years)) {
      ++monitored;
      EXPECT_EQ(h[1], 1.0);
      EXPECT_GT(x[1], 1.0);
    } else {
      EXPECT_EQ(h[1], x[1]);
      EXPECT_LE(h[1], 1.0);
    }
    for (size_t j = 0; j < num_params; ++j) {
      if (j != 1) {
        EXPECT_EQ(h[j], x[j]);
      }
    }
  }
  EXPECT_GT(monitored, 0);
}

TEST(PixelModel, ComputeNdvi) {
  const uint32_t mask = (1u << 4) | (1u << 5) | (1u << 6);
  // red 0.1, nir 0.5 -> (0.5 - 0.1) / (0.5 + 0.1) = 2/3.
  EXPECT_NEAR(ComputeNdvi<double>(1000, 5000, 4, 1e-4, 0.0, mask), 2.0 / 3.0,
              1e-12);
  EXPECT_NEAR(ComputeNdvi<float>(1000, 5000, 6, 1e-4f, 0.0f, mask), 2.0 / 3.0,
              1e-6);
  // Class not in the mask (cloud 9, class 0, class 40 beyond the mask width).
  EXPECT_EQ(ComputeNdvi<float>(1000, 5000, 9, 1e-4f, 0.0f, mask), kInvalidNdvi);
  EXPECT_EQ(ComputeNdvi<float>(1000, 5000, 0, 1e-4f, 0.0f, mask), kInvalidNdvi);
  EXPECT_EQ(ComputeNdvi<float>(1000, 5000, 40, 1e-4f, 0.0f, mask),
            kInvalidNdvi);
  // No data: red or nir equal to 0.
  EXPECT_EQ(ComputeNdvi<float>(0, 5000, 4, 1e-4f, 0.0f, mask), kInvalidNdvi);
  EXPECT_EQ(ComputeNdvi<float>(1000, 0, 4, 1e-4f, 0.0f, mask), kInvalidNdvi);
  // Reflectance sum <= 1e-4.
  EXPECT_EQ(ComputeNdvi<float>(1, 1, 4, 1e-5f, 0.0f, mask), kInvalidNdvi);
  // Offset applies before the ratio: (0.4 - 0.0) / (0.4 + 0.0) is exactly 1,
  // and values beyond 1 are clamped.
  EXPECT_NEAR(ComputeNdvi<double>(1000, 5000, 4, 1e-4, -0.1, mask), 1.0, 1e-12);
  EXPECT_NEAR(ComputeNdvi<double>(900, 5000, 4, 1e-4, -0.1, mask), 1.0, 1e-12);
}

TEST(PixelModel, RawSeriesMatchesComputeNdvi) {
  // Two pixels side by side, three dates: stride is the plane size (2).
  const uint16_t red[6] = {1000, 1500, 1200, 0, 900, 800};
  const uint16_t nir[6] = {5000, 4000, 4500, 3000, 4800, 400};
  const uint8_t scl[6] = {4, 4, 9, 4, 5, 4};
  const float scale[3] = {1e-4f, 1e-4f, 2e-4f};
  const float offset[3] = {0.0f, 0.0f, 0.0f};
  const uint32_t mask = (1u << 4) | (1u << 5) | (1u << 6);
  for (int pixel = 0; pixel < 2; ++pixel) {
    const RawSeries<float> raw{red + pixel, nir + pixel, scl + pixel, 2,
                               scale,       offset,      mask};
    for (int t = 0; t < 3; ++t) {
      const int i = t * 2 + pixel;
      EXPECT_EQ(raw(t), ComputeNdvi<float>(red[i], nir[i], scl[i], scale[t],
                                           offset[t], mask));
    }
  }
  const RawSeries<float> second{red + 1, nir + 1, scl + 1, 2,
                                scale,   offset,  mask};
  EXPECT_EQ(second(0), ComputeNdvi<float>(1500, 4000, 4, 1e-4f, 0.0f, mask));
  EXPECT_EQ(second(1), kInvalidNdvi);  // red == 0
}

TEST(PixelModel, NdviArraySeriesStride) {
  const float values[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  const NdviArraySeries<float> strided{values + 1, 2};
  EXPECT_EQ(strided(0), 0.2f);
  EXPECT_EQ(strided(1), 0.4f);
  EXPECT_EQ(strided(2), 0.6f);
  const NdviArraySeries<double> widened{values, 3};
  EXPECT_EQ(widened(1), static_cast<double>(0.4f));
}

template <typename Real>
void CheckCholeskyKnownSystem() {
  // Classic SPD example with L = [[2,0,0],[6,1,0],[-8,5,3]].
  Real a[9] = {4, 12, -16, 12, 37, -43, -16, -43, 98};
  Real b[3] = {-20, -43, 192};  // A * [1, 2, 3]
  ASSERT_TRUE((CholeskySolve<Real, 3>(a, b)));
  EXPECT_NEAR(b[0], 1.0, 1e-5);
  EXPECT_NEAR(b[1], 2.0, 1e-5);
  EXPECT_NEAR(b[2], 3.0, 1e-5);
  // The lower triangle holds the factor.
  EXPECT_NEAR(a[0], 2.0, 1e-6);
  EXPECT_NEAR(a[3], 6.0, 1e-6);
  EXPECT_NEAR(a[4], 1.0, 1e-6);
  EXPECT_NEAR(a[6], -8.0, 1e-6);
  EXPECT_NEAR(a[7], 5.0, 1e-6);
  EXPECT_NEAR(a[8], 3.0, 1e-6);
}

TEST(PixelModel, CholeskyKnownSystem) {
  CheckCholeskyKnownSystem<double>();
  CheckCholeskyKnownSystem<float>();
}

template <typename Real>
void CheckCholeskyFailures() {
  // Singular: second pivot is 0.
  Real singular[4] = {1, 1, 1, 1};
  Real b[2] = {1, 1};
  EXPECT_FALSE((CholeskySolve<Real, 2>(singular, b)));
  // Pivot 2e-5 is below 1e-4 times the largest diagonal entry.
  Real near_singular[4] = {1, Real(0.99999), Real(0.99999), 1};
  Real b2[2] = {1, 1};
  EXPECT_FALSE((CholeskySolve<Real, 2>(near_singular, b2)));
  // Same matrix with a clearly positive second pivot (0.19) works.
  Real fine[4] = {1, Real(0.9), Real(0.9), 1};
  Real b3[2] = {1, 1};
  EXPECT_TRUE((CholeskySolve<Real, 2>(fine, b3)));
  // Not positive definite, all zero, and non-finite input.
  Real indefinite[4] = {1, 2, 2, 1};
  Real b4[2] = {1, 1};
  EXPECT_FALSE((CholeskySolve<Real, 2>(indefinite, b4)));
  Real zero[4] = {0, 0, 0, 0};
  Real b5[2] = {1, 1};
  EXPECT_FALSE((CholeskySolve<Real, 2>(zero, b5)));
  Real not_a_number[4] = {Real(NAN), 0, 0, 1};
  Real b6[2] = {1, 1};
  EXPECT_FALSE((CholeskySolve<Real, 2>(not_a_number, b6)));
}

TEST(PixelModel, CholeskyFailsOnSingularSystems) {
  CheckCholeskyFailures<double>();
  CheckCholeskyFailures<float>();
}

TEST(PixelModel, PredictNdviIsDotProduct) {
  const double coef[4] = {0.5, 0.25, -1.0, 2.0};
  const double row[4] = {1.0, 2.0, 3.0, 4.0};
  EXPECT_NEAR((PredictNdvi<double, 1>(coef, row)), 0.5 + 0.5 - 3.0 + 8.0,
              1e-15);
}

template <typename Real, int kHarmonics>
void CheckExactRecovery(double coef_tolerance) {
  Scenario<Real, kHarmonics> scenario;
  const PixelOutput<Real> out = scenario.Run();
  ASSERT_TRUE(out.fit.ok);
  EXPECT_EQ(out.fit.n_train, scenario.num_train());
  for (int j = 0; j < 2 + 2 * kHarmonics; ++j) {
    EXPECT_NEAR(out.fit.coef[j], scenario.coef()[static_cast<size_t>(j)],
                coef_tolerance);
  }
  // Unused coefficient slots stay zero.
  for (int j = 2 + 2 * kHarmonics; j < kMaxParams; ++j) {
    EXPECT_EQ(out.fit.coef[j], Real(0));
  }
  // The residual of an exact fit is far below the floor, so rmse is the floor.
  EXPECT_EQ(out.fit.rmse, static_cast<Real>(scenario.params().rmse_floor));
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kStable));
  EXPECT_EQ(out.detect.break_index, -1);
  EXPECT_EQ(out.detect.magnitude, Real(0));
}

TEST(PixelModel, ExactDataRecoveryDouble) {
  CheckExactRecovery<double, 1>(1e-9);
  CheckExactRecovery<double, 2>(1e-9);
  CheckExactRecovery<double, 3>(1e-9);
}

TEST(PixelModel, ExactDataRecoveryFloat) {
  CheckExactRecovery<float, 1>(1e-4);
  CheckExactRecovery<float, 2>(1e-4);
  CheckExactRecovery<float, 3>(1e-4);
}

TEST(PixelModel, RidgeShrinksButKeepsFit) {
  Scenario<double, 2> scenario;
  const PixelOutput<double> plain = scenario.Run();
  scenario.params().ridge = 0.5f;
  const PixelOutput<double> ridged = scenario.Run();
  ASSERT_TRUE(plain.fit.ok);
  ASSERT_TRUE(ridged.fit.ok);
  // The intercept is not penalised, so it stays close; the trend column
  // (index 1) is shrunk towards zero.
  EXPECT_LT(std::fabs(ridged.fit.coef[1]), std::fabs(plain.fit.coef[1]));
  EXPECT_NEAR(ridged.fit.coef[0], plain.fit.coef[0], 0.05);
}

template <typename Real>
void CheckCleanStep() {
  Scenario<Real, 2> scenario;
  const int first = scenario.num_train() + 4;
  scenario.AddStep(first, scenario.num_dates() - 1, -0.3);
  const PixelOutput<Real> out = scenario.Run();
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kBreak));
  EXPECT_EQ(out.detect.break_index, first);
  EXPECT_NEAR(out.detect.magnitude, -0.3, 1e-4);
}

TEST(PixelModel, CleanStepIsDetected) {
  CheckCleanStep<double>();
  CheckCleanStep<float>();
}

TEST(PixelModel, PositiveStepHasPositiveMagnitude) {
  Scenario<double, 2> scenario;
  const int first = scenario.num_train() + 2;
  scenario.AddStep(first, scenario.num_dates() - 1, 0.25);
  const PixelOutput<double> out = scenario.Run();
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kBreak));
  EXPECT_EQ(out.detect.break_index, first);
  EXPECT_NEAR(out.detect.magnitude, 0.25, 1e-9);
}

TEST(PixelModel, AlternatingSignExceedancesDoNotBreak) {
  Scenario<double, 2> scenario;
  // +-0.2 is 10 sigma at the 0.02 floor, but the sign flips every date, so
  // no run ever reaches 3 observations.
  for (int t = scenario.num_train(); t < scenario.num_dates(); ++t) {
    const double delta = (t % 2 == 0) ? 0.2 : -0.2;
    scenario.AddStep(t, t, delta);
  }
  const PixelOutput<double> out = scenario.Run();
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kStable));
  EXPECT_EQ(out.detect.break_index, -1);
  EXPECT_EQ(out.detect.magnitude, 0.0);
}

TEST(PixelModel, SignChangeStartsNewRun) {
  Scenario<double, 2> scenario;
  const int a = scenario.num_train() + 1;
  // Two positive exceedances, then three negative ones: the break is the first
  // negative date, not the first exceedance.
  scenario.AddStep(a, a + 1, 0.2);
  scenario.AddStep(a + 2, a + 4, -0.2);
  const PixelOutput<double> out = scenario.Run();
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kBreak));
  EXPECT_EQ(out.detect.break_index, a + 2);
  EXPECT_NEAR(out.detect.magnitude, -0.2, 1e-9);
}

TEST(PixelModel, NonExceedanceResetsRun) {
  Scenario<double, 2> scenario;
  const int a = scenario.num_train() + 1;
  scenario.AddStep(a, a + 1, -0.2);      // two exceedances ...
  scenario.AddStep(a + 3, a + 5, -0.2);  // ... a clean date, then a run of 3
  const PixelOutput<double> out = scenario.Run();
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kBreak));
  EXPECT_EQ(out.detect.break_index, a + 3);
}

TEST(PixelModel, InvalidObservationsDoNotResetRun) {
  Scenario<double, 2> scenario;
  const int a = scenario.num_train() + 2;
  scenario.AddStep(a, scenario.num_dates() - 1, -0.3);
  // Invalid observations inside the run are skipped: the run is a, a+2, a+4.
  scenario.Invalidate(a + 1);
  scenario.Invalidate(a + 3);
  const PixelOutput<double> out = scenario.Run();
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kBreak));
  EXPECT_EQ(out.detect.break_index, a);
  EXPECT_NEAR(out.detect.magnitude, -0.3, 1e-9);
}

TEST(PixelModel, InvalidFirstObservationOfRunMovesBreak) {
  Scenario<double, 2> scenario;
  const int a = scenario.num_train() + 2;
  scenario.AddStep(a, scenario.num_dates() - 1, -0.3);
  scenario.Invalidate(a);
  const PixelOutput<double> out = scenario.Run();
  EXPECT_EQ(out.detect.break_index, a + 1);
}

TEST(PixelModel, LossesOnlyIgnoresPositiveStep) {
  Scenario<double, 2> scenario(80, 15);
  const int up = scenario.num_train() + 1;
  const int down = up + 8;
  // A short gain that ends, then a later loss that lasts.
  scenario.AddStep(up, up + 3, 0.3);
  scenario.AddStep(down, scenario.num_dates() - 1, -0.3);
  scenario.params().losses_only = false;
  const PixelOutput<double> both = scenario.Run();
  EXPECT_EQ(both.detect.break_index, up);
  EXPECT_GT(both.detect.magnitude, 0.0);
  scenario.params().losses_only = true;
  const PixelOutput<double> losses = scenario.Run();
  EXPECT_EQ(losses.detect.status, Code(PixelStatus::kBreak));
  EXPECT_EQ(losses.detect.break_index, down);
  EXPECT_LT(losses.detect.magnitude, 0.0);
}

TEST(PixelModel, TooFewTrainingObservations) {
  Scenario<double, 2> scenario;
  // Keep only 10 valid training observations; the default minimum is 18.
  for (int t = 10; t < scenario.num_train(); ++t) {
    scenario.Invalidate(t);
  }
  const PixelOutput<double> out = scenario.Run();
  EXPECT_FALSE(out.fit.ok);
  EXPECT_EQ(out.fit.n_train, 10);
  EXPECT_EQ(out.fit.rmse, 0.0);
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kInsufficientData));
  EXPECT_EQ(out.detect.break_index, -1);
  EXPECT_EQ(out.detect.magnitude, 0.0);
}

TEST(PixelModel, MinimumTrainingObservationsIsInclusive) {
  Scenario<double, 2> scenario;
  const int needed = scenario.params().min_train_obs;
  for (int t = needed; t < scenario.num_train(); ++t) {
    scenario.Invalidate(t);
  }
  EXPECT_TRUE(scenario.Run().fit.ok);
  scenario.Invalidate(needed - 1);
  EXPECT_FALSE(scenario.Run().fit.ok);
}

// Dates for which the first `clustered` dates fall on `cluster_days` distinct
// days from the start, followed by regular monitoring dates.
DateTable ClusteredDates(int clustered, int cluster_days) {
  std::vector<int> days;
  for (int i = 0; i < clustered; ++i) {
    days.push_back(cputest::kFirstDay + (i * cluster_days) / clustered);
  }
  for (int i = 0; i < 20; ++i) {
    days.push_back(cputest::kFirstDay + 800 + 15 * i);
  }
  return cputest::MakeDateTableFromDays(days);
}

TEST(PixelModel, AllTrainingObservationsOnOneDateIsSingular) {
  Scenario<double, 2> scenario(ClusteredDates(30, 1));
  ASSERT_GE(scenario.num_train(), 30);
  const PixelOutput<double> out = scenario.Run();
  EXPECT_GE(out.fit.n_train, 30);
  EXPECT_FALSE(out.fit.ok);
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kInsufficientData));
}

TEST(PixelModel, ClusteredTrainingDatesGiveSameStatusInBothPrecisions) {
  // Thirty observations inside six days: the design is nearly singular and
  // the relative pivot test must reject it in FP32 and FP64 alike.
  Scenario<double, 2> wide(ClusteredDates(30, 6));
  Scenario<float, 2> narrow(ClusteredDates(30, 6));
  const PixelOutput<double> d = wide.Run();
  const PixelOutput<float> f = narrow.Run();
  EXPECT_EQ(d.detect.status, f.detect.status);
  EXPECT_EQ(d.detect.status, Code(PixelStatus::kInsufficientData));
  EXPECT_EQ(d.fit.ok, f.fit.ok);
}

TEST(PixelModel, BreakAtLastPossibleIndex) {
  Scenario<double, 2> scenario;
  const int last = scenario.num_dates() - 1;
  const int m = scenario.params().consecutive;  // 3
  // A run of exactly M exceedances ending on the last date starts at last-M+1.
  scenario.AddStep(last - m + 1, last, -0.3);
  PixelOutput<double> out = scenario.Run();
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kBreak));
  EXPECT_EQ(out.detect.break_index, last - m + 1);
  // One observation fewer than M: stable.
  Scenario<double, 2> short_run;
  short_run.AddStep(last - m + 2, last, -0.3);
  out = short_run.Run();
  EXPECT_EQ(out.detect.status, Code(PixelStatus::kStable));
  // M exceedances needing an invalid date to be skipped: only M-1 valid.
  Scenario<double, 2> gap;
  gap.AddStep(last - m + 1, last, -0.3);
  gap.Invalidate(last - 1);
  EXPECT_EQ(gap.Run().detect.status, Code(PixelStatus::kStable));
}

TEST(PixelModel, ConsecutiveParameterIsHonoured) {
  Scenario<double, 2> scenario;
  const int a = scenario.num_train() + 2;
  scenario.AddStep(a, a, -0.3);
  scenario.params().consecutive = 1;
  EXPECT_EQ(scenario.Run().detect.break_index, a);
  scenario.params().consecutive = 2;
  EXPECT_EQ(scenario.Run().detect.status, Code(PixelStatus::kStable));
}

TEST(PixelModel, ThresholdBoundaryIsExclusive) {
  Scenario<double, 2> scenario;
  const int a = scenario.num_train() + 2;
  // rmse is the 0.02 floor; threshold 3 means |r| > 0.06 is an exceedance.
  scenario.AddStep(a, scenario.num_dates() - 1, -0.059);
  EXPECT_EQ(scenario.Run().detect.status, Code(PixelStatus::kStable));
  scenario.AddStep(a, scenario.num_dates() - 1, -0.002);  // total -0.061
  EXPECT_EQ(scenario.Run().detect.status, Code(PixelStatus::kBreak));
}

// Random series with noise, invalid observations and occasional steps.
template <typename Real>
void CheckFitAndDetectEqualsParts() {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, 0.03);
  int breaks = 0;
  int insufficient = 0;
  for (int trial = 0; trial < 1000; ++trial) {
    Scenario<Real, 2> scenario;
    const double invalid_rate = trial % 7 == 0 ? 0.7 : 0.15;
    const bool step = uniform(rng) < 0.4;
    const int step_at =
        scenario.num_train() + static_cast<int>(uniform(rng) * 10.0);
    for (int t = 0; t < scenario.num_dates(); ++t) {
      scenario.y()[static_cast<size_t>(t)] +=
          static_cast<Real>(noise(rng) + (step && t >= step_at ? -0.25 : 0.0));
      if (uniform(rng) < invalid_rate) {
        scenario.Invalidate(t);
      }
    }
    const PixelOutput<Real> combined = scenario.Run();
    const NdviArraySeries<Real, Real> series{scenario.y().data(), 1};
    const FitOutput<Real> fit =
        FitModel<Real, 2>(series, scenario.design().data(),
                          scenario.num_train(), scenario.params());
    const DetectOutput<Real> detect = DetectBreak<Real, 2>(
        fit, series, scenario.design().data(), scenario.num_dates(),
        scenario.num_train(), scenario.params());
    EXPECT_EQ(combined.fit.ok, fit.ok);
    EXPECT_EQ(combined.fit.n_train, fit.n_train);
    EXPECT_EQ(combined.fit.rmse, fit.rmse);
    for (int j = 0; j < kMaxParams; ++j) {
      EXPECT_EQ(combined.fit.coef[j], fit.coef[j]);
    }
    EXPECT_EQ(combined.detect.break_index, detect.break_index);
    EXPECT_EQ(combined.detect.magnitude, detect.magnitude);
    EXPECT_EQ(combined.detect.status, detect.status);
    breaks += detect.status == Code(PixelStatus::kBreak) ? 1 : 0;
    insufficient +=
        detect.status == Code(PixelStatus::kInsufficientData) ? 1 : 0;
  }
  // The random series must exercise all three outcomes.
  EXPECT_GT(breaks, 50);
  EXPECT_GT(insufficient, 50);
}

TEST(PixelModel, FitAndDetectEqualsFitThenDetect) {
  CheckFitAndDetectEqualsParts<double>();
  CheckFitAndDetectEqualsParts<float>();
}

TEST(PixelModel, QuickDiffFlags) {
  float diff = 99.0f;
  uint8_t flag = 99;
  QuickDiffPixel(0.6f, 0.2f, 0.15f, false, &diff, &flag);
  EXPECT_NEAR(diff, -0.4, 1e-6);
  EXPECT_EQ(flag, 2);
  QuickDiffPixel(0.2f, 0.6f, 0.15f, false, &diff, &flag);
  EXPECT_NEAR(diff, 0.4, 1e-6);
  EXPECT_EQ(flag, 3);
  QuickDiffPixel(0.2f, 0.6f, 0.15f, true, &diff, &flag);
  EXPECT_EQ(flag, 1);  // gain ignored with losses_only
  EXPECT_NEAR(diff, 0.4, 1e-6);
  QuickDiffPixel(0.5f, 0.4f, 0.15f, false, &diff, &flag);
  EXPECT_EQ(flag, 1);
  QuickDiffPixel(kInvalidNdvi, 0.4f, 0.15f, false, &diff, &flag);
  EXPECT_EQ(flag, 0);
  EXPECT_EQ(diff, 0.0f);
  QuickDiffPixel(0.4f, kInvalidNdvi, 0.15f, false, &diff, &flag);
  EXPECT_EQ(flag, 0);
}

}  // namespace
}  // namespace cd
