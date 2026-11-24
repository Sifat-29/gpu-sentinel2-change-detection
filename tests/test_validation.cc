// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "test_framework.h"
#include "types.h"
#include "validation.h"

namespace cd {
namespace {

// Dates 0..9 in the years 2019, 2019, 2020, 2020, ..., 2023, 2023; the first
// four are training dates, so the monitoring window is 2021..2023.
DateTable MakeYearTable() {
  DateTable dates;
  for (int t = 0; t < 10; ++t) {
    dates.year.push_back(static_cast<int16_t>(2019 + t / 2));
  }
  dates.num_train_dates = 4;
  return dates;
}

// One pixel of the hand-made 4 x 4 case: reference value and detection.
struct Pixel {
  uint8_t reference;
  PixelStatus status;
  int break_index;  // date index, ignored unless status is kBreak
  float magnitude;
};

// The 16 pixels, with the expected class in the comment of each line.
std::vector<Pixel> HandMadePixels() {
  const PixelStatus s = PixelStatus::kStable;
  const PixelStatus b = PixelStatus::kBreak;
  const PixelStatus n = PixelStatus::kInsufficientData;
  return {
      {0, s, -1, 0.0f},    // 0  TN
      {255, s, -1, 0.0f},  // 1  excluded
      {22, b, 6, -0.3f},   // 2  TP, year exact (2022)
      {22, b, 4, -0.3f},   // 3  TP, year off by one (2021 vs 2022)
      {22, b, 2, -0.3f},   // 4  FN (detected 2020 is before the window)
      {22, s, -1, 0.0f},   // 5  FN
      {0, b, 6, -0.3f},    // 6  FP
      {0, b, 6, 0.3f},     // 7  TN (a gain is not a detection)
      {19, s, -1, 0.0f},   // 8  TN (reference year 2019 is outside)
      {19, b, 6, -0.3f},   // 9  FP
      {23, b, 9, -0.1f},   // 10 TP, year exact (2023)
      {23, b, 4, -0.1f},   // 11 TP, year off by two
      {255, b, 6, -0.3f},  // 12 excluded
      {0, n, -1, 0.0f},    // 13 TN
      {22, n, -1, 0.0f},   // 14 FN
      {24, s, -1, 0.0f},   // 15 TN (reference year 2024 is outside)
  };
}

void FillCase(DetectionResult* result, std::vector<uint8_t>* reference) {
  AllocateResult(4, 4, 6, false, result);
  reference->assign(16, 0);
  const std::vector<Pixel> pixels = HandMadePixels();
  for (size_t i = 0; i < pixels.size(); ++i) {
    (*reference)[i] = pixels[i].reference;
    result->status[i] = static_cast<uint8_t>(pixels[i].status);
    result->break_index[i] = static_cast<int16_t>(pixels[i].break_index);
    result->magnitude[i] = pixels[i].magnitude;
  }
}

TEST(Validation, WindowFromDateTable) {
  const ValidationWindow window = MakeValidationWindow(MakeYearTable());
  EXPECT_EQ(window.first_year, 2021);
  EXPECT_EQ(window.last_year, 2023);
  const ValidationWindow empty = MakeValidationWindow(DateTable());
  EXPECT_EQ(empty.first_year, 0);
  EXPECT_EQ(empty.last_year, 0);
}

TEST(Validation, HandMadeFourByFourCounts) {
  DetectionResult result;
  std::vector<uint8_t> reference;
  FillCase(&result, &reference);
  const DateTable dates = MakeYearTable();
  ConfusionCounts counts;
  double ms = -1;
  ASSERT_TRUE(ComputeConfusionCpu(result, reference, dates,
                                  MakeValidationWindow(dates), &counts, &ms)
                  .ok());
  EXPECT_EQ(counts.tp, 4u);
  EXPECT_EQ(counts.fp, 2u);
  EXPECT_EQ(counts.fn, 3u);
  EXPECT_EQ(counts.tn, 5u);
  EXPECT_EQ(counts.excluded, 2u);
  EXPECT_EQ(counts.year_exact, 2u);
  EXPECT_EQ(counts.year_within1, 3u);
  EXPECT_GE(ms, 0.0);
}

TEST(Validation, MetricsOfHandMadeCase) {
  DetectionResult result;
  std::vector<uint8_t> reference;
  FillCase(&result, &reference);
  const DateTable dates = MakeYearTable();
  ConfusionCounts counts;
  ASSERT_TRUE(ComputeConfusionCpu(result, reference, dates,
                                  MakeValidationWindow(dates), &counts, nullptr)
                  .ok());
  const ValidationMetrics m = MetricsFromCounts(counts);
  EXPECT_NEAR(m.precision, 4.0 / 6.0, 1e-12);
  EXPECT_NEAR(m.recall, 4.0 / 7.0, 1e-12);
  EXPECT_NEAR(m.f1, 2.0 * m.precision * m.recall / (m.precision + m.recall),
              1e-12);
  EXPECT_NEAR(m.year_exact_frac, 0.5, 1e-12);
  EXPECT_NEAR(m.year_within1_frac, 0.75, 1e-12);
  EXPECT_FALSE(m.undefined);
}

TEST(Validation, MetricsEdgeCasesNeverNaN) {
  const ConfusionCounts none;  // all zero: 0/0 everywhere
  const ValidationMetrics m = MetricsFromCounts(none);
  EXPECT_EQ(m.precision, 0.0);
  EXPECT_EQ(m.recall, 0.0);
  EXPECT_EQ(m.f1, 0.0);
  EXPECT_EQ(m.year_exact_frac, 0.0);
  EXPECT_TRUE(m.undefined);

  ConfusionCounts only_fp;  // no positives in the reference, nothing right
  only_fp.fp = 5;
  const ValidationMetrics f = MetricsFromCounts(only_fp);
  EXPECT_EQ(f.precision, 0.0);
  EXPECT_EQ(f.recall, 0.0);
  EXPECT_EQ(f.f1, 0.0);
  EXPECT_FALSE(std::isnan(f.f1));
  EXPECT_TRUE(f.undefined);

  ConfusionCounts perfect;
  perfect.tp = 7;
  perfect.year_exact = 7;
  perfect.year_within1 = 7;
  const ValidationMetrics p = MetricsFromCounts(perfect);
  EXPECT_EQ(p.precision, 1.0);
  EXPECT_EQ(p.recall, 1.0);
  EXPECT_EQ(p.f1, 1.0);
  EXPECT_EQ(p.year_exact_frac, 1.0);
  EXPECT_FALSE(p.undefined);
}

TEST(Validation, SizeMismatchIsAnError) {
  DetectionResult result;
  std::vector<uint8_t> reference;
  FillCase(&result, &reference);
  reference.pop_back();
  const DateTable dates = MakeYearTable();
  ConfusionCounts counts;
  EXPECT_FALSE(ComputeConfusionCpu(result, reference, dates,
                                   MakeValidationWindow(dates), &counts,
                                   nullptr)
                   .ok());
}

TEST(Validation, BreakIndexOutOfRangeIsAnError) {
  DetectionResult result;
  std::vector<uint8_t> reference;
  FillCase(&result, &reference);
  result.break_index[2] = 99;
  const DateTable dates = MakeYearTable();
  ConfusionCounts counts;
  EXPECT_FALSE(ComputeConfusionCpu(result, reference, dates,
                                   MakeValidationWindow(dates), &counts,
                                   nullptr)
                   .ok());
}

TEST(Validation, LineAndCsvText) {
  ConfusionCounts counts;
  counts.tp = 4;
  counts.fp = 2;
  counts.fn = 3;
  counts.tn = 5;
  counts.excluded = 2;
  counts.year_exact = 2;
  counts.year_within1 = 3;
  const ValidationMetrics m = MetricsFromCounts(counts);
  const std::string line = FormatValidationLine(counts, m);
  EXPECT_EQ(line,
            "precision 0.667 | recall 0.571 | f1 0.615 | year_exact 0.500 | "
            "year_within1 0.750 (tp 4 fp 2 fn 3)");
  ValidationWindow window;
  window.first_year = 2021;
  window.last_year = 2023;
  const std::string csv = FormatValidationCsv(counts, m, window);
  EXPECT_EQ(csv.substr(0, 17), "metric,value\ntp,4");
  EXPECT_TRUE(csv.find("\nexcluded,2\n") != std::string::npos);
  EXPECT_TRUE(csv.find("\nyear_exact,0.500000\n") != std::string::npos);
  EXPECT_TRUE(csv.find("\nfirst_year,2021\nlast_year,2023\n") !=
              std::string::npos);
  // An undefined ratio is flagged in the log text, never printed as NaN.
  const ValidationMetrics undefined = MetricsFromCounts(ConfusionCounts());
  const std::string text = FormatValidationLine(ConfusionCounts(), undefined);
  EXPECT_TRUE(text.find("undefined") != std::string::npos);
  EXPECT_TRUE(text.find("nan") == std::string::npos);
}

}  // namespace
}  // namespace cd
