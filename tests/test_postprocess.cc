// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <cstdint>
#include <string>
#include <vector>

#include "colormap.h"
#include "cpu_test_cube.h"
#include "postprocess.h"
#include "test_framework.h"
#include "types.h"

namespace cd {
namespace {

constexpr uint8_t Code(PixelStatus s) { return static_cast<uint8_t>(s); }

constexpr uint8_t kStable = Code(PixelStatus::kStable);
constexpr uint8_t kBreak = Code(PixelStatus::kBreak);
constexpr uint8_t kInsufficient = Code(PixelStatus::kInsufficientData);

// Date table with 10 dates, 100 days apart; the first 4 are training dates.
DateTable TenDates() {
  DateTable dates;
  for (int i = 0; i < 10; ++i) {
    dates.day.push_back(18000 + 100 * i);
  }
  dates.num_train_dates = 4;
  return dates;
}

// Map of the given status characters ('.' stable, 'B' break, 'I'
// insufficient), row-major, with break_index 5 and magnitude -0.2 on breaks.
DetectionResult MapFromText(const std::vector<std::string>& rows) {
  DetectionResult r;
  const int height = static_cast<int>(rows.size());
  const int width = static_cast<int>(rows[0].size());
  AllocateResult(height, width, 6, false, &r);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t i = static_cast<size_t>(y) * width + x;
      const char c = rows[static_cast<size_t>(y)][static_cast<size_t>(x)];
      if (c == 'B') {
        r.status[i] = kBreak;
        r.break_index[i] = 5;
        r.magnitude[i] = -0.2f;
      } else if (c == 'I') {
        r.status[i] = kInsufficient;
      }
    }
  }
  return r;
}

// Text form of the status map, the inverse of MapFromText for statuses.
std::vector<std::string> StatusText(const DetectionResult& r) {
  std::vector<std::string> rows;
  for (int y = 0; y < r.height; ++y) {
    std::string row;
    for (int x = 0; x < r.width; ++x) {
      const uint8_t s = r.status[static_cast<size_t>(y) * r.width + x];
      row += s == kBreak ? 'B' : (s == kInsufficient ? 'I' : '.');
    }
    rows.push_back(row);
  }
  return rows;
}

TEST(Postprocess, ColorizeKnownPixels) {
  const DateTable dates = TenDates();  // monitoring days 18400 .. 18900
  DetectionResult r;
  AllocateResult(1, 6, 6, false, &r);
  // 0: loss at the first monitoring date, 1: gain at the last date,
  // 2: loss in the middle, 3: stable, 4: insufficient, 5: break_index out of
  // the date range is treated as the first monitoring day.
  r.status = {kBreak, kBreak, kBreak, kStable, kInsufficient, kStable};
  r.break_index = {4, 9, 7, -1, -1, -1};
  r.magnitude = {-0.3f, 0.2f, -0.1f, 0.0f, 0.0f, 0.0f};
  std::vector<uint8_t> rgba;
  double ms = -1.0;
  ASSERT_TRUE(CpuColorizeChangeMap(r, dates, &rgba, &ms).ok());
  EXPECT_GE(ms, 0.0);
  ASSERT_EQ(rgba.size(), 24u);
  // Stop 0 (68, 1, 84), opaque because it is a loss.
  EXPECT_EQ(rgba[0], 68);
  EXPECT_EQ(rgba[1], 1);
  EXPECT_EQ(rgba[2], 84);
  EXPECT_EQ(rgba[3], 255);
  // Stop 7 (253, 231, 37), translucent because it is a gain.
  EXPECT_EQ(rgba[4], 253);
  EXPECT_EQ(rgba[5], 231);
  EXPECT_EQ(rgba[6], 37);
  EXPECT_EQ(rgba[7], 160);
  // Day 18700: offset 300 of span 500 -> stop 8 * 300 / 500 = 4 (31, 161, 135).
  EXPECT_EQ(rgba[8], 31);
  EXPECT_EQ(rgba[9], 161);
  EXPECT_EQ(rgba[10], 135);
  EXPECT_EQ(rgba[11], 255);
  // Stable is dark grey, insufficient data is transparent.
  EXPECT_EQ(rgba[12], 40);
  EXPECT_EQ(rgba[15], 255);
  EXPECT_EQ(rgba[16], 0);
  EXPECT_EQ(rgba[19], 0);
  EXPECT_EQ(rgba[20], 40);
}

TEST(Postprocess, ColorizeMatchesChangeColorForEveryBreakDate) {
  const DateTable dates = TenDates();
  DetectionResult r;
  AllocateResult(1, 6, 6, false, &r);
  for (int i = 0; i < 6; ++i) {
    r.status[static_cast<size_t>(i)] = kBreak;
    r.break_index[static_cast<size_t>(i)] = static_cast<int16_t>(4 + i);
    r.magnitude[static_cast<size_t>(i)] = -0.1f;
  }
  std::vector<uint8_t> rgba;
  ASSERT_TRUE(CpuColorizeChangeMap(r, dates, &rgba, nullptr).ok());
  for (int i = 0; i < 6; ++i) {
    uint8_t expected[4];
    ChangeColor(static_cast<int16_t>(4 + i), -0.1f, kBreak, dates.day[4],
                dates.day[9], dates.day[static_cast<size_t>(4 + i)], expected);
    for (int c = 0; c < 4; ++c) {
      EXPECT_EQ(rgba[static_cast<size_t>(i) * 4 + c], expected[c]);
    }
  }
}

TEST(Postprocess, ColorizeRejectsBadInput) {
  DateTable dates = TenDates();
  DetectionResult r;
  AllocateResult(2, 2, 6, false, &r);
  std::vector<uint8_t> rgba;
  EXPECT_FALSE(CpuColorizeChangeMap(r, dates, nullptr, nullptr).ok());
  dates.num_train_dates = 10;  // no monitoring dates
  EXPECT_FALSE(CpuColorizeChangeMap(r, dates, &rgba, nullptr).ok());
  dates = TenDates();
  r.status.pop_back();
  EXPECT_FALSE(CpuColorizeChangeMap(r, dates, &rgba, nullptr).ok());
}

TEST(Postprocess, CleanupRemovesIsolatedPixelsAndKeepsClusters) {
  DetectionResult r = MapFromText({
      "B..B.B.",  // (0,0) isolated corner, (0,3) and (0,5) isolated
      "...B...",  //   (0,3)-(1,3): pair -> each has 1 neighbour
      "..BBB..",  // row of three plus (1,3) above the middle
      "..I....",
      "BB...BB",  // left pair at the border, right pair at the border
      "BB...B.",  // 2x2 block at the left border
  });
  double ms = -1.0;
  ASSERT_TRUE(CpuCleanup(&r, &ms).ok());
  EXPECT_GE(ms, 0.0);
  // (0,0) has neighbours (0,1) '.', (1,0) '.', (1,1) '.': removed.
  // (0,3): neighbours (1,3) only (and (1,2), (1,4) are '.'): 1 -> removed.
  // (0,5): none -> removed. (1,3): neighbours (0,3), (2,2), (2,3), (2,4) = 4
  // -> kept. (2,2): neighbours (1,3), (2,3) = 2 -> kept. (2,3): (1,3), (2,2),
  // (2,4) = 3 -> kept. (2,4): (1,3), (2,3) = 2 -> kept. (3,2) is insufficient
  // data and stays so. Row 4/5: the 2x2 block at the left border: each pixel
  // has 3 neighbours -> kept. Right side: (4,5) has (4,6), (5,5) = 2 -> kept;
  // (4,6) has (4,5), (5,5) = 2 -> kept; (5,5) has (4,5), (4,6) = 2 -> kept.
  const std::vector<std::string> expected = {
      ".......", "...B...", "..BBB..", "..I....", "BB...BB", "BB...B.",
  };
  EXPECT_TRUE(StatusText(r) == expected);
}

TEST(Postprocess, CleanupEvaluatesOnTheUnmodifiedInput) {
  // A row of three: the two ends each have 1 neighbour and are removed, but
  // the middle pixel had 2 neighbours in the input and must stay even though
  // its neighbours disappear. A sequential in-place rule would remove it too.
  DetectionResult r = MapFromText({"......", ".BBB..", "......"});
  ASSERT_TRUE(CpuCleanup(&r, nullptr).ok());
  const std::vector<std::string> expected = {"......", "..B...", "......"};
  EXPECT_TRUE(StatusText(r) == expected);
  // Removed pixels become stable with break_index -1 and magnitude 0.
  EXPECT_EQ(r.break_index[1 * 6 + 1], -1);
  EXPECT_EQ(r.magnitude[1 * 6 + 1], 0.0f);
  EXPECT_EQ(r.status[1 * 6 + 1], kStable);
  // The surviving pixel is untouched.
  EXPECT_EQ(r.break_index[1 * 6 + 2], 5);
  EXPECT_EQ(r.magnitude[1 * 6 + 2], -0.2f);
}

TEST(Postprocess, CleanupHandlesCornersAndEdges) {
  // Out-of-image neighbours count as not-break: a corner pixel needs two of
  // its three in-image neighbours.
  DetectionResult corner = MapFromText({"BB", "B."});
  ASSERT_TRUE(CpuCleanup(&corner, nullptr).ok());
  const std::vector<std::string> keep = {"BB", "B."};
  // (0,0) has two neighbours and stays; (0,1) and (1,0) have two as well.
  EXPECT_TRUE(StatusText(corner) == keep);

  DetectionResult single_row = MapFromText({"BBBB"});
  ASSERT_TRUE(CpuCleanup(&single_row, nullptr).ok());
  // Ends have one neighbour, the middle pixels have two.
  const std::vector<std::string> middle = {".BB."};
  EXPECT_TRUE(StatusText(single_row) == middle);

  DetectionResult one_pixel = MapFromText({"B"});
  ASSERT_TRUE(CpuCleanup(&one_pixel, nullptr).ok());
  EXPECT_TRUE(StatusText(one_pixel) == std::vector<std::string>({"."}));
}

TEST(Postprocess, CleanupDoesNotCountInsufficientDataAsBreak) {
  DetectionResult r = MapFromText({"BII", "III"});
  ASSERT_TRUE(CpuCleanup(&r, nullptr).ok());
  EXPECT_TRUE(StatusText(r) == std::vector<std::string>({".II", "III"}));
}

TEST(Postprocess, CleanupRejectsBadInput) {
  EXPECT_FALSE(CpuCleanup(nullptr, nullptr).ok());
  DetectionResult r = MapFromText({"B."});
  r.magnitude.clear();
  EXPECT_FALSE(CpuCleanup(&r, nullptr).ok());
}

// A 3 x 4 cube with 4 dates and hand-set NDVI values.
class QuickDiffCube {
 public:
  QuickDiffCube() : cube_(3, 4, cputest::MakeRegularDates(4, 30)) {
    // Dates 1 and 3 are compared. Pixel index = y * 4 + x.
    SetPair(0, 0, 0.7, 0.3);              // loss (-0.4)
    SetPair(0, 1, 0.3, 0.7);              // gain (+0.4)
    SetPair(0, 2, 0.5, 0.45);             // small change
    SetPair(0, 3, 0.5, 0.5);              // no change
    SetPair(1, 0, 0.5, 0.5, 9);           // cloud in date 1
    SetPair(1, 1, 0.6, 0.2, 4, 9);        // cloud in date 3
    SetPair(1, 2, 0.6, 0.2, 4, 4, true);  // no data (red = 0) in date 1
    SetPair(1, 3, 0.2, 0.5);              // gain
    SetPair(2, 0, 0.9, 0.2);              // loss
    SetPair(2, 1, 0.1, 0.0);              // small loss
    SetPair(2, 2, 0.5, 0.351);            // just inside the threshold
    SetPair(2, 3, 0.4, 0.45);             // small gain
  }

  cputest::MemoryCubeSource* cube() { return &cube_; }

 private:
  void SetPair(int y, int x, double a, double b, uint8_t scl_a = 4,
               uint8_t scl_b = 4, bool nodata_a = false) {
    cube_.SetNdvi(1, y, x, a, scl_a);
    cube_.SetNdvi(3, y, x, b, scl_b);
    if (nodata_a) {
      cube_.Set(1, y, x, 0, 0, 4);
    }
    // Other dates carry a distinct value so reading the wrong plane fails.
    cube_.SetNdvi(0, y, x, 0.0 + 0.01 * x);
    cube_.SetNdvi(2, y, x, 0.8);
  }

  cputest::MemoryCubeSource cube_;
};

TEST(Postprocess, QuickDiffFlags) {
  QuickDiffCube fixture;
  ModelParams params;
  RunConfig config;
  QuickDiffResult out;
  StageTimings timings;
  ASSERT_TRUE(
      CpuQuickDiff(fixture.cube(), 1, 3, 0.15f, params, config, &out, &timings)
          .ok());
  EXPECT_EQ(out.height, 3);
  EXPECT_EQ(out.width, 4);
  ASSERT_EQ(out.flag.size(), 12u);
  const std::vector<uint8_t> expected_flags = {2, 3, 1, 1,   // row 0
                                               0, 0, 0, 3,   // row 1
                                               2, 1, 1, 1};  // row 2
  EXPECT_TRUE(out.flag == expected_flags);
  EXPECT_NEAR(out.diff[0], -0.4, 1e-3);
  EXPECT_NEAR(out.diff[1], 0.4, 1e-3);
  EXPECT_NEAR(out.diff[2], -0.05, 1e-3);
  EXPECT_NEAR(out.diff[7], 0.3, 1e-3);
  EXPECT_NEAR(out.diff[8], -0.7, 1e-3);
  // Invalid pixels report diff 0.
  EXPECT_EQ(out.diff[4], 0.0f);
  EXPECT_EQ(out.diff[5], 0.0f);
  EXPECT_EQ(out.diff[6], 0.0f);
  EXPECT_GE(timings.read_ms, 0.0);
  EXPECT_GE(timings.total_ms, 0.0);
}

TEST(Postprocess, QuickDiffLossesOnlyAndDateOrder) {
  QuickDiffCube fixture;
  ModelParams params;
  params.losses_only = true;
  RunConfig config;
  QuickDiffResult out;
  ASSERT_TRUE(
      CpuQuickDiff(fixture.cube(), 1, 3, 0.15f, params, config, &out, nullptr)
          .ok());
  // Gains (pixels 1 and 7) become "no change" with losses_only.
  const std::vector<uint8_t> expected_flags = {2, 1, 1, 1, 0, 0,
                                               0, 1, 2, 1, 1, 1};
  EXPECT_TRUE(out.flag == expected_flags);
  // Swapping the dates negates the difference and swaps loss and gain.
  params.losses_only = false;
  QuickDiffResult swapped;
  ASSERT_TRUE(CpuQuickDiff(fixture.cube(), 3, 1, 0.15f, params, config,
                           &swapped, nullptr)
                  .ok());
  EXPECT_EQ(swapped.flag[0], 3);
  EXPECT_EQ(swapped.flag[1], 2);
  EXPECT_NEAR(swapped.diff[0], 0.4, 1e-3);
}

TEST(Postprocess, QuickDiffIndependentOfTileRows) {
  QuickDiffCube fixture;
  ModelParams params;
  RunConfig config;
  QuickDiffResult reference;
  ASSERT_TRUE(CpuQuickDiff(fixture.cube(), 1, 3, 0.15f, params, config,
                           &reference, nullptr)
                  .ok());
  for (const int rows : {1, 2, 3, 100}) {
    config.tile_rows = rows;
    QuickDiffResult out;
    ASSERT_TRUE(
        CpuQuickDiff(fixture.cube(), 1, 3, 0.15f, params, config, &out, nullptr)
            .ok());
    EXPECT_TRUE(out.flag == reference.flag);
    EXPECT_TRUE(out.diff == reference.diff);
  }
}

TEST(Postprocess, QuickDiffRejectsBadInput) {
  QuickDiffCube fixture;
  ModelParams params;
  RunConfig config;
  QuickDiffResult out;
  EXPECT_FALSE(
      CpuQuickDiff(fixture.cube(), -1, 3, 0.15f, params, config, &out, nullptr)
          .ok());
  EXPECT_FALSE(
      CpuQuickDiff(fixture.cube(), 1, 4, 0.15f, params, config, &out, nullptr)
          .ok());
  EXPECT_FALSE(
      CpuQuickDiff(nullptr, 1, 3, 0.15f, params, config, &out, nullptr).ok());
  fixture.cube()->FailReadsAfter(0);
  EXPECT_FALSE(
      CpuQuickDiff(fixture.cube(), 1, 3, 0.15f, params, config, &out, nullptr)
          .ok());
}

}  // namespace
}  // namespace cd
