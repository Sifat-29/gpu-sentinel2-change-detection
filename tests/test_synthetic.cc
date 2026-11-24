// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): portable paths and mkdir -p
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cube_io.h"
#include "cube_source.h"
#include "status.h"
#include "synthetic_cube.h"
#include "test_framework.h"
#include "types.h"

namespace cd {
namespace {

namespace fs = std::filesystem;

// A unique temporary directory removed on destruction.
class TempDir {
 public:
  TempDir() {
    std::string pattern =
        (fs::temp_directory_path() / "cd_test_synthetic_XXXXXX").string();
    const char* made = mkdtemp(pattern.data());
    path_ = made != nullptr ? made : "";
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir() {
    if (!path_.empty()) {
      std::error_code error;
      fs::remove_all(path_, error);
    }
  }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

SyntheticOptions SmallOptions() {
  SyntheticOptions options;
  options.height = 37;
  options.width = 29;
  options.num_dates = 24;
  options.years = 4.0f;
  options.seed = 7;
  return options;
}

TEST(Synthetic, IsIdenticalForAnyThreadCount) {
  SyntheticOptions options = SmallOptions();
  options.threads = 1;
  SyntheticCube one;
  ASSERT_TRUE(GenerateSyntheticCube(options, &one).ok());
  options.threads = 8;
  SyntheticCube eight;
  ASSERT_TRUE(GenerateSyntheticCube(options, &eight).ok());
  EXPECT_TRUE(one.red == eight.red);
  EXPECT_TRUE(one.nir == eight.nir);
  EXPECT_TRUE(one.scl == eight.scl);
  EXPECT_TRUE(one.truth_break_index == eight.truth_break_index);
  options.seed = 8;
  SyntheticCube other;
  ASSERT_TRUE(GenerateSyntheticCube(options, &other).ok());
  EXPECT_FALSE(one.red == other.red);
}

TEST(Synthetic, DiskPathMatchesMemoryPath) {
  SyntheticOptions options = SmallOptions();
  options.threads = 3;
  SyntheticCube memory;
  ASSERT_TRUE(GenerateSyntheticCube(options, &memory).ok());
  const TempDir dir;
  ASSERT_FALSE(dir.path().empty());
  ASSERT_TRUE(GenerateSyntheticCubeToDisk(options, dir.path()).ok());

  CubeInfo info;
  ASSERT_TRUE(ReadManifest(dir.path() + "/manifest.txt", &info).ok());
  EXPECT_EQ(info.source, "synthetic");
  EXPECT_EQ(info.height, 37);
  EXPECT_EQ(info.width, 29);
  EXPECT_EQ(info.num_dates, 24);
  EXPECT_TRUE(info.dates.day == memory.info.dates.day);
  EXPECT_TRUE(info.dates.iso == memory.info.dates.iso);
  EXPECT_TRUE(info.dates.offset == memory.info.dates.offset);
  EXPECT_EQ(info.dates.day[0], 17897);
  EXPECT_EQ(info.dates.iso[0], "2019-01-01");

  const size_t plane = 37u * 29u;
  std::vector<uint16_t> red;
  std::vector<uint16_t> nir;
  std::vector<uint8_t> scl;
  std::vector<int16_t> truth;
  ASSERT_TRUE(ReadRawArray(info.red_file, plane * 24, &red).ok());
  ASSERT_TRUE(ReadRawArray(info.nir_file, plane * 24, &nir).ok());
  ASSERT_TRUE(ReadRawArray(info.scl_file, plane * 24, &scl).ok());
  ASSERT_TRUE(
      ReadRawArray(dir.path() + "/truth_break_index.i16", plane, &truth).ok());
  EXPECT_TRUE(red == memory.red);
  EXPECT_TRUE(nir == memory.nir);
  EXPECT_TRUE(scl == memory.scl);
  EXPECT_TRUE(truth == memory.truth_break_index);

  // WriteSyntheticCube produces the same files.
  const TempDir second;
  ASSERT_TRUE(WriteSyntheticCube(memory, second.path()).ok());
  std::vector<uint16_t> red2;
  ASSERT_TRUE(ReadRawArray(second.path() + "/red.u16", plane * 24, &red2).ok());
  EXPECT_TRUE(red2 == red);
}

TEST(Synthetic, DatesAndOffsetsFollowTheRules) {
  SyntheticOptions options = SmallOptions();
  options.num_dates = 48;
  SyntheticCube cube;
  ASSERT_TRUE(GenerateSyntheticCube(options, &cube).ok());
  const DateTable& dates = cube.info.dates;
  ASSERT_EQ(dates.day.size(), 48u);
  for (int i = 0; i < 48; ++i) {
    EXPECT_EQ(dates.day[i],
              17897 + static_cast<int>(std::llround(i * 365.25 * 4.0 / 48)));
    EXPECT_EQ(dates.offset[i], dates.day[i] > 19017 ? -0.1f : 0.0f);
    EXPECT_EQ(dates.scale[i], 0.0001f);
  }
  EXPECT_EQ(dates.num_train_dates, 24);
  EXPECT_EQ(dates.iso[0], "2019-01-01");
  EXPECT_EQ(dates.iso[12], "2020-01-01");
  // Offset -0.1 starts after 2022-01-25 (day 19017): date 36 is day 18993
  // = 2022-01-01 (offset 0), date 37 is day 19023 (offset -0.1).
  EXPECT_EQ(dates.offset[36], 0.0f);
  EXPECT_EQ(dates.offset[37], -0.1f);
}

TEST(Synthetic, TruthLiesInMonitoringWindowWithDatesToSpare) {
  SyntheticOptions options = SmallOptions();
  options.height = 64;
  options.width = 64;
  options.num_dates = 48;
  options.break_fraction = 0.5f;
  SyntheticCube cube;
  ASSERT_TRUE(GenerateSyntheticCube(options, &cube).ok());
  const int num_train = cube.info.dates.num_train_dates;
  int min_index = 1000;
  int max_index = -1;
  size_t count = 0;
  for (const int16_t index : cube.truth_break_index) {
    if (index < 0) {
      continue;
    }
    ++count;
    min_index = std::min<int>(min_index, index);
    max_index = std::max<int>(max_index, index);
  }
  EXPECT_GT(count, 0u);
  EXPECT_GE(min_index, num_train);
  EXPECT_LE(max_index, 48 - options.consecutive);
  // Both ends of the allowed range are used.
  EXPECT_EQ(min_index, num_train);
  EXPECT_EQ(max_index, 48 - options.consecutive);
}

TEST(Synthetic, FractionsOfBreaksCloudsAndNodata) {
  SyntheticOptions options;
  options.height = 256;
  options.width = 256;
  options.num_dates = 48;
  SyntheticCube cube;
  ASSERT_TRUE(GenerateSyntheticCube(options, &cube).ok());
  size_t breaks = 0;
  for (const int16_t index : cube.truth_break_index) {
    breaks += index >= 0 ? 1 : 0;
  }
  const double break_fraction =
      static_cast<double>(breaks) / cube.truth_break_index.size();
  EXPECT_NEAR(break_fraction, 0.15, 0.02);

  size_t clouds = 0;
  size_t nodata = 0;
  size_t valid = 0;
  size_t zero_dn_valid = 0;
  for (size_t i = 0; i < cube.scl.size(); ++i) {
    clouds += cube.scl[i] == 9 ? 1 : 0;
    nodata += cube.scl[i] == 0 ? 1 : 0;
    valid += cube.scl[i] == 4 ? 1 : 0;
    if (cube.scl[i] == 0) {
      EXPECT_EQ(cube.red[i], 0);
      EXPECT_EQ(cube.nir[i], 0);
    } else if (cube.red[i] == 0 || cube.nir[i] == 0) {
      ++zero_dn_valid;
    }
  }
  const double total = static_cast<double>(cube.scl.size());
  EXPECT_NEAR(clouds / total, 0.2, 0.02);
  EXPECT_NEAR(nodata / total, 0.005, 0.002);
  EXPECT_EQ(clouds + nodata + valid, cube.scl.size());
  EXPECT_EQ(zero_dn_valid, 0u);
}

TEST(Synthetic, CloudsAreBright) {
  SyntheticOptions options = SmallOptions();
  options.cloud_fraction = 0.5f;
  SyntheticCube cube;
  ASSERT_TRUE(GenerateSyntheticCube(options, &cube).ok());
  size_t checked = 0;
  for (int t = 0; t < options.num_dates; ++t) {
    const size_t plane = static_cast<size_t>(options.height) * options.width;
    for (size_t i = 0; i < plane; ++i) {
      const size_t cell = t * plane + i;
      if (cube.scl[cell] != 9) {
        continue;
      }
      const double offset = cube.info.dates.offset[t];
      const double red = cube.red[cell] * 0.0001 + offset;
      const double nir = cube.nir[cell] * 0.0001 + offset;
      EXPECT_GE(red, 0.4 - 1e-3);
      EXPECT_LE(red, 0.6 + 1e-3);
      EXPECT_GE(nir, 0.4 - 1e-3);
      EXPECT_LE(nir, 0.6 + 1e-3);
      ++checked;
    }
  }
  EXPECT_GT(checked, 10000u);
}

TEST(Synthetic, CleanNdviReproducesSeasonalModel) {
  SyntheticOptions options;
  options.height = 24;
  options.width = 24;
  options.num_dates = 48;
  options.noise = 0.0f;
  options.cloud_fraction = 0.0f;
  options.break_fraction = 0.3f;
  SyntheticCube cube;
  ASSERT_TRUE(GenerateSyntheticCube(options, &cube).ok());
  const size_t plane = 24u * 24u;
  double worst = 0;
  size_t compared = 0;
  size_t steps_seen = 0;
  for (size_t pixel = 0; pixel < plane; ++pixel) {
    double before = 0;
    for (int t = 0; t < 48; ++t) {
      const size_t cell = t * plane + pixel;
      if (cube.scl[cell] != 4) {
        continue;  // the 0.5 % nodata observations
      }
      const double offset = cube.info.dates.offset[t];
      const double red = cube.red[cell] * 0.0001 + offset;
      const double nir = cube.nir[cell] * 0.0001 + offset;
      const double ndvi = (nir - red) / (nir + red);
      const double model =
          SyntheticCleanNdvi(options, static_cast<int64_t>(pixel), t);
      worst = std::max(worst, std::fabs(ndvi - model));
      ++compared;
      // A break is a step of at least 0.25 on top of the seasonal change
      // between neighbouring dates (at most about 0.16).
      if (t == cube.truth_break_index[pixel] && t > 0) {
        EXPECT_GE(std::fabs(model - before), 0.05);
        ++steps_seen;
      }
      before = model;
    }
  }
  EXPECT_GT(compared, 20000u);
  EXPECT_LT(worst, 2e-3);
  EXPECT_GT(steps_seen, 30u);
}

TEST(Synthetic, RejectsBadOptions) {
  SyntheticCube cube;
  SyntheticOptions options = SmallOptions();
  options.height = 0;
  EXPECT_FALSE(GenerateSyntheticCube(options, &cube).ok());
  options = SmallOptions();
  options.num_dates = 2000;
  EXPECT_FALSE(GenerateSyntheticCube(options, &cube).ok());
  options = SmallOptions();
  options.train_years = 5.0f;  // nothing left to monitor
  EXPECT_FALSE(GenerateSyntheticCube(options, &cube).ok());
  options = SmallOptions();
  options.cloud_fraction = 1.5f;
  EXPECT_FALSE(GenerateSyntheticCube(options, &cube).ok());
  options = SmallOptions();
  options.consecutive = 0;
  EXPECT_FALSE(GenerateSyntheticCube(options, &cube).ok());
  options = SmallOptions();
  options.height = -1;
  const TempDir dir;
  EXPECT_FALSE(GenerateSyntheticCubeToDisk(options, dir.path()).ok());
}

// Builds a 1 x 7 cube of 12 dates (30 days apart, all observations valid
// unless a test changes them) for the recovery cases.
SyntheticCube MakeRecoveryCube() {
  SyntheticCube cube;
  CubeInfo& info = cube.info;
  info.height = 1;
  info.width = 7;
  info.num_dates = 12;
  for (int t = 0; t < 12; ++t) {
    info.dates.day.push_back(17897 + 30 * t);
    info.dates.iso.push_back("2019-01-01");
    info.dates.scale.push_back(0.0001f);
    info.dates.offset.push_back(0.0f);
  }
  const size_t cells = 7u * 12u;
  cube.red.assign(cells, 1000);
  cube.nir.assign(cells, 3000);
  cube.scl.assign(cells, 4);
  return cube;
}

TEST(Synthetic, ComputeRecoveryCases) {
  SyntheticCube cube = MakeRecoveryCube();
  const auto at = [](int t, int pixel) { return t * 7 + pixel; };
  // Pixel 2: the date after the injected index is cloudy, so only two valid
  // observations remain at/after index 10 -> ineligible.
  // Pixel 5: the injected date 7 has a zero DN and date 8 is cloudy, so the
  // first valid observation is date 9 and the next one date 10.
  cube.scl[at(11, 2)] = 9;
  cube.nir[at(7, 5)] = 0;
  cube.scl[at(8, 5)] = 9;
  // Pixel 6: dates 0..5 are cloudy, so the training window (dates 0..6) has
  // too few valid observations.
  for (int t = 0; t < 6; ++t) {
    cube.scl[at(t, 6)] = 9;
  }
  const std::vector<int16_t> truth = {8, 8, 10, -1, -1, 7, 8};

  DetectionResult result;
  AllocateResult(1, 7, 8, false, &result);
  const auto set_break = [&](int pixel, int index) {
    result.status[pixel] = static_cast<uint8_t>(PixelStatus::kBreak);
    result.break_index[pixel] = static_cast<int16_t>(index);
  };
  set_break(0, 8);   // exact -> recovered
  set_break(1, 10);  // two observations late -> not recovered
  set_break(2, 10);  // ineligible, whatever the result
  set_break(3, 9);   // false break
  // pixel 4: stable, no break
  set_break(5, 9);  // first valid >= 7 is 9 -> recovered
  set_break(6, 8);  // insufficient training -> not eligible

  ModelParams params;
  params.train_years = 0.5f;  // dates 0..6 (tau < 0.5 years)
  params.harmonics = 1;
  params.min_train_obs = 3;
  params.consecutive = 3;
  const std::unique_ptr<CubeSource> source = MakeMemoryCube(std::move(cube));
  RecoveryStats stats;
  ASSERT_TRUE(
      ComputeRecovery(source.get(), truth, result, params, &stats).ok());
  EXPECT_EQ(stats.injected, 5u);
  EXPECT_EQ(stats.insufficient_training, 1u);
  EXPECT_EQ(stats.eligible, 3u);
  EXPECT_EQ(stats.recovered, 2u);
  EXPECT_EQ(stats.stable_truth, 2u);
  EXPECT_EQ(stats.false_breaks, 1u);

  // Pixel 5: the next valid observation (date 10) is still within one
  // observation, date 11 is not, and a non-break status never counts.
  result.break_index[5] = 10;
  ASSERT_TRUE(
      ComputeRecovery(source.get(), truth, result, params, &stats).ok());
  EXPECT_EQ(stats.recovered, 2u);
  result.break_index[5] = 11;
  ASSERT_TRUE(
      ComputeRecovery(source.get(), truth, result, params, &stats).ok());
  EXPECT_EQ(stats.recovered, 1u);
  result.break_index[5] = 9;
  result.status[5] = static_cast<uint8_t>(PixelStatus::kStable);
  ASSERT_TRUE(
      ComputeRecovery(source.get(), truth, result, params, &stats).ok());
  EXPECT_EQ(stats.recovered, 1u);

  // Size mismatches are errors.
  const std::vector<int16_t> short_truth(3, -1);
  EXPECT_FALSE(
      ComputeRecovery(source.get(), short_truth, result, params, &stats).ok());
}

}  // namespace
}  // namespace cd
