// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): temporary directories
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark.h"
#include "outputs.h"
#include "postprocess.h"
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
        (fs::temp_directory_path() / "cd_test_outputs_XXXXXX").string();
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

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

size_t CountChar(const std::string& text, char c) {
  return static_cast<size_t>(std::count(text.begin(), text.end(), c));
}

// Writes the pixel index into the magnitude map, so that the pixel a block
// chose can be read back from the downsampled map.
void MarkPixels(DetectionResult* result) {
  for (size_t i = 0; i < result->magnitude.size(); ++i) {
    result->magnitude[i] = static_cast<float>(i);
  }
}

// ---------------------------------------------------------------------------
// Downsampling.

TEST(Outputs, DownsampleFactor) {
  EXPECT_EQ(DownsampleFactor(100, 80, 2048), 1);
  EXPECT_EQ(DownsampleFactor(5000, 5000, 2048), 3);
  EXPECT_EQ(DownsampleFactor(4096, 100, 2048), 2);
  EXPECT_EQ(DownsampleFactor(10, 10, 0), 1);
}

TEST(Outputs, ChangeMapBlockPriorityBreakStableInsufficient) {
  // 4 x 4 pixels, blocks of 2 x 2. Pixel index = 4 * row + col.
  DetectionResult result;
  AllocateResult(4, 4, 6, false, &result);
  MarkPixels(&result);
  const uint8_t kBreak = static_cast<uint8_t>(PixelStatus::kBreak);
  const uint8_t kInsufficient =
      static_cast<uint8_t>(PixelStatus::kInsufficientData);
  // Block (0,0) pixels 0,1,4,5: insufficient, stable, break at 7, break at 3.
  result.status[0] = kInsufficient;
  result.status[1] = static_cast<uint8_t>(PixelStatus::kStable);
  result.status[4] = kBreak;
  result.break_index[4] = 7;
  result.status[5] = kBreak;
  result.break_index[5] = 3;
  // Block (0,1) pixels 2,3,6,7: insufficient, stable, insufficient, stable.
  result.status[2] = kInsufficient;
  result.status[3] = static_cast<uint8_t>(PixelStatus::kStable);
  result.status[6] = kInsufficient;
  result.status[7] = static_cast<uint8_t>(PixelStatus::kStable);
  // Block (1,0) pixels 8,9,12,13: all insufficient.
  for (const int i : {8, 9, 12, 13}) {
    result.status[i] = kInsufficient;
  }
  // Block (1,1) pixels 10,11,14,15: all stable.
  for (const int i : {10, 11, 14, 15}) {
    result.status[i] = static_cast<uint8_t>(PixelStatus::kStable);
  }
  const DetectionResult small = DownsampleDetection(result, 2);
  ASSERT_EQ(small.width, 2);
  ASSERT_EQ(small.height, 2);
  ASSERT_EQ(small.status.size(), 4u);
  ASSERT_EQ(small.break_index.size(), 4u);
  ASSERT_EQ(small.magnitude.size(), 4u);
  EXPECT_EQ(small.magnitude[0], 5.0f);  // the earliest break (index 3) wins
  EXPECT_EQ(small.break_index[0], 3);
  EXPECT_EQ(small.status[0], kBreak);
  EXPECT_EQ(small.magnitude[1], 3.0f);  // first stable pixel beats insufficient
  EXPECT_EQ(small.magnitude[2], 8.0f);  // all insufficient: first pixel
  EXPECT_EQ(small.magnitude[3], 10.0f);  // all stable: first pixel
}

TEST(Outputs, ChangeMapKeepsSingleBreakInLargeBlock) {
  DetectionResult result;
  AllocateResult(9, 9, 6, false, &result);
  MarkPixels(&result);
  std::fill(result.status.begin(), result.status.end(),
            static_cast<uint8_t>(PixelStatus::kStable));
  const size_t lonely = 7 * 9 + 5;
  result.status[lonely] = static_cast<uint8_t>(PixelStatus::kBreak);
  result.break_index[lonely] = 12;
  const DetectionResult small = DownsampleDetection(result, 3);
  ASSERT_EQ(small.width, 3);
  ASSERT_EQ(small.height, 3);
  // Pixel (7, 5) lies in block row 2, block column 1.
  EXPECT_EQ(small.magnitude[2 * 3 + 1], static_cast<float>(lonely));
  EXPECT_EQ(small.break_index[2 * 3 + 1], 12);
}

TEST(Outputs, ChangeMapEdgeBlocksAndNoDownsampling) {
  DetectionResult result;
  AllocateResult(5, 3, 6, false, &result);
  MarkPixels(&result);
  std::fill(result.status.begin(), result.status.end(),
            static_cast<uint8_t>(PixelStatus::kStable));
  const DetectionResult same = DownsampleDetection(result, 100);
  EXPECT_EQ(same.width, 3);
  EXPECT_EQ(same.height, 5);
  EXPECT_TRUE(same.magnitude == result.magnitude);
  EXPECT_TRUE(same.status == result.status);
  EXPECT_TRUE(same.break_index == result.break_index);
  const DetectionResult blocks = DownsampleDetection(result, 2);
  EXPECT_EQ(blocks.width, 1);            // factor 3: 3 columns -> 1 block
  EXPECT_EQ(blocks.height, 2);           // 5 rows -> blocks of 3 and 2
  EXPECT_EQ(blocks.magnitude[1], 9.0f);  // second block starts at pixel (3, 0)
}

// Colouring the downsampled map gives the picture that colouring the full map
// and then picking one pixel per block gives (the order used before the
// downsampling moved ahead of the colouring).
TEST(Outputs, ColouringAfterDownsamplingEqualsDownsamplingColours) {
  const DateTable dates = [] {
    DateTable table;
    for (int i = 0; i < 10; ++i) {
      table.day.push_back(18000 + 100 * i);
    }
    table.num_train_dates = 4;
    return table;
  }();
  const int kHeight = 23;
  const int kWidth = 31;
  DetectionResult result;
  AllocateResult(kHeight, kWidth, 6, false, &result);
  const uint8_t codes[3] = {
      static_cast<uint8_t>(PixelStatus::kBreak),
      static_cast<uint8_t>(PixelStatus::kStable),
      static_cast<uint8_t>(PixelStatus::kInsufficientData)};
  uint32_t state = 12345;
  for (size_t i = 0; i < result.status.size(); ++i) {
    state = state * 1664525u + 1013904223u;
    result.status[i] = codes[(state >> 24) % 3];
    result.break_index[i] = static_cast<int16_t>(4 + (state >> 16) % 6);
    result.magnitude[i] = (state >> 8) % 2 == 0 ? -0.2f : 0.2f;
  }
  std::vector<uint8_t> full;
  ASSERT_TRUE(CpuColorizeChangeMap(result, dates, &full, nullptr).ok());
  const int max_dim = 8;
  const int factor = DownsampleFactor(kHeight, kWidth, max_dim);
  const DetectionResult small = DownsampleDetection(result, max_dim);
  // The number of worker threads does not change the map.
  for (const int threads : {2, 3, 64}) {
    const DetectionResult threaded =
        DownsampleDetection(result, max_dim, threads);
    EXPECT_TRUE(threaded.break_index == small.break_index);
    EXPECT_TRUE(threaded.magnitude == small.magnitude);
    EXPECT_TRUE(threaded.status == small.status);
  }
  std::vector<uint8_t> after;
  ASSERT_TRUE(CpuColorizeChangeMap(small, dates, &after, nullptr).ok());
  ASSERT_EQ(after.size(), static_cast<size_t>(small.height) * small.width * 4);
  // Reference: scan each block for the winning pixel and take its colour.
  for (int by = 0; by < small.height; ++by) {
    for (int bx = 0; bx < small.width; ++bx) {
      int best_class = 3;
      int best_break = 1 << 30;
      size_t best = 0;
      for (int r = by * factor; r < std::min(kHeight, (by + 1) * factor); ++r) {
        for (int c = bx * factor; c < std::min(kWidth, (bx + 1) * factor);
             ++c) {
          const size_t i = static_cast<size_t>(r) * kWidth + c;
          const int cls = result.status[i] == codes[0]
                              ? 0
                              : (result.status[i] == codes[1] ? 1 : 2);
          if (cls < best_class || (cls == 0 && best_class == 0 &&
                                   result.break_index[i] < best_break)) {
            best_class = cls;
            best_break = result.break_index[i];
            best = i;
          }
        }
      }
      const size_t out = (static_cast<size_t>(by) * small.width + bx) * 4;
      for (size_t k = 0; k < 4; ++k) {
        EXPECT_EQ(after[out + k], full[best * 4 + k]);
      }
    }
  }
}

TEST(Outputs, QuickDiffPriorityAndColours) {
  QuickDiffResult result;
  result.height = 2;
  result.width = 4;
  // Left 2 x 2 block: no data, no change, gain, loss. Right block: no data,
  // no change, no change, gain.
  result.diff.assign(8, 0.0f);
  result.flag = {0, 1, 0, 0, 3, 3, 2, 1};
  // Row-major: row 0 = 0 1 0 0, row 1 = 3 3 2 1.
  result.flag = {0, 1, 0, 0, 3, 3, 2, 1};
  const Image image = RenderQuickDiff(result, 2);
  ASSERT_EQ(image.width, 2);
  ASSERT_EQ(image.height, 1);
  // Left block holds flags 0, 1, 3, 3: gain wins. Right block holds 0, 0, 2,
  // 1: loss wins.
  EXPECT_EQ(image.rgba[0], 44);
  EXPECT_EQ(image.rgba[1], 160);
  EXPECT_EQ(image.rgba[2], 44);
  EXPECT_EQ(image.rgba[3], 255);
  EXPECT_EQ(image.rgba[4], 214);
  EXPECT_EQ(image.rgba[5], 39);
  EXPECT_EQ(image.rgba[6], 40);
  // Without downsampling each flag has its own colour.
  const Image full = RenderQuickDiff(result, 100);
  EXPECT_EQ(full.rgba[0], 0);    // no data: black
  EXPECT_EQ(full.rgba[4], 128);  // no change: grey
}

TEST(Outputs, WritePngProducesPngFile) {
  TempDir dir;
  ASSERT_FALSE(dir.path().empty());
  Image image;
  image.width = 3;
  image.height = 2;
  image.rgba.assign(24, 200);
  const std::string path = dir.path() + "/x.png";
  ASSERT_TRUE(WritePng(path, image).ok());
  const std::string bytes = ReadFile(path);
  ASSERT_GE(bytes.size(), 8u);
  EXPECT_EQ(bytes.substr(1, 3), "PNG");
  Image bad;
  EXPECT_FALSE(WritePng(dir.path() + "/bad.png", bad).ok());
}

// ---------------------------------------------------------------------------
// Result manifest.

ResultManifestInput MakeManifestInput() {
  ResultManifestInput input;
  input.cube_manifest = "synthetic";
  input.name = "unit test cube";
  input.height = 4;
  input.width = 5;
  input.num_dates = 3;
  input.backend = "cpu-mt";
  input.params.harmonics = 2;
  input.params.train_years = 2.0f;
  input.params.min_train_obs = 18;
  input.params.threshold = 3.5f;
  input.params.consecutive = 3;
  input.params.rmse_floor = 0.02f;
  input.params.ridge = 0.0f;
  input.params.losses_only = true;
  input.params.valid_class_mask = (1u << 4) | (1u << 5) | (1u << 6);
  input.cleanup = true;
  input.reference_file = "none";
  input.dates.iso = {"2020-01-01", "2021-02-03", "2022-03-04"};
  input.dates.day = {18262, 18661, 19055};
  input.dates.scale = {1e-4f, 1e-4f, 1e-4f};
  input.dates.offset = {0.0f, 0.0f, -0.1f};
  input.dates.year = {2020, 2021, 2022};
  input.dates.num_train_dates = 1;
  input.files.push_back({"status.u8", "u8", "status map"});
  return input;
}

TEST(Outputs, ResultManifestContent) {
  const std::string text = BuildResultManifest(MakeManifestInput());
  EXPECT_EQ(text.substr(0, 23), "change_detect_result 1\n");
  const std::vector<std::string> expected = {
      "cube_manifest synthetic\n",
      "name unit test cube\n",
      "height 4\n",
      "width 5\n",
      "num_dates 3\n",
      "backend cpu-mt\n",
      "harmonics 2\n",
      "trend hold\n",
      "train_years 2\n",
      "min_train_obs 18\n",
      "threshold 3.5\n",
      "consecutive 3\n",
      "rmse_floor 0.02\n",
      "ridge 0\n",
      "losses_only 1\n",
      "valid_classes 4,5,6\n",
      "cleanup 1\n",
      "reference_file none\n",
      "num_train_dates 1\n",
      "first_monitor_year 2021\n",
      "last_year 2022\n",
      "date 0 2020-01-01 18262 0.0001 0 -1 -\n",
      "date 2 2022-03-04 19055 0.0001 -0.1 -1 -\n",
      "file status.u8 u8 status map\n"};
  for (const std::string& line : expected) {
    EXPECT_TRUE(text.find("\n" + line) != std::string::npos);
  }
  ResultManifestInput extrapolating = MakeManifestInput();
  extrapolating.params.hold_trend = false;
  EXPECT_TRUE(
      BuildResultManifest(extrapolating).find("\ntrend extrapolate\n") !=
      std::string::npos);
  // Keys come in the documented order.
  EXPECT_LT(text.find("\nheight "), text.find("\nharmonics "));
  EXPECT_LT(text.find("\nreference_file "), text.find("\ndate 0 "));
  EXPECT_LT(text.find("\ndate 2 "), text.find("\nfile "));
}

TEST(Outputs, ResultManifestWriteAndMissingDirectory) {
  TempDir dir;
  ASSERT_FALSE(dir.path().empty());
  ASSERT_TRUE(WriteResultManifest(dir.path() + "/result_manifest.txt",
                                  MakeManifestInput())
                  .ok());
  EXPECT_EQ(ReadFile(dir.path() + "/result_manifest.txt"),
            BuildResultManifest(MakeManifestInput()));
  EXPECT_FALSE(WriteResultManifest(dir.path() + "/no/such/dir/m.txt",
                                   MakeManifestInput())
                   .ok());
}

// ---------------------------------------------------------------------------
// Sample pixels.

TEST(Outputs, SamplePixelsPreferConfirmedLossesAndAreDeterministic) {
  DetectionResult result;
  AllocateResult(10, 10, 6, false, &result);
  const uint8_t stable = static_cast<uint8_t>(PixelStatus::kStable);
  std::fill(result.status.begin(), result.status.end(), stable);
  std::vector<uint8_t> reference(100, 0);
  ValidationWindow window;
  window.first_year = 2021;
  window.last_year = 2023;
  // Pixels 10..19: unconfirmed losses; 40..44: losses confirmed by the
  // reference; 60, 61: gains.
  for (int i = 10; i < 20; ++i) {
    result.status[i] = static_cast<uint8_t>(PixelStatus::kBreak);
    result.break_index[i] = 5;
    result.magnitude[i] = -0.3f;
  }
  for (int i = 40; i < 45; ++i) {
    result.status[i] = static_cast<uint8_t>(PixelStatus::kBreak);
    result.break_index[i] = 5;
    result.magnitude[i] = -0.3f;
    reference[i] = 22;
  }
  for (const int i : {60, 61}) {
    result.status[i] = static_cast<uint8_t>(PixelStatus::kBreak);
    result.break_index[i] = 5;
    result.magnitude[i] = 0.3f;
  }
  const std::vector<std::pair<int, int>> a =
      SelectSamplePixels(result, &reference, window, 8);
  ASSERT_EQ(a.size(), 8u);
  // Four break pixels (all confirmed losses: rows 4, cols 0..4) and four
  // stable pixels.
  int confirmed = 0;
  int broken = 0;
  for (const std::pair<int, int>& pixel : a) {
    const size_t i = static_cast<size_t>(pixel.first) * 10 + pixel.second;
    broken += result.status[i] != stable ? 1 : 0;
    confirmed += (i >= 40 && i < 45) ? 1 : 0;
  }
  EXPECT_EQ(broken, 4);
  EXPECT_EQ(confirmed, 4);
  EXPECT_TRUE(std::is_sorted(a.begin(), a.end()));
  EXPECT_TRUE(a == SelectSamplePixels(result, &reference, window, 8));
  // Without a reference the unconfirmed losses (pixels 10..19) come first.
  const std::vector<std::pair<int, int>> b =
      SelectSamplePixels(result, nullptr, window, 2);
  ASSERT_EQ(b.size(), 2u);
  int losses = 0;
  for (const std::pair<int, int>& pixel : b) {
    const size_t i = static_cast<size_t>(pixel.first) * 10 + pixel.second;
    losses += result.status[i] != stable ? 1 : 0;
  }
  EXPECT_EQ(losses, 1);
}

TEST(Outputs, SamplePixelsShortageIsMadeUpByTheOtherSide) {
  DetectionResult result;
  AllocateResult(2, 3, 6, false, &result);
  // One break, one insufficient pixel, four stable pixels; ask for 6.
  std::fill(result.status.begin(), result.status.end(),
            static_cast<uint8_t>(PixelStatus::kStable));
  result.status[1] = static_cast<uint8_t>(PixelStatus::kBreak);
  result.break_index[1] = 2;
  result.magnitude[1] = -0.2f;
  result.status[4] = static_cast<uint8_t>(PixelStatus::kInsufficientData);
  const std::vector<std::pair<int, int>> pixels =
      SelectSamplePixels(result, nullptr, ValidationWindow(), 6);
  EXPECT_EQ(pixels.size(), 5u);  // only 5 candidates exist
  EXPECT_TRUE(
      SelectSamplePixels(result, nullptr, ValidationWindow(), 0).empty());
}

TEST(Outputs, ExplicitSamplePixelsAreCheckedAgainstTheBounds) {
  const std::vector<std::pair<int, int>> inside = {{0, 0}, {3, 4}};
  EXPECT_TRUE(ValidateSamplePixels(inside, 4, 5).ok());
  const std::vector<std::pair<int, int>> outside = {{0, 0}, {4, 0}};
  EXPECT_FALSE(ValidateSamplePixels(outside, 4, 5).ok());
  const std::vector<std::pair<int, int>> wide = {{0, 5}};
  EXPECT_FALSE(ValidateSamplePixels(wide, 4, 5).ok());
}

// ---------------------------------------------------------------------------
// pixel_stats.csv and trajectories.csv.

PixelTrajectory MakeTrajectory(int row, int col, uint8_t status) {
  PixelTrajectory t;
  t.row = row;
  t.col = col;
  t.ndvi = {0.5, kInvalidNdvi, 0.4};
  t.predicted = {0.5, 0.45, 0.42};
  t.z = {0.1, 0.0, -0.5};
  t.in_break_run = {0, 0, 1};
  t.coef[0] = 0.5;
  t.rmse = 0.03;
  t.n_train = 2;
  t.status = status;
  t.break_index = status == 1 ? 2 : -1;
  t.magnitude = status == 1 ? -0.4 : 0.0;
  return t;
}

TEST(Outputs, TrajectoriesCsvColumns) {
  const DateTable dates = MakeManifestInput().dates;
  DateTable full = dates;
  full.tau = {0.0, 1.09, 2.18};
  EXPECT_EQ(TrajectoriesCsvHeader(),
            "row,col,date_index,date,day,tau,ndvi,valid,in_training,predicted,"
            "z,is_break_run");
  const std::vector<PixelTrajectory> trajectories = {MakeTrajectory(1, 2, 1),
                                                     MakeTrajectory(3, 4, 2)};
  const std::string csv = BuildTrajectoriesCsv(trajectories, full);
  std::istringstream lines(csv);
  std::string line;
  std::vector<std::string> rows;
  while (std::getline(lines, line)) {
    rows.push_back(line);
  }
  ASSERT_EQ(rows.size(), 1u + 2u * 3u);
  EXPECT_EQ(rows[0], TrajectoriesCsvHeader());
  for (const std::string& row : rows) {
    EXPECT_EQ(CountChar(row, ','), 11u);
  }
  EXPECT_EQ(rows[1],
            "1,2,0,2020-01-01,18262,0.000000,0.500000,1,1,0.500000,"
            "0.100000,0");
  // Invalid observation: ndvi and z are empty, valid is 0.
  EXPECT_EQ(rows[2], "1,2,1,2021-02-03,18661,1.090000,,0,0,0.450000,,0");
  EXPECT_EQ(rows[3],
            "1,2,2,2022-03-04,19055,2.180000,0.400000,1,0,0.420000,"
            "-0.500000,1");
  // Failed fit (insufficient data): no prediction and no z-score.
  EXPECT_EQ(rows[4], "3,4,0,2020-01-01,18262,0.000000,0.500000,1,1,,,0");
}

TEST(Outputs, PixelStatsCsvHasTwoRowsPerPixel) {
  DetectionResult result;
  AllocateResult(4, 5, 6, true, &result);
  const size_t pixel = 1 * 5 + 2;  // (1, 2)
  result.status[pixel] = static_cast<uint8_t>(PixelStatus::kBreak);
  result.break_index[pixel] = 2;
  result.magnitude[pixel] = -0.4f;
  result.rmse[pixel] = 0.03f;
  result.train_obs[pixel] = 2;
  result.coefficients[0 * 20 + pixel] = 0.5f;
  const DateTable dates = MakeManifestInput().dates;
  const std::vector<PixelTrajectory> trajectories = {MakeTrajectory(1, 2, 1)};
  const std::string csv =
      BuildPixelStatsCsv(result, "cpu-mt", trajectories, dates);
  std::istringstream lines(csv);
  std::string header;
  std::string run_row;
  std::string fp64_row;
  ASSERT_TRUE(static_cast<bool>(std::getline(lines, header)));
  ASSERT_TRUE(static_cast<bool>(std::getline(lines, run_row)));
  ASSERT_TRUE(static_cast<bool>(std::getline(lines, fp64_row)));
  EXPECT_EQ(header,
            "row,col,backend,status,break_index,break_date,magnitude,rmse,"
            "n_train,coef0,coef1,coef2,coef3,coef4,coef5");
  EXPECT_EQ(CountChar(run_row, ','), CountChar(header, ','));
  EXPECT_EQ(CountChar(fp64_row, ','), CountChar(header, ','));
  EXPECT_EQ(run_row.substr(0, 40), "1,2,cpu-mt,break,2,2022-03-04,-0.4,0.03,");
  EXPECT_TRUE(run_row.find(",0.5,0,0,0,0,0") != std::string::npos);
  EXPECT_EQ(fp64_row.substr(0, 25), "1,2,cpu-fp64-recompute,br");
  // Without saved coefficients the run's coefficient cells are blank.
  DetectionResult bare;
  AllocateResult(4, 5, 6, false, &bare);
  const std::string blank =
      BuildPixelStatsCsv(bare, "gpu", trajectories, dates);
  EXPECT_TRUE(blank.find(",gpu,stable,-1,,0,0,0,,,,,,\n") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Benchmark CSV and agreement.

// The expected benchmark CSV header line.
std::string PlanHeader() {
  return "run_id,timestamp,mode,backend,precision,kernels,input_name,height,"
         "width,"
         "dates,pixels,samples,tile_rows,num_tiles,streams,block_size,threads,"
         "io_threads,repeat_index,input_bytes,read_ms,read_wait_ms,setup_ms,"
         "h2d_ms,ndvi_ms,"
         "fit_ms,detect_ms,fused_ms,d2h_ms,kernel_ms,total_ms,throughput_"
         "msamples_s,"
         "kernel_throughput_msamples_s,device_mem_peak_mb,changed_pixels,"
         "compare_to,agree_break_frac,agree_status_frac,max_coef_diff,gpu_name,"
         "gpu_mem_mb,cpu_model,cpu_threads,cuda_runtime,driver_version,git_"
         "commit";
}

BenchmarkRow MakeRow(const std::string& cpu_model) {
  CubeInfo info;
  info.name = "synthetic, 4x5";
  info.height = 4;
  info.width = 5;
  info.num_dates = 10;
  RunConfig config;
  config.backend = Backend::kCpuMt;
  StageTimings timings;
  timings.read_ms = 1.0;
  timings.setup_ms = 0.5;
  timings.kernel_ms = 2.0;
  timings.total_ms = 4.0;
  RunStats stats;
  stats.tile_rows_used = 4;
  stats.num_tiles = 1;
  stats.threads_used = 8;
  SystemColumns system;
  system.cpu_model = cpu_model;
  system.cpu_threads = 16;
  system.git_commit = "abc1234";
  return MakeBenchmarkRow("run1", info, config, timings, stats, 3, 7, system);
}

TEST(Outputs, BenchmarkCsvHeaderMatchesPlan) {
  EXPECT_EQ(BenchmarkCsvHeader(), PlanHeader());
}

TEST(Outputs, BenchmarkRowFormatting) {
  const BenchmarkRow row = MakeRow("Fast, CPU \"X\"");
  const std::string line = FormatBenchmarkRow(row);
  // The comma-containing name and model are quoted (a quote is doubled), so
  // the unquoted commas still number one less than the columns.
  EXPECT_TRUE(line.find("\"synthetic, 4x5\"") != std::string::npos);
  EXPECT_TRUE(line.find("\"Fast, CPU \"\"X\"\"\"") != std::string::npos);
  EXPECT_EQ(CountChar(line, ',') - 2u, CountChar(BenchmarkCsvHeader(), ','));
  // CPU row: fp32, no kernel variant, streams and block size not applicable.
  EXPECT_TRUE(line.find(",cpu-mt,fp32,n/a,") != std::string::npos);
  EXPECT_TRUE(line.find(",4,1,-1,-1,8,8,3,") != std::string::npos);
  // 3 decimals for milliseconds, -1 for not-applicable stages.
  EXPECT_TRUE(line.find(",1.000,-1,0.500,-1,-1,-1,-1,-1,-1,2.000,4.000,") !=
              std::string::npos);
  // Throughput = 200 samples / 4 ms / 1e3 = 0.050 Msamples/s.
  EXPECT_TRUE(line.find(",0.050,0.100,") != std::string::npos);
  EXPECT_TRUE(line.find(",none,-1,") != std::string::npos);  // gpu_name, mem
}

TEST(Outputs, BenchmarkCsvAppendRules) {
  TempDir dir;
  ASSERT_FALSE(dir.path().empty());
  const std::string path = dir.path() + "/sub/bench.csv";
  const std::vector<BenchmarkRow> rows = {MakeRow("cpu"), MakeRow("cpu")};
  ASSERT_TRUE(AppendBenchmarkRows(path, rows).ok());
  ASSERT_TRUE(AppendBenchmarkRows(path, rows).ok());
  const std::string text = ReadFile(path);
  EXPECT_EQ(CountChar(text, '\n'), 5u);  // one header, four rows
  EXPECT_EQ(text.substr(0, BenchmarkCsvHeader().size()), BenchmarkCsvHeader());
  // An existing file with another header is refused and left untouched.
  const std::string other = dir.path() + "/other.csv";
  {
    std::ofstream out(other);
    out << "a,b,c\n1,2,3\n";
  }
  EXPECT_FALSE(AppendBenchmarkRows(other, rows).ok());
  EXPECT_EQ(ReadFile(other), "a,b,c\n1,2,3\n");
  // An empty row list still creates the file with its header.
  const std::string empty = dir.path() + "/empty.csv";
  ASSERT_TRUE(AppendBenchmarkRows(empty, {}).ok());
  EXPECT_EQ(ReadFile(empty), BenchmarkCsvHeader() + "\n");
}

TEST(Outputs, AgreementFractionsAndCoefficientDifference) {
  DetectionResult a;
  DetectionResult b;
  AllocateResult(2, 5, 6, true, &a);
  AllocateResult(2, 5, 6, true, &b);
  b.break_index[3] = 4;
  b.status[3] = static_cast<uint8_t>(PixelStatus::kBreak);
  b.status[7] = static_cast<uint8_t>(PixelStatus::kInsufficientData);
  b.coefficients[11] = 0.25f;
  Agreement agreement;
  ASSERT_TRUE(ComputeAgreement(a, b, &agreement).ok());
  EXPECT_NEAR(agreement.break_frac, 0.9, 1e-12);
  EXPECT_NEAR(agreement.status_frac, 0.8, 1e-12);
  EXPECT_NEAR(agreement.max_coef_diff, 0.25, 1e-9);
  // Coefficients are compared only when both results hold them.
  DetectionResult c;
  AllocateResult(2, 5, 6, false, &c);
  ASSERT_TRUE(ComputeAgreement(a, c, &agreement).ok());
  EXPECT_EQ(agreement.max_coef_diff, -1.0);
  DetectionResult d;
  AllocateResult(3, 5, 6, false, &d);
  EXPECT_FALSE(ComputeAgreement(a, d, &agreement).ok());
}

TEST(Outputs, MedianAndMedianTable) {
  EXPECT_EQ(Median({}), 0.0);
  EXPECT_EQ(Median({5.0}), 5.0);
  EXPECT_EQ(Median({9.0, 1.0, 5.0}), 5.0);
  EXPECT_EQ(Median({4.0, 1.0, 2.0, 3.0}), 2.5);
  BackendSamples cpu;
  cpu.backend = "cpu";
  cpu.total_ms = {100.0, 120.0, 110.0};
  cpu.kernel_ms = {90.0, 95.0, 92.0};
  BackendSamples mt;
  mt.backend = "cpu-mt";
  mt.total_ms = {10.0, 12.0, 11.0};
  mt.kernel_ms = {-1, -1, -1};
  const std::vector<std::string> table = FormatMedianTable({cpu, mt});
  ASSERT_EQ(table.size(), 3u);
  EXPECT_EQ(table[0],
            "backend | median backend_total ms | min | max | median kernel ms "
            "| speedup vs cpu-mt");
  EXPECT_EQ(table[1], "cpu | 110.000 | 100.000 | 120.000 | 92.000 | 0.10x");
  EXPECT_EQ(table[2], "cpu-mt | 11.000 | 10.000 | 12.000 | - | 1.00x");
  // Without cpu-mt in the list the speedup column is "-".
  const std::vector<std::string> alone = FormatMedianTable({cpu});
  EXPECT_EQ(alone[1], "cpu | 110.000 | 100.000 | 120.000 | 92.000 | -");
}

}  // namespace
}  // namespace cd
