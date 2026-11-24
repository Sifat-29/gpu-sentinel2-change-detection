// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Tests of the cuFFT phase-correlation co-registration check: recovery of
// known integer and sub-pixel shifts of a synthetic texture (with noise and a
// masked cloud blob), the reference date, flag rules and rejected arguments.
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "coreg.h"
#include "cuda/gpu_backend.h"
#include "flags.h"
#include "test_framework.h"

namespace cd {
namespace {

constexpr int kWindow = 256;
constexpr float kScale = 1e-4f;

// A smooth random texture: a sum of Gaussian blobs, evaluable at any real
// position so that a sub-pixel shift is exact (no interpolation).
class Texture {
 public:
  explicit Texture(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> position(-80.0f, kWindow + 80.0f);
    std::uniform_real_distribution<float> width(1.2f, 6.0f);
    std::uniform_real_distribution<float> amplitude(-1.0f, 1.0f);
    for (int i = 0; i < 900; ++i) {
      blobs_.push_back(
          {position(rng), position(rng), width(rng), amplitude(rng)});
    }
  }

  // Texture value at row y, column x (about -1..1).
  float At(float y, float x) const {
    float value = 0.0f;
    for (const Blob& b : blobs_) {
      const float dy = y - b.y;
      const float dx = x - b.x;
      value += b.amplitude *
               std::exp(-(dy * dy + dx * dx) / (2.0f * b.sigma * b.sigma));
    }
    return value;
  }

 private:
  struct Blob {
    float y;
    float x;
    float sigma;
    float amplitude;
  };
  std::vector<Blob> blobs_;
};

// Input arrays of the synthetic dates plus the pointers handed to the GPU.
struct TestScenes {
  std::vector<uint16_t> band;
  std::vector<uint8_t> valid;
  std::vector<float> scale;
  std::vector<float> offset;
  int num_dates = 0;

  CoregGpuInput Input(int reference) const {
    CoregGpuInput in;
    in.window = kWindow;
    in.num_dates = num_dates;
    in.reference_date = reference;
    in.band = band.data();
    in.valid = valid.data();
    in.scale = scale.data();
    in.offset = offset.data();
    return in;
  }
};

// Builds one scene per (dy, dx): the texture displaced by that amount, noise
// of sd 0.02 reflectance and, per date, a masked disc of bright "cloud".
TestScenes MakeScenes(const std::vector<std::pair<float, float>>& shifts) {
  const Texture texture(7);
  std::mt19937 rng(11);
  std::normal_distribution<float> noise(0.0f, 0.002f);
  TestScenes scenes;
  scenes.num_dates = static_cast<int>(shifts.size());
  const size_t pixels = static_cast<size_t>(kWindow) * kWindow;
  scenes.band.assign(pixels * shifts.size(), 0);
  scenes.valid.assign(pixels * shifts.size(), 1);
  scenes.scale.assign(shifts.size(), kScale);
  scenes.offset.assign(shifts.size(), 0.0f);
  for (size_t t = 0; t < shifts.size(); ++t) {
    const float cloud_y = 40.0f + 50.0f * static_cast<float>(t);
    const float cloud_x = 200.0f - 40.0f * static_cast<float>(t);
    for (int y = 0; y < kWindow; ++y) {
      for (int x = 0; x < kWindow; ++x) {
        const size_t index = t * pixels + static_cast<size_t>(y) * kWindow + x;
        float reflectance =
            0.3f +
            0.1f * texture.At(y - shifts[t].first, x - shifts[t].second) +
            noise(rng);
        const float cy = y - cloud_y;
        const float cx = x - cloud_x;
        if (cy * cy + cx * cx < 28.0f * 28.0f) {
          reflectance = 0.9f;  // bright cloud, flagged invalid below
          scenes.valid[index] = 0;
        }
        scenes.band[index] = static_cast<uint16_t>(reflectance / kScale);
      }
    }
  }
  return scenes;
}

TEST(Coreg, RecoversKnownShifts) {
  if (!GpuAvailable()) {
    return;
  }
  const std::vector<std::pair<float, float>> shifts = {
      {0.0f, 0.0f}, {3.0f, -5.0f}, {0.5f, 1.25f}, {-7.3f, 2.6f}};
  const TestScenes scenes = MakeScenes(shifts);
  CoregGpuOutput out;
  ASSERT_TRUE(GpuPhaseCorrelation(scenes.Input(0), &out).ok());
  ASSERT_EQ(out.dy.size(), shifts.size());
  EXPECT_NEAR(out.dy[0], 0.0, 1e-4);
  EXPECT_NEAR(out.dx[0], 0.0, 1e-4);
  EXPECT_NEAR(out.peak[0], 1.0f, 1e-3);
  for (size_t t = 1; t < shifts.size(); ++t) {
    const bool integer = shifts[t].first == std::floor(shifts[t].first) &&
                         shifts[t].second == std::floor(shifts[t].second);
    const double tolerance = integer ? 0.1 : 0.2;
    EXPECT_NEAR(out.dy[t], shifts[t].first, tolerance);
    EXPECT_NEAR(out.dx[t], shifts[t].second, tolerance);
    EXPECT_GT(out.peak[t], LowPeakThreshold(kWindow));
  }
}

TEST(Coreg, ReferenceDateGivesZeroShift) {
  if (!GpuAvailable()) {
    return;
  }
  const TestScenes scenes =
      MakeScenes({{0.0f, 0.0f}, {3.0f, -5.0f}, {-2.0f, 4.0f}});
  CoregGpuOutput out;
  ASSERT_TRUE(GpuPhaseCorrelation(scenes.Input(1), &out).ok());
  EXPECT_NEAR(out.dy[1], 0.0, 1e-4);
  EXPECT_NEAR(out.dx[1], 0.0, 1e-4);
  // Relative to date 1 the other dates move by the difference of the shifts.
  EXPECT_NEAR(out.dy[0], -3.0, 0.1);
  EXPECT_NEAR(out.dx[0], 5.0, 0.1);
  EXPECT_NEAR(out.dy[2], -5.0, 0.1);
  EXPECT_NEAR(out.dx[2], 9.0, 0.1);
}

TEST(Coreg, UnrelatedSceneHasLowPeak) {
  if (!GpuAvailable()) {
    return;
  }
  TestScenes scenes = MakeScenes({{0.0f, 0.0f}, {0.0f, 0.0f}});
  std::mt19937 rng(5);
  std::uniform_int_distribution<int> dn(1000, 5000);
  const size_t pixels = static_cast<size_t>(kWindow) * kWindow;
  for (size_t i = 0; i < pixels; ++i) {
    scenes.band[pixels + i] = static_cast<uint16_t>(dn(rng));
    scenes.valid[pixels + i] = 1;
  }
  CoregGpuOutput out;
  ASSERT_TRUE(GpuPhaseCorrelation(scenes.Input(0), &out).ok());
  EXPECT_LT(out.peak[1], LowPeakThreshold(kWindow));
}

TEST(Coreg, RejectsBadGpuInput) {
  if (!GpuAvailable()) {
    return;
  }
  TestScenes scenes = MakeScenes({{0.0f, 0.0f}, {1.0f, 1.0f}});
  CoregGpuOutput out;
  CoregGpuInput in = scenes.Input(0);
  in.window = 100;  // not a power of two
  EXPECT_FALSE(GpuPhaseCorrelation(in, &out).ok());
  in = scenes.Input(2);  // reference outside the dates
  EXPECT_FALSE(GpuPhaseCorrelation(in, &out).ok());
}

TEST(Coreg, FlagRules) {
  EXPECT_EQ(std::string(CoregFlag(0.3f, 0.1f, 1024)), "ok");
  EXPECT_EQ(std::string(CoregFlag(0.3f, 2.5f, 1024)), "large_shift");
  EXPECT_EQ(std::string(CoregFlag(0.001f, 0.1f, 1024)), "low_peak");
  EXPECT_EQ(std::string(CoregFlag(0.001f, 9.0f, 1024)), "low_peak");
  // The noise floor of small windows raises the threshold.
  EXPECT_EQ(std::string(CoregFlag(0.1f, 0.1f, 64)), "low_peak");
  EXPECT_EQ(std::string(CoregFlag(0.1f, 0.1f, 2048)), "ok");
}

TEST(Coreg, WindowResolution) {
  CoregWindow window;
  ASSERT_TRUE(ResolveCoregWindow(1024, -1, -1, 5000, 4000, &window).ok());
  EXPECT_EQ(window.row0, 1988);
  EXPECT_EQ(window.col0, 1488);
  ASSERT_TRUE(ResolveCoregWindow(64, 10, 20, 128, 128, &window).ok());
  EXPECT_EQ(window.row0, 10);
  EXPECT_FALSE(ResolveCoregWindow(1000, -1, -1, 5000, 5000, &window).ok());
  EXPECT_FALSE(ResolveCoregWindow(32, -1, -1, 5000, 5000, &window).ok());
  EXPECT_FALSE(ResolveCoregWindow(8192, -1, -1, 9000, 9000, &window).ok());
  EXPECT_FALSE(ResolveCoregWindow(1024, -1, -1, 128, 128, &window).ok());
  EXPECT_FALSE(ResolveCoregWindow(64, 100, 0, 128, 128, &window).ok());
  EXPECT_TRUE(CheckCoregBand("nir").ok());
  EXPECT_TRUE(CheckCoregBand("red").ok());
  EXPECT_FALSE(CheckCoregBand("green").ok());
}

// Parses `arguments` like the command line; returns the status.
Status Parse(const std::vector<std::string>& arguments, Flags* flags) {
  std::vector<std::string> storage = {"change_detect"};
  storage.insert(storage.end(), arguments.begin(), arguments.end());
  std::vector<char*> argv;
  for (std::string& text : storage) {
    argv.push_back(text.data());
  }
  return ParseFlags(static_cast<int>(argv.size()), argv.data(), flags);
}

TEST(Coreg, FlagParsing) {
  Flags flags;
  ASSERT_TRUE(Parse({"--mode", "coreg", "--input", "x", "--coreg-window",
                     "2048", "--coreg-origin", "10,20", "--coreg-band", "red",
                     "--reference-date", "3"},
                    &flags)
                  .ok());
  EXPECT_EQ(flags.coreg_window, 2048);
  EXPECT_EQ(flags.coreg_origin_row, 10);
  EXPECT_EQ(flags.coreg_origin_col, 20);
  EXPECT_EQ(flags.coreg_band, "red");
  EXPECT_EQ(flags.reference_date, 3);
  const std::vector<std::vector<std::string>> bad = {
      {"--coreg-window", "1000"}, {"--coreg-window", "32"},
      {"--coreg-window", "8192"}, {"--coreg-band", "green"},
      {"--coreg-origin", "5"},    {"--coreg-origin", "-1,2"},
      {"--reference-date", "-2"}};
  for (const std::vector<std::string>& arguments : bad) {
    Flags fresh;
    std::vector<std::string> all = {"--mode", "coreg", "--input", "x"};
    all.insert(all.end(), arguments.begin(), arguments.end());
    EXPECT_FALSE(Parse(all, &fresh).ok());
  }
}

}  // namespace
}  // namespace cd
