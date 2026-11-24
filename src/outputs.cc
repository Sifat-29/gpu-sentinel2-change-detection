// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "outputs.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17): portable paths and mkdir -p
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "cube_io.h"

// The vendored PNG writer is a single-header library: exactly one translation
// unit (this one) defines STB_IMAGE_WRITE_IMPLEMENTATION to compile its
// function bodies. The header is third-party code, so its warnings (unused
// functions and parameters under -Wextra) are silenced around the include.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace cd {
namespace {

namespace fs = std::filesystem;

constexpr uint8_t kStatusBreak = static_cast<uint8_t>(PixelStatus::kBreak);
constexpr uint8_t kStatusStable = static_cast<uint8_t>(PixelStatus::kStable);

// printf-style formatting of one number.
std::string Num(const char* format, double value) {
  char text[64];
  std::snprintf(text, sizeof(text), format, value);
  return text;
}

const char* StatusName(uint8_t status) {
  switch (status) {
    case static_cast<uint8_t>(PixelStatus::kStable):
      return "stable";
    case static_cast<uint8_t>(PixelStatus::kBreak):
      return "break";
    default:
      return "insufficient";
  }
}

// Number of blocks of size `factor` needed to cover `length` pixels.
int BlockCount(int length, int factor) {
  return (length + factor - 1) / factor;
}

// Registers one written file in the optional list.
void AddFile(std::vector<OutputFile>* files, const std::string& name,
             const std::string& type, const std::string& description) {
  if (files != nullptr) {
    files->push_back({name, type, description});
  }
}

// ---------------------------------------------------------------------------
// Sample pixel selection.

// Selection tiers in order of preference. Break tiers are filled first
// (losses the reference confirms, other losses, gains), then stable pixels.
enum Tier : int {
  kConfirmedLoss = 0,
  kLoss = 1,
  kGain = 2,
  kStable = 3,
  kNoTier = 4
};
constexpr int kNumTiers = 4;

// True when the reference marks a loss inside the monitoring window.
bool ReferenceConfirms(uint8_t reference, const ValidationWindow& window) {
  if (reference == 0 || reference == 255) {
    return false;
  }
  const int year = 2000 + static_cast<int>(reference);
  return year >= window.first_year && year <= window.last_year;
}

Tier TierOfPixel(const DetectionResult& result,
                 const std::vector<uint8_t>* reference,
                 const ValidationWindow& window, size_t i) {
  if (result.status[i] == kStatusBreak) {
    if (result.magnitude[i] >= 0.0f) {
      return kGain;
    }
    return reference != nullptr && ReferenceConfirms((*reference)[i], window)
               ? kConfirmedLoss
               : kLoss;
  }
  return result.status[i] == kStatusStable ? kStable : kNoTier;
}

// Ranks 0 <= r_0 < r_1 < ... < r_{n-1} < m spread evenly over m candidates
// (the middle of each of n equal parts).
std::vector<uint64_t> StrideRanks(uint64_t m, uint64_t n) {
  std::vector<uint64_t> ranks;
  for (uint64_t j = 0; j < n; ++j) {
    ranks.push_back((2 * j + 1) * m / (2 * n));
  }
  return ranks;
}

}  // namespace

Status EnsureDirectory(const std::string& dir) {
  std::error_code error;
  fs::create_directories(dir, error);
  if (error) {
    return Status::Error("cannot create directory '" + dir +
                         "': " + error.message());
  }
  return Status::Ok();
}

Status WriteTextFile(const std::string& path, const std::string& text) {
  return WriteRawFile(path, text.data(), text.size());
}

int DownsampleFactor(int height, int width, int max_dim) {
  const int longest = std::max(height, width);
  if (max_dim < 1 || longest <= max_dim) {
    return 1;
  }
  return (longest + max_dim - 1) / max_dim;
}

DetectionResult DownsampleDetection(const DetectionResult& result, int max_dim,
                                    int threads) {
  DetectionResult small;
  const int height = result.height;
  const int width = result.width;
  const int factor = DownsampleFactor(height, width, max_dim);
  small.height = BlockCount(height, factor);
  small.width = BlockCount(width, factor);
  small.num_params = result.num_params;
  const size_t small_pixels =
      static_cast<size_t>(small.height) * static_cast<size_t>(small.width);
  small.break_index.assign(small_pixels, -1);
  small.magnitude.assign(small_pixels, 0.0f);
  small.status.assign(small_pixels, 0);
  // Fills the block rows [first, last) of the output. The rows are
  // independent, so workers take disjoint ranges.
  const auto fill_rows = [&](int first, int last) {
    for (int by = first; by < last; ++by) {
      const int row_end = std::min(height, (by + 1) * factor);
      for (int bx = 0; bx < small.width; ++bx) {
        const int col_end = std::min(width, (bx + 1) * factor);
        // Priority class: 0 break, 1 stable, 2 anything else, 3 none yet.
        int best_class = 3;
        int best_break = INT_MAX;
        size_t best = 0;
        for (int r = by * factor; r < row_end; ++r) {
          for (int c = bx * factor; c < col_end; ++c) {
            const size_t i = static_cast<size_t>(r) * width + c;
            const uint8_t status = result.status[i];
            const int cls =
                status == kStatusBreak ? 0 : (status == kStatusStable ? 1 : 2);
            const int break_index = result.break_index[i];
            if (cls < best_class ||
                (cls == 0 && best_class == 0 && break_index < best_break)) {
              best_class = cls;
              best_break = break_index;
              best = i;
            }
          }
        }
        const size_t out = static_cast<size_t>(by) * small.width + bx;
        small.break_index[out] = result.break_index[best];
        small.magnitude[out] = result.magnitude[best];
        small.status[out] = result.status[best];
      }
    }
  };
  const int workers = std::max(1, std::min(threads, small.height));
  std::vector<std::thread> pool;
  for (int w = 1; w < workers; ++w) {
    pool.emplace_back(fill_rows, small.height * w / workers,
                      small.height * (w + 1) / workers);
  }
  fill_rows(0, small.height / workers);  // the calling thread is worker 0
  for (std::thread& thread : pool) {
    thread.join();
  }
  return small;
}

Image RenderQuickDiff(const QuickDiffResult& result, int max_dim) {
  // Colour and block priority (lower rank wins) of the flags 0..3.
  static constexpr uint8_t kColors[4][3] = {
      {0, 0, 0}, {128, 128, 128}, {214, 39, 40}, {44, 160, 44}};
  static constexpr int kRank[4] = {3, 2, 0, 1};
  Image image;
  const int factor = DownsampleFactor(result.height, result.width, max_dim);
  image.height = BlockCount(result.height, factor);
  image.width = BlockCount(result.width, factor);
  image.rgba.assign(
      static_cast<size_t>(image.height) * static_cast<size_t>(image.width) * 4,
      255);
  for (int by = 0; by < image.height; ++by) {
    const int row_end = std::min(result.height, (by + 1) * factor);
    for (int bx = 0; bx < image.width; ++bx) {
      const int col_end = std::min(result.width, (bx + 1) * factor);
      int best_flag = 0;
      int best_rank = 4;
      for (int r = by * factor; r < row_end; ++r) {
        for (int c = bx * factor; c < col_end; ++c) {
          const int flag = std::min<int>(
              3, result.flag[static_cast<size_t>(r) * result.width + c]);
          if (kRank[flag] < best_rank) {
            best_rank = kRank[flag];
            best_flag = flag;
          }
        }
      }
      const size_t out = (static_cast<size_t>(by) * image.width + bx) * 4;
      for (int k = 0; k < 3; ++k) {
        image.rgba[out + k] = kColors[best_flag][k];
      }
    }
  }
  return image;
}

Status WritePng(const std::string& path, const Image& image) {
  if (image.width <= 0 || image.height <= 0 ||
      image.rgba.size() != static_cast<size_t>(image.width) *
                               static_cast<size_t>(image.height) * 4) {
    return Status::Error("cannot write PNG '" + path + "': invalid image");
  }
  const int ok = stbi_write_png(path.c_str(), image.width, image.height, 4,
                                image.rgba.data(), image.width * 4);
  if (ok == 0) {
    return Status::Error("cannot write PNG '" + path + "'");
  }
  return Status::Ok();
}

Status WriteDetectionMaps(const std::string& dir, const DetectionResult& result,
                          std::vector<OutputFile>* files) {
  struct Map {
    const char* name;
    const char* type;
    const char* description;
    const void* data;
    size_t bytes;
  };
  const size_t n = static_cast<size_t>(result.height) * result.width;
  const std::vector<Map> maps = {
      {"break_index.i16", "i16", "break date index per pixel, -1 = none",
       result.break_index.data(), n * sizeof(int16_t)},
      {"magnitude.f32", "f32", "signed mean residual of the break run",
       result.magnitude.data(), n * sizeof(float)},
      {"status.u8", "u8", "0 stable, 1 break, 2 insufficient data",
       result.status.data(), n * sizeof(uint8_t)},
      {"rmse.f32", "f32", "floored training RMSE", result.rmse.data(),
       n * sizeof(float)},
      {"train_obs.u16", "u16", "valid training observations",
       result.train_obs.data(), n * sizeof(uint16_t)}};
  for (const Map& map : maps) {
    const Status status =
        WriteRawFile(dir + "/" + map.name, map.data, map.bytes);
    if (!status.ok()) {
      return status;
    }
    AddFile(files, map.name, map.type, map.description);
  }
  if (!result.coefficients.empty()) {
    const Status status =
        WriteRawFile(dir + "/coefficients.f32", result.coefficients.data(),
                     result.coefficients.size() * sizeof(float));
    if (!status.ok()) {
      return status;
    }
    AddFile(files, "coefficients.f32", "f32",
            "model coefficients, coefficient-major (p x H x W)");
  }
  return Status::Ok();
}

Status WriteQuickDiffOutputs(const std::string& dir,
                             const QuickDiffResult& result, int png_max_dim,
                             std::vector<OutputFile>* files) {
  Status status = WriteRawFile(dir + "/quick_diff.f32", result.diff.data(),
                               result.diff.size() * sizeof(float));
  if (!status.ok()) {
    return status;
  }
  AddFile(files, "quick_diff.f32", "f32", "NDVI difference (date b - date a)");
  status = WriteRawFile(dir + "/quick_diff_flag.u8", result.flag.data(),
                        result.flag.size());
  if (!status.ok()) {
    return status;
  }
  AddFile(files, "quick_diff_flag.u8", "u8",
          "0 no data, 1 no change, 2 loss, 3 gain");
  status =
      WritePng(dir + "/quick_diff.png", RenderQuickDiff(result, png_max_dim));
  if (!status.ok()) {
    return status;
  }
  AddFile(files, "quick_diff.png", "png",
          "quick-diff map (loss red, gain green, no change grey, no data "
          "black)");
  return Status::Ok();
}

std::string BuildResultManifest(const ResultManifestInput& input) {
  const ModelParams& p = input.params;
  std::string classes;
  for (int c = 0; c < 16; ++c) {
    if ((p.valid_class_mask >> c) & 1u) {
      classes += (classes.empty() ? "" : ",") + std::to_string(c);
    }
  }
  std::string text = "change_detect_result 1\n";
  const auto add = [&text](const std::string& key, const std::string& value) {
    text += key + " " + value + "\n";
  };
  add("cube_manifest", input.cube_manifest);
  add("name", input.name);
  add("height", std::to_string(input.height));
  add("width", std::to_string(input.width));
  add("num_dates", std::to_string(input.num_dates));
  add("backend", input.backend);
  add("harmonics", std::to_string(p.harmonics));
  add("trend", p.hold_trend ? "hold" : "extrapolate");
  add("train_years", Num("%g", p.train_years));
  add("min_train_obs", std::to_string(p.min_train_obs));
  add("threshold", Num("%g", p.threshold));
  add("consecutive", std::to_string(p.consecutive));
  add("rmse_floor", Num("%g", p.rmse_floor));
  add("ridge", Num("%g", p.ridge));
  add("losses_only", p.losses_only ? "1" : "0");
  add("valid_classes", classes);
  add("cleanup", input.cleanup ? "1" : "0");
  add("reference_file", input.reference_file);
  const DateTable& d = input.dates;
  add("num_train_dates", std::to_string(d.num_train_dates));
  if (!d.year.empty()) {
    const size_t first =
        std::min(static_cast<size_t>(d.num_train_dates), d.year.size() - 1);
    add("first_monitor_year", std::to_string(d.year[first]));
    add("last_year", std::to_string(d.year.back()));
  }
  for (size_t t = 0; t < d.iso.size(); ++t) {
    // date <index> <iso> <days since 1970> <scale> <offset> <valid> <scene>;
    // the last two are unknown for a result, written as -1 and "-".
    add("date", std::to_string(t) + " " + d.iso[t] + " " +
                    std::to_string(d.day[t]) + " " + Num("%g", d.scale[t]) +
                    " " + Num("%g", d.offset[t]) + " -1 -");
  }
  for (const OutputFile& file : input.files) {
    add("file", file.name + " " + file.type + " " + file.description);
  }
  return text;
}

Status WriteResultManifest(const std::string& path,
                           const ResultManifestInput& input) {
  return WriteTextFile(path, BuildResultManifest(input));
}

std::vector<std::pair<int, int>> SelectSamplePixels(
    const DetectionResult& result, const std::vector<uint8_t>* reference,
    const ValidationWindow& window, int count) {
  std::vector<std::pair<int, int>> picked;
  const size_t pixels =
      static_cast<size_t>(result.height) * static_cast<size_t>(result.width);
  if (count <= 0 || pixels == 0 || result.status.size() != pixels ||
      result.magnitude.size() != pixels ||
      (reference != nullptr && reference->size() != pixels)) {
    return picked;
  }
  // Pass 1: how many candidates does each tier have?
  uint64_t available[kNumTiers] = {0, 0, 0, 0};
  for (size_t i = 0; i < pixels; ++i) {
    const Tier tier = TierOfPixel(result, reference, window, i);
    if (tier != kNoTier) {
      ++available[tier];
    }
  }
  // Quotas: half (rounded up) from breaks, the rest from stable pixels; either
  // side makes up for a shortage on the other.
  const uint64_t wanted = static_cast<uint64_t>(count);
  const uint64_t break_available =
      available[kConfirmedLoss] + available[kLoss] + available[kGain];
  uint64_t take_break = std::min((wanted + 1) / 2, break_available);
  const uint64_t take_stable =
      std::min(wanted - take_break, available[kStable]);
  take_break = std::min(wanted - take_stable, break_available);
  uint64_t quota[kNumTiers] = {0, 0, 0, take_stable};
  uint64_t remaining = take_break;
  for (int tier = kConfirmedLoss; tier <= kGain; ++tier) {
    quota[tier] = std::min(remaining, available[tier]);
    remaining -= quota[tier];
  }
  std::vector<uint64_t> ranks[kNumTiers];
  size_t cursor[kNumTiers] = {0, 0, 0, 0};
  uint64_t seen[kNumTiers] = {0, 0, 0, 0};
  for (int tier = 0; tier < kNumTiers; ++tier) {
    ranks[tier] = StrideRanks(available[tier], quota[tier]);
  }
  // Pass 2: walk the pixels in row-major order and keep the ranked ones.
  for (size_t i = 0; i < pixels; ++i) {
    const Tier tier = TierOfPixel(result, reference, window, i);
    if (tier == kNoTier) {
      continue;
    }
    if (cursor[tier] < ranks[tier].size() &&
        seen[tier] == ranks[tier][cursor[tier]]) {
      picked.emplace_back(static_cast<int>(i / result.width),
                          static_cast<int>(i % result.width));
      ++cursor[tier];
    }
    ++seen[tier];
  }
  std::sort(picked.begin(), picked.end());
  return picked;
}

Status ValidateSamplePixels(const std::vector<std::pair<int, int>>& pixels,
                            int height, int width) {
  for (const std::pair<int, int>& pixel : pixels) {
    if (pixel.first < 0 || pixel.first >= height || pixel.second < 0 ||
        pixel.second >= width) {
      return Status::Error(
          "--sample-pixels: pixel (" + std::to_string(pixel.first) + ", " +
          std::to_string(pixel.second) + ") is outside the " +
          std::to_string(height) + " x " + std::to_string(width) + " image");
    }
  }
  return Status::Ok();
}

std::string BuildPixelStatsCsv(const DetectionResult& result,
                               const std::string& backend,
                               const std::vector<PixelTrajectory>& trajectories,
                               const DateTable& dates) {
  const int p = result.num_params;
  std::string text =
      "row,col,backend,status,break_index,break_date,magnitude,rmse,n_train";
  for (int c = 0; c < p; ++c) {
    text += ",coef" + std::to_string(c);
  }
  text += "\n";
  const size_t plane = static_cast<size_t>(result.height) * result.width;
  const auto date_of = [&dates](int index) {
    return index >= 0 && static_cast<size_t>(index) < dates.iso.size()
               ? dates.iso[static_cast<size_t>(index)]
               : std::string();
  };
  for (const PixelTrajectory& t : trajectories) {
    const size_t i = static_cast<size_t>(t.row) * result.width + t.col;
    // Row 1: what the run's backend produced.
    const int break_index = result.break_index[i];
    text += std::to_string(t.row) + "," + std::to_string(t.col) + "," +
            backend + "," + StatusName(result.status[i]) + "," +
            std::to_string(break_index) + "," + date_of(break_index) + "," +
            Num("%.7g", result.magnitude[i]) + "," +
            Num("%.7g", result.rmse[i]) + "," +
            std::to_string(result.train_obs[i]);
    for (int c = 0; c < p; ++c) {
      text += ",";
      if (!result.coefficients.empty()) {
        text += Num("%.9g",
                    result.coefficients[static_cast<size_t>(c) * plane + i]);
      }
    }
    text += "\n";
    // Row 2: the FP64 recomputation of the same pixel.
    text += std::to_string(t.row) + "," + std::to_string(t.col) +
            ",cpu-fp64-recompute," + StatusName(t.status) + "," +
            std::to_string(t.break_index) + "," + date_of(t.break_index) + "," +
            Num("%.7g", t.magnitude) + "," + Num("%.7g", t.rmse) + "," +
            std::to_string(t.n_train);
    for (int c = 0; c < p; ++c) {
      text += "," + Num("%.12g", t.coef[c]);
    }
    text += "\n";
  }
  return text;
}

const std::string& TrajectoriesCsvHeader() {
  static const std::string header =
      "row,col,date_index,date,day,tau,ndvi,valid,in_training,predicted,z,"
      "is_break_run";
  return header;
}

std::string BuildTrajectoriesCsv(
    const std::vector<PixelTrajectory>& trajectories, const DateTable& dates) {
  std::string text = TrajectoriesCsvHeader() + "\n";
  for (const PixelTrajectory& t : trajectories) {
    // A failed fit (insufficient data) has no prediction or z-score.
    const bool fitted =
        t.status != static_cast<uint8_t>(PixelStatus::kInsufficientData);
    for (size_t k = 0; k < t.ndvi.size() && k < dates.iso.size(); ++k) {
      const bool valid = t.ndvi[k] > static_cast<double>(kInvalidNdvi) + 0.5;
      text += std::to_string(t.row) + "," + std::to_string(t.col) + "," +
              std::to_string(k) + "," + dates.iso[k] + "," +
              std::to_string(dates.day[k]) + "," + Num("%.6f", dates.tau[k]) +
              "," + (valid ? Num("%.6f", t.ndvi[k]) : std::string()) + "," +
              (valid ? "1" : "0") + "," +
              (static_cast<int>(k) < dates.num_train_dates ? "1" : "0") + "," +
              (fitted ? Num("%.6f", t.predicted[k]) : std::string()) + "," +
              (fitted && valid ? Num("%.6f", t.z[k]) : std::string()) + "," +
              (t.in_break_run[k] != 0 ? "1" : "0") + "\n";
    }
  }
  return text;
}

}  // namespace cd
