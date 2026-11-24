// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Files written by the program (format: docs/FORMATS.md, "Detect outputs"): raw
// result maps, the result manifest, PNG images with block-priority
// downsampling, sample-pixel selection, pixel_stats.csv and trajectories.csv.
#ifndef SRC_OUTPUTS_H_
#define SRC_OUTPUTS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "status.h"
#include "types.h"

namespace cd {

// One line "file <name> <type> <description>" of result_manifest.txt.
struct OutputFile {
  std::string name;
  std::string type;  // i16, f32, u8, u16, png, csv or txt
  std::string description;
};

// An 8-bit RGBA image, row-major.
struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgba;  // 4 * width * height bytes
};

// Creates `dir` and its parents (an existing directory is fine).
Status EnsureDirectory(const std::string& dir);

// Writes `text` to `path`, truncating the file.
Status WriteTextFile(const std::string& path, const std::string& text);

// Block size of the PNG downsampling: the smallest integer f >= 1 such that
// max(height, width) / f <= max_dim (rounded up). max_dim < 1 gives 1.
int DownsampleFactor(int height, int width, int max_dim);

// Block-priority downsampling of the detection maps so that the long side is
// <= max_dim. Each f x f block (f = DownsampleFactor) is represented by the
// break_index, magnitude and status of one of its pixels, chosen by priority:
// a break pixel wins over a stable pixel, which wins over an insufficient-data
// pixel; among break pixels the earliest break date wins (ties: first in
// row-major order). The returned result has ceil(height / f) x ceil(width / f)
// pixels and only those three maps; colouring it (CpuColorizeChangeMap or
// GpuColorizeChangeMap) gives the colour of the chosen pixel of each block,
// without colouring the full-resolution map. The block rows are split over
// `threads` worker threads (1 = the calling thread only); the result does not
// depend on it.
DetectionResult DownsampleDetection(const DetectionResult& result, int max_dim,
                                    int threads = 1);

// Renders a quick-diff result as an image (loss red, gain green, no change
// grey, no data black) downsampled to max_dim. Block priority: loss > gain >
// no change > no data.
Image RenderQuickDiff(const QuickDiffResult& result, int max_dim);

// Writes `image` as a PNG file.
Status WritePng(const std::string& path, const Image& image);

// Writes break_index.i16, magnitude.f32, status.u8, rmse.f32, train_obs.u16
// and, when result.coefficients is not empty, coefficients.f32 into `dir`.
// Appends one entry per file to `files`.
Status WriteDetectionMaps(const std::string& dir, const DetectionResult& result,
                          std::vector<OutputFile>* files);

// Writes quick_diff.f32, quick_diff_flag.u8 and quick_diff.png into `dir`;
// appends the entries to `files` (optional).
Status WriteQuickDiffOutputs(const std::string& dir,
                             const QuickDiffResult& result, int png_max_dim,
                             std::vector<OutputFile>* files);

// Everything result_manifest.txt records.
struct ResultManifestInput {
  std::string cube_manifest;  // manifest path or "synthetic"
  std::string name;
  int height = 0;
  int width = 0;
  int num_dates = 0;
  std::string backend;  // BackendName of the run
  ModelParams params;   // resolved
  bool cleanup = false;
  std::string reference_file = "none";  // path or "none"
  DateTable dates;                      // iso, day, scale, offset, years
  std::vector<OutputFile> files;
};

// The text of result_manifest.txt (grammar of the cube manifest with the
// first line "change_detect_result 1"; docs/FORMATS.md). Keys: cube_manifest,
// name, height, width, num_dates, backend, harmonics, train_years,
// min_train_obs, threshold, consecutive, rmse_floor, ridge, losses_only (0/1),
// valid_classes (comma list), cleanup (0/1), reference_file, num_train_dates,
// first_monitor_year, last_year, then the date lines and the file lines.
std::string BuildResultManifest(const ResultManifestInput& input);

Status WriteResultManifest(const std::string& path,
                           const ResultManifestInput& input);

// Chooses the pixels whose trajectories are exported, deterministically: half
// (rounded up) from break pixels, the rest from stable pixels, each by a
// fixed stride over the row-major candidate list. Break candidates are taken
// in this order of preference: greenness losses that the reference confirms
// (only when `reference` is not null), other losses, gains. Returns at most
// `count` pixels sorted row-major as (row, col); fewer when the maps hold
// fewer candidates.
std::vector<std::pair<int, int>> SelectSamplePixels(
    const DetectionResult& result, const std::vector<uint8_t>* reference,
    const ValidationWindow& window, int count);

// Errors when an explicit (row, col) pixel is outside height x width.
Status ValidateSamplePixels(const std::vector<std::pair<int, int>>& pixels,
                            int height, int width);

// Text of pixel_stats.csv: header, then two rows per trajectory (the run's
// own backend, then "cpu-fp64-recompute"). Coefficients of the run's row are
// blank unless the result holds them.
std::string BuildPixelStatsCsv(const DetectionResult& result,
                               const std::string& backend,
                               const std::vector<PixelTrajectory>& trajectories,
                               const DateTable& dates);

// Text of trajectories.csv: header, then one row per sample pixel and date.
std::string BuildTrajectoriesCsv(
    const std::vector<PixelTrajectory>& trajectories, const DateTable& dates);

// The header line of trajectories.csv (without newline).
const std::string& TrajectoriesCsvHeader();

}  // namespace cd

#endif  // SRC_OUTPUTS_H_
