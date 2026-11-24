// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "tile_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace cd {
namespace {

// Bytes of input per pixel and date: 2 (red) + 2 (nir) + 1 (scl).
constexpr uint64_t kInputBytesPerObservation = 5;

}  // namespace

TilePlan PlanTiles(int height, int width, int num_dates, int requested_rows,
                   size_t free_bytes, int slots, size_t device_bytes_per_pixel,
                   size_t max_tile_input_bytes) {
  TilePlan plan;
  const int safe_height = std::max(height, 1);
  const uint64_t row_input_bytes =
      static_cast<uint64_t>(std::max(width, 1)) *
      static_cast<uint64_t>(std::max(num_dates, 1)) * kInputBytesPerObservation;
  int64_t rows = 0;
  // Memory limit: three (or `slots`) tiles in flight use at most half of the
  // free device memory.
  const uint64_t denominator = static_cast<uint64_t>(std::max(slots, 0)) *
                               static_cast<uint64_t>(std::max(width, 1)) *
                               static_cast<uint64_t>(device_bytes_per_pixel);
  if (requested_rows > 0) {
    rows = std::min(requested_rows, safe_height);
    if (denominator > 0 && free_bytes > 0) {
      int64_t fitting = static_cast<int64_t>(free_bytes / (2 * denominator));
      fitting = std::max<int64_t>(fitting, 1);
      if (fitting >= 8) {
        fitting -= fitting % 8;
      }
      if (rows > fitting) {
        rows = fitting;
        plan.reduced_for_memory = true;
      }
    }
  } else {
    rows = safe_height;
    if (denominator > 0) {
      rows = std::min<int64_t>(
          rows, static_cast<int64_t>(free_bytes / (2 * denominator)));
    }
    // Input-size cap: keeps pinned host buffers and pipeline granularity sane.
    rows = std::min<int64_t>(
        rows, static_cast<int64_t>(max_tile_input_bytes / row_input_bytes));
    rows = std::max<int64_t>(rows, 1);
    if (rows >= 8) {
      rows -= rows % 8;
    }
  }
  plan.tile_rows = static_cast<int>(rows);
  plan.num_tiles = (safe_height + plan.tile_rows - 1) / plan.tile_rows;
  plan.input_bytes_per_tile =
      static_cast<size_t>(row_input_bytes * static_cast<uint64_t>(rows));
  return plan;
}

std::vector<std::pair<int, int>> TileRanges(int height, int tile_rows) {
  std::vector<std::pair<int, int>> ranges;
  if (height <= 0) {
    return ranges;
  }
  if (tile_rows <= 0 || tile_rows > height) {
    tile_rows = height;
  }
  for (int row0 = 0; row0 < height; row0 += tile_rows) {
    ranges.emplace_back(row0, std::min(tile_rows, height - row0));
  }
  return ranges;
}

}  // namespace cd
