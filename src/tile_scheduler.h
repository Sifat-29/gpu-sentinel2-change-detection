// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Tile planning: how many full-width rows go into one tile and how the image
// is split into row bands.
#ifndef SRC_TILE_SCHEDULER_H_
#define SRC_TILE_SCHEDULER_H_

#include <cstddef>
#include <utility>
#include <vector>

namespace cd {

// Largest cube input of one tile that every backend plans with, so that the
// GPU and CPU backends read the cube with the same request sizes.
constexpr size_t kMaxTileInputBytes = 64ull << 20;

// Result of PlanTiles.
struct TilePlan {
  int tile_rows = 0;
  int num_tiles = 0;
  // True when a positive `requested_rows` was reduced to fit the memory
  // budget (the requested value is then larger than tile_rows).
  bool reduced_for_memory = false;
  // Bytes of cube input (red + nir + scl, all dates) in one full tile:
  // width * tile_rows * num_dates * 5.
  size_t input_bytes_per_tile = 0;
};

// Chooses the tile height. With automatic sizing (`requested_rows` = 0)
// rows = floor(0.5 * free_bytes / (slots * width * device_bytes_per_pixel)),
// further capped so that one tile's input is at most `max_tile_input_bytes`,
// clamped to [1, height] and rounded down to a multiple of 8 when it is at
// least 8. A positive `requested_rows` (clamped to `height`) is honoured
// except that, when `free_bytes` is known (> 0), it is reduced to the same
// half-of-free-memory limit (rounded the same way); reduced_for_memory is then
// set. The input cap does not apply to requested rows.
// num_tiles = ceil(height / tile_rows). A `slots` or `device_bytes_per_pixel`
// of 0 is treated as "no memory limit from that term".
TilePlan PlanTiles(int height, int width, int num_dates, int requested_rows,
                   size_t free_bytes, int slots, size_t device_bytes_per_pixel,
                   size_t max_tile_input_bytes = 256ull << 20);

// Splits [0, height) into consecutive (row0, rows) bands of `tile_rows` rows;
// the last band may be shorter. A non-positive `tile_rows` gives one band
// covering the whole image; a non-positive `height` gives no bands.
std::vector<std::pair<int, int>> TileRanges(int height, int tile_rows);

}  // namespace cd

#endif  // SRC_TILE_SCHEDULER_H_
