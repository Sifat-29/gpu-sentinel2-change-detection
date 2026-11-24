// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <cstddef>
#include <utility>
#include <vector>

#include "test_framework.h"
#include "tile_scheduler.h"

namespace cd {
namespace {

constexpr size_t kMiB = 1ull << 20;
constexpr size_t kGiB = 1ull << 30;

TEST(Tile, RequestedRowsWinAndAreClamped) {
  const TilePlan plan = PlanTiles(1000, 500, 48, 100, 0, 3, 100);
  EXPECT_EQ(plan.tile_rows, 100);
  EXPECT_EQ(plan.num_tiles, 10);
  EXPECT_EQ(plan.input_bytes_per_tile, 100u * 500u * 48u * 5u);

  // Not a multiple of 8 and larger than the input cap: still honoured when
  // the memory budget allows it.
  const TilePlan odd = PlanTiles(1000, 4000, 48, 77, 1 * kGiB, 3, 100);
  EXPECT_EQ(odd.tile_rows, 77);
  EXPECT_EQ(odd.num_tiles, 13);
  EXPECT_FALSE(odd.reduced_for_memory);

  const TilePlan clamped = PlanTiles(300, 500, 48, 5000, 0, 3, 100);
  EXPECT_EQ(clamped.tile_rows, 300);
  EXPECT_EQ(clamped.num_tiles, 1);
}

TEST(Tile, RequestedRowsAreReducedToHalfOfFreeMemory) {
  // 3 slots, W = 5000, 100 B per pixel: 1500 rows need 3 * 1500 * 5000 * 100
  // = 2145 MiB, more than half of 3500 MiB free. The limit is
  // floor(1750 MiB / (3 * 5000 * 100)) = 1223 rows -> 1216 (multiple of 8).
  const TilePlan plan = PlanTiles(5000, 5000, 70, 1500, 3500 * kMiB, 3, 100);
  EXPECT_TRUE(plan.reduced_for_memory);
  EXPECT_EQ(plan.tile_rows, 1216);
  EXPECT_EQ(plan.num_tiles, 5);

  // A request that fits is left alone, and so is one without known memory.
  const TilePlan fits = PlanTiles(5000, 5000, 70, 1000, 3500 * kMiB, 3, 100);
  EXPECT_FALSE(fits.reduced_for_memory);
  EXPECT_EQ(fits.tile_rows, 1000);
  const TilePlan unknown = PlanTiles(5000, 5000, 70, 1500, 0, 3, 100);
  EXPECT_FALSE(unknown.reduced_for_memory);
  EXPECT_EQ(unknown.tile_rows, 1500);

  // Less than one row of memory still leaves one row.
  const TilePlan starved = PlanTiles(100, 1000, 10, 50, 10, 3, 40);
  EXPECT_TRUE(starved.reduced_for_memory);
  EXPECT_EQ(starved.tile_rows, 1);
}

TEST(Tile, MemoryLimitUsesHalfOfFreeMemoryForAllSlots) {
  // 4 GiB free, 3 slots, W = 10000, 40 B per pixel:
  // floor(0.5 * 4 GiB / (3 * 10000 * 40)) = 1789 rows -> 1784 (multiple of 8).
  // The input cap is raised out of the way.
  const TilePlan plan =
      PlanTiles(20000, 10000, 4, 0, 4 * kGiB, 3, 40, 100 * kGiB);
  EXPECT_EQ(plan.tile_rows, 1784);
  EXPECT_EQ(plan.num_tiles, 12);
  EXPECT_EQ(plan.input_bytes_per_tile, 1784u * 10000u * 4u * 5u);
}

TEST(Tile, InputCapLimitsRows) {
  // Plenty of memory; one row needs 4000 * 48 * 5 = 960000 bytes, so the
  // default 256 MiB cap allows floor(268435456 / 960000) = 279 -> 272 rows.
  const TilePlan plan = PlanTiles(10000, 4000, 48, 0, 100 * kGiB, 3, 40);
  EXPECT_EQ(plan.tile_rows, 272);
  EXPECT_LE(plan.input_bytes_per_tile, 256 * kMiB);
  EXPECT_EQ(plan.num_tiles, 37);
}

TEST(Tile, ClampsToImageAndRoundsToEight) {
  // Everything fits, so rows = H = 100, rounded down to a multiple of 8.
  const TilePlan all = PlanTiles(100, 64, 8, 0, 100 * kGiB, 3, 40);
  EXPECT_EQ(all.tile_rows, 96);
  EXPECT_EQ(all.num_tiles, 2);

  // Fewer than 8 rows fit: no rounding, but at least one row.
  const TilePlan few =
      PlanTiles(100, 1000, 10, 0, 3 * 1000 * 40 * 2 * 5, 3, 40);
  EXPECT_EQ(few.tile_rows, 5);
  EXPECT_EQ(few.num_tiles, 20);

  const TilePlan starved = PlanTiles(100, 1000, 10, 0, 10, 3, 40);
  EXPECT_EQ(starved.tile_rows, 1);
  EXPECT_EQ(starved.num_tiles, 100);
}

TEST(Tile, RangesCoverEveryRowOnce) {
  const std::vector<std::pair<int, int>> ranges = TileRanges(1000, 384);
  ASSERT_EQ(ranges.size(), 3u);
  EXPECT_EQ(ranges[0].first, 0);
  EXPECT_EQ(ranges[0].second, 384);
  EXPECT_EQ(ranges[1].first, 384);
  EXPECT_EQ(ranges[1].second, 384);
  EXPECT_EQ(ranges[2].first, 768);
  EXPECT_EQ(ranges[2].second, 232);

  const std::vector<std::pair<int, int>> single = TileRanges(10, 100);
  ASSERT_EQ(single.size(), 1u);
  EXPECT_EQ(single[0].second, 10);
  EXPECT_EQ(TileRanges(10, 0).size(), 1u);
  EXPECT_TRUE(TileRanges(0, 8).empty());

  const TilePlan plan = PlanTiles(1000, 100, 8, 384, 0, 3, 1);
  EXPECT_EQ(static_cast<size_t>(plan.num_tiles),
            TileRanges(1000, plan.tile_rows).size());
}

}  // namespace
}  // namespace cd
