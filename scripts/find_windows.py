"""Non-GPU helper tooling.

Ranks pixel windows of a Sentinel-2 tile by forest loss so the development
and demo windows can be chosen by evidence. It builds the tile grid from one
STAC item (``proj:transform``, ``proj:shape``, ``proj:code``), reprojects the
Hansen GFC-2024-v1.12 ``lossyear`` and ``treecover2000`` layers onto that
grid, counts loss pixels in the monitoring years (default 2021-2024) on
forest pixels (tree cover >= 30 %), and prints the best non-overlapping
windows of each requested size together with tile-wide totals.

This script does not use the GPU. Example:

    .venv/bin/python scripts/find_windows.py --tile 20KQA \
        --sizes 2048,5000 --exclude 3000,2000,2048,2048
"""

import argparse
import pathlib
import sys

import numpy as np

import cube_common as cc
import fetch_hansen
import fetch_sentinel2

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
BLOCK = 4


def parse_args(argv=None):
    """Builds the command line."""
    parser = argparse.ArgumentParser(
        description="Rank tile windows by Hansen forest loss "
        "(non-GPU helper tooling).")
    parser.add_argument("--tile", default="20KQA", help="MGRS tile")
    parser.add_argument("--sizes", default="2048,5000",
                        help="comma-separated square window sizes (pixels); "
                        "the full tile is always reported as well")
    parser.add_argument("--monitor-start-year", type=int, default=2021)
    parser.add_argument("--monitor-end-year", type=int, default=2024)
    parser.add_argument("--step", type=int, default=64,
                        help="window origin step in pixels (multiple of 4)")
    parser.add_argument("--top", type=int, default=5,
                        help="windows reported per size")
    parser.add_argument("--exclude", action="append", default=[],
                        help="r0,c0,h,w; candidates overlapping it are "
                        "dropped (repeatable)")
    parser.add_argument("--reference-dir", default="")
    parser.add_argument("--stac-url", default=cc.STAC_URL)
    parser.add_argument("--collection", default=cc.COLLECTION)
    return parser.parse_args(argv)


def tile_item(args):
    """One STAC item of the tile (any date) to read the grid from."""
    from pystac_client import Client
    client = Client.open(args.stac_url)
    search = client.search(collections=[args.collection],
                           query={"grid:code": {"eq": f"MGRS-{args.tile}"}},
                           datetime="2019-01-01/2024-12-31", max_items=1)
    items = list(search.items())
    if not items:
        raise SystemExit(f"no STAC item for tile {args.tile}")
    return items[0]


def block_sums(mask, block):
    """Sums a boolean mask over block x block cells."""
    h, w = mask.shape
    trimmed = mask[:h - h % block, :w - w % block]
    return trimmed.reshape(h // block, block, w // block, block).sum(
        axis=(1, 3), dtype=np.int64)


def integral(coarse):
    """Zero-padded 2-D summed-area table."""
    table = np.zeros((coarse.shape[0] + 1, coarse.shape[1] + 1), np.int64)
    table[1:, 1:] = coarse.cumsum(axis=0).cumsum(axis=1)
    return table


def window_sums(table, size_blocks, step_blocks):
    """Sums for every window origin on a coarse grid; returns (sums, rows)."""
    n_rows = table.shape[0] - 1 - size_blocks
    n_cols = table.shape[1] - 1 - size_blocks
    rows = np.arange(0, n_rows + 1, step_blocks)
    cols = np.arange(0, n_cols + 1, step_blocks)
    r, c = np.meshgrid(rows, cols, indexing="ij")
    sums = (table[r + size_blocks, c + size_blocks] - table[r, c + size_blocks]
            - table[r + size_blocks, c] + table[r, c])
    return sums, rows, cols


def rank_windows(sums, rows, cols, size, args, excluded):
    """Greedy non-overlapping top windows as (count, r0, c0)."""
    sums = sums.copy().astype(np.int64)
    r0s = rows[:, None] * BLOCK
    c0s = cols[None, :] * BLOCK
    for er0, ec0, eh, ew in excluded:
        overlap = ((r0s < er0 + eh) & (r0s + size > er0)
                   & (c0s < ec0 + ew) & (c0s + size > ec0))
        sums[overlap] = -1
    picked = []
    for _ in range(args.top):
        index = np.unravel_index(np.argmax(sums), sums.shape)
        if sums[index] < 0:
            break
        r0, c0 = int(r0s[index[0], 0]), int(c0s[0, index[1]])
        picked.append((int(sums[index]), r0, c0))
        overlap = ((r0s < r0 + size) & (r0s + size > r0)
                   & (c0s < c0 + size) & (c0s + size > c0))
        sums[overlap] = -1
    return picked


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    cc.apply_gdal_options()
    if args.step % BLOCK:
        raise SystemExit("--step must be a multiple of 4")
    item = tile_item(args)
    crs, transform_tuple, height, width = fetch_sentinel2.tile_grid(item)
    from affine import Affine
    transform = Affine(*transform_tuple)
    bounds = fetch_hansen.lonlat_bounds(crs, transform, height, width)
    names = fetch_hansen.tiles_for_bounds(*bounds)
    cc.log(f"tile {args.tile}: {crs}, {height}x{width}, lon/lat bounds "
           f"{tuple(round(v, 3) for v in bounds)}, Hansen tiles {names}")
    reference_dir = (pathlib.Path(args.reference_dir) if args.reference_dir
                     else REPO_DIR / "data" / "reference")
    loss_paths = fetch_hansen.fetch_tiles("lossyear", names, reference_dir)
    cover_paths = fetch_hansen.fetch_tiles("treecover2000", names,
                                           reference_dir)
    lossyear = fetch_hansen.warp_to_grid(loss_paths, crs, transform, height,
                                         width)
    treecover = fetch_hansen.warp_to_grid(cover_paths, crs, transform, height,
                                          width)
    forest = treecover >= fetch_hansen.TREE_COVER_MIN
    first = args.monitor_start_year - 2000
    last = args.monitor_end_year - 2000
    loss = forest & (lossyear >= first) & (lossyear <= last)
    cc.log(f"tile totals: forest(2000) {int(forest.sum())} px "
           f"({forest.mean():.3f}); loss {args.monitor_start_year}-"
           f"{args.monitor_end_year} {int(loss.sum())} px ({loss.mean():.4f} "
           f"of tile, {loss.sum() / max(forest.sum(), 1):.4f} of forest)")
    for year in range(first, last + 1):
        cc.log(f"  {2000 + year}: {int((forest & (lossyear == year)).sum())}"
               " px")
    table = integral(block_sums(loss, BLOCK))
    forest_table = integral(block_sums(forest, BLOCK))
    excluded = [cc.parse_window(text, height, width) for text in args.exclude]
    print("\nsize   rank  r0,c0  loss_px  loss_frac  forest_frac")
    for size in [int(v) for v in args.sizes.split(",")]:
        if size > min(height, width):
            continue
        sums, rows, cols = window_sums(table, size // BLOCK,
                                       args.step // BLOCK)
        ranked = rank_windows(sums, rows, cols, size, args, excluded)
        for rank, (count, r0, c0) in enumerate(ranked, start=1):
            fsum = (forest_table[(r0 + size) // BLOCK, (c0 + size) // BLOCK]
                    - forest_table[r0 // BLOCK, (c0 + size) // BLOCK]
                    - forest_table[(r0 + size) // BLOCK, c0 // BLOCK]
                    + forest_table[r0 // BLOCK, c0 // BLOCK])
            print(f"{size:5d}  {rank:3d}  {r0},{c0}  {count:9d}  "
                  f"{count / size ** 2:.4f}  {fsum / size ** 2:.3f}")
    print(f"{height:5d}  full  0,0  {int(loss.sum()):9d}  {loss.mean():.4f}  "
          f"{forest.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
