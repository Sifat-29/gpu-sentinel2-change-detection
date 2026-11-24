"""Non-GPU helper tooling.

Downloads the Hansen Global Forest Change (GFC-2024-v1.12) ``lossyear`` and
``treecover2000`` tiles that cover a cube, reprojects them (nearest neighbour)
onto the cube grid and writes ``reference_lossyear.u8`` next to the cube:
values 0..24 are the loss year minus 2000 (0 = no loss), and 255 marks pixels
that are excluded from validation (tree cover in 2000 below 30 %, no data, or
loss before the monitoring start year). The manifest gets ``reference_file``,
``reference_source`` and a sha256 line.

This script does not use the GPU. Example:

    .venv/bin/python scripts/fetch_hansen.py \
        --cube data/cubes/chaco_dev2048/manifest.txt
"""

import argparse
import math
import pathlib
import sys

import numpy as np

import cube_common as cc

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
HANSEN_BASE = ("https://storage.googleapis.com/earthenginepartners-hansen/"
               "GFC-2024-v1.12")
HANSEN_DOWNLOAD_PAGE = HANSEN_BASE + "/download.html"
HANSEN_RELEASE = "GFC-2024-v1.12"
TREE_COVER_MIN = 30
EXCLUDED = 255


def parse_args(argv=None):
    """Builds the command line."""
    parser = argparse.ArgumentParser(
        description="Download Hansen forest-loss tiles and write the cube "
        "reference layer (non-GPU helper tooling).")
    parser.add_argument("--cube", required=True,
                        help="path to the cube manifest.txt")
    parser.add_argument("--monitor-start-year", type=int, default=2021,
                        help="loss before this year is excluded (255)")
    parser.add_argument("--reference-dir", default="",
                        help="tile cache (default data/reference)")
    return parser.parse_args(argv)


def tile_name(top, left):
    """Hansen tile name from its north-west corner, e.g. 20S_070W."""
    ns = "N" if top >= 0 else "S"
    ew = "E" if left >= 0 else "W"
    return f"{abs(top):02d}{ns}_{abs(left):03d}{ew}"


def tiles_for_bounds(minx, miny, maxx, maxy):
    """Names of the 10x10 degree tiles intersecting a lon/lat box."""
    tops = range(int(math.ceil(maxy / 10.0)) * 10,
                 int(math.floor(miny / 10.0)) * 10, -10)
    lefts = range(int(math.floor(minx / 10.0)) * 10,
                  int(math.ceil(maxx / 10.0)) * 10, 10)
    return [tile_name(top, left) for top in tops for left in lefts]


def tile_url(layer, name):
    """Download URL of one Hansen layer tile."""
    return f"{HANSEN_BASE}/Hansen_{HANSEN_RELEASE}_{layer}_{name}.tif"


def fetch_tiles(layer, names, directory):
    """Downloads missing tiles (resumable) and returns their local paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    progress = cc.Progress()
    for name in names:
        path = directory / f"Hansen_{HANSEN_RELEASE}_{layer}_{name}.tif"
        if not path.exists():
            cc.log(f"downloading {tile_url(layer, name)}")
            cc.download_file(tile_url(layer, name), path, progress)
            cc.log(f"  {path.stat().st_size / 1e6:.1f} MB, "
                   f"{progress.rate():.1f} MB/s")
        paths.append(path)
    return paths


def warp_to_grid(paths, crs, transform, height, width):
    """Nearest-neighbour reprojection of Hansen tiles onto a target grid.

    Pixels outside every tile stay 0 (treated as no data by the callers).
    """
    import rasterio
    from rasterio.warp import Resampling, reproject
    out = np.zeros((height, width), dtype=np.uint8)
    for path in paths:
        with rasterio.Env(), rasterio.open(path) as src:
            reproject(source=rasterio.band(src, 1), destination=out,
                      dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest, init_dest_nodata=False,
                      num_threads=4)
    return out


def grid_from_manifest(man):
    """Affine transform of the cube grid from the manifest geotransform."""
    from affine import Affine
    gt = man.geotransform
    return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])


def lonlat_bounds(crs, transform, height, width):
    """Lon/lat bounding box of a raster grid."""
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    left, bottom, right, top = array_bounds(height, width, transform)
    return transform_bounds(crs, "EPSG:4326", left, bottom, right, top,
                            densify_pts=21)


def build_reference(lossyear, treecover, start_year):
    """Applies the reference rule to grids already on the cube grid."""
    ref = lossyear.copy()
    excluded = treecover < TREE_COVER_MIN
    # Loss in years 2001..start_year-1 happened before/during training.
    excluded |= (lossyear >= 1) & (lossyear <= start_year - 2001)
    ref[excluded] = EXCLUDED
    return ref


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    cc.apply_gdal_options()
    man = cc.read_manifest(args.cube)
    reference_dir = (pathlib.Path(args.reference_dir) if args.reference_dir
                     else REPO_DIR / "data" / "reference")
    transform = grid_from_manifest(man)
    bounds = lonlat_bounds(man.crs, transform, man.height, man.width)
    names = tiles_for_bounds(*bounds)
    cc.log(f"cube lon/lat bounds {tuple(round(v, 3) for v in bounds)}; "
           f"Hansen tiles {names}")
    loss_paths = fetch_tiles("lossyear", names, reference_dir)
    cover_paths = fetch_tiles("treecover2000", names, reference_dir)
    lossyear = warp_to_grid(loss_paths, man.crs, transform, man.height,
                            man.width)
    treecover = warp_to_grid(cover_paths, man.crs, transform, man.height,
                             man.width)
    ref = build_reference(lossyear, treecover, args.monitor_start_year)
    out_name = "reference_lossyear.u8"
    out_path = man.path(out_name)
    part = out_path.with_name(out_name + ".part")
    ref.tofile(part)
    part.replace(out_path)
    man.reference_file = out_name
    man.reference_source = (
        f"Hansen GFC-2024-v1.12 lossyear, treecover2000>=30, "
        f"loss before {args.monitor_start_year} excluded")
    man.sha256[out_name] = cc.sha256_file(out_path)
    cc.write_manifest(man)
    valid = ref != EXCLUDED
    loss = (ref >= 1) & valid
    cc.log(f"wrote {out_path}: eligible {valid.mean():.3f}, "
           f"loss pixels {int(loss.sum())} "
           f"({loss.sum() / max(valid.sum(), 1):.4f} of eligible)")
    for year in range(args.monitor_start_year - 2000, 25):
        cc.log(f"  lossyear {year}: {int((ref == year).sum())} pixels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
