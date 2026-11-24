"""Non-GPU helper tooling.

Searches the Earth Search STAC catalogue for Sentinel-2 L2A scenes of one
MGRS tile, keeps the clearest scene per calendar month, downloads the red
(B04), near-infrared (B08) and scene-classification (SCL) bands for a pixel
window, and assembles them into the date-major cube format of docs/FORMATS.md
(manifest.txt, red.u16, nir.u16, scl.u8, previews/). It also saves true-colour
previews from the visual (TCI) asset. The download is resumable: every date is
staged under data/staging/<name>/ and skipped on a re-run once complete.

This script does not use the GPU. Example (development cube):

    .venv/bin/python scripts/fetch_sentinel2.py --name chaco_dev2048 \
        --tile 20KQA --window 3000,2000,2048,2048 --workers 4
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

import numpy as np

import cube_common as cc

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
CSV_FIELDS = ["date", "scene_id", "cloud_cover_pct", "nodata_pct",
              "processing_baseline", "scale", "offset", "stac_offset",
              "relative_orbit", "collection"]
# A scene whose metadata offset is -0.1 holds DN = reflectance * 10000 + 1000.
# If the first percentile of valid red DN is below this fraction of that
# offset (in DN) the stored values cannot contain the offset.
OFFSET_CHECK_FRACTION = 0.8
OFFSET_CHECK_SAMPLE = 1 << 20
PREVIEW_MAX_SIDE = 1024


def parse_args(argv=None):
    """Builds the command line."""
    parser = argparse.ArgumentParser(
        description="Search, download and assemble a Sentinel-2 L2A cube "
        "(non-GPU helper tooling).")
    parser.add_argument("--name", required=True,
                        help="cube name, e.g. chaco_dev2048")
    parser.add_argument("--tile", default="",
                        help="MGRS tile such as 20KQA (grid:code filter)")
    parser.add_argument("--bbox", default="",
                        help="minx,miny,maxx,maxy in lon/lat; with --tile it "
                        "narrows the search, without --tile it also picks "
                        "the window")
    parser.add_argument("--window", default="",
                        help="r0,c0,h,w in tile pixel coordinates "
                        "(default: bbox extent, else the full tile)")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--max-cloud", type=float, default=20.0,
                        help="keep scenes with eo:cloud_cover below this (%%)")
    parser.add_argument("--max-nodata", type=float, default=1.0,
                        help="keep scenes with s2:nodata_pixel_percentage "
                        "below this (%%)")
    parser.add_argument("--per-month", type=int, default=1,
                        help="clearest scenes kept per calendar month")
    parser.add_argument("--orbit", default="auto",
                        help="relative orbit: 'auto' prefers the orbit with "
                        "most candidates, 'any' ignores it, or a number")
    parser.add_argument("--workers", type=int, default=4,
                        help="dates downloaded in parallel")
    parser.add_argument("--read-mode", choices=["auto", "window", "download"],
                        default="auto",
                        help="window: GDAL range reads of the COG (auto "
                        "default; measured faster than download on the "
                        "development machine); download: whole COG via "
                        "requests, then a local read")
    parser.add_argument("--list-only", action="store_true",
                        help="write the scene list CSV and stop")
    parser.add_argument("--out", default="",
                        help="cube directory (default data/cubes/<name>)")
    parser.add_argument("--staging", default="",
                        help="staging directory (default data/staging/<name>)")
    parser.add_argument("--manifests-dir", default="",
                        help="where scene list and manifest copies go "
                        "(default data/manifests)")
    parser.add_argument("--no-full-previews", action="store_true",
                        help="skip the full-resolution first/last previews")
    parser.add_argument("--delete-staging", action="store_true",
                        help="remove the staging directory after assembling")
    parser.add_argument("--stac-url", default=cc.STAC_URL)
    parser.add_argument("--collection", default=cc.COLLECTION)
    parser.add_argument("--fallback-collection",
                        default=cc.FALLBACK_COLLECTION,
                        help="collection used for months that have no "
                        "qualifying scene in --collection ('' disables); the "
                        "per-date scale/offset come from each item")
    return parser.parse_args(argv)


def relative_orbit(item):
    """Relative orbit number parsed from the product URI (..._R110_...)."""
    uri = item.properties.get("s2:product_uri", "")
    match = re.search(r"_R(\d{3})_", uri)
    return match.group(1) if match else "unknown"


def scene_record(item, collection):
    """Flattens the STAC properties used for selection and the CSV."""
    red = item.assets["red"].extra_fields["raster:bands"][0]
    nir = item.assets["nir"].extra_fields["raster:bands"][0]
    if (red.get("scale"), red.get("offset")) != (nir.get("scale"),
                                                 nir.get("offset")):
        raise ValueError(f"{item.id}: red and nir scale/offset differ")
    return {
        "date": item.properties["datetime"][:10],
        "scene_id": item.id,
        "cloud_cover_pct": float(item.properties["eo:cloud_cover"]),
        "nodata_pct": float(item.properties["s2:nodata_pixel_percentage"]),
        "processing_baseline": item.properties.get("s2:processing_baseline",
                                                   ""),
        "relative_orbit": relative_orbit(item),
        "scale": red.get("scale", 0.0001),
        "offset": red.get("offset", -0.1),
        "stac_offset": red.get("offset", -0.1),
        "collection": collection,
        "item": item,
    }


def search_collection(args, collection):
    """Queries one STAC collection; returns records passing the filters."""
    from pystac_client import Client
    query = {}
    if args.tile:
        query["grid:code"] = {"eq": f"MGRS-{args.tile}"}
    kwargs = {}
    if args.bbox:
        kwargs["bbox"] = [float(v) for v in args.bbox.split(",")]
    if not query and not kwargs:
        raise SystemExit("give --tile and/or --bbox")
    client = Client.open(args.stac_url)
    search = client.search(collections=[collection], query=query or None,
                           datetime=f"{args.start}/{args.end}",
                           max_items=5000, **kwargs)
    records = [scene_record(item, collection) for item in search.items()]
    cc.log(f"{collection}: STAC returned {len(records)} items")
    if not args.tile:
        # Without --tile keep the MGRS tile that has the most candidates.
        counts = {}
        for rec in records:
            code = rec["item"].properties.get("grid:code", "")
            counts[code] = counts.get(code, 0) + 1
        best = max(counts, key=counts.get)
        cc.log(f"tiles seen {counts}; using {best}")
        records = [r for r in records
                   if r["item"].properties.get("grid:code") == best]
    return [r for r in records
            if r["cloud_cover_pct"] < args.max_cloud
            and r["nodata_pct"] < args.max_nodata]


def search_scenes(args):
    """Candidates from the primary collection plus the fallback collection."""
    candidates = search_collection(args, args.collection)
    if args.fallback_collection:
        candidates += search_collection(args, args.fallback_collection)
    return candidates


def choose_scenes(candidates, per_month, orbit, primary):
    """Clearest scene(s) per month.

    Ranking inside a month: primary collection first, then the preferred
    relative orbit (scenes with unknown orbit are not penalised), then cloud.
    """
    counts = {}
    for rec in candidates:
        key = rec["relative_orbit"]
        counts[key] = counts.get(key, 0) + 1
    cc.log(f"candidates per relative orbit: {counts}")
    preferred = None
    if orbit == "auto" and counts:
        preferred = max(counts, key=counts.get)
    elif orbit not in ("auto", "any"):
        preferred = orbit
    months = {}
    for rec in candidates:
        months.setdefault(rec["date"][:7], []).append(rec)
    chosen = []
    for month in sorted(months):
        ranked = sorted(months[month],
                        key=lambda r: (
                            r["collection"] != primary,
                            r["relative_orbit"] not in (preferred, "unknown"),
                            r["cloud_cover_pct"], r["nodata_pct"]))
        used_days = set()
        picked = 0
        for rec in ranked:
            if picked >= per_month:
                break
            if rec["date"] in used_days:
                continue
            used_days.add(rec["date"])
            chosen.append(rec)
            picked += 1
    chosen.sort(key=lambda r: r["date"])
    off_orbit = [r for r in chosen
                 if preferred and r["relative_orbit"] not in (preferred,
                                                              "unknown")]
    from_fallback = [r for r in chosen if r["collection"] != primary]
    cc.log(f"selected {len(chosen)} scenes in {len(months)} months; "
           f"preferred orbit {preferred}; {len(off_orbit)} from other orbits; "
           f"{len(from_fallback)} from the fallback collection")
    return chosen


def write_scene_csv(chosen, path):
    """Writes the chosen scene list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in chosen:
            writer.writerow({key: rec[key] for key in CSV_FIELDS})
    cc.log(f"wrote {path}")


def tile_grid(item):
    """(crs, transform tuple, height, width) of the 10 m grid of an item."""
    asset = item.assets["red"].extra_fields
    height, width = asset["proj:shape"]
    crs = asset.get("proj:code") or item.properties["proj:code"]
    return crs, tuple(asset["proj:transform"][:6]), height, width


def window_from_bbox(item, bbox):
    """Pixel window (r0,c0,h,w) of a lon/lat bbox on the tile grid."""
    from rasterio.warp import transform_bounds
    crs, transform, height, width = tile_grid(item)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, *bbox)
    c0 = max(int((minx - transform[2]) // transform[0]), 0)
    c1 = min(int(-(-(maxx - transform[2]) // transform[0])), width)
    r0 = max(int((transform[5] - maxy) // -transform[4]), 0)
    r1 = min(int(-(-(transform[5] - miny) // -transform[4])), height)
    return r0, c0, r1 - r0, c1 - c0


def read_band_window(path_or_url, window):
    """Reads one band window (r0,c0,h,w) as a 2-D array."""
    import rasterio
    from rasterio.windows import Window
    r0, c0, h, w = window
    with rasterio.Env(), rasterio.open(path_or_url) as src:
        return src.read(1, window=Window(c0, r0, w, h))


def read_scl_upsampled(path_or_url, window):
    """Reads SCL (20 m) for a 10 m window and upsamples x2 by nearest."""
    r0, c0, h, w = window
    sr0, sc0 = r0 // 2, c0 // 2
    sr1, sc1 = (r0 + h - 1) // 2 + 1, (c0 + w - 1) // 2 + 1
    coarse = read_band_window(path_or_url, (sr0, sc0, sr1 - sr0, sc1 - sc0))
    fine = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
    top, left = r0 - 2 * sr0, c0 - 2 * sc0
    return fine[top:top + h, left:left + w]


def read_tci(url, window, max_side=None):
    """Reads the visual (TCI) window as (rows, cols, 3) uint8.

    With max_side the read is decimated (GDAL picks an overview) so the long
    side is at most max_side pixels.
    """
    import rasterio
    from rasterio.windows import Window
    r0, c0, h, w = window
    out_h, out_w = h, w
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        out_h, out_w = max(int(h * scale), 1), max(int(w * scale), 1)
    with rasterio.Env(), rasterio.open(url) as src:
        data = src.read(window=Window(c0, r0, w, h), out_shape=(3, out_h,
                                                                 out_w))
    return np.moveaxis(data, 0, -1)


def save_png(array, path, compress_level=3):
    """Atomically writes an RGB uint8 array as PNG."""
    from PIL import Image
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    Image.fromarray(array).save(part, format="PNG",
                                compress_level=compress_level)
    os.replace(part, path)


def write_array(array, path, dtype):
    """Writes a little-endian raw array via a .part file."""
    part = path.with_name(path.name + ".part")
    np.ascontiguousarray(array, dtype=dtype).tofile(part)
    os.replace(part, path)


def effective_offset(red, scl, scale, offset):
    """Checks that the DN values really contain the offset the STAC claims.

    Some Earth Search scenes of the fallback collection (processing baseline
    04.00, sensed after March 2022) declare offset -0.1 while their DN lack
    the +1000 shift (first percentile of valid red DN far below 1000), which
    would give negative reflectance and nonsense NDVI. For those the offset
    actually applying to the stored data is 0. Returns (offset, overridden).
    """
    if offset >= 0:
        return offset, False
    valid = np.isin(scl, cc.SCL_VALID_CLASSES) & (red > 0)
    values = red[valid]
    if values.size == 0:
        return offset, False
    if values.size > OFFSET_CHECK_SAMPLE:
        step = values.size // OFFSET_CHECK_SAMPLE
        values = values[::step]
    first_percentile = float(np.percentile(values, 1))
    if first_percentile < OFFSET_CHECK_FRACTION * (-offset / scale):
        return 0.0, True
    return offset, False


def stage_date(rec, window, mode, staging, progress):
    """Downloads and stores one date; the .json file marks completion."""
    day = rec["date"]
    marker = staging / f"{day}.json"
    r0, c0, h, w = window
    meta_wanted = {"scene_id": rec["scene_id"], "window": list(window)}
    if marker.exists():
        meta = json.loads(marker.read_text())
        if (meta.get("scene_id"), meta.get("window")) == (
                meta_wanted["scene_id"], meta_wanted["window"]):
            return meta, True
    assets = rec["item"].assets
    sources = {}
    cache = staging / "_cache"
    for key in ("red", "nir", "scl"):
        url = assets[key].href
        if mode == "download":
            local = cache / f"{day}_{key}.tif"
            if not local.exists():
                cc.download_file(url, local, progress)
            sources[key] = str(local)
        else:
            sources[key] = url
    tag = f"{day} {rec['scene_id']}"
    red = cc.retry(lambda: read_band_window(sources["red"], window), tag)
    nir = cc.retry(lambda: read_band_window(sources["nir"], window), tag)
    scl = cc.retry(lambda: read_scl_upsampled(sources["scl"], window), tag)
    if mode == "window":
        progress.add(red.nbytes + nir.nbytes + scl.nbytes // 4)
    preview = cc.retry(lambda: read_tci(assets["visual"].href, window,
                                     PREVIEW_MAX_SIDE), tag)
    save_png(preview, staging / "previews" / f"rgb_{day}.png")
    write_array(red, staging / f"{day}_red.u16", "<u2")
    write_array(nir, staging / f"{day}_nir.u16", "<u2")
    write_array(scl, staging / f"{day}_scl.u8", "u1")
    if mode == "download":
        for key in ("red", "nir", "scl"):
            pathlib.Path(sources[key]).unlink(missing_ok=True)
    offset, overridden = effective_offset(red, scl, rec["scale"],
                                          rec["offset"])
    if overridden:
        cc.log(f"WARNING {tag}: STAC offset {rec['offset']} but the DN lack "
               "it; using offset 0 for this date")
    meta = dict(meta_wanted, date=day, scale=rec["scale"], offset=offset,
                stac_offset=rec["stac_offset"], offset_overridden=overridden,
                valid_fraction=cc.scl_valid_fraction(scl), height=h, width=w)
    part = marker.with_name(marker.name + ".part")
    part.write_text(json.dumps(meta))
    os.replace(part, marker)
    return meta, False


def ensure_full_preview(rec, window, staging):
    """Full-resolution TCI window for one date (before/after crops)."""
    target = staging / "previews" / f"rgb_full_{rec['date']}.png"
    if target.exists():
        return
    cc.log(f"full-resolution preview {rec['date']} ...")
    tci = cc.retry(lambda: read_tci(rec["item"].assets["visual"].href, window),
                rec["date"])
    save_png(tci, target, compress_level=1)


def assemble(chosen, window, args, staging, out_dir, grid):
    """Concatenates staged dates into cube files and writes the manifest."""
    crs, transform, _, _ = grid
    r0, c0, h, w = window
    out_dir.mkdir(parents=True, exist_ok=True)
    man = cc.Manifest(directory=out_dir, name=args.name)
    used = sorted({r["collection"] for r in chosen},
                  key=lambda name: name != args.collection)
    man.source = ("Sentinel-2 L2A, Earth Search collection "
                  + " + ".join(used))
    man.crs = crs
    man.geotransform = (transform[2] + c0 * transform[0], transform[0], 0.0,
                        transform[5] + r0 * transform[4], 0.0, transform[4])
    man.height, man.width = h, w
    metas = []
    for index, rec in enumerate(chosen):
        meta = json.loads((staging / f"{rec['date']}.json").read_text())
        metas.append(meta)
        man.dates.append(cc.DateRecord(index, rec["date"], meta["scale"],
                                       meta["offset"], meta["valid_fraction"],
                                       rec["scene_id"]))
    for file_name, suffix in ((man.red_file, "red.u16"),
                              (man.nir_file, "nir.u16"),
                              (man.scl_file, "scl.u8")):
        digest = hashlib.sha256()
        part = out_dir / (file_name + ".part")
        with open(part, "wb") as dest:
            for rec in chosen:
                source = staging / f"{rec['date']}_{suffix}"
                with open(source, "rb") as src:
                    while True:
                        block = src.read(cc.CHUNK_BYTES)
                        if not block:
                            break
                        digest.update(block)
                        dest.write(block)
        os.replace(part, out_dir / file_name)
        man.sha256[file_name] = digest.hexdigest()
        size_mb = (out_dir / file_name).stat().st_size / 1e6
        cc.log(f"assembled {file_name}: {size_mb:.1f} MB")
    previews = out_dir / "previews"
    previews.mkdir(exist_ok=True)
    keep = {f"rgb_{r['date']}.png" for r in chosen}
    keep |= {f"rgb_full_{chosen[0]['date']}.png",
             f"rgb_full_{chosen[-1]['date']}.png"}
    for name in sorted(keep):
        source = staging / "previews" / name
        if source.exists():
            shutil.copyfile(source, previews / name)
    cc.write_manifest(man)
    return man


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    cc.apply_gdal_options()
    manifests_dir = (pathlib.Path(args.manifests_dir) if args.manifests_dir
                     else REPO_DIR / "data" / "manifests")
    out_dir = (pathlib.Path(args.out) if args.out
               else REPO_DIR / "data" / "cubes" / args.name)
    staging = (pathlib.Path(args.staging) if args.staging
               else REPO_DIR / "data" / "staging" / args.name)

    candidates = search_scenes(args)
    if not candidates:
        print("no scenes passed the filters", file=sys.stderr)
        return 1
    chosen = choose_scenes(candidates, args.per_month, args.orbit,
                           args.collection)
    write_scene_csv(chosen, manifests_dir / f"{args.name}_scenes.csv")
    for rec in chosen:
        print(f"  {rec['date']} {rec['scene_id']} cloud "
              f"{rec['cloud_cover_pct']:.1f}% orbit {rec['relative_orbit']} "
              f"{rec['collection']} baseline {rec['processing_baseline']}")
    if args.list_only:
        return 0

    grid = tile_grid(chosen[0]["item"])
    for rec in chosen:
        if tile_grid(rec["item"]) != grid:
            raise SystemExit(f"{rec['scene_id']}: grid differs from the first "
                             "scene; cannot stack")
    _, _, height, width = grid
    if args.window:
        window = cc.parse_window(args.window, height, width)
    elif args.bbox and not args.tile:
        window = window_from_bbox(chosen[0]["item"],
                                  [float(v) for v in args.bbox.split(",")])
    else:
        window = (0, 0, height, width)
    mode = "window" if args.read_mode == "auto" else args.read_mode
    cc.log(f"window r0,c0,h,w = {window}; read mode {mode}; "
           f"{len(chosen)} dates; staging {staging}")
    staging.mkdir(parents=True, exist_ok=True)

    progress = cc.Progress()
    done = 0
    failures = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(stage_date, rec, window, mode, staging,
                               progress): rec for rec in chosen}
        for future in concurrent.futures.as_completed(futures):
            rec = futures[future]
            try:
                _, skipped = future.result()
            except Exception as error:  # Keep going; report at the end.
                failures.append(rec["date"])
                cc.log(f"FAILED {rec['date']}: {error!r}")
                continue
            done += 1
            state = "already staged" if skipped else "staged"
            cc.log(f"{done}/{len(chosen)} {rec['date']} {state}; "
                   f"{progress.bytes / 1e6:.0f} MB so far, "
                   f"{progress.rate():.1f} MB/s average")
    if failures:
        cc.log(f"{len(failures)} dates failed ({failures}); re-run to resume")
        return 1
    for rec in chosen:
        meta = json.loads((staging / f"{rec['date']}.json").read_text())
        rec["offset"] = meta["offset"]
    write_scene_csv(chosen, manifests_dir / f"{args.name}_scenes.csv")
    if not args.no_full_previews:
        for rec in (chosen[0], chosen[-1]):
            ensure_full_preview(rec, window, staging)
    man = assemble(chosen, window, args, staging, out_dir, grid)
    shutil.copyfile(out_dir / "manifest.txt",
                    manifests_dir / f"{args.name}_manifest.txt")
    if args.delete_staging:
        shutil.rmtree(staging)
        cc.log(f"deleted {staging}")
    cc.log(f"cube ready: {out_dir} ({man.num_dates} dates, "
           f"{man.height}x{man.width})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
