"""Non-GPU helper tooling.

Crops an existing cube (docs/FORMATS.md) in space and/or time into a new
cube: ``--window r0,c0,h,w`` selects pixels, ``--dates i0:i1`` selects date
indices (Python slice, end exclusive). The raw arrays are streamed one date
at a time through memory maps, so a 43 GB source is never loaded whole. The
sha256 lines and per-date valid fractions are recomputed; the reference layer
is cropped with the same window and the previews are cropped proportionally
(the 1024 px previews) or exactly (the full-resolution ones).

This script does not use the GPU. Example (demo cube from the full tile):

    .venv/bin/python scripts/crop_cube.py \
        --input data/cubes/chaco_full/manifest.txt \
        --window 0,5504,5000,5000 --out data/cubes/chaco_demo5000
"""

import argparse
import hashlib
import os
import pathlib
import sys

import numpy as np

import cube_common as cc

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    """Builds the command line."""
    parser = argparse.ArgumentParser(
        description="Crop a cube by window and/or date range "
        "(non-GPU helper tooling).")
    parser.add_argument("--input", required=True, help="source manifest.txt")
    parser.add_argument("--out", required=True, help="output cube directory")
    parser.add_argument("--window", default="",
                        help="r0,c0,h,w in source pixels (default: all)")
    parser.add_argument("--dates", default="",
                        help="i0:i1 date index slice (default: all)")
    parser.add_argument("--name", default="",
                        help="cube name (default: output directory name)")
    parser.add_argument("--manifests-dir", default="",
                        help="also copy the manifest here "
                        "(default data/manifests; 'none' disables)")
    return parser.parse_args(argv)


def parse_dates(text, count):
    """Parses 'i0:i1' into a validated (start, stop) pair."""
    if not text:
        return 0, count
    start_text, _, stop_text = text.partition(":")
    start = int(start_text) if start_text else 0
    stop = int(stop_text) if stop_text else count
    if not 0 <= start < stop <= count:
        raise ValueError(f"--dates {text} outside 0..{count}")
    return start, stop


def crop_png(source, target, box_fraction, exact_box=None):
    """Crops a PNG by fractional box (l, t, r, b in 0..1) or exact pixels."""
    from PIL import Image
    with Image.open(source) as image:
        if exact_box is not None:
            box = exact_box
        else:
            width, height = image.size
            left, top, right, bottom = box_fraction
            x0, y0 = int(round(left * width)), int(round(top * height))
            x1, y1 = int(round(right * width)), int(round(bottom * height))
            box = (x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))
        image.crop(box).save(target, format="PNG", compress_level=3)


def crop_previews(src_man, dst_dir, window, kept_days):
    """Copies/crops previews for the kept dates; returns how many were made."""
    r0, c0, h, w = window
    src_previews = src_man.directory / "previews"
    if not src_previews.is_dir():
        return 0
    out = dst_dir / "previews"
    out.mkdir(parents=True, exist_ok=True)
    fraction = (c0 / src_man.width, r0 / src_man.height,
                (c0 + w) / src_man.width, (r0 + h) / src_man.height)
    made = 0
    for day in kept_days:
        small = src_previews / f"rgb_{day}.png"
        if small.exists():
            crop_png(small, out / small.name, fraction)
            made += 1
        full = src_previews / f"rgb_full_{day}.png"
        if full.exists():
            crop_png(full, out / full.name, None,
                     exact_box=(c0, r0, c0 + w, r0 + h))
            made += 1
    return made


def crop_cube(src_man, dst_dir, window, date_range, name):
    """Streams the crop into dst_dir and returns the new Manifest."""
    r0, c0, h, w = window
    d0, d1 = date_range
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = src_man.num_dates
    shape = (count, src_man.height, src_man.width)
    man = cc.Manifest(directory=dst_dir, name=name)
    man.source = src_man.source
    man.license = src_man.license
    man.crs = src_man.crs
    gt = src_man.geotransform
    man.geotransform = (gt[0] + c0 * gt[1], gt[1], gt[2],
                        gt[3] + r0 * gt[5], gt[4], gt[5])
    man.height, man.width = h, w
    man.extra = list(src_man.extra)
    valid = {}
    for file_name, dtype in ((src_man.red_file, "<u2"),
                             (src_man.nir_file, "<u2"),
                             (src_man.scl_file, "u1")):
        source = np.memmap(src_man.path(file_name), dtype=dtype, mode="r",
                           shape=shape)
        digest = hashlib.sha256()
        part = dst_dir / (file_name + ".part")
        with open(part, "wb") as handle:
            for t in range(d0, d1):
                block = np.ascontiguousarray(source[t, r0:r0 + h, c0:c0 + w])
                raw = block.tobytes()
                digest.update(raw)
                handle.write(raw)
                if file_name == src_man.scl_file:
                    valid[t] = cc.scl_valid_fraction(block)
        os.replace(part, dst_dir / file_name)
        man.sha256[file_name] = digest.hexdigest()
        del source
    for new_index, t in enumerate(range(d0, d1)):
        rec = src_man.dates[t]
        man.dates.append(cc.DateRecord(new_index, rec.day, rec.scale,
                                       rec.offset, valid[t], rec.scene_id))
    if src_man.reference_file:
        ref = np.memmap(src_man.path(src_man.reference_file), dtype="u1",
                        mode="r", shape=(src_man.height, src_man.width))
        cropped = np.ascontiguousarray(ref[r0:r0 + h, c0:c0 + w])
        cropped.tofile(dst_dir / src_man.reference_file)
        man.reference_file = src_man.reference_file
        man.reference_source = src_man.reference_source
        man.sha256[man.reference_file] = cc.sha256_file(
            dst_dir / man.reference_file)
    made = crop_previews(src_man, dst_dir, window,
                         [rec.day for rec in man.dates])
    cc.log(f"cropped previews written: {made}")
    cc.write_manifest(man)
    return man


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    src_man = cc.read_manifest(args.input)
    window = cc.parse_window(args.window, src_man.height, src_man.width)
    date_range = parse_dates(args.dates, src_man.num_dates)
    out_dir = pathlib.Path(args.out)
    name = args.name or out_dir.name
    cc.log(f"cropping {args.input} window {window} dates "
           f"{date_range[0]}:{date_range[1]} -> {out_dir}")
    man = crop_cube(src_man, out_dir, window, date_range, name)
    manifests_dir = (REPO_DIR / "data" / "manifests"
                     if not args.manifests_dir else args.manifests_dir)
    if str(manifests_dir) != "none":
        target = pathlib.Path(manifests_dir)
        target.mkdir(parents=True, exist_ok=True)
        cc.write_manifest(man, target / f"{name}_manifest.txt")
    cc.log(f"cube ready: {out_dir} ({man.num_dates} dates, "
           f"{man.height}x{man.width})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
