"""Non-GPU helper tooling. Compares the outputs of two detect runs.

Each run directory is read through its ``result_manifest.txt`` (image size and
the names of ``break_index.i16``, ``status.u8`` and ``magnitude.f32``, see
docs/FORMATS.md). The files are streamed in chunks, so a full 10980 x 10980
tile needs little memory. For every pair given with ``--pair A B`` the script
prints and writes one CSV row

  a,b,pixels,break_index_differ,status_differ,differ_fraction,
  either_differ,magnitude_max_abs_diff

where ``break_index_differ`` and ``status_differ`` are the numbers of pixels
whose break index (resp. status code) differs, ``either_differ`` the number of
pixels where at least one of them differs, ``differ_fraction`` is
``either_differ / pixels`` and ``magnitude_max_abs_diff`` is the largest
absolute difference of the break magnitude over pixels that are breaks with
the same break index in both runs (the rounding difference of the two
precisions; pixels with different break dates have different runs and are not
compared). ``a`` and ``b`` are the names of the run directories.

Usage:
  compare_outputs.py --pair out/full_gpu out/full_cpu \
      --pair out/full_gpu out/full_cpu-mt --out results/csv/agreement_full.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

CHUNK = 1 << 24  # pixels per chunk


def read_manifest(run_dir):
    """Returns (height, width, {file key: path}) of a result directory."""
    path = os.path.join(run_dir, "result_manifest.txt")
    if not os.path.exists(path):
        raise SystemExit("error: %s not found" % path)
    values = {}
    files = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("height", "width"):
                values[parts[0]] = int(parts[1])
            elif len(parts) >= 2 and parts[0] == "file":
                files[parts[1]] = os.path.join(run_dir, parts[1])
    for key in ("height", "width"):
        if key not in values:
            raise SystemExit("error: %s has no '%s' line" % (path, key))
    for name in ("break_index.i16", "status.u8", "magnitude.f32"):
        if name not in files or not os.path.exists(files[name]):
            raise SystemExit("error: %s lists no readable %s" % (path, name))
    return values["height"], values["width"], files


def compare(dir_a, dir_b):
    """Compares two runs; returns the CSV row as a dict."""
    height_a, width_a, files_a = read_manifest(dir_a)
    height_b, width_b, files_b = read_manifest(dir_b)
    if (height_a, width_a) != (height_b, width_b):
        raise SystemExit("error: %s and %s have different image sizes" % (
            dir_a, dir_b))
    pixels = height_a * width_a
    maps = {}
    for name, dtype in (("break_index.i16", "<i2"), ("status.u8", "u1"),
                        ("magnitude.f32", "<f4")):
        for tag, files in (("a", files_a), ("b", files_b)):
            maps[(name, tag)] = np.memmap(files[name], dtype=dtype, mode="r",
                                          shape=(pixels,))
    break_differ = status_differ = either = 0
    magnitude_max = 0.0
    for start in range(0, pixels, CHUNK):
        sl = slice(start, min(start + CHUNK, pixels))
        index_diff = maps[("break_index.i16", "a")][sl] != \
            maps[("break_index.i16", "b")][sl]
        status_a = maps[("status.u8", "a")][sl]
        status_b = maps[("status.u8", "b")][sl]
        status_diff = status_a != status_b
        break_differ += int(index_diff.sum())
        status_differ += int(status_diff.sum())
        either += int((index_diff | status_diff).sum())
        both_break = (status_a == 1) & (status_b == 1) & ~index_diff
        if both_break.any():
            diff = np.abs(maps[("magnitude.f32", "a")][sl][both_break].astype(
                np.float64) - maps[("magnitude.f32", "b")][sl][both_break])
            magnitude_max = max(magnitude_max, float(diff.max()))
    return {
        "a": os.path.basename(os.path.normpath(dir_a)),
        "b": os.path.basename(os.path.normpath(dir_b)),
        "pixels": pixels,
        "break_index_differ": break_differ,
        "status_differ": status_differ,
        "differ_fraction": "%.3e" % (either / pixels),
        "either_differ": either,
        "magnitude_max_abs_diff": "%.3e" % magnitude_max,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the break index, status and magnitude maps of "
                    "detect output directories.")
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("A", "B"), help="two detect output "
                        "directories (repeatable)")
    parser.add_argument("--out", help="CSV file to write")
    args = parser.parse_args(argv)
    rows = []
    for dir_a, dir_b in args.pair:
        row = compare(dir_a, dir_b)
        rows.append(row)
        print("%s vs %s: %d pixels, break_index differ %d, status differ %d, "
              "either %d (fraction %s), max |magnitude diff| %s" % (
                  row["a"], row["b"], row["pixels"],
                  row["break_index_differ"], row["status_differ"],
                  row["either_differ"], row["differ_fraction"],
                  row["magnitude_max_abs_diff"]))
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
