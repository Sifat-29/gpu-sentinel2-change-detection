"""Non-GPU helper tooling. Shows how much of a detection the reference scores.

The validation of a detect run (precision, recall, F1) is computed only on
pixels the reference can judge: pixels where the reference raster is 255 (no
forest in 2000, no data, or loss before the monitoring start) are excluded.
Detected break pixels on excluded pixels are counted neither as true nor as
false positives. This script reads a detect output directory
(``result_manifest.txt``, ``status.u8``, ``break_index.i16``,
``magnitude.f32``) and the reference raster named in the manifest, applies the
same rule as the program (src/validation_rules.h) and writes one CSV row per
run:

  run, pixels, eligible_pixels, reference_positive,
  reference_positive_share_of_eligible, detected, detected_scored,
  detected_unscored, unscored_fraction, tp, fp, precision_scored,
  precision_unscored_as_fp

``precision_scored`` is tp / (tp + fp), as the program reports it;
``precision_unscored_as_fp`` also counts the detections on excluded pixels as
false positives: tp / (tp + fp + detected_unscored). The reference path in the
manifest is relative to the directory the detect run was started from, so run
this script from the repository root.

Usage:
  validation_coverage.py --run out/full_gpu --run out/demo_gpu \
      --out results/csv/validation_coverage.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

CHUNK = 1 << 24  # pixels per chunk


def read_manifest(run_dir):
    """Returns (height, width, reference, first year, last year, years)."""
    path = os.path.join(run_dir, "result_manifest.txt")
    if not os.path.exists(path):
        raise SystemExit("error: %s not found" % path)
    values = {}
    years = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2 and parts[0] in (
                    "height", "width", "reference_file", "first_monitor_year",
                    "last_year"):
                values[parts[0]] = parts[1]
            elif len(parts) >= 3 and parts[0] == "date":
                years.append(int(parts[2][:4]))
    for key in ("height", "width", "reference_file", "first_monitor_year",
                "last_year"):
        if key not in values:
            raise SystemExit("error: %s has no '%s' line" % (path, key))
    return (int(values["height"]), int(values["width"]),
            values["reference_file"], int(values["first_monitor_year"]),
            int(values["last_year"]), np.array(years, dtype=np.int32))


def coverage(run_dir):
    """Returns the CSV row (a dict) of one detect output directory."""
    height, width, reference, first, last, years = read_manifest(run_dir)
    if not os.path.exists(reference):
        raise SystemExit("error: reference %s not found (run from the "
                         "repository root)" % reference)
    pixels = height * width
    ref = np.memmap(reference, dtype="u1", mode="r", shape=(pixels,))
    status = np.memmap(os.path.join(run_dir, "status.u8"), dtype="u1",
                       mode="r", shape=(pixels,))
    index = np.memmap(os.path.join(run_dir, "break_index.i16"), dtype="<i2",
                      mode="r", shape=(pixels,))
    magnitude = np.memmap(os.path.join(run_dir, "magnitude.f32"), dtype="<f4",
                          mode="r", shape=(pixels,))
    eligible = positive = detected = unscored = tp = fp = 0
    for start in range(0, pixels, CHUNK):
        sl = slice(start, min(start + CHUNK, pixels))
        r = ref[sl].astype(np.int32)
        idx = index[sl].astype(np.int64)
        valid_idx = (idx >= 0) & (idx < len(years))
        year = np.where(valid_idx, years[np.clip(idx, 0, len(years) - 1)], 0)
        det = ((status[sl] == 1) & (magnitude[sl] < 0) & valid_idx &
               (year >= first) & (year <= last))
        is_eligible = r != 255
        ref_year = 2000 + r
        ref_pos = is_eligible & (r != 0) & (ref_year >= first) & \
            (ref_year <= last)
        eligible += int(is_eligible.sum())
        positive += int(ref_pos.sum())
        detected += int(det.sum())
        unscored += int((det & ~is_eligible).sum())
        tp += int((det & ref_pos).sum())
        fp += int((det & is_eligible & ~ref_pos).sum())
    return {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "pixels": pixels,
        "eligible_pixels": eligible,
        "reference_positive": positive,
        "reference_positive_share_of_eligible": "%.4f" % (positive / eligible),
        "detected": detected,
        "detected_scored": detected - unscored,
        "detected_unscored": unscored,
        "unscored_fraction": "%.4f" % (unscored / detected),
        "tp": tp,
        "fp": fp,
        "precision_scored": "%.4f" % (tp / (tp + fp)),
        "precision_unscored_as_fp": "%.4f" % (tp / (tp + fp + unscored)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Count detected break pixels the reference does not "
                    "score and recompute the precision with them as false "
                    "positives.")
    parser.add_argument("--run", action="append", required=True,
                        metavar="DIR", help="detect output directory "
                        "(repeatable)")
    parser.add_argument("--out", help="CSV file to write")
    args = parser.parse_args(argv)
    rows = []
    for run_dir in args.run:
        row = coverage(run_dir)
        rows.append(row)
        print("%s: %d detected, %d unscored (%s), precision %s scored, %s "
              "with unscored as false positives; reference positive %s of "
              "eligible" % (
                  row["run"], row["detected"], row["detected_unscored"],
                  row["unscored_fraction"], row["precision_scored"],
                  row["precision_unscored_as_fp"],
                  row["reference_positive_share_of_eligible"]))
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
