#!/usr/bin/env python3
"""Non-GPU helper tooling. Parameter sweep driver for bin/change_detect.

The script launches the detector (``--mode detect --backend gpu``) once per
parameter configuration on one cube, parses the ``--- results ---`` and
``validation:`` lines of its log, and appends one CSV row per configuration.
Nothing is computed here: every number comes from the program's own log.

Plans:
  coord  coordinate search around the base configuration (one parameter
         changed at a time) plus three context rows (trend extrapolate with
         and without --losses-only, trend hold without --losses-only).
  grid   full grid over the two parameters whose coordinate rows spread the
         F1 score the most (read from the CSV), or over --grid-params.

The CSV is resumable: configurations already present are skipped. A ranked
table (precision >= --min-precision first, by F1) is printed at the end.

Example:
  python scripts/tune_parameters.py \\
      --cube data/cubes/chaco_dev2048/manifest.txt \\
      --out results/csv/tuning_dev.csv --plan coord
"""

import argparse
import csv
import itertools
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Brief defaults; the sweep changes one entry at a time around these.
BASE = {
    "threshold": 3.0,
    "consecutive": 3,
    "harmonics": 2,
    "rmse_floor": 0.02,
    "train_years": 2.0,
    "valid_classes": "4,5,6",
    "cleanup": 0,
    "trend": "hold",
    "losses_only": 1,
}

# Candidate values per tunable parameter.
VALUES = {
    "threshold": [2.5, 3.0, 3.5, 4.0, 5.0],
    "consecutive": [2, 3, 4],
    "harmonics": [1, 2, 3],
    "rmse_floor": [0.01, 0.02, 0.03, 0.05],
    "train_years": [1.5, 2.0],
    "valid_classes": ["4,5,6", "4,5,6,7"],
    "cleanup": [0, 1],
}

# Parameters that define a configuration (CSV columns, resume key).
PARAMS = list(BASE)

# Parameters counted when judging how far a configuration is from the brief.
TUNABLE = list(VALUES)

COUNT_FIELDS = ["stable", "break", "loss", "gain", "insufficient"]
METRIC_FIELDS = ["tp", "fp", "fn", "precision", "recall", "f1",
                 "year_exact", "year_within1"]
COLUMNS = (PARAMS
           + COUNT_FIELDS + [f + "_frac" for f in COUNT_FIELDS]
           + METRIC_FIELDS + ["wall_seconds", "git_commit"])

RESULTS_RE = re.compile(
    r"stable:\s*(\d+)\s*\([\d.]+%\)\s*\|\s*break:\s*(\d+)\s*\([\d.]+%\)"
    r"\s*\[loss\s+(\d+),\s*gain\s+(\d+)\]\s*\|\s*insufficient_data:\s*(\d+)")
VALID_RE = re.compile(
    r"validation:\s*precision\s+([\d.]+)\s*\|\s*recall\s+([\d.]+)\s*\|\s*"
    r"f1\s+([\d.]+)\s*\|\s*year_exact\s+([\d.]+)\s*\|\s*"
    r"year_within1\s+([\d.]+)\s*\(tp\s+(\d+)\s+fp\s+(\d+)\s+fn\s+(\d+)\)")
COMMIT_RE = re.compile(r"git_commit:\s*(\S+)")


def key_of(config):
    """Canonical hashable key of a configuration (resume identity)."""
    return tuple(_norm(config[p]) for p in PARAMS)


def _norm(value):
    """Normalises a value so that CSV text and Python values compare equal."""
    if isinstance(value, str):
        try:
            return _norm(float(value))
        except ValueError:
            return value
    return round(float(value), 6)


def coord_plan():
    """Returns the list of configurations of the coordinate search."""
    plan = [dict(BASE)]
    for name, values in VALUES.items():
        for value in values:
            if _norm(value) == _norm(BASE[name]):
                continue
            config = dict(BASE)
            config[name] = value
            plan.append(config)
    # Context rows for the README: the brief's model and the loss/gain choice.
    plan.append(dict(BASE, trend="extrapolate", losses_only=1))
    plan.append(dict(BASE, trend="extrapolate", losses_only=0))
    plan.append(dict(BASE, trend="hold", losses_only=0))
    return plan


def read_rows(path):
    """Reads an existing result CSV (empty list if missing)."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def influential_parameters(rows):
    """Returns the two tunable parameters with the largest F1 spread.

    Only coordinate rows are considered: rows that differ from the base in
    exactly that parameter (the base row itself belongs to every parameter).
    """
    spread = {}
    for name in TUNABLE:
        f1s = []
        for row in rows:
            others_base = all(
                _norm(row[p]) == _norm(BASE[p]) for p in PARAMS if p != name)
            if others_base:
                f1s.append(float(row["f1"]))
        spread[name] = max(f1s) - min(f1s) if len(f1s) > 1 else 0.0
    ranked = sorted(TUNABLE, key=lambda n: -spread[n])
    return ranked[:2], spread


def grid_plan(rows, grid_params):
    """Returns the full grid over two parameters, others at the base."""
    if grid_params:
        names = grid_params
    else:
        names, _ = influential_parameters(rows)
    plan = []
    for combo in itertools.product(*(VALUES[n] for n in names)):
        config = dict(BASE)
        config.update(dict(zip(names, combo)))
        plan.append(config)
    return plan, names


def command_for(binary, cube, config, out_dir, base_args):
    """Builds the change_detect command line for one configuration."""
    cmd = [str(binary), "--input", str(cube), "--mode", "detect",
           "--backend", "gpu", "--quiet", "--sample-pixels", "0",
           "--png-max-dim", "512", "--output-dir", str(out_dir),
           "--threshold", str(config["threshold"]),
           "--consecutive", str(config["consecutive"]),
           "--harmonics", str(config["harmonics"]),
           "--rmse-floor", str(config["rmse_floor"]),
           "--train-years", str(config["train_years"]),
           "--valid-classes", str(config["valid_classes"]),
           "--trend", str(config["trend"])]
    if int(config["losses_only"]):
        cmd.append("--losses-only")
    if int(config["cleanup"]):
        cmd.append("--cleanup")
    return cmd + base_args


def run_one(binary, cube, config, base_args, tmp_root):
    """Runs one configuration; returns the CSV row dict or None on failure."""
    out_dir = Path(tempfile.mkdtemp(prefix="tune_", dir=tmp_root))
    try:
        cmd = command_for(binary, cube, config, out_dir, base_args)
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              check=False)
        wall = time.perf_counter() - start
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    text = proc.stdout + proc.stderr
    results = RESULTS_RE.search(text)
    valid = VALID_RE.search(text)
    if proc.returncode != 0 or not results or not valid:
        sys.stderr.write("run failed (rc %d): %s\n"
                         % (proc.returncode, " ".join(cmd)))
        return None
    counts = dict(zip(COUNT_FIELDS, (int(g) for g in results.groups())))
    total = sum(counts[f] for f in ("stable", "break", "insufficient"))
    row = dict(config)
    for name in COUNT_FIELDS:
        row[name] = counts[name]
        row[name + "_frac"] = "%.5f" % (counts[name] / max(total, 1))
    precision, recall, f1, exact, within1, tp, fp, fn = valid.groups()
    row.update(precision=precision, recall=recall, f1=f1, year_exact=exact,
               year_within1=within1, tp=tp, fp=fp, fn=fn)
    row["wall_seconds"] = "%.1f" % wall
    commit = COMMIT_RE.search(text)
    row["git_commit"] = commit.group(1) if commit else "unknown"
    return row


def distance_from_brief(row):
    """Number of tunable parameters that differ from the brief defaults."""
    return sum(_norm(row[p]) != _norm(BASE[p]) for p in TUNABLE)


def print_ranking(rows, min_precision, top):
    """Prints the ranked table and the rule-based recommendation."""
    ranked = sorted(rows, key=lambda r: -float(r["f1"]))
    print("\nRanked by F1 (rows below precision %.2f are marked '-'):"
          % min_precision)
    header = ("rank thr  M K floor  train classes cln trend       lo "
              "prec   rec    F1     yr_exact  yr+-1")
    print(header)
    for i, row in enumerate(ranked[:top], 1):
        ok = float(row["precision"]) >= min_precision
        print("%3d%s %-4s %d %d %-5s %-5s %-7s %d   %-11s %d  %s  %s  %s"
              "  %s     %s" % (
                  i, " " if ok else "-", row["threshold"],
                  int(row["consecutive"]), int(row["harmonics"]),
                  row["rmse_floor"], row["train_years"],
                  row["valid_classes"], int(row["cleanup"]), row["trend"],
                  int(row["losses_only"]), row["precision"], row["recall"],
                  row["f1"], row["year_exact"], row["year_within1"]))
    eligible = [r for r in rows
                if float(r["precision"]) >= min_precision
                and r["trend"] == "hold" and int(r["losses_only"]) == 1]
    if not eligible:
        return
    best = max(float(r["f1"]) for r in eligible)
    near = [r for r in eligible if best - float(r["f1"]) < 0.01]
    pick = min(near, key=lambda r: (distance_from_brief(r),
                                    -float(r["precision"]),
                                    -float(r["f1"])))
    print("\nBest F1 %.3f; simplest within 0.01: %s (F1 %s, P %s, R %s)" % (
        best, {p: pick[p] for p in TUNABLE}, pick["f1"], pick["precision"],
        pick["recall"]))


def main():
    """Parses arguments, runs missing configurations, prints the ranking."""
    parser = argparse.ArgumentParser(
        description="Sweep change_detect parameters on one cube.")
    parser.add_argument("--cube", required=True, type=Path,
                        help="cube manifest.txt")
    parser.add_argument("--out", required=True, type=Path, help="result CSV")
    parser.add_argument("--base-args", default="",
                        help="extra change_detect flags (quoted string)")
    parser.add_argument("--plan", choices=["coord", "grid"], default="coord")
    parser.add_argument("--grid-params", default="",
                        help="comma list of two parameters for --plan grid "
                        "(default: the two with the largest F1 spread)")
    parser.add_argument("--binary", type=Path, default=ROOT / "bin" /
                        "change_detect", help="detector executable")
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--top", type=int, default=15,
                        help="rows shown in the ranked table")
    args = parser.parse_args()

    rows = read_rows(args.out)
    grid_params = [p for p in args.grid_params.split(",") if p]
    if args.plan == "coord":
        plan = coord_plan()
    else:
        plan, names = grid_plan(rows, grid_params)
        print("grid over %s" % ", ".join(names))
    done = {key_of(r) for r in rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_args = shlex.split(args.base_args)
    todo = [c for c in plan if key_of(c) not in done]
    print("%d configurations planned, %d already in %s"
          % (len(plan), len(plan) - len(todo), args.out))
    write_header = not args.out.exists() or not rows
    with args.out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for index, config in enumerate(todo, 1):
            row = run_one(args.binary, args.cube, config, base_args, None)
            if row is None:
                continue
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print("[%d/%d] %s -> P %s R %s F1 %s (%ss)" % (
                index, len(todo),
                " ".join("%s=%s" % (p, config[p]) for p in PARAMS),
                row["precision"], row["recall"], row["f1"],
                row["wall_seconds"]))
    if rows:
        print_ranking(rows, args.min_precision, args.top)
        if args.plan == "coord":
            names, spread = influential_parameters(rows)
            print("\nF1 spread per parameter: " + ", ".join(
                "%s %.3f" % (n, spread[n]) for n in TUNABLE))
            print("two most influential: %s" % ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
