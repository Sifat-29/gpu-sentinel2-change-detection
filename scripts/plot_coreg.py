"""Non-GPU helper tooling. Plots the result of ``change_detect --mode coreg``.

Reads ``coreg_shifts.csv`` (one row per date, see docs/FORMATS.md) and draws
the estimated row shift (dy) and column shift (dx) of every date relative to
the reference date in two stacked panels that share the date axis, with the
+-0.5 pixel band shaded. Dates flagged ``low_peak`` or ``large_shift`` are
marked with a red diamond and their flag name, so they are recognisable
without relying on colour.

Example:
  plot_coreg.py --csv out/coreg_demo/coreg_shifts.csv \\
      --out out/coreg_demo/coreg_shifts.png
"""

import argparse
import csv
import datetime
import sys

import matplotlib  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLUMNS = ["date_index", "date", "valid_fraction", "dy_px", "dx_px",
           "shift_px", "peak", "flag"]
LINE_COLORS = {"dy_px": "#2a6fbb", "dx_px": "#d9822b"}
FLAG_COLOR = "#c0392b"
BAND_PX = 0.5

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold"})


def load_rows(path):
    """Returns the CSV rows as dicts with numeric fields; exits on error."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != COLUMNS:
            sys.exit("error: %s is not a coreg_shifts.csv file (header %s)"
                     % (path, header))
        rows = []
        for line in reader:
            row = dict(zip(COLUMNS, line))
            row["when"] = datetime.date.fromisoformat(row["date"])
            for key in COLUMNS[2:7]:
                row[key] = float(row[key])
            rows.append(row)
    if not rows:
        sys.exit("error: %s has no rows" % path)
    return rows


def draw_panel(ax, rows, key, label):
    """Draws the shift of one axis against the date, flagged dates marked."""
    dates = [r["when"] for r in rows]
    values = [r[key] for r in rows]
    ax.axhspan(-BAND_PX, BAND_PX, color="#999999", alpha=0.18, zorder=1,
               label="+-%.1f px" % BAND_PX)
    ax.axhline(0.0, color="#555555", linewidth=0.8, zorder=2)
    ax.plot(dates, values, color=LINE_COLORS[key], marker="o", markersize=3.5,
            linewidth=1.0, zorder=3, label="shift per date")
    flagged = [r for r in rows if r["flag"] != "ok"]
    if flagged:
        ax.scatter([r["when"] for r in flagged], [r[key] for r in flagged],
                   marker="D", s=48, color=FLAG_COLOR, zorder=4,
                   label="flagged date")
        for row in flagged:
            ax.annotate(row["flag"], (row["when"], row[key]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=8, color=FLAG_COLOR)
    ax.set_ylabel(label)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    limit = max(0.75, 1.15 * max(abs(v) for v in values))
    ax.set_ylim(-limit, limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", required=True, help="coreg_shifts.csv")
    parser.add_argument("--out", required=True, help="output PNG path")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    draw_panel(axes[0], rows, "dy_px", "row shift dy (px)")
    draw_panel(axes[1], rows, "dx_px", "column shift dx (px)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    axes[1].set_xlabel("acquisition date")
    flagged = sum(1 for r in rows if r["flag"] != "ok")
    shifts = sorted(r["shift_px"] for r in rows)
    mid = len(shifts) // 2
    median = (shifts[mid] if len(shifts) % 2 else
              0.5 * (shifts[mid - 1] + shifts[mid]))
    fig.suptitle("Co-registration check: shift of each date relative to the "
                 "reference scene (phase correlation)", fontsize=12,
                 fontweight="bold")
    fig.text(0.5, 0.005,
             "%d dates; median shift %.3f px, maximum %.3f px; %d flagged "
             "(low_peak or large_shift); grey band = +-%.1f px"
             % (len(rows), median, shifts[-1], flagged, BAND_PX),
             ha="center", fontsize=9, color="#333333")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
