"""Non-GPU helper tooling. Plots the measured pipeline timeline of a GPU run.

Reads one or two trace files written by ``change_detect --trace-csv`` (one row
per tile and stage, see docs/FORMATS.md) and draws a Gantt chart: one lane per
resource (host read, host-to-device stream, compute stream, device-to-host
stream), one bar per tile and stage, coloured by the buffer slot the tile
used, with the tile number on (or above) the bar. With two traces the panels
are stacked and share the time axis, so a 3-stream run can be compared with a
1-stream run directly. Only the first ``--max-tiles`` tiles are drawn; the
caption gives the total tile count and the time of the whole tile loop.

Example:
  plot_timeline.py --trace s3.csv --trace s1.csv \\
      --labels "3 streams" "1 stream" --out pipeline_timeline_measured.png
"""

import argparse
import csv
import os
import sys

import matplotlib  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Buffer-slot colours (distinct under colour-vision deficiency); the tile
# number on every bar is the second, non-colour cue.
SLOT_COLORS = ["#2a6fbb", "#d9822b", "#8a5cc2"]
# Lanes from top to bottom: (title, trace stage names shown in the lane).
LANES = [
    ("host read (disk to pinned)", ("read",)),
    ("H2D copy", ("h2d",)),
    ("kernels", ("fused", "ndvi", "fit", "detect", "validate")),
    ("D2H copy", ("d2h",)),
]
HEADER = ["tile", "slot", "stage", "stream", "start_ms", "end_ms"]
BAR_HEIGHT = 0.62
MIN_LABEL_FRACTION = 0.012  # bar width / axis span needed for an inside label

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold"})


def load_trace(path):
    """Returns the rows of a trace CSV as dicts; exits on a malformed file."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != HEADER:
            sys.exit("error: %s is not a --trace-csv file (header %s)"
                     % (path, header))
        rows = []
        for line in reader:
            rows.append({"tile": int(line[0]), "slot": int(line[1]),
                         "stage": line[2], "stream": int(line[3]),
                         "start": float(line[4]), "end": float(line[5])})
    if not rows:
        sys.exit("error: %s has no rows" % path)
    return rows


def union_length(intervals):
    """Total length covered by a list of (start, end) intervals."""
    total, edge = 0.0, None
    for start, end in sorted(intervals):
        if edge is None or start > edge:
            total += end - start
            edge = end
        elif end > edge:
            total += end - edge
            edge = end
    return total


def summarize(rows):
    """Returns (tiles, loop_ms, busy_by_lane_ms, hidden_copy_fraction).

    The loop time runs from the start of the tile loop to the end of the last
    stage. The hidden copy fraction is the share of the copy time (H2D and D2H
    busy time) that runs while a kernel or the host read is also active.
    """
    tiles = max(r["tile"] for r in rows) + 1
    loop_ms = max(r["end"] for r in rows)
    busy = {}
    for title, stages in LANES:
        spans = [(r["start"], r["end"]) for r in rows if r["stage"] in stages]
        busy[title] = union_length(spans)
    copies = [(r["start"], r["end"]) for r in rows
              if r["stage"] in ("h2d", "d2h")]
    others = [(r["start"], r["end"]) for r in rows
              if r["stage"] not in ("h2d", "d2h")]
    copy_union = union_length(copies)
    together = union_length(copies + others)
    other_union = union_length(others)
    # |copies ∩ others| = |copies| + |others| - |copies ∪ others|
    overlap = copy_union + other_union - together
    hidden = overlap / copy_union if copy_union > 0 else 0.0
    return tiles, loop_ms, busy, hidden


def draw_panel(ax, rows, label, max_tiles, span):
    """Draws one Gantt panel for the first max_tiles tiles."""
    shown = [r for r in rows if r["tile"] < max_tiles]
    num_lanes = len(LANES)
    lane_of = {}
    for index, (_, stages) in enumerate(LANES):
        for stage in stages:
            lane_of[stage] = num_lanes - 1 - index
    labelled = set()
    for row in shown:
        lane = lane_of.get(row["stage"])
        if lane is None:
            continue
        width = row["end"] - row["start"]
        color = SLOT_COLORS[row["slot"] % len(SLOT_COLORS)]
        # A white outline separates touching bars but would swallow a bar
        # that is only a few pixels wide.
        wide = width >= 0.01 * span
        ax.barh(lane, width, left=row["start"], height=BAR_HEIGHT,
                color=color, edgecolor="white" if wide else color,
                linewidth=0.6, zorder=3)
        text = str(row["tile"])
        # Several stages can share a lane (the kernel lane): number each
        # tile once per lane so the labels do not overprint.
        if (lane, row["tile"]) in labelled:
            continue
        labelled.add((lane, row["tile"]))
        if width >= MIN_LABEL_FRACTION * span * (1 + 0.6 * len(text)):
            ax.text(row["start"] + width / 2, lane, text, ha="center",
                    va="center", color="white", fontsize=7.5,
                    fontweight="bold", zorder=4)
        else:
            # Too narrow for an inside label: put the number above the bar.
            ax.text(row["start"] + width / 2, lane + BAR_HEIGHT / 2 + 0.04,
                    text, ha="center", va="bottom", color="#333333",
                    fontsize=7, zorder=4)
    ax.set_yticks(range(num_lanes))
    ax.set_yticklabels([title for title, _ in reversed(LANES)])
    ax.set_ylim(-0.6, num_lanes - 0.25)
    ax.set_xlim(0, span)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(label, loc="left", fontsize=11)


def main():
    """Parses arguments, draws the figure and prints a text summary."""
    parser = argparse.ArgumentParser(
        description="Plot the measured pipeline timeline (--trace-csv).")
    parser.add_argument("--trace", action="append", required=True,
                        help="trace CSV (give twice to compare two runs)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="panel titles, one per --trace")
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--max-tiles", type=int, default=12,
                        help="draw only the first N tiles (default 12)")
    args = parser.parse_args()
    if len(args.trace) > 2:
        sys.exit("error: at most two --trace files are supported")
    if args.max_tiles < 1:
        sys.exit("error: --max-tiles must be >= 1")
    labels = args.labels or [os.path.basename(p) for p in args.trace]
    if len(labels) != len(args.trace):
        sys.exit("error: give one label per --trace")

    traces = [load_trace(path) for path in args.trace]
    # One shared time axis: the end of the last drawn tile over all traces.
    span = 0.0
    for rows in traces:
        shown = [r["end"] for r in rows if r["tile"] < args.max_tiles]
        span = max(span, max(shown))
    span *= 1.02

    fig, axes = plt.subplots(len(traces), 1, sharex=True, squeeze=False,
                             figsize=(12, 3.6 * len(traces) + 1.2))
    caption = []
    for ax, rows, label in zip(axes[:, 0], traces, labels):
        tiles, loop_ms, busy, hidden = summarize(rows)
        draw_panel(ax, rows, "%s: tile loop %.0f ms for %d tiles"
                   % (label, loop_ms, tiles), args.max_tiles, span)
        shown_tiles = min(tiles, args.max_tiles)
        caption.append("%s: first %d of %d tiles drawn; all %d tiles take "
                       "%.0f ms; %.0f%% of the copy time runs while a "
                       "kernel or a read is active."
                       % (label, shown_tiles, tiles, tiles, loop_ms,
                          100 * max(0.0, hidden)))
        lanes = ", ".join("%s %.0f%%" % (t.split(" (")[0], 100 * b / loop_ms)
                          for t, b in busy.items())
        print("%s: loop %.1f ms, %d tiles, lane busy: %s; copy hidden %.0f%%"
              % (label, loop_ms, tiles, lanes, 100 * max(0.0, hidden)))
    axes[-1, 0].set_xlabel("time since the start of the tile loop (ms)")
    handles = [Patch(color=c, label="buffer slot %d" % i)
               for i, c in enumerate(SLOT_COLORS)]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False,
               bbox_to_anchor=(0.98, 0.985))
    fig.suptitle("Pipeline timeline (measured, CUDA events)", x=0.01,
                 ha="left", fontsize=13, fontweight="bold")
    note = ("Bars are tiles (number on the bar), coloured by buffer slot. "
            "Device times are CUDA event timestamps; host reads use the "
            "steady clock, aligned to the device timeline by a wait on the "
            "reference event.")
    fig.text(0.01, 0.01, "\n".join(caption + [note]), ha="left", va="bottom",
             fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.04 + 0.025 * len(caption), 1, 0.95))
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
