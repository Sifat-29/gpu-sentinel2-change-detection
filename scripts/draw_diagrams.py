"""Non-GPU helper tooling. Draws the two explanatory diagrams of the project.

Writes into ``--out`` (default ``docs/figures``):

  algorithm_diagram.png  one synthetic pixel series with the harmonic fit on
                         the training window, the +/- threshold x RMSE band,
                         the run of M consecutive exceedances and the break
                         date. The curve is computed here from the documented
                         formulas (README.md); the series is an illustration,
                         not measured data.
  pipeline_diagram.png   schematic timeline of tiles flowing through read,
                         host-to-device copy, kernels and device-to-host copy
                         on three CUDA streams, overlapping in time. A
                         schematic, not a measurement.
"""

import argparse
import os
import sys

import numpy as np

import matplotlib  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

MAX_BYTES = 1000 * 1000
HARMONICS = 2
TRAIN_YEARS = 2.0
THRESHOLD = 3.0
CONSECUTIVE = 3
RMSE_FLOOR = 0.02
SLOT_COLORS = ["#2a6fbb", "#e8833a", "#3a9d5d"]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold"})


def design_matrix(tau):
    """Design rows [1, s, cos 2 pi k tau, sin 2 pi k tau, ...] (README.md)."""
    s = (tau - TRAIN_YEARS / 2.0) / (TRAIN_YEARS / 2.0)
    cols = [np.ones_like(tau), s]
    for k in range(1, HARMONICS + 1):
        cols += [np.cos(2 * np.pi * k * tau), np.sin(2 * np.pi * k * tau)]
    return np.stack(cols, axis=1)


def simulate_series():
    """Synthetic NDVI series: seasonal forest, cloud gaps, clearing at 3.3 y.

    Returns tau, observed ndvi, valid mask, the break tau and the fitted
    quantities computed with the documented algorithm.
    """
    rng = np.random.default_rng(11)
    tau = np.arange(0, 5.0, 1.0 / 18.0)
    truth = 0.72 + 0.12 * np.cos(2 * np.pi * (tau - 0.15))
    truth += 0.02 * np.cos(4 * np.pi * (tau - 0.4))
    clear_tau = 3.3
    observed = truth + rng.normal(0, 0.025, tau.size)
    observed = np.where(tau >= clear_tau, observed - 0.38, observed)
    valid = rng.random(tau.size) > 0.22
    x = design_matrix(tau)
    train = (tau < TRAIN_YEARS) & valid
    beta = np.linalg.solve(x[train].T @ x[train], x[train].T @ observed[train])
    predicted = x @ beta
    n, p = int(train.sum()), x.shape[1]
    rss = float(np.sum((observed[train] - predicted[train]) ** 2))
    rmse = max(np.sqrt(rss / (n - p)), RMSE_FLOOR)
    z = (observed - predicted) / rmse
    monitor = np.where((tau >= TRAIN_YEARS) & valid)[0]
    run, sign, start, break_idx = 0, 0, -1, -1
    run_members = []
    for i in monitor:
        exceed = abs(z[i]) > THRESHOLD
        sg = int(np.sign(z[i]))
        if exceed and run > 0 and sg == sign:
            run += 1
            run_members.append(i)
        elif exceed:
            run, sign, start, run_members = 1, sg, i, [i]
        else:
            run, run_members = 0, []
        if run >= CONSECUTIVE:
            break_idx = start
            break
    return {"tau": tau, "y": observed, "valid": valid, "pred": predicted,
            "rmse": rmse, "z": z, "train": train, "break_idx": break_idx,
            "run": run_members, "n_train": n, "monitor": monitor}


def draw_algorithm(path):
    d = simulate_series()
    year0 = 2019.0
    t = year0 + d["tau"]
    fig, (ax, az) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.3, 1]})
    ax.axvspan(year0, year0 + TRAIN_YEARS, color="#9ecae1", alpha=0.35, lw=0)
    ax.text(year0 + TRAIN_YEARS / 2, 1.02, "training window\n(fit by least "
            "squares on valid dates)", ha="center", va="bottom", fontsize=9,
            color="#08519c")
    band = THRESHOLD * d["rmse"]
    mon = d["tau"] >= TRAIN_YEARS
    ax.fill_between(t[mon], d["pred"][mon] - band, d["pred"][mon] + band,
                    color="#bdbdbd", alpha=0.5, lw=0)
    ax.plot(t[~mon], d["pred"][~mon], color="#08519c", lw=2.2)
    ax.plot(t[mon], d["pred"][mon], color="#08519c", lw=2, ls="--")
    ok = d["valid"]
    ax.scatter(t[~ok], d["y"][~ok], marker="x", color="#999", s=26, zorder=3)
    ax.scatter(t[ok], d["y"][ok], color="#222", s=24, zorder=4)
    run = d["run"]
    ax.scatter(t[run], d["y"][run], color="#d62728", s=52, zorder=5)
    bi = d["break_idx"]
    ax.axvline(t[bi], color="#d62728", lw=1.8)
    ax.annotate("break date = first date of the run\n(%d consecutive "
                "exceedances)" % CONSECUTIVE, xy=(t[bi], 0.12),
                xytext=(t[bi] + 0.12, 0.05), fontsize=9.5, color="#d62728",
                va="center")
    arrow_y = float(d["y"][run].min()) - 0.07
    ax.annotate("", xy=(t[run[-1]] + 0.03, arrow_y),
                xytext=(t[run[0]] - 0.03, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="#d62728"))
    ax.set_ylim(-0.02, 1.15)
    ax.set_ylabel("NDVI")
    ax.set_title("Per-pixel break detection (illustration: synthetic series, "
                 "not real data)")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#222",
                   label="valid observation"),
        plt.Line2D([0], [0], marker="x", color="#999", ls="", label="masked "
                   "(cloud / shadow / nodata)"),
        plt.Line2D([0], [0], color="#08519c", lw=2.2, label="harmonic fit "
                   "(training)"),
        plt.Line2D([0], [0], color="#08519c", lw=2, ls="--",
                   label="model prediction (monitoring)"),
        Patch(facecolor="#bdbdbd", alpha=0.5,
              label="+/- %g x RMSE band" % THRESHOLD),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   label="exceedance run")], loc="lower left", fontsize=8.5,
        ncol=2, framealpha=0.95)
    ax.text(0.995, 0.97, "NDVI(tau) = b0 + b1*s + sum_k [a_k cos(2 pi k tau) "
            "+ c_k sin(2 pi k tau)]\nz = (y - prediction) / RMSE,  "
            "RMSE = max(sqrt(RSS/(n-p)), floor)\nbreak: |z| > %g for %d "
            "consecutive valid dates (same sign)" % (THRESHOLD, CONSECUTIVE),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            family="monospace", bbox=dict(boxstyle="round", facecolor="white",
                                          edgecolor="#888", alpha=0.95))
    az.axhspan(-THRESHOLD, THRESHOLD, color="#bdbdbd", alpha=0.4, lw=0)
    az.axhline(THRESHOLD, color="#555", lw=1, ls=":")
    az.axhline(-THRESHOLD, color="#555", lw=1, ls=":")
    zm = d["monitor"]
    az.vlines(t[zm], 0, d["z"][zm], color="#888", lw=1)
    az.scatter(t[zm], d["z"][zm], s=20, color="#222", zorder=3)
    az.scatter(t[run], d["z"][run], s=40, color="#d62728", zorder=4)
    az.axvline(t[bi], color="#d62728", lw=1.8)
    az.set_ylabel("z-score")
    az.set_xlabel("time (years)")
    az.text(year0 + 0.05, -THRESHOLD - 0.4, "threshold +/- %g" % THRESHOLD,
            fontsize=8.5, color="#555", va="top")
    az.set_title("Standardised residuals in the monitoring period",
                 fontsize=10)
    fig.tight_layout()
    save(fig, path)


def pipeline_schedule(num_tiles=8):
    """Greedy schedule of the role-stream pipeline

    The host main thread reads tile k into pinned buffer slot k % 3 (only
    once that slot's previous tile has left the device-to-host stream), then
    the H2D stream, the compute stream and the D2H stream process it in turn,
    ordered by events. Each resource runs one task at a time. Durations are
    arbitrary schematic units.
    """
    dur = {"read": 2.4, "h2d": 1.5, "kernel": 2.2, "d2h": 1.0}
    free = {"read": 0.0, "h2d": 0.0, "kernel": 0.0, "d2h": 0.0}
    slot_free = [0.0, 0.0, 0.0]
    events = []
    for tile in range(num_tiles):
        slot = tile % 3
        t = max(free["read"], slot_free[slot])
        for op in ("read", "h2d", "kernel", "d2h"):
            start = max(t, free[op])
            end = start + dur[op]
            free[op] = end
            events.append((op, tile, slot, start, end))
            t = end
        slot_free[slot] = t
    return events


def sequential_schedule(num_tiles=3):
    """The same stages without overlap: one tile fully finishes first."""
    dur = {"read": 2.4, "h2d": 1.5, "kernel": 2.2, "d2h": 1.0}
    events, t = [], 0.0
    for tile in range(num_tiles):
        for op in ("read", "h2d", "kernel", "d2h"):
            events.append((op, tile, tile % 3, t, t + dur[op]))
            t += dur[op]
    return events


def draw_box(ax, y, start, end, slot, label, height=0.76):
    ax.add_patch(FancyBboxPatch((start + 0.04, y - height / 2),
                                end - start - 0.08, height,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=SLOT_COLORS[slot],
                                edgecolor="white", lw=1.2))
    ax.text((start + end) / 2, y, label, ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")


def draw_pipeline(path):
    events = pipeline_schedule()
    lanes = ["read", "h2d", "kernel", "d2h"]
    names = {"read": "Host main thread\nread tile into pinned slot",
             "h2d": "H2D stream\n(copy engine)",
             "kernel": "Compute stream\n(kernels)",
             "d2h": "D2H stream\n(copy engine)"}
    fig, (ax, seq) = plt.subplots(2, 1, figsize=(11.5, 7.0),
                                  gridspec_kw={"height_ratios": [4, 1.3]})
    for op, tile, slot, start, end in events:
        draw_box(ax, len(lanes) - 1 - lanes.index(op), start, end, slot,
                 "T%d" % (tile + 1))
    total = max(e[4] for e in events)
    ax.set_xlim(0, total + 0.5)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([names[lane] for lane in reversed(lanes)], fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel("time (schematic, not measured)")
    ax.set_title("Tiled streaming pipeline with overlap: three role streams, "
                 "three buffer slots (schematic, not measured)")
    ax.legend(handles=[Patch(facecolor=c, label="buffer slot %d (tiles %s)" % (
        i, ", ".join("T%d" % (t + 1) for t in range(i, 8, 3))))
        for i, c in enumerate(SLOT_COLORS)], loc="upper right", fontsize=8.5,
        ncol=3, bbox_to_anchor=(1.0, -0.06), frameon=False)
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.01, 0.02, "Events order the stages of a tile; tile k uses slot "
            "k mod 3, so while tile k computes, tile k+1 uploads and tile "
            "k+2 is being read.", transform=ax.transAxes, fontsize=8,
            color="#444", va="bottom")
    stage_label = {"read": "read", "h2d": "H2D", "kernel": "kern",
                   "d2h": "D2H"}
    for op, tile, slot, start, end in sequential_schedule():
        draw_box(seq, 0, start, end, slot,
                 "%s\nT%d" % (stage_label[op], tile + 1), height=0.8)
    seq.set_xlim(0, max(e[4] for e in sequential_schedule()) + 0.3)
    seq.set_ylim(-0.6, 0.6)
    seq.set_yticks([0])
    seq.set_yticklabels(["Without overlap\n(one tile at a time)"],
                        fontsize=9)
    seq.set_xticks([])
    seq.set_xlabel("time (schematic): every stage waits for the previous "
                   "one, so copies and disk reads are not hidden")
    fig.tight_layout()
    save(fig, path)


def save(fig, path, dpi=140):
    for _ in range(6):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if os.path.getsize(path) <= MAX_BYTES:
            break
        dpi = int(dpi * 0.8)
    plt.close(fig)
    print("wrote %s (%.0f KB)" % (path, os.path.getsize(path) / 1000.0))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw algorithm_diagram.png and pipeline_diagram.png.")
    parser.add_argument("--out", default="docs/figures",
                        help="output directory (default docs/figures)")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    draw_algorithm(os.path.join(args.out, "algorithm_diagram.png"))
    draw_pipeline(os.path.join(args.out, "pipeline_diagram.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
