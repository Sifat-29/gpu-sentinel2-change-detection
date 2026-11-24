"""Non-GPU helper tooling. Turns benchmark CSV rows into charts and a summary.

Reads one or more benchmark CSV files (one row per timed repetition; files
written before the ``setup_ms`` column existed are accepted too) and writes
into ``--out``:

  benchmark_summary.csv        one row per configuration: median/min/max of
                               kernel_ms and total_ms, speedups, repeats
  benchmark_backends.png       kernel-only and end-to-end time per backend;
                               the fixed setup time (allocations, table
                               upload, included in the end-to-end time) is
                               hatched inside the end-to-end bar
  benchmark_streams_tiles.png  1 vs 3 streams and the tile-row sweep
  benchmark_fusion.png         fused vs unfused kernel stage breakdown
  benchmark_scaling.png        time against pixels and against dates
  benchmark_block_size.png     time against threads per block
  benchmark_io_threads.png     read and total time against --io-threads

A configuration is the tuple (backend, input_name, height, width, dates,
tile_rows, streams, block_size, threads, io_threads; CSVs written before
the io_threads column existed group with io_threads missing). Charts whose
data are absent are skipped with a printed reason. Speedups are cpu-mt
median divided by the configuration median for the same input (name, size,
dates).

Rows of identical configurations from several CSV files are pooled in the
summary and in every chart except the ones drawn from ``--backends-csv``:
each file listed there gets its own backend chart (benchmark_backends_demo.csv
-> benchmark_backends_demo.png) built from that file's rows only. Without
``--backends-csv`` one pooled benchmark_backends.png is written. Chart
captions list every git commit found in the plotted rows.
"""

import argparse
import csv
import os
import sys
import textwrap

import numpy as np

import cdio

import matplotlib  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIG_KEYS = ("backend", "input_name", "height", "width", "dates",
               "tile_rows", "streams", "block_size", "threads", "io_threads")
DEFAULT_IO_THREADS = 8  # the --io-threads default
BACKEND_ORDER = ["cpu", "cpu-fp32", "cpu-mt", "gpu-unfused", "gpu"]
BACKEND_LABEL = {
    "cpu": "CPU 1 thread\nFP64", "cpu-fp32": "CPU 1 thread\nFP32",
    "cpu-mt": "CPU multi-thread\nFP32", "gpu-unfused": "GPU\nunfused",
    "gpu": "GPU\nfused",
}
KERNEL_COLOR = "#2a6fbb"
TOTAL_COLOR = "#e8833a"
MAX_PNG_BYTES = 2 * 1000 * 1000

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold"})


TEXT_COLUMNS = ("run_id", "timestamp", "mode", "backend", "precision",
                "kernels", "input_name", "compare_to", "gpu_name", "cpu_model",
                "cuda_runtime", "driver_version", "git_commit")


def load_rows(paths):
    """Loads rows from all CSVs; -1 (not applicable) becomes NaN."""
    rows = []
    for path in paths:
        for row in cdio.load_csv(path, TEXT_COLUMNS):
            for key, value in list(row.items()):
                if isinstance(value, (int, float)) and value == -1 and \
                        key not in ("tile_rows", "streams", "block_size",
                                    "threads", "io_threads"):
                    row[key] = float("nan")
                elif value is None:
                    row[key] = float("nan") if key.endswith(
                        ("_ms", "_s", "_mb", "_frac", "diff")) else None
            row["_file"] = path
            rows.append(row)
    return rows


def finite(values):
    arr = np.array([v for v in values if isinstance(v, (int, float))],
                   dtype=float)
    return arr[np.isfinite(arr)]


def stat(values, func):
    arr = finite(values)
    return float(func(arr)) if arr.size else float("nan")


def group_rows(rows):
    groups = {}
    for row in rows:
        key = tuple(row.get(k) for k in CONFIG_KEYS)
        groups.setdefault(key, []).append(row)
    return groups


def build_summary(rows):
    """Returns a list of summary dicts, one per configuration."""
    summary = []
    for key, members in group_rows(rows).items():
        item = dict(zip(CONFIG_KEYS, key))
        first = members[0]
        item.update({
            "pixels": first.get("pixels"), "samples": first.get("samples"),
            "precision": first.get("precision"),
            "kernels": first.get("kernels"), "n_repeats": len(members),
            "mode": first.get("mode"),
        })
        for col in ("kernel_ms", "total_ms", "read_ms"):
            vals = [m.get(col) for m in members]
            item[col + "_median"] = stat(vals, np.median)
            item[col + "_min"] = stat(vals, np.min)
            item[col + "_max"] = stat(vals, np.max)
        item["throughput_msamples_s_median"] = stat(
            [m.get("throughput_msamples_s") for m in members], np.median)
        item["kernel_throughput_msamples_s_median"] = stat(
            [m.get("kernel_throughput_msamples_s") for m in members],
            np.median)
        item["device_mem_peak_mb_max"] = stat(
            [m.get("device_mem_peak_mb") for m in members], np.max)
        item["agree_break_frac_median"] = stat(
            [m.get("agree_break_frac") for m in members], np.median)
        item["max_coef_diff_max"] = stat(
            [m.get("max_coef_diff") for m in members], np.max)
        for col in ("gpu_name", "gpu_mem_mb", "cpu_model", "cpu_threads",
                    "cuda_runtime", "driver_version", "git_commit"):
            item[col] = first.get(col)
        for col in ("setup_ms", "h2d_ms", "ndvi_ms", "fit_ms",
                    "detect_ms", "fused_ms", "d2h_ms"):
            item[col + "_median"] = stat([m.get(col) for m in members],
                                         np.median)
        summary.append(item)
    attach_speedups(summary)
    summary.sort(key=lambda s: (str(s["input_name"]), s["samples"] or 0,
                                BACKEND_ORDER.index(s["backend"])
                                if s["backend"] in BACKEND_ORDER else 99,
                                s["streams"] or 0, s["tile_rows"] or 0,
                                s["block_size"] or 0))
    return summary


def attach_speedups(summary):
    """Adds speedup_{kernel,total}_vs_{cpu_mt,cpu} to each summary dict."""
    base = {}
    for item in summary:
        if item["backend"] in ("cpu-mt", "cpu"):
            key = (item["backend"], item["input_name"], item["height"],
                   item["width"], item["dates"])
            old = base.get(key)
            if old is None or item["n_repeats"] > old["n_repeats"]:
                base[key] = item
    for item in summary:
        for ref, tag in (("cpu-mt", "cpu_mt"), ("cpu", "cpu")):
            ref_item = base.get((ref, item["input_name"], item["height"],
                                 item["width"], item["dates"]))
            for col, name in (("kernel_ms_median", "kernel"),
                              ("total_ms_median", "total")):
                value = float("nan")
                if ref_item is not None and item[col] > 0:
                    value = ref_item[col] / item[col]
                item["speedup_%s_vs_%s" % (name, tag)] = value


def write_summary(summary, path):
    if not summary:
        return
    columns = list(summary[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for item in summary:
            writer.writerow(["" if _is_nan(item[c]) else
                             ("%.6g" % item[c] if isinstance(item[c], float)
                              else item[c]) for c in columns])


def _is_nan(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def hardware_caption(rows):
    """Hardware / software caption built from the CSV columns."""
    first = rows[0]
    parts = []
    gpu = first.get("gpu_name")
    if gpu and gpu != "none":
        mem = first.get("gpu_mem_mb")
        parts.append("GPU: %s%s" % (gpu, " (%d MiB)" % mem if isinstance(
            mem, (int, float)) and mem > 0 else ""))
    if first.get("cpu_model"):
        parts.append("CPU: %s, %s threads" % (first["cpu_model"],
                                              first.get("cpu_threads")))
    commits = sorted({str(r["git_commit"]) for r in rows
                      if r.get("git_commit")})
    versions = [("CUDA runtime", first.get("cuda_runtime")),
                ("driver", first.get("driver_version")),
                ("commit" if len(commits) < 2 else "commits",
                 ", ".join(commits))]
    parts.append(", ".join("%s %s" % (k, v) for k, v in versions if v))
    return " | ".join(p for p in parts if p)


def pick_main_input(summary):
    """Input with the most backends (largest by samples on ties); needs at
    least two backends."""
    sizes = {}
    for item in summary:
        key = (item["input_name"], item["height"], item["width"],
               item["dates"])
        sizes.setdefault(key, set()).add(item["backend"])
    ok = [k for k, v in sizes.items() if len(v) >= 2]
    if not ok:
        return None
    return max(ok, key=lambda k: (len(sizes[k]), k[1] * k[2] * k[3]))


def same_input(item, key):
    return (item["input_name"], item["height"], item["width"],
            item["dates"]) == key


def has_default_io(item):
    """True for the default reader setting (and for older CSVs, which have no
    io_threads column)."""
    return item.get("io_threads") in (None, DEFAULT_IO_THREADS)


def default_config_rank(item):
    """Larger is better: prefer 3 streams, block 256, default reader threads,
    most repeats."""
    return (item["n_repeats"], item["streams"] == 3,
            item["block_size"] == 256, has_default_io(item),
            -(item["total_ms_median"] or 0))


def default_per_backend(summary, key):
    best = {}
    for item in summary:
        if not same_input(item, key):
            continue
        cur = best.get(item["backend"])
        if cur is None or default_config_rank(item) > default_config_rank(cur):
            best[item["backend"]] = item
    return best


def sweep_rows(rows, columns):
    """Rows of the CSV files in which the fused GPU rows of one input vary in
    any of `columns` (the sweep that varies them), so that charts of a sweep
    do not pool rows of identical configurations from other sweeps. Falls
    back to all rows when no file varies them."""
    chosen = set()
    for path in {r["_file"] for r in rows}:
        values = {}
        for row in rows:
            if row["_file"] == path and row["backend"] == "gpu":
                for col in columns:
                    values.setdefault((row["input_name"], col),
                                      set()).add(row.get(col))
        if any(len(v) > 1 for v in values.values()):
            chosen.add(path)
    return [r for r in rows if r["_file"] in chosen] or rows


def source_text(rows):
    """'Rows from: a.csv, b.csv.' for captions."""
    return "Rows from: %s." % ", ".join(
        sorted({os.path.basename(r["_file"]) for r in rows}))


def caption_text(text, width=165):
    """Wraps each line of a figure caption so it never widens the figure."""
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def save(fig, path, dpi=140):
    for _ in range(6):
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if os.path.getsize(path) <= MAX_PNG_BYTES:
            break
        dpi = int(dpi * 0.8)
    plt.close(fig)
    print("wrote %s (%.0f KB)" % (path, os.path.getsize(path) / 1000.0))


def whisker(item, col):
    med, lo, hi = (item[col + "_median"], item[col + "_min"],
                   item[col + "_max"])
    return med, max(0.0, med - lo), max(0.0, hi - med)


def input_text(key):
    return "%s %d x %d x %d dates (%.1f M samples)" % (
        key[0], key[1], key[2], key[3], key[1] * key[2] * key[3] / 1e6)


def format_ms(value):
    return "%.0f ms" % value if value >= 100 else "%.2f ms" % value


def speedup_text(value):
    """Speedup vs cpu-mt as a short label (e.g. 53.3x, 0.17x)."""
    return ("%.1fx" if value >= 10 else "%.2fx") % value


def repeats_text(item):
    """'n=5' or 'single run' for a backend bar."""
    n = item["n_repeats"]
    return "single run" if n == 1 else "n=%d" % n


def plot_backends(summary, rows, out, log_mode,
                  name="benchmark_backends.png", note=None):
    key = pick_main_input(summary)
    if key is None:
        print("SKIP %s: no input with two or more backends" % name)
        return
    best = default_per_backend(summary, key)
    names = [b for b in BACKEND_ORDER if b in best]
    names += [b for b in best if b not in names]
    vals = [best[b] for b in names]
    kern = [whisker(v, "kernel_ms") for v in vals]
    tot = [whisker(v, "total_ms") for v in vals]
    all_pos = [m for m, _, _ in kern + tot if m > 0 and np.isfinite(m)]
    use_log = (log_mode == "on" or (log_mode == "auto" and all_pos and
                                    max(all_pos) / min(all_pos) > 30))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(names))
    width = 0.38
    for offset, data, color, label in ((-width / 2, kern, KERNEL_COLOR,
                                        "kernel only"),
                                       (width / 2, tot, TOTAL_COLOR,
                                        "end to end (incl. I/O, copies)")):
        meds = np.array([d[0] for d in data])
        err = np.array([[d[1] for d in data], [d[2] for d in data]])
        ax.bar(x + offset, np.nan_to_num(meds), width, color=color,
               label=label, yerr=err, capsize=3,
               error_kw={"elinewidth": 1.1, "ecolor": "#333"})
    if use_log:
        ax.set_yscale("log")
    top = max(m + hi for m, _, hi in kern + tot if np.isfinite(m))
    ax.set_ylim(top=top * (12 if use_log else 1.3))
    for i, v in enumerate(vals):
        for off, col, sp in (
                (-width / 2, "kernel", "speedup_kernel_vs_cpu_mt"),
                (width / 2, "total", "speedup_total_vs_cpu_mt")):
            med, _, hi = whisker(v, col + "_ms")
            text = format_ms(med)
            if np.isfinite(v[sp]) and v["backend"] != "cpu-mt":
                text += "\n%s" % speedup_text(v[sp])
            setup_med = v.get("setup_ms_median", float("nan"))
            if col == "total" and np.isfinite(setup_med):
                text += "\nsetup %s" % format_ms(setup_med)
            ax.text(x[i] + off, (med + hi) * (1.08 if use_log else 1.0) +
                    (0 if use_log else 0.01 * top), text, ha="center",
                    va="bottom", fontsize=7)
    # Fixed setup cost (buffer allocation, constant-table upload) is part of
    # the end-to-end time: hatch it from zero up to its median. Files written
    # before the setup_ms column existed give NaN and draw nothing.
    setup = np.array([v.get("setup_ms_median", float("nan")) for v in vals],
                     dtype=float)
    if np.isfinite(setup).any():
        ax.bar(x + width / 2, np.nan_to_num(setup), width, fill=False,
               hatch="///", edgecolor="#7a3b0c", linewidth=0.0,
               label="of which setup (allocation, table upload)")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(%s)" % (BACKEND_LABEL.get(n, n),
                                      repeats_text(best[n])) for n in names])
    ax.set_ylabel("time (ms, %s)" % ("log scale" if use_log else "linear"))
    ax.set_title("Backend comparison: kernel-only vs end-to-end time "
                 "(labels: median time and speedup vs cpu-mt)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    repeats = ", ".join("%s %s" % (n.replace("\n", " "), repeats_text(
        best[n])) for n in names)
    files = sorted({os.path.basename(r["_file"]) for r in rows})
    caption = ("Input: %s. Bars = median of the timed repeats per backend "
               "(%s); one warm-up run is discarded when there are repeats; "
               "a single run has no warm-up and no whiskers. Whiskers = "
               "min-max. Speedup = cpu-mt median / backend median (cpu-mt is "
               "the headline CPU baseline). Hatched part of the end-to-end "
               "bar = fixed setup time.\nRows from: %s.%s\n%s" % (
                   input_text(key), repeats, ", ".join(files),
                   " " + note if note else "", hardware_caption(rows)))
    fig.text(0.01, -0.02, caption_text(caption, 190), fontsize=7.5,
             va="top")
    save(fig, os.path.join(out, name))


def gpu_fused(summary, key):
    """Fused-GPU configurations of an input with the default reader setting."""
    return [s for s in summary if same_input(s, key) and
            s["backend"] == "gpu" and has_default_io(s)]


def pick_gpu_input(summary, field_a, field_b=None):
    """Largest input whose fused GPU rows vary in field_a (or field_b)."""
    best = None
    for key in {(s["input_name"], s["height"], s["width"], s["dates"])
                for s in summary if s["backend"] == "gpu"}:
        items = gpu_fused(summary, key)
        varied = len({i[field_a] for i in items}) > 1 or (
            field_b and len({i[field_b] for i in items}) > 1)
        size = key[1] * key[2] * key[3]
        if varied and (best is None or size > best[0]):
            best = (size, key)
    return best[1] if best else None


def plot_streams_tiles(summary, rows, out):
    """Left: 1 vs 3 streams on the largest input that varies the stream
    count. Right: tile-row sweep on the largest input that varies it (the
    two inputs can differ, and are named in the panel titles)."""
    rows_s = sweep_rows(rows, ("streams",))
    rows_t = sweep_rows(rows, ("tile_rows",))
    sum_s, sum_t = build_summary(rows_s), build_summary(rows_t)
    key_s = pick_gpu_input(sum_s, "streams")
    key_t = pick_gpu_input(sum_t, "tile_rows")
    if key_s is None and key_t is None:
        print("SKIP benchmark_streams_tiles.png: no fused-GPU rows varying "
              "streams or tile_rows")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.5]})
    notes = []
    by_stream = []
    if key_s is not None:
        items = gpu_fused(sum_s, key_s)
        blocks = [i for i in items if i["block_size"] == 256] or items
        tiles = sorted({i["tile_rows"] for i in blocks})
        # Prefer the tile size the default (auto) run used, else the median.
        ref_tile = max(tiles, key=lambda t: sum(
            i["n_repeats"] for i in blocks if i["tile_rows"] == t))
        by_stream = sorted([i for i in blocks if i["tile_rows"] == ref_tile],
                           key=lambda i: i["streams"])
    if len(by_stream) > 1:
        labels = ["%d stream%s" % (i["streams"], "" if i["streams"] == 1
                                   else "s") for i in by_stream]
        meds = [whisker(i, "total_ms") for i in by_stream]
        ax1.bar(labels, [m[0] for m in meds],
                yerr=[[m[1] for m in meds], [m[2] for m in meds]], capsize=4,
                color=["#9aa7b8", TOTAL_COLOR][:len(labels)] + [TOTAL_COLOR] *
                max(0, len(labels) - 2))
        base = meds[0][0]
        for i, m in enumerate(meds):
            ax1.text(i, m[0] + m[2], "%s\n%.2fx" % (format_ms(m[0]),
                     base / m[0]), ha="center", va="bottom", fontsize=8)
        ax1.set_ylim(top=max(m[0] + m[2] for m in meds) * 1.3)
        ax1.set_title("Streams: %s %d x %d, tile rows %d" % (
            key_s[0], key_s[1], key_s[2], ref_tile), fontsize=10)
        ax1.set_ylabel("end-to-end time (ms)")
        ax1.grid(axis="y", alpha=0.3)
        notes.append("Left: %s." % input_text(key_s))
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "only one stream setting measured", ha="center")
    if key_t is not None:
        blocks = [i for i in gpu_fused(sum_t, key_t)
                  if i["block_size"] == 256] or gpu_fused(sum_t, key_t)
        for streams, color in ((1, "#7f8fa6"), (3, TOTAL_COLOR)):
            pts = sorted([i for i in blocks if i["streams"] == streams],
                         key=lambda i: i["tile_rows"])
            if len(pts) < 2:
                continue
            xs = [p["tile_rows"] for p in pts]
            w = [whisker(p, "total_ms") for p in pts]
            ax2.errorbar(xs, [m[0] for m in w],
                         yerr=[[m[1] for m in w], [m[2] for m in w]],
                         marker="o", capsize=3, color=color,
                         label="%d stream%s" % (streams, "" if streams == 1
                                                else "s"))
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(sorted({p["tile_rows"] for p in blocks}))
        ax2.set_xticklabels([str(t) for t in sorted(
            {p["tile_rows"] for p in blocks})])
        ax2.set_xlabel("tile rows (rows per tile, full width)")
        ax2.set_ylabel("end-to-end time (ms)")
        ax2.set_title("Tile-size sweep: %s %d x %d" % (
            key_t[0], key_t[1], key_t[2]), fontsize=10)
        ax2.grid(alpha=0.3)
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend()
        notes.append("Right: %s." % input_text(key_t))
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "tile size not varied", ha="center")
    fig.suptitle("GPU streaming: streams and tile size", fontsize=13,
                 fontweight="bold")
    fig.text(0.01, -0.01, caption_text(
        "%s Fused kernel, block size 256, median with min-max whiskers. "
        "Left %s Right %s\n%s" % (
            " ".join(notes), source_text(rows_s).lower(),
            source_text(rows_t).lower(),
            hardware_caption(rows_s + rows_t))), fontsize=7.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, os.path.join(out, "benchmark_streams_tiles.png"))


def plot_fusion(summary, rows, out):
    key = None
    for cand in {(s["input_name"], s["height"], s["width"], s["dates"])
                 for s in summary}:
        backends = {s["backend"] for s in summary if same_input(s, cand)}
        if {"gpu", "gpu-unfused"} <= backends and (
                key is None or cand[1] * cand[2] * cand[3] >
                key[1] * key[2] * key[3]):
            key = cand
    if key is None:
        print("SKIP benchmark_fusion.png: need both gpu and gpu-unfused rows "
              "on the same input")
        return
    best = default_per_backend(summary, key)
    unf, fus = best["gpu-unfused"], best["gpu"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    stages = [("ndvi_ms_median", "NDVI kernel", "#8dc6a8"),
              ("fit_ms_median", "fit kernel", "#4a90c2"),
              ("detect_ms_median", "detect kernel", "#6a5acd"),
              ("fused_ms_median", "fused kernel", TOTAL_COLOR)]
    bottoms = np.zeros(2)
    for col, label, color in stages:
        vals = np.array([np.nan_to_num(unf[col]), np.nan_to_num(fus[col])])
        if vals.sum() <= 0:
            continue
        ax1.bar(["unfused\n(3 kernels)", "fused\n(1 kernel)"], vals,
                bottom=bottoms, label=label, color=color, width=0.55)
        bottoms += vals
    ax1.set_ylabel("kernel time (ms, median)")
    ax1.set_title("Kernel-stage breakdown")
    for i, tot in enumerate(bottoms):
        ax1.text(i, tot, format_ms(tot), ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(top=bottoms.max() * 1.15)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    w = [whisker(unf, "total_ms"), whisker(fus, "total_ms")]
    ax2.bar(["unfused", "fused"], [m[0] for m in w],
            yerr=[[m[1] for m in w], [m[2] for m in w]], capsize=4,
            color=["#9aa7b8", TOTAL_COLOR], width=0.55)
    for i, m in enumerate(w):
        ax2.text(i, m[0] + m[2], format_ms(m[0]), ha="center", va="bottom",
                 fontsize=9)
    ax2.set_ylim(top=max(m[0] + m[2] for m in w) * 1.15)
    ax2.set_title("End-to-end time")
    ax2.set_ylabel("ms (median, min-max whiskers)")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Effect of kernel fusion (GPU)", fontsize=13,
                 fontweight="bold")
    kernel_speed = unf["kernel_ms_median"] / fus["kernel_ms_median"]
    fig.text(0.01, -0.01, caption_text(
        "Input: %s. Fused kernel is %.2fx faster kernel-only and %.2fx end "
        "to end. Stage times are not additive with disk reads and copies, "
        "which overlap with compute when streaming. %s\n%s" % (
            input_text(key), kernel_speed,
            unf["total_ms_median"] / fus["total_ms_median"],
            source_text(rows), hardware_caption(rows)), 150),
        fontsize=7.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, os.path.join(out, "benchmark_fusion.png"))


def scaling_series(summary, backend, fixed, varied):
    """(x list, items) for default configs of backend varying `varied` while
    the other of (pixels, dates) equals `fixed`."""
    best = {}
    for item in summary:
        if item["backend"] != backend:
            continue
        pixels, dates = item["height"] * item["width"], item["dates"]
        if (dates if varied == "pixels" else pixels) != fixed:
            continue
        x = pixels if varied == "pixels" else dates
        if x not in best or default_config_rank(item) > default_config_rank(
                best[x]):
            best[x] = item
    xs = sorted(best)
    return xs, [best[x] for x in xs]


def plot_scaling(summary, rows, out):
    gpu_cpu = [b for b in ("gpu", "cpu-mt") if any(
        s["backend"] == b for s in summary)]
    panels = []
    for varied, other in (("pixels", "dates"), ("dates", "pixels")):
        best = None
        for fixed in {(s["dates"] if varied == "pixels"
                       else s["height"] * s["width"]) for s in summary}:
            n = max((len(scaling_series(summary, b, fixed, varied)[0])
                     for b in gpu_cpu), default=0)
            if n >= 2 and (best is None or n > best[0]):
                best = (n, fixed)
        if best:
            panels.append((varied, best[1]))
    if not panels:
        print("SKIP benchmark_scaling.png: need >= 2 sizes or date counts "
              "for gpu or cpu-mt")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.8),
                             squeeze=False)
    colors = {"gpu": KERNEL_COLOR, "cpu-mt": "#c0392b"}
    for ax, (varied, fixed) in zip(axes[0], panels):
        for backend in gpu_cpu:
            xs, items = scaling_series(summary, backend, fixed, varied)
            if len(xs) < 2:
                continue
            for col, style, tag in (("total_ms", "-", "end to end"),
                                    ("kernel_ms", "--", "kernel only")):
                w = [whisker(i, col) for i in items]
                ax.errorbar(xs, [m[0] for m in w],
                            yerr=[[m[1] for m in w], [m[2] for m in w]],
                            marker="o", ls=style, capsize=2,
                            color=colors[backend],
                            label="%s %s" % (backend, tag))
        ax.set_xscale("log")
        ax.set_yscale("log")
        all_x = sorted({x for b in gpu_cpu for x in scaling_series(
            summary, b, fixed, varied)[0]})
        ax.set_xticks(all_x)
        ax.set_xticklabels([("%.1f M" % (x / 1e6)) if varied == "pixels"
                            else str(x) for x in all_x], fontsize=8)
        ax.minorticks_off()
        ax.grid(alpha=0.3, which="major")
        ax.set_ylabel("time (ms)")
        if varied == "pixels":
            ax.set_xlabel("pixels per scene (H x W)")
            ax.set_title("Time vs pixels (%d dates)" % fixed)
        else:
            ax.set_xlabel("number of dates T")
            ax.set_title("Time vs dates (%.1f M pixels)" % (fixed / 1e6))
        ax.legend(fontsize=8)
    fig.suptitle("Scaling with problem size", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.01, caption_text(
        "Median with min-max whiskers; log-log axes. Only measured sizes "
        "are plotted (no extrapolation). Rows of identical configurations "
        "from different CSV files are pooled (the point shared by both "
        "panels).\n"
        "%s" %
        hardware_caption(rows)), fontsize=7.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, os.path.join(out, "benchmark_scaling.png"))


def plot_block_size(summary, rows, out):
    rows = sweep_rows(rows, ("block_size",))
    summary = build_summary(rows)
    key = pick_gpu_input(summary, "block_size")
    if key is None:
        print("SKIP benchmark_block_size.png: no fused-GPU rows varying "
              "block_size")
        return
    items = [i for i in gpu_fused(summary, key)
             if i["streams"] == 3 or len({j["streams"] for j in gpu_fused(
                 summary, key)}) == 1]
    best = {}
    for item in items:
        cur = best.get(item["block_size"])
        if cur is None or default_config_rank(item) > default_config_rank(cur):
            best[item["block_size"]] = item
    xs = sorted(best)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for col, color, label in (("kernel_ms", KERNEL_COLOR, "kernel only"),
                              ("total_ms", TOTAL_COLOR, "end to end")):
        w = [whisker(best[x], col) for x in xs]
        ax.errorbar(range(len(xs)), [m[0] for m in w],
                    yerr=[[m[1] for m in w], [m[2] for m in w]], marker="o",
                    capsize=3, color=color, label=label)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("threads per block")
    ax.set_yscale("log")
    ax.set_ylabel("time (ms, log scale)")
    ax.set_title("Block-size sweep (fused GPU kernel)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.text(0.01, -0.01, caption_text(
        "Input: %s. Median with min-max whiskers. %s\n%s" % (
            input_text(key), source_text(rows), hardware_caption(rows)),
        100), fontsize=7.5,
        va="top")
    save(fig, os.path.join(out, "benchmark_block_size.png"))


def plot_io_threads(summary, rows, out):
    """Read time and end-to-end time against parallel disk reads per tile."""
    rows = sweep_rows(rows, ("io_threads",))
    summary = build_summary(rows)
    series = {}
    for item in summary:
        if item.get("io_threads") is None or item["backend"] not in (
                "gpu", "cpu-mt"):
            continue
        key = (item["input_name"], item["height"], item["width"],
               item["dates"])
        series.setdefault(key, {}).setdefault(item["backend"], []).append(
            item)
    candidates = [k for k, v in series.items()
                  if any(len({i["io_threads"] for i in items}) > 1
                         for items in v.values())]
    if not candidates:
        print("SKIP benchmark_io_threads.png: no rows varying io_threads")
        return
    key = max(candidates, key=lambda k: k[1] * k[2] * k[3])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    colors = {"gpu": TOTAL_COLOR, "cpu-mt": KERNEL_COLOR}
    for backend, items in sorted(series[key].items()):
        items = sorted(items, key=lambda i: i["io_threads"])
        xs = [str(int(i["io_threads"])) for i in items]
        for ax, col in ((ax1, "read_ms"), (ax2, "total_ms")):
            w = [whisker(i, col) for i in items]
            ax.errorbar(range(len(xs)), [m[0] for m in w],
                        yerr=[[m[1] for m in w], [m[2] for m in w]],
                        marker="o", capsize=3, color=colors[backend],
                        label=backend)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(xs)
    ax1.set_title("Time spent reading tiles")
    ax2.set_title("End-to-end time")
    for ax in (ax1, ax2):
        ax.set_xlabel("parallel disk reads per tile (--io-threads)")
        ax.set_ylabel("ms (median, min-max whiskers)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Parallel tile reader", fontsize=13, fontweight="bold")
    fig.text(0.01, -0.01, caption_text("Input: %s. %s\n%s" % (
        input_text(key), source_text(rows), hardware_caption(rows)), 150),
        fontsize=7.5, va="top")
    save(fig, os.path.join(out, "benchmark_io_threads.png"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot benchmark CSV rows and write "
                    "benchmark_summary.csv.")
    parser.add_argument("--csv", nargs="+", required=True,
                        help="benchmark CSV file(s)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--summary-out",
                        help="where to write benchmark_summary.csv "
                        "(default: <out>/benchmark_summary.csv)")
    parser.add_argument("--backends-csv", nargs="+", metavar="CSV",
                        help="also draw one backend chart per listed CSV, "
                        "using only that file's rows (so rows of identical "
                        "default configurations from other sweeps are not "
                        "pooled); the chart is named after the file, e.g. "
                        "benchmark_backends_demo.csv -> "
                        "benchmark_backends_demo.png")
    parser.add_argument("--backends-note", action="append", default=[],
                        metavar="STEM=TEXT", help="sentence appended to the "
                        "caption of the --backends-csv chart whose file "
                        "stem is STEM (e.g. cache state of the run)")
    parser.add_argument("--log-scale", choices=("auto", "on", "off"),
                        default="auto", help="log y axis of the backend "
                        "chart (auto: when the range exceeds 30x)")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = load_rows(args.csv)
    if not rows:
        print("error: no benchmark rows in the given CSV files")
        return 1
    summary = build_summary(rows)
    summary_path = args.summary_out or os.path.join(args.out,
                                                    "benchmark_summary.csv")
    write_summary(summary, summary_path)
    print("wrote %s (%d configurations from %d rows)" % (
        summary_path, len(summary), len(rows)))
    if args.backends_csv:
        notes = dict(n.split("=", 1) for n in args.backends_note if "=" in n)
        for path in args.backends_csv:
            subset = [r for r in rows if r["_file"] == path]
            if not subset:
                print("SKIP %s: not among the --csv files" % path)
                continue
            stem = os.path.splitext(os.path.basename(path))[0]
            plot_backends(build_summary(subset), subset, args.out,
                          args.log_scale, stem + ".png", notes.get(stem))
    else:
        plot_backends(summary, rows, args.out, args.log_scale)
    plot_streams_tiles(summary, rows, args.out)
    if args.backends_csv:
        fusion_rows = [r for r in rows if r["_file"] in args.backends_csv]
        plot_fusion(build_summary(fusion_rows), fusion_rows, args.out)
    else:
        plot_fusion(summary, rows, args.out)
    plot_scaling(summary, rows, args.out)
    plot_block_size(summary, rows, args.out)
    plot_io_threads(summary, rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
