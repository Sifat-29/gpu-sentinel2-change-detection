"""Non-GPU helper tooling. Fills measured results into README and docs.

Replaces the text between ``<!-- AUTO:<name> -->`` and
``<!-- /AUTO:<name> -->`` markers in Markdown files with tables built only
from the given benchmark summary CSV (plot_benchmarks.py) and the run logs
(docs/FORMATS.md, "Log format"). Marker names:

  bench_table   backend comparison on the largest multi-backend input
  headline_run  device, sizes, tiles, streams, stage timings and the SUMMARY
                line of the largest GPU detect log
  accuracy      agreement of runs with a reference backend, synthetic recovery
  validation    precision / recall / F1 / year agreement against the reference
  hardware      GPU, CPU, CUDA runtime, driver and commit of the measurements

Nothing is invented: a table or cell without data reads ``TO BE MEASURED``.
The script is idempotent. With ``--check`` nothing is written and the exit
status is 1 if any marker would still contain ``TO BE MEASURED``.
"""

import argparse
import os
import re
import sys

import cdio
import plot_benchmarks as pb

MISSING = "TO BE MEASURED"
MARKER_RE = re.compile(
    r"(<!-- AUTO:([A-Za-z0-9_]+) -->)(.*?)(<!-- /AUTO:\2 -->)", re.DOTALL)
BACKEND_NAMES = {
    "cpu": "CPU, 1 thread, FP64", "cpu-fp32": "CPU, 1 thread, FP32",
    "cpu-mt": "CPU, multi-thread, FP32", "gpu-unfused": "GPU, unfused kernels",
    "gpu": "GPU, fused kernel",
}


def escape(text):
    return str(text).replace("|", "\\|")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(escape(c) for c in row) + " |")
    return "\n".join(lines)


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and value != value):
        return MISSING
    if isinstance(value, float):
        return ("%.0f" % value) if abs(value) >= 1000 else (
            "%.*f" % (digits, value))
    return str(value)


def commas(value):
    return "{:,}".format(int(value)) if value is not None else MISSING


# ------------------------------------------------------------- data loading

def load_logs(directory):
    """Parses every change_detect log under directory (sorted by path)."""
    logs = []
    if not directory or not os.path.isdir(directory):
        return logs
    for root, _, files in sorted(os.walk(directory)):
        for name in sorted(files):
            if not name.endswith((".txt", ".log")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as h:
                    first = h.readline()
            except OSError:
                continue
            if first.startswith("=== change_detect"):
                logs.append(cdio.parse_log(path))
    return logs


def load_summary(path):
    if not path or not os.path.exists(path):
        return []
    return cdio.load_csv(path, pb.TEXT_COLUMNS)


def log_backend(log):
    value = log.get("backend", "")
    return value.split(" (")[0].split()[0] if value else (
        log.summary["backend"] if log.summary else "")


# --------------------------------------------------------------- the tables

def to_items(rows):
    """Summary CSV rows -> dicts with the keys plot_benchmarks expects."""
    items = []
    for row in rows:
        item = dict(row)
        for key, value in item.items():
            if value is None and key not in pb.TEXT_COLUMNS:
                item[key] = float("nan")
        items.append(item)
    return items


def speed(value):
    return ("%.2fx" % value) if value == value else MISSING


def bench_table(summary, logs):
    items = to_items(summary)
    key = pb.pick_main_input(items) if items else None
    if key is None:
        return MISSING + " (needs a benchmark_summary.csv with at least two " \
            "backends on one input)"
    best = pb.default_per_backend(items, key)
    rows = []
    for backend in pb.BACKEND_ORDER:
        item = best.get(backend)
        if item is None:
            continue
        kern = "%s [%s-%s]" % (fmt(item["kernel_ms_median"]),
                               fmt(item["kernel_ms_min"]),
                               fmt(item["kernel_ms_max"]))
        tot = "%s [%s-%s]" % (fmt(item["total_ms_median"]),
                              fmt(item["total_ms_min"]),
                              fmt(item["total_ms_max"]))
        rows.append([BACKEND_NAMES.get(backend, backend), kern, tot,
                     fmt(item["throughput_msamples_s_median"], 1),
                     "%s / %s" % (speed(item["speedup_kernel_vs_cpu_mt"]),
                                  speed(item["speedup_total_vs_cpu_mt"])),
                     speed(item["speedup_total_vs_cpu"]),
                     str(item["n_repeats"])])
    head = ("Input: %s %d x %d, %d dates (%s samples). Times in ms, median "
            "[min-max] over the repeats." % (key[0], key[1], key[2], key[3],
                                             commas(key[1] * key[2] * key[3])))
    body = table(["Backend", "Kernel only (ms)", "End to end (ms)",
                  "End-to-end throughput (M samples/s)",
                  "Speedup vs cpu-mt (kernel / end to end)",
                  "End-to-end speedup vs CPU FP64", "Repeats"], rows)
    return head + "\n\n" + body


def headline_log(logs):
    """Largest GPU detect-mode log that has a SUMMARY line."""
    best = None
    for log in logs:
        if log.mode != "detect" or log.summary is None:
            continue
        if not log_backend(log).startswith("gpu"):
            continue
        if best is None or log.summary["samples"] > best.summary["samples"]:
            best = log
    return best


def headline_run(summary, logs):
    log = headline_log(logs)
    if log is None:
        return MISSING + " (no GPU detect log with a SUMMARY line found)"
    timings = log.timings
    rows = [
        ["Log file", os.path.basename(log.path)],
        ["Command", "`%s`" % log.command],
        ["Device", log.get("gpu", MISSING)],
        ["CPU", log.get("cpu", MISSING)],
        ["Scenes (dates) T", log.get("scenes (dates) T", MISSING)],
        ["Image size H x W", log.get("image size H x W", MISSING)],
        ["Pixels", commas(log.get_int("pixels"))],
        ["Pixel time series", commas(log.get_int("pixel time series"))],
        ["Samples (pixel-dates)",
         commas(log.get_int("samples (pixel-dates)"))],
        ["Backend", log.get("backend", MISSING)],
        ["Tiles / tile rows / streams", "%s / %s / %s" % (
            log.get("tiles", MISSING), log.get("tile_rows", MISSING),
            log.get("streams", MISSING))],
    ]
    for key, label in (("read_input", "read input"), ("h2d", "host to device"),
                       ("ndvi", "NDVI kernel"), ("fit", "fit kernel"),
                       ("detect", "detect kernel"), ("fused", "fused kernel"),
                       ("d2h", "device to host"),
                       ("kernel_total", "kernel total"),
                       ("backend_total", "backend total (read to results)"),
                       ("post", "post-processing"),
                       ("write_outputs", "write outputs"),
                       ("wall_total", "wall total")):
        if key in timings:
            rows.append(["Time: %s (ms)" % label, fmt(timings[key])])
    rows.append(["Throughput", log.get("throughput", MISSING)])
    rows.append(["Peak device memory", log.get("device_memory_peak", MISSING)])
    result_line = log.summary["text"][len("SUMMARY: "):]
    rows.append(["Result line", "`%s`" % result_line])
    return table(["Item", "Value"], rows)


_AGREE_RE = re.compile(
    r"identical_break_index ([0-9.eE+-]+) \| identical_status ([0-9.eE+-]+)"
    r"(?: \| max_coef_diff ([0-9.eE+-]+))?")


def accuracy(summary, logs):
    rows = []
    for log in logs:
        for section in log.sections.values():
            for key, value in section.items():
                if key.startswith("agreement_vs_"):
                    match = _AGREE_RE.search(value)
                    if not match:
                        continue
                    text = "identical break index %.2f%%, identical status " \
                        "%.2f%%" % (100 * float(match.group(1)),
                                    100 * float(match.group(2)))
                    if match.group(3):
                        text += ", max coefficient diff %s" % match.group(3)
                    rows.append([os.path.basename(log.path),
                                 log_backend(log),
                                 "vs " + key[len("agreement_vs_"):], text])
                elif key == "synthetic_recovery":
                    rows.append([os.path.basename(log.path),
                                 log_backend(log), "synthetic recovery",
                                 value.replace(" | ", "; ")])
    if not rows:
        return MISSING + " (no agreement or recovery lines in the logs)"
    return table(["Log", "Backend", "Check", "Result"], rows)


_VALID_NUM = re.compile(r"(precision|recall|f1|year_exact|year_within1) "
                        r"([0-9.]+)")
_VALID_COUNTS = re.compile(r"tp (\d+) fp (\d+) fn (\d+)")


def percent(values, key):
    return "%.1f%%" % (100 * float(values[key])) if key in values else MISSING


def validation(summary, logs):
    rows = []
    for log in logs:
        text = log.get("validation")
        if not text:
            continue
        values = dict(_VALID_NUM.findall("precision " + text))
        counts = _VALID_COUNTS.search(text)
        if "precision" not in values or "recall" not in values:
            continue
        rows.append([
            os.path.basename(log.path), log_backend(log),
            "%.3f" % float(values["precision"]),
            "%.3f" % float(values["recall"]),
            "%.3f" % float(values.get("f1", 0)),
            percent(values, "year_exact"), percent(values, "year_within1"),
            "%s / %s / %s" % counts.groups() if counts else MISSING])
    if not rows:
        return MISSING + " (no validation line in the logs)"
    return table(["Log", "Backend", "Precision", "Recall", "F1",
                  "Year exact", "Year within 1", "TP / FP / FN"], rows)


def hardware(summary, logs):
    rows = []
    first = summary[0] if summary else None
    headline = headline_log(logs) or (logs[0] if logs else None)

    def pick(csv_key, log_key):
        if first and first.get(csv_key) not in (None, ""):
            return str(first[csv_key])
        if headline is not None and headline.get(log_key):
            return headline.get(log_key)
        return MISSING

    gpu = pick("gpu_name", "gpu")
    if first and first.get("gpu_mem_mb") not in (None, "") and \
            isinstance(first.get("gpu_mem_mb"), (int, float)):
        gpu += " (%d MiB)" % first["gpu_mem_mb"]
    cpu = pick("cpu_model", "cpu")
    if first and first.get("cpu_threads") not in (None, "") and \
            first.get("cpu_model"):
        cpu += ", %s threads" % first["cpu_threads"]
    rows = [["GPU", gpu], ["CPU", cpu],
            ["CUDA runtime", pick("cuda_runtime", "runtime")],
            ["Driver", pick("driver_version", "driver")],
            ["Git commit", pick("git_commit", "git_commit")]]
    return table(["Item", "Value"], rows)


BUILDERS = {"bench_table": bench_table, "headline_run": headline_run,
            "accuracy": accuracy, "validation": validation,
            "hardware": hardware}


def fill_text(text, summary, logs, source):
    """Returns (new text, [(marker, has_missing)])."""
    seen = []

    def replace(match):
        name = match.group(2)
        builder = BUILDERS.get(name)
        if builder is None:
            print("warning: %s: unknown marker '%s', left unchanged" % (
                source, name))
            return match.group(0)
        body = builder(summary, logs)
        seen.append((name, MISSING in body))
        return "%s\n%s\n%s" % (match.group(1), body, match.group(4))

    return MARKER_RE.sub(replace, text), seen


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fill AUTO markers in Markdown files from the benchmark "
                    "summary CSV and the run logs.")
    parser.add_argument("--readme", required=True, help="README.md to fill")
    parser.add_argument("--docs", nargs="*", default=[],
                        help="further Markdown files to fill")
    parser.add_argument("--summary", help="benchmark_summary.csv")
    parser.add_argument("--logs", help="directory with change_detect logs")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if a marker would still "
                             "contain TO BE MEASURED")
    args = parser.parse_args(argv)
    summary = load_summary(args.summary)
    logs = load_logs(args.logs)
    print("inputs: %d summary rows, %d logs" % (len(summary), len(logs)))
    status = 0
    for path in [args.readme] + list(args.docs):
        if not os.path.exists(path):
            print("error: %s not found" % path)
            return 2
        with open(path, "r", encoding="utf-8") as handle:
            old = handle.read()
        new, seen = fill_text(old, summary, logs, path)
        for name, missing in seen:
            print("%s: %s %s" % (path, name, MISSING if missing else "filled"))
            if missing:
                status = 1
        if not args.check and new != old:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(new)
            print("updated %s" % path)
    if args.check:
        print("CHECK: %s" % ("FAIL (TO BE MEASURED remains)" if status
                             else "PASS"))
        return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
