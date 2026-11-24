"""Non-GPU helper tooling. Checks that a benchmark CSV is complete and sane.

The checker is run by ``run_benchmarks.sh`` after every sweep and can be run by
hand on any benchmark CSV written by ``change_detect --mode benchmark``. It
verifies that

  * the header equals the documented benchmark schema, column for column;
  * every row has the right number of fields and numeric columns parse;
  * ``total_ms > 0`` everywhere; GPU rows have ``0 < kernel_ms <= total_ms``;
    CPU rows have every GPU-only stage column (h2d, ndvi, fit, detect, fused,
    d2h) equal to -1;
  * every configuration has at least ``--min-repeats`` rows (a configuration
    is the tuple backend, input_name, height, width, dates, tile_rows,
    streams, block_size, threads, io_threads). Configurations of the backends named by
    ``--allow-single`` may consist of one row (a labelled single run);
  * optionally, the number of configurations and rows matches what the sweep
    was expected to produce.

It prints a table of configurations with row counts and the median total_ms,
then ``CSV: PASS`` or ``CSV: FAIL <reasons>`` and exits 0 or 1.
"""

import argparse
import csv
import statistics
import sys

EXPECTED_HEADER = (
    "run_id,timestamp,mode,backend,precision,kernels,input_name,height,"
    "width,dates,pixels,samples,tile_rows,num_tiles,streams,block_size,"
    "threads,io_threads,repeat_index,input_bytes,read_ms,read_wait_ms,setup_ms,h2d_ms,"
    "ndvi_ms,fit_ms,detect_ms,fused_ms,d2h_ms,kernel_ms,total_ms,"
    "throughput_msamples_s,kernel_throughput_msamples_s,device_mem_peak_mb,"
    "changed_pixels,compare_to,agree_break_frac,agree_status_frac,"
    "max_coef_diff,gpu_name,gpu_mem_mb,cpu_model,cpu_threads,cuda_runtime,"
    "driver_version,git_commit").split(",")

TEXT_COLUMNS = frozenset((
    "run_id", "timestamp", "mode", "backend", "precision", "kernels",
    "input_name", "compare_to", "gpu_name", "cpu_model", "cuda_runtime",
    "driver_version", "git_commit"))

CONFIG_KEYS = ("backend", "input_name", "height", "width", "dates",
               "tile_rows", "streams", "block_size", "threads", "io_threads")

GPU_ONLY_STAGES = ("h2d_ms", "ndvi_ms", "fit_ms", "detect_ms", "fused_ms",
                   "d2h_ms")

MAX_REASONS_SHOWN = 8


def parse_args():
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description="Check a benchmark CSV for completeness and sanity.")
    parser.add_argument("--csv", required=True, help="benchmark CSV to check")
    parser.add_argument("--min-repeats", type=int, default=5,
                        help="minimum rows per configuration (default 5)")
    parser.add_argument("--allow-single", default="",
                        help="comma list of backends whose configurations "
                             "may have a single row (labelled single runs)")
    parser.add_argument("--expect-configs", type=int, default=None,
                        help="expected number of configurations")
    parser.add_argument("--config-slack", type=int, default=0,
                        help="accept up to this many fewer configurations "
                             "than --expect-configs (an automatic tile size "
                             "can equal an explicit one) (default 0)")
    parser.add_argument("--expect-rows", type=int, default=None,
                        help="expected total number of data rows")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the CSV: PASS / FAIL line")
    return parser.parse_args()


def read_rows(path, reasons):
    """Reads the CSV; returns (rows as dicts of parsed values, row count)."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            records = list(csv.reader(handle))
    except OSError as error:
        reasons.append("cannot read %s (%s)" % (path, error.strerror))
        return [], 0
    if not records:
        reasons.append("file is empty")
        return [], 0
    header = records[0]
    if header != EXPECTED_HEADER:
        if len(header) != len(EXPECTED_HEADER):
            reasons.append("header has %d columns, expected %d" %
                           (len(header), len(EXPECTED_HEADER)))
        for index, (found, wanted) in enumerate(zip(header, EXPECTED_HEADER)):
            if found != wanted:
                reasons.append("header column %d is '%s', expected '%s'" %
                               (index + 1, found, wanted))
                break
        return [], len(records) - 1
    rows = []
    for number, record in enumerate(records[1:], start=2):
        if len(record) != len(header):
            reasons.append("line %d has %d fields, expected %d" %
                           (number, len(record), len(header)))
            continue
        row = {"_line": number}
        bad = False
        for name, text in zip(header, record):
            if name in TEXT_COLUMNS:
                row[name] = text
                continue
            try:
                row[name] = float(text)
            except ValueError:
                reasons.append("line %d: column %s is not numeric ('%s')" %
                               (number, name, text))
                bad = True
                break
        if not bad:
            rows.append(row)
    return rows, len(records) - 1


def check_row(row, reasons):
    """Checks the per-row timing rules."""
    line = row["_line"]
    if not row["total_ms"] > 0:
        reasons.append("line %d: total_ms %g is not > 0" %
                       (line, row["total_ms"]))
    if row["backend"].startswith("gpu"):
        if not row["kernel_ms"] > 0:
            reasons.append("line %d: GPU row has kernel_ms %g (not > 0)" %
                           (line, row["kernel_ms"]))
        elif row["kernel_ms"] > row["total_ms"]:
            reasons.append("line %d: kernel_ms %g exceeds total_ms %g" %
                           (line, row["kernel_ms"], row["total_ms"]))
    else:
        for stage in GPU_ONLY_STAGES:
            if row[stage] != -1:
                reasons.append("line %d: CPU row has %s = %g, expected -1" %
                               (line, stage, row[stage]))


def group_configs(rows):
    """Groups rows by configuration, keeping first-seen order."""
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in CONFIG_KEYS)
        groups.setdefault(key, []).append(row)
    return groups


def number_text(value):
    """Formats a numeric config field as an integer when it is whole."""
    return str(int(value)) if float(value).is_integer() else str(value)


def print_table(groups):
    """Prints one line per configuration."""
    print("%-12s %-30s %6s %6s %7s %6s %4s %7s %5s %12s" %
          ("backend", "input", "tile", "strm", "block", "thrd", "io", "size",
           "rows", "median_total_ms"))
    for key, members in groups.items():
        (backend, name, height, width, dates, tile, streams, block, threads,
         io_threads) = key
        size = "%sx%sx%s" % (number_text(height), number_text(width),
                             number_text(dates))
        median = statistics.median(m["total_ms"] for m in members)
        print("%-12s %-30s %6s %6s %7s %6s %4s %7s %5d %12.3f" %
              (backend, name, number_text(tile), number_text(streams),
               number_text(block), number_text(threads),
               number_text(io_threads), size, len(members), median))


def check_counts(args, groups, row_count, reasons):
    """Checks repeats per configuration and the expected totals."""
    singles = {name for name in args.allow_single.split(",") if name}
    for key, members in groups.items():
        if len(members) >= args.min_repeats:
            continue
        backend = key[0]
        if backend in singles and len(members) == 1:
            continue
        reasons.append("config %s %s tile_rows=%s streams=%s block=%s has %d "
                       "rows, need >= %d" %
                       (backend, key[1], number_text(key[5]),
                        number_text(key[6]), number_text(key[7]),
                        len(members), args.min_repeats))
    if args.expect_configs is not None:
        low = args.expect_configs - args.config_slack
        if not low <= len(groups) <= args.expect_configs:
            wanted = str(args.expect_configs) if low == \
                args.expect_configs else "%d to %d" % (low,
                                                       args.expect_configs)
            reasons.append("%d configurations, expected %s" %
                           (len(groups), wanted))
    if args.expect_rows is not None and row_count != args.expect_rows:
        reasons.append("%d data rows, expected %d" %
                       (row_count, args.expect_rows))


def main():
    """Runs all checks and prints the verdict."""
    args = parse_args()
    reasons = []
    rows, row_count = read_rows(args.csv, reasons)
    groups = {}
    if rows:
        for row in rows:
            check_row(row, reasons)
        groups = group_configs(rows)
        if not args.quiet:
            print_table(groups)
    if not reasons or rows:
        if row_count == 0:
            reasons.append("no data rows")
        else:
            check_counts(args, groups, row_count, reasons)
    if reasons:
        shown = reasons[:MAX_REASONS_SHOWN]
        if len(reasons) > len(shown):
            shown.append("... and %d more" % (len(reasons) - len(shown)))
        print("CSV: FAIL " + "; ".join(shown))
        return 1
    print("CSV: PASS (%d rows, %d configurations)" % (row_count, len(groups)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
