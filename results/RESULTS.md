# Results: how to read the proof of execution

This folder holds the curated outputs of the real runs: logs, CSV files and
figures. [`MANIFEST.md`](MANIFEST.md) lists every file with the command that
produced it. This page says what the files show and how to tell that the
program did what is expected. Every number below is taken from, or computed
directly from, the named files. Paths are relative to `results/`.

**Provenance.** All detection runs and benchmarks were made with the build at
commit `8fd110a` (every log records it in its `git_commit:` line;
`logs/measurement_run_console.txt` records the clean-tree check, the idle GPU,
AC power and the load average before the runs). The date-scaling sweep and the
textbook-defaults comparison on the demo cube were run at `cfb0fd0`, which
changed only a script. The co-registration run is at `1003e00`
(co-registration code unchanged since `6cf6c38`; later commits are documentation).
Later commits add the co-registration mode, documentation and one fix to the
optional `--trace-csv` timeline (host/device clock alignment, `3f963be`); the
detection path is unchanged (`git diff --stat 8fd110a HEAD -- src` lists only
the co-registration files, the flags, the mode dispatch and the trace code). The
pipeline traces (`csv/pipeline_trace_*`, `figures/pipeline_timeline_measured.png`)
and the unit-test logs (`logs/log_test_*.txt`) were produced at `3f963be`, and the installation check `logs/log_smoke_check_synthetic_512x512x48.txt` (ends with `SMOKE: PASS`) at `ac472b9` (see
the tests' `git_commit:` line). These hashes refer to the development repository; the public history was created from the final tree in seven commits, so they cannot be checked out there.

## 1. One execution over many per-pixel time series

The question: did a single run process many pieces of data? Open
`logs/log_detect_gpu_10980x10980x70_full.txt` (the headline run). It states:

| Line | Value |
| --- | --- |
| `gpu:` | NVIDIA GeForce RTX 4070 Laptop GPU, 8187 MiB, runtime 12.0 |
| `scenes (dates) T:` | 70 (2019-01-20 .. 2024-12-29) |
| `image size H x W:` | 10980 x 10980 |
| `pixels:` / `pixel time series:` | 120560400 / 120560400 |
| `samples (pixel-dates):` / `input bytes:` | 8439228000 / 42.20 GB |
| `backend:` | gpu (fused kernel), block_size 256, streams 3, io_threads 8, 687 tiles of 16 rows |
| `--- timings (ms) ---` | read_input 8266, h2d 4761, fused 252, kernel_total 264, backend_total 9660, wall_total 11589 |
| `SUMMARY:` | `processed 120560400 pixel time series (8439228000 samples) in 11.589 seconds (backend gpu)` |

The log also holds one `[tile k/687]` line per tile (the run was streamed in
687 pieces, the 42.2 GB input does not fit in GPU memory or in RAM), the
parameters, the change counts (4,082,473 break pixels, 3.39 %, all losses;
5,752 insufficient-data pixels) and the validation line.

To see the same fields for the smaller real runs:
`grep -h '^SUMMARY' logs/log_detect_*.txt logs/log_quick-diff_*.txt`. (The test
logs also contain `SUMMARY` lines from individual test cases.) The runs are the
full tile (120.6 M series), the 5,000 x 5,000 demo cube (25 M series, 3.319 s, a first read that was partly
cold),
the 2,048 x 2,048 development cube (4.19 M series, 0.994 s) and the 128 x 128
sample (16,384 series).

## 2. The CPU produced the same answer

`logs/log_detect_cpu-mt_10980x10980x70_full.txt` (16 threads, 14.111 s) and
`logs/log_detect_cpu_10980x10980x70_full.txt` (one thread, FP64, 98.260 s) are
the same command on the same tile with `--backend` changed. Compare the
`--- results ---` blocks: break pixels 4,082,473 (GPU), 4,082,472 (`cpu-mt`) and
4,082,470 (`cpu`) of 120.6 M, identical validation to three decimals.

The exact agreement is in `csv/agreement_full.csv`, written by
`scripts/compare_outputs.py`, which compares the break index and status maps of
two detect runs pixel by pixel (120,560,400 pixels per pair):

| Pair | Break index differs | Status differs | Fraction that differ |
| --- | --- | --- | --- |
| GPU vs `cpu` (FP64) | 6 | 5 | 5.0e-08 |
| GPU vs `cpu-mt` (FP32) | 9 | 5 | 7.5e-08 |
| `cpu-mt` (FP32) vs `cpu` (FP64) | 9 | 4 | 7.5e-08 |

So 6 of 120,560,400 pixels differ between the GPU and the FP64 reference, and the
FP32 CPU build differs from FP64 on 9: FP32 versus FP64 rounding at pixels whose
statistic lies exactly on the threshold, as much a property of FP32 as of the
GPU (the cause is inferred from this pattern, not checked pixel by pixel).
Where the break index is equal the break magnitudes agree to 7.7e-07 or better.
The `agreement_vs_cpu: ... 1.0000` lines in the logs are these fractions rounded
to four decimals. On the development cube
(`logs/log_detect_gpu_2048x2048x70_dev.txt`, run with `--save-coefficients`) the
largest difference of any model coefficient between the GPU and the FP64
reference is 5.4e-07. `csv/pixel_stats_demo.csv` shows the same comparison for
8 demo pixels, with the GPU result and an FP64 recomputation in adjacent rows
(same break dates, magnitudes agreeing to 5-6 digits).

The unit tests (`logs/log_test_gpu.txt` and `logs/log_test_cpu.txt`, each ending
in `PASS (N tests)`; produced from the final commit, see the `git_commit:` line)
compare the GPU and the CPU on
synthetic cubes with injected breaks. `logs/log_smoke_synthetic_512x512x48.txt`
shows one such run (no data needed): `agreement_vs_cpu` 1.0000 and
`synthetic_recovery: recovered 30492 of 30876 eligible (98.8%)`, with 1.2 % false
breaks on stable pixels.

## 3. Agreement with the Hansen reference (sanity check)

The reference is the Hansen forest-loss layer (annual, 30 m, forest only), a
sanity check rather than ground truth. Precision, recall and F1 are scored only
on the pixels the reference can judge (43.5 M of the 120.6 M pixels of the full
tile): `csv/validation_coverage.csv` (from `scripts/validation_coverage.py`)
shows that 877,787 of the 4,082,473 detected pixels (21.5 %) lie outside that
area and are not scored. Counting them as false positives would give a precision
of 0.635 (demo 0.736, dev 0.933). The same file shows that the reference marks
19.7 % (dev), 17.6 % (demo) and 7.2 % (full tile) of its eligible pixels as loss;
dev and demo were chosen where Hansen shows loss, which flatters precision on
those two windows.

* `csv/validation_full.csv` (headline run): precision 0.809, recall 0.823,
  F1 0.816; the break year equals Hansen's year for 92.5 % of true positives and
  is within one year for 98.9 %.
  The full tile contains both the dev and the demo window, so this score is not
  an independent test.
* `csv/validation_dev.csv` (development cube, used for tuning): precision 0.968,
  recall 0.852, F1 0.906. `csv/validation_demo.csv` (demo cube: the parameters
  were fixed on dev and not changed for demo; the demo cube was afterwards also
  scored with other settings for context only, none adopted, and that sweep is
  not in this folder): precision 0.835, recall 0.841, F1 0.838. The drop in
  precision from dev to demo is discussed in `docs/DESCRIPTION.md`.
* `csv/validation_demo_brief_defaults.csv`: the same demo cube with the trend
  extrapolated and the textbook-style threshold (3) and run length (3): F1 0.667.
  `csv/tuning_dev.csv` shows how the 28 parameter sets compare on the dev cube.
* `figures/change_year_map.png`: the break year of every detected loss pixel over
  the full tile on a grey base image. Large rectangular blocks (clearings) are
  coloured by year; the same blocks are the true positives in
  `figures/validation_overlay.png` (green = detected and in the reference, red =
  detected but not in the reference, orange = reference loss that was missed).
* `figures/before_2019-01-20.png`, `figures/after_2024-12-29.png` and
  `figures/before_after_crop.png`: the same 256 x 256 pixel area at the first
  and last date; forest on the left, cleared land on the right, with the pixels
  flagged as loss outlined (29,272 px, from the figure legend).
* `figures/ndvi_trajectory_examples.png` with `csv/trajectories_demo.csv`: six
  pixels with their observed NDVI, fitted seasonal model, prediction band and the
  detected break (for example row 187, col 3970: loss from 2024-06-02, mean
  residual -0.29); two stable pixels show no break.
* `figures/ndvi_2021-04-29.png`, `ndvi_2022-10-21.png`, `ndvi_2024-12-29.png`,
  `figures/change_year_map_demo.png` and `figures/timelapse.gif`: NDVI maps and
  the change map of the demo cube, and a 36-frame animation of true-colour
  frames with the loss detected so far in magenta and a running pixel count (0 on
  2019-01-20, 1,976,111 on 2024-12-29).
* `logs/log_detect_gpu_128x128x70_sample.txt`: the committed sample cube
  (`./run.sh sample`) with F1 0.945.
* `logs/log_quick-diff_gpu_2048x2048x70_dev.txt`: the simple two-date NDVI
  difference of the dev cube, the minor baseline mode.

### Co-registration (cuFFT phase correlation)

`logs/log_coreg_gpu_2048x2048x70_demo.txt` (`--mode coreg`, 2,048 x 2,048 centre
window of the demo cube, all 70 dates, reference date 2022-05-04, 1.443 s) ends
with `median_shift_px: 0.1009 | max_shift_px: 0.5077 (date 24, 2021-01-19) |
flagged: 0`. `csv/coreg_shifts_demo.csv` has the shift (dy, dx), peak height and
flag of every date; `figures/coreg_shifts_demo.png` plots the row and column
shift against the acquisition date: median shift 0.10 px, maximum 0.51 px (the
largest is on 2021-01-19, a date with a low correlation peak of 0.138), row shifts
between -0.51 and +0.28 px and column shifts between -0.15 and +0.09 px, none
flagged. This is consistent with the dates lying on the same pixel grid, the
assumption of the per-pixel time series, within this window; it does not prove
sub-pixel alignment.

## 4. Benchmarks

Per-sweep CSVs (`csv/benchmark_*.csv`, one row per timed repetition, warm-up
discarded) and their figures (`figures/benchmark_*.png`); the method and
every table are in [`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md). Headline
numbers, medians of 5 repeats (the full-tile `cpu` is a single run):

* full tile, `csv/benchmark_backends_full.csv`: kernel only 244.5 ms (GPU) vs
  11,553 ms (16-thread CPU), 47.2x; end to end 9,046 ms vs 11,858 ms (1.31x) vs
  91,710 ms on one FP64 thread (10.1x, single run). The 42.2 GB input exceeds
  the RAM, so it is uncached in practice (the cache could not be dropped).
* demo cube, `csv/benchmark_backends_demo.csv`: kernel only 50.4 ms vs 2,097 ms
  (41.6x); end to end 1,076 ms vs 2,137 ms (1.99x) vs 21,102 ms (19.6x). The
  input is read from the page cache.
* Where the time goes: on the full tile the read takes 8.2 s of the 9.0 s GPU
  run; the kernels take 0.24 s. `figures/pipeline_timeline_measured.png` shows the
  measured overlap of read, copy and kernels (847 ms with 3 streams vs 1,342 ms
  with 1 on the demo cube; `scripts/plot_timeline.py` prints these numbers and the
  59 % / 0 % copy overlap from `csv/pipeline_trace_streams3_demo.csv` and
  `csv/pipeline_trace_streams1_demo.csv`). `figures/benchmark_fusion.png` shows the full-tile
  kernel breakdown (572 ms unfused vs 245 ms fused) and the equal end-to-end
  times; `figures/benchmark_streams_tiles.png` the streams comparison on the full
  tile (13,683 vs 8,858 ms) and the demo tile-size sweep;
  `figures/benchmark_block_size.png` the fused block-size sweep;
  `figures/benchmark_io_threads.png` the reader-thread sweep;
  `figures/benchmark_scaling.png` the scaling with pixels and dates.
* The GPU is not faster everywhere: at 1 M pixels the 16-thread CPU wins end to
  end (`csv/benchmark_scaling_pixels.csv`); at 4 M pixels the ranges still
  overlap, from about 9 M pixels they do not.

`csv/benchmark_summary.csv` pools identical default configurations from several
sweeps (so its n differs from 5); the numbers above and in `docs/BENCHMARKS.md`
come from the per-sweep CSVs, not from it. `logs/hardware.txt` records the
machine and whether each sweep read its input from the page cache. The
`scaling_dates` section of `logs/bench_console.txt` is the first, superseded
version of that sweep (12-96 dates); `csv/benchmark_scaling_dates.csv` holds the
re-run (36-120 dates).

## 5. Code quality

`lint/clang_format.txt` (exit status 0) and `lint/cpplint.txt` (one `Done
processing` line per file and no error line: cpplint prints only its findings)
are the clean outputs of
`scripts/lint.sh` (clang-format with the Google style, `--dry-run --Werror`, and
cpplint 2.0.2) on `src/` and `tests/`, regenerated by `make proof`.

## 6. Mapping to commands

Each log starts with the exact command line that produced it (`command:`); the
same commands are in `MANIFEST.md`. `./run.sh help` lists the documented examples
(`sample`, `dev`, `demo`, `full`, `cpu-baseline`, `benchmark`, ...), and
`README.md` explains the flags. The data cubes themselves (up to 42 GB) are not in
the repository; `input/*_manifest.txt` list their files with sha256 checksums and
`../data/README.md` gives the commands that recreate them.

## 7. Limits of this evidence

* Everything ran on one laptop (RTX 4070 Laptop GPU, i9-14900HX with 16 of 32
  threads visible under WSL2); timings depend on the hardware.
* The Hansen reference is not ground truth; precision and recall measure
  agreement with it.
* Full-tile `cpu` and `cpu-fp32` benchmark numbers are single runs; the headline
  detect run and the benchmark medians are separate measurements.
* Nsight Systems and compute-sanitizer could not be used on this machine; the
  timeline is drawn from CUDA event timestamps, and memory-checker runs are
  missing.
* The co-registration check covers one window of one cube.
