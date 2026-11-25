# File formats

This document describes every file the program reads and writes: the input
cube, the outputs of `--mode detect` (including the optional pipeline trace),
the text log and the benchmark CSV. The examples are taken from real runs: the
committed sample cube `data/sample/manifest.txt` (128 x 128 pixels, 70 dates)
and the programs' own output. The algorithm itself is described in
`README.md`; the command-line flags are listed there and in
`bin/change_detect --help`.

Conventions used throughout:

* Binary arrays are raw, headerless and **little-endian**. Two-dimensional
  maps are row-major: pixel `(r, c)` of an `H x W` map is element `r * W + c`.
* Text files are UTF-8, one record per line, fields separated by single
  spaces (manifests, logs) or commas (CSV).
* A numeric value of `-1` in the benchmark CSV and a `-` in the log mean "not
  applicable" for that backend or mode.
* Dates are written `YYYY-MM-DD`; a "day number" is days since 1970-01-01.

Contents:

1. [Input cube](#1-input-cube)
2. [Detect outputs](#2-detect-outputs)
3. [Log format](#3-log-format)
4. [Benchmark CSV](#4-benchmark-csv)
5. [Co-registration output](#5-co-registration-output)

---

## 1. Input cube

### 1.1 Directory layout

A cube is a directory (for example `data/cubes/<name>/`) that holds a
`manifest.txt` and the raw arrays it names. The program is started with
`--input <dir>/manifest.txt`; file names inside the manifest are resolved
relative to the manifest's directory (absolute paths are also accepted).

| File | Type | Shape | Content |
| --- | --- | --- | --- |
| `manifest.txt` | text | | metadata, one `date` line per scene, checksums |
| `red.u16` | uint16 | T x H x W | Sentinel-2 band B04 digital numbers (DN); 0 = no data |
| `nir.u16` | uint16 | T x H x W | Sentinel-2 band B08 digital numbers (DN); 0 = no data |
| `scl.u8` | uint8 | T x H x W | scene classification (SCL) code of each observation |
| `reference_lossyear.u8` (optional) | uint8 | H x W | forest-loss reference, see 1.5 |
| `truth_break_index.i16` (synthetic cubes only) | int16 | H x W | injected break date index, `-1` = none |
| `previews/rgb_<YYYY-MM-DD>.png` (optional) | PNG | long side <= 1024 | true-colour preview of each date |
| `previews/rgb_full_<YYYY-MM-DD>.png` (optional) | PNG | full resolution | true colour of the first and last date |

The file names `red.u16`, `nir.u16`, `scl.u8`, `reference_lossyear.u8` are the
defaults written by the data scripts; the manifest keys `red_file`, `nir_file`,
`scl_file` and `reference_file` may name other files.

### 1.2 `manifest.txt`

The first line is exactly `change_detect_cube 1`. Every other line is blank,
a comment (first character `#`) or a record `key value`. The key ends at the
first space; the value is the rest of the line (leading and trailing blanks
removed), so text values may contain spaces. Excerpt of the sample cube:

```
change_detect_cube 1
name chaco_sample128
source Sentinel-2 L2A, Earth Search collection sentinel-2-c1-l2a + sentinel-2-l2a
license Contains modified Copernicus Sentinel data 2019-2024
crs EPSG:32720
geotransform 748280 10 0 7591360 0 -10
height 128
width 128
num_dates 70
red_file red.u16
nir_file nir.u16
scl_file scl.u8
reference_file reference_lossyear.u8
reference_source Hansen GFC-2024-v1.12 lossyear, treecover2000>=30, loss before 2021 excluded
# date <index> <YYYY-MM-DD> <days_since_1970> <scale> <offset> <valid_fraction> <scene_id>
date 0 2019-01-20 17916 0.0001 0 1.0000 S2A_20KQA_20190120_0_L2A
date 1 2019-02-19 17946 0.0001 0 1.0000 S2A_20KQA_20190219_0_L2A
...
date 69 2024-12-29 20086 0.0001 -0.1 1.0000 S2A_T20KQA_20241229T141749_L2A
sha256 red.u16 7960750bab1e5f97c180cd05fca1d01bc7efeca3da7c5c87c3d596ea70f79e70
sha256 nir.u16 4d6536f17a42885822c5c95a1ad92c5b2eeea6f49e150155dfef1486758edcd5
sha256 scl.u8 2e967398ad87bcb7df885f6112faf13f40bdee049c960ade9c6c42db788a0264
sha256 reference_lossyear.u8 ffa8ed7166d5eab6c61b7fec56e5c573c4c222e7a2202c78754ed710099c8aee
```

| Key | Required | Value |
| --- | --- | --- |
| `name` | yes | cube name (free text, shown in logs and CSV) |
| `source`, `license`, `crs`, `reference_source` | no | free text (provenance, attribution, coordinate reference system, how the reference was made); may be empty |
| `geotransform` | no | six numbers `x0 dx rx y0 ry dy` (GDAL order: origin and pixel size in the units of `crs`) |
| `height`, `width` | yes | pixels per column / row; positive integers |
| `num_dates` | yes | T, the number of `date` lines; positive and at most 1024 |
| `red_file`, `nir_file`, `scl_file` | yes | file names of the three raw arrays |
| `reference_file` | no | file name of the reference raster |
| `date` | yes, `num_dates` times | one scene, see below |
| `sha256` | no | `sha256 <file name> <64 hex digits>`, one per raw file |

Rules checked by the C++ reader (`ReadManifest`); a violation is an error that
names the file and line:

* a key other than `date` and `sha256` may appear only once;
* a known key (except `source`, `license`, `crs`, `reference_source`) needs a
  value; an unknown key is ignored with a `warning:` line in the log;
* the number of `date` lines must equal `num_dates`;
* every raw file must exist and have exactly the size implied by the shape
  (`T*H*W*2` bytes for `red.u16`/`nir.u16`, `T*H*W` for `scl.u8`, `H*W` for the
  reference).

The `sha256` lines are **not** verified by the C++ program (hashing 40 GB
would dominate the run time); `scripts/verify_cube.py` checks them together
with the file sizes and the per-date valid fractions.

**`date` line.** `date <index> <YYYY-MM-DD> <day> <scale> <offset>
<valid_fraction> <scene_id>`:

| Field | Meaning |
| --- | --- |
| `index` | 0, 1, ..., T-1, in order; the position of the scene along the first axis of the raw arrays |
| `YYYY-MM-DD` | acquisition date; its year is used for validation and the change-map colours |
| `day` | the same date as days since 1970-01-01; must be strictly increasing |
| `scale`, `offset` | calibration of this date: reflectance = `DN * scale + offset` (`scale` > 0) |
| `valid_fraction` | fraction of the window's pixels whose SCL class is 4, 5 or 6; informational |
| `scene_id` | identifier of the source scene, without spaces; informational |

The C++ reader needs the first six fields; the Python scripts always write and
require all seven. In the sample most scenes have `offset -0.1` (their DN
include the +1000 shift of newer processing baselines) and the others
`offset 0`; the reasons are listed in `data/README.md`.

**Derived date quantities.** From the `day` numbers the program computes
`tau = (day - day[0]) / 365.25` (years since the first date). The *training
dates* are the leading dates with `tau < --train-years`; all later dates are
*monitoring dates*. It is an error if either set is empty. For the sample
cube with the default 2 years this gives 25 training dates (the first
monitoring date is index 25, 2021-02-08) and 45 monitoring dates.

### 1.3 Raw arrays

`red.u16`, `nir.u16` and `scl.u8` are **date-major**: the three-dimensional
element `(t, r, c)` (date `t`, row `r`, column `c`) is stored at index

```
index = t * H * W + r * W + c
```

so each date is one contiguous `H x W` image, and a band file has
`T * H * W` elements. Rows are numbered from the top (north) of the window.
`0` in `red.u16` or `nir.u16` means no data. The manifest sizes are checked at
start-up. A date image can be loaded with NumPy as

```python
red = np.fromfile("red.u16", dtype="<u2").reshape(T, H, W)
```

### 1.4 SCL classes and the validity rule

`scl.u8` holds the Sentinel-2 scene classification of the L2A product (20 m,
upsampled to the 10 m grid by nearest neighbour):

| SCL | Meaning | SCL | Meaning |
| --- | --- | --- | --- |
| 0 | no data | 6 | water |
| 1 | saturated or defective | 7 | unclassified |
| 2 | dark area pixels | 8 | cloud, medium probability |
| 3 | cloud shadows | 9 | cloud, high probability |
| 4 | vegetation | 10 | thin cirrus |
| 5 | not vegetated (bare soil) | 11 | snow or ice |

An observation (pixel, date) is **valid** when all of these hold:

1. its SCL class is in the set given by `--valid-classes` (default `4,5,6`;
   codes 0..15 are accepted);
2. `red` and `nir` are both non-zero;
3. the reflectance sum `nir + red` is greater than `1e-4`, with
   reflectance = `DN * scale + offset` of that date.

For a valid observation NDVI = `(nir - red) / (nir + red)`, clamped to
[-1, 1]. An invalid observation is skipped by the model: it does not enter the
fit, cannot start, extend or reset a run of exceedances, and appears as an
empty `ndvi` in `trajectories.csv`. The raw SCL plane is stored unchanged; the
valid set is a run-time flag.

### 1.5 Reference raster

`reference_lossyear.u8` has `H x W` elements (no time axis) and encodes the
Hansen Global Forest Change `lossyear` layer, resampled to the cube grid:

| Value | Meaning |
| --- | --- |
| `0` | no forest loss |
| `1` .. `24` | forest loss in year 2000 + value (1 = 2001, ..., 24 = 2024) |
| `255` | **excluded**: no meaningful reference, see below |

`255` marks pixels that must not count as either a correct or a wrong
detection: pixels whose tree cover in 2000 was below 30 % (not forest, so
"forest loss" is undefined), pixels without Hansen data, and pixels whose loss
happened before the monitoring period (for the default monitoring start year
2021, `lossyear` 1..20): such a clearing took place before or during the
training years, so the model learns the cleared state and there is no change
left to detect. `255` cannot be confused with a year because valid values stop
at 24. The sample cube holds only `0` and `23` (loss in 2023).

The validator uses the monitoring window `first_year .. last_year` (the year
of the first monitoring date to the year of the last date) and treats a
reference pixel as a *loss* only when `1 <= value <= 24` and `2000 + value`
lies inside the window. Values `1..20` that were not turned into `255` are
therefore counted as "no loss". `--reference auto` (default) uses the
manifest's `reference_file` if present, `--reference none` disables
validation, any other value is a path to a raster of the same shape.

### 1.6 Previews

`previews/rgb_<date>.png` (every date, long side <= 1024 px) and
`previews/rgb_full_<date>.png` (full resolution, first and last date only) are
true-colour PNGs made from the Sentinel-2 `visual` asset. The detector does
not read them; `scripts/render_figures.py` uses them for the time-lapse and
the before/after figures and falls back to a NIR-red false colour from the raw
cube when they are absent.

### 1.7 Synthetic cubes

`--input synthetic` builds a cube in memory; `--mode generate` writes the same
cube to `--output-dir` as `manifest.txt`, `red.u16`, `nir.u16`, `scl.u8` and
`truth_break_index.i16` (`int16`, `H x W`, the date index at which a step
change was injected into the pixel, `-1` for none). Its manifest follows the
grammar above with `source synthetic`, scene id `synthetic` and a
`valid_fraction` equal to the fraction of pixels with SCL 4.

---

## 2. Detect outputs

`--mode detect` writes the following files into `--output-dir` (created if
missing). `H x W` maps are headerless little-endian arrays in the row-major
order of 1.3; their element `(r, c)` is the pixel of the cube at row `r`,
column `c`. With `--cleanup` the maps, `change_map.png`, `validation.csv` and
the backend row of `pixel_stats.csv` describe the cleaned map (isolated break
pixels turned into stable ones); the FP64 recomputation is never cleaned.

| File | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `break_index.i16` | int16 | H x W | date index (0..T-1) of the first observation of the detected change; `-1` = no break |
| `magnitude.f32` | float32 | H x W | signed mean of the absolute NDVI residuals over the break run; negative = greenness loss, positive = gain; `0` without a break |
| `status.u8` | uint8 | H x W | `0` stable, `1` break, `2` insufficient data |
| `rmse.f32` | float32 | H x W | training RMSE of the fit, floored at `--rmse-floor`; `0` when the fit failed |
| `train_obs.u16` | uint16 | H x W | number of valid observations among the training dates |
| `coefficients.f32` (only with `--save-coefficients`) | float32 | p x H x W | fitted coefficients, coefficient-major: coefficient `j` of pixel `(r, c)` is at `j * H * W + r * W + c`; order: intercept, trend, `cos 2*pi*tau`, `sin 2*pi*tau`, then the next harmonics; `p = 2 + 2K` |
| `change_map.png` | PNG (RGBA) | see below | colour-coded change map |
| `pixel_stats.csv` | CSV | | fit statistics of the sample pixels |
| `trajectories.csv` | CSV | | NDVI series, fit and z-scores of the sample pixels |
| `validation.csv` (only with a reference) | CSV | | confusion counts and metrics |
| `result_manifest.txt` | text | | what was run and which files were written |

Status values: *stable* = fitted and no run of `--consecutive` exceedances;
*break* = such a run was found (`break_index` is its first observation,
`magnitude` its mean absolute residual with the sign of the residuals);
*insufficient data* = fewer than `max(--min-train-obs, p + 1)` valid training
observations or a singular fit (`rmse` and `magnitude` are 0).

**`change_map.png`.** An RGBA image. Break pixels are coloured by break date
on an 8-step viridis ramp from the first monitoring date (dark purple) to the
last date (yellow); greenness losses are opaque, gains translucent (alpha
160). Stable pixels are dark grey (40, 40, 40), pixels with insufficient data
are fully transparent. If the longer side exceeds `--png-max-dim` (default
2048), the map is reduced by an integer factor `ceil(long_side / max_dim)`;
each output pixel shows one pixel of its block chosen by priority: any break
pixel (the earliest break in the block) wins, then stable, then insufficient,
so that small clearings stay visible. The reduction is done on the detection
maps first and only the reduced map is coloured (on the GPU for GPU backends),
which gives the same colours as colouring the full map and then picking the
block's pixel.

**Sample pixels.** `pixel_stats.csv` and `trajectories.csv` cover the pixels
selected by `--sample-pixels`: either an explicit list `r,c;r,c;...` or a
count `N`. For a count, half of the pixels (rounded up) are break pixels
(losses that the reference confirms first, then other losses, then gains) and
the rest stable pixels; either kind makes up for a shortage of the other.
They are taken at an even stride through the row-major list of candidates, so
that every backend picks the same pixels, and are listed in row-major order.

**`pixel_stats.csv`.** Two rows per sample pixel: the run's own backend, and
`cpu-fp64-recompute`, an independent double-precision recomputation from the
raw cube. Columns:

| Column | Meaning |
| --- | --- |
| `row`, `col` | pixel position |
| `backend` | `gpu`, `gpu-unfused`, `cpu`, `cpu-fp32`, `cpu-mt`, or `cpu-fp64-recompute` |
| `status` | `stable`, `break` or `insufficient` |
| `break_index` | as in `break_index.i16` (`-1` = none) |
| `break_date` | date of `break_index`; empty when there is no break |
| `magnitude` | as in `magnitude.f32` |
| `rmse` | as in `rmse.f32` |
| `n_train` | as in `train_obs.u16` |
| `coef0` .. `coef{p-1}` | fitted coefficients, in the order of `coefficients.f32`; empty in the backend row unless `--save-coefficients` was given (all 0 in the recompute row when the fit failed) |

Example (first rows of a run with `--sample-pixels 3`):

```
row,col,backend,status,break_index,break_date,magnitude,rmse,n_train,coef0,coef1,coef2,coef3,coef4,coef5
35,53,gpu,stable,-1,,0,0.04711432,24,,,,,,
35,53,cpu-fp64-recompute,stable,-1,,0,0.04711434,24,0.591854831513,-0.0509852134489,0.0397970132913,0.12346214463,0.0290215447297,-0.0374792637519
70,84,gpu,break,51,2023-05-14,-0.3293186,0.06726959,24,,,,,,
```

**`trajectories.csv`.** One row per sample pixel and date (`T` rows per
pixel), recomputed in FP64 from the raw cube:

| Column | Meaning |
| --- | --- |
| `row`, `col` | pixel position |
| `date_index`, `date`, `day` | date index, `YYYY-MM-DD`, day number |
| `tau` | years since the first date, `(day - day[0]) / 365.25` |
| `ndvi` | NDVI of the observation; empty when the observation is invalid |
| `valid` | `1` if the observation is valid (see 1.4), else `0` |
| `in_training` | `1` for training dates, `0` for monitoring dates |
| `predicted` | model prediction at this date; empty when the fit failed |
| `z` | standardised residual `(ndvi - predicted) / rmse`; empty for invalid observations or a failed fit |
| `is_break_run` | `1` for the `--consecutive` valid observations that form the detected break run, from `break_index` on |

```
row,col,date_index,date,day,tau,ndvi,valid,in_training,predicted,z,is_break_run
35,53,0,2019-01-20,17916,0.000000,0.730852,1,1,0.711659,0.407375,0
35,53,1,2019-02-19,17946,0.082136,0.685234,1,1,0.716906,-0.672248,0
```

**`validation.csv`.** Rows `metric,value`, written only when a reference is
available:

| Metric | Meaning |
| --- | --- |
| `tp`, `fp`, `fn`, `tn` | pixel counts (see below) |
| `excluded` | pixels with reference `255`, not counted anywhere |
| `precision` | `tp / (tp + fp)` |
| `recall` | `tp / (tp + fn)` |
| `f1` | harmonic mean of precision and recall |
| `year_exact` | fraction of true positives whose break year equals the reference year |
| `year_within1` | fraction of true positives whose break year is within one year of the reference year (includes exact matches) |
| `first_year`, `last_year` | the monitoring window used (years) |

A pixel is a *detection* when its status is break, its magnitude is negative
(a greenness loss; gains never count) and the year of its break date lies in
`first_year .. last_year`. It is a *reference loss* when its reference value
is in `1..24` and `2000 + value` lies in the same window. Then `tp` =
detection and reference loss, `fp` = detection only, `fn` = reference loss
only, `tn` = neither; pixels with reference `255` are `excluded`. A ratio
with a zero denominator is reported as 0. Example:

```
metric,value
tp,6507
fp,32
fn,1712
tn,8133
excluded,0
precision,0.995106
recall,0.791702
f1,0.881827
year_exact,0.992162
year_within1,1.000000
first_year,2021
last_year,2024
```

**`result_manifest.txt`.** Same `key value` grammar as the cube manifest, so
the run can be reproduced and the files interpreted without the command line.
First line `change_detect_result 1`, then these keys in this order:

| Key | Value |
| --- | --- |
| `cube_manifest` | path of the input manifest as given (relative to the working directory when inside it), or `synthetic` |
| `name`, `height`, `width`, `num_dates` | copied from the cube |
| `backend` | the backend that produced the maps |
| `harmonics` | K (p = 2 + 2K) |
| `trend` | `hold` or `extrapolate`: whether the trend term is held at its end-of-training value or extrapolated over the monitoring dates (`--trend`) |
| `train_years`, `min_train_obs`, `threshold`, `consecutive`, `rmse_floor`, `ridge` | model parameters (`min_train_obs` after resolving `0` to `3p`) |
| `losses_only`, `cleanup` | `1` or `0` |
| `valid_classes` | comma list of SCL codes |
| `reference_file` | reference raster used, or `none` |
| `num_train_dates` | number of training dates |
| `first_monitor_year`, `last_year` | the monitoring window of `validation.csv` |
| `date` (T lines) | `date <index> <YYYY-MM-DD> <day> <scale> <offset> -1 -`; the last two fields (valid fraction, scene id) are unknown for a result |
| `file` (one per output) | `file <name> <type> <description>`; the description is the rest of the line |

Excerpt:

```
change_detect_result 1
cube_manifest data/sample/manifest.txt
name chaco_sample128
height 128
width 128
num_dates 70
backend gpu
harmonics 2
trend hold
train_years 2
min_train_obs 18
threshold 3.5
consecutive 2
rmse_floor 0.02
ridge 0
losses_only 0
valid_classes 4,5,6
cleanup 0
reference_file data/sample/reference_lossyear.u8
num_train_dates 25
first_monitor_year 2021
last_year 2024
date 0 2019-01-20 17916 0.0001 0 -1 -
...
file break_index.i16 i16 break date index per pixel, -1 = none
file magnitude.f32 f32 signed mean residual of the break run
file status.u8 u8 0 stable, 1 break, 2 insufficient data
...
```

**Other modes.** `--mode quick-diff` writes `quick_diff.f32` (float32, `H x W`,
NDVI of `--date-b` minus NDVI of `--date-a`, 0 where a date is invalid),
`quick_diff_flag.u8` (`0` no data, `1` no change, `2` loss, `3` gain) and
`quick_diff.png` (loss red, gain green, no change grey, no data black); it
writes no result manifest. `--mode benchmark` writes only the CSV of section 4.

**Pipeline trace (`--trace-csv PATH`).** Detect mode with a GPU backend
(`gpu` or `gpu-unfused`) can export the timeline of the streamed pipeline; the
flag is rejected for CPU backends and other modes, and the log gets a
`trace_csv: PATH` line. One row per tile and stage, tile by tile:

```
tile,slot,stage,stream,start_ms,end_ms
0,0,read,-1,0.0017,5.0413
0,0,h2d,0,5.1110,10.3585
0,0,fused,1,10.4438,10.8923
0,0,d2h,2,10.9445,11.1817
1,1,read,-1,5.4089,9.0382
```

| Column | Meaning |
| --- | --- |
| `tile` | tile index, 0-based, in row order |
| `slot` | buffer slot the tile used (`tile % slots`; slots = `min(3, tiles)` with `--streams 3`, 1 with `--streams 1`) |
| `stage` | `read` (host: disk to pinned buffer), `h2d` (host-to-device copy), the kernel stage(s) `fused`, or `ndvi`, `fit`, `detect` for `gpu-unfused`, followed by `validate` (the confusion kernel) when the run counts the validation inside the pipeline (a reference is present and `--cleanup` is off), then `d2h` (device-to-host copy) |
| `stream` | CUDA stream that ran the stage: 0 copy-in, 1 compute, 2 copy-out with `--streams 3`; always 0 with one stream (or a single tile, which uses one stream); `-1` for the host `read` |
| `start_ms`, `end_ms` | interval in milliseconds since the start of the tile loop (after allocation, before the first read) |

Device stages are CUDA events: the time of each event is
`cudaEventElapsedTime` from a reference event recorded on the first stream when
the loop starts, so device intervals are exact relative to each other. A
`start_ms` is when the stream reached the stage (a copy waits in its stream
behind the previous tile's copy, so `start_ms` is the actual start, not the
enqueue time). The `read` interval is host `steady_clock` time relative to the
host time taken right after a wait on the reference event (one
`cudaEventSynchronize` before the loop), so host and device clocks agree to
within the wake-up latency of that wait. Apart from that wait, recording adds
no synchronisation: the events are kept in a vector and converted after the final
drain, and the results are byte-identical with and without the flag. The file
is written after the run, outside the timed region.
`scripts/plot_timeline.py` draws it:

```
python scripts/plot_timeline.py --trace s3.csv --trace s1.csv \
    --labels "3 streams" "1 stream" --out pipeline_timeline_measured.png \
    [--max-tiles 12]
```

It makes a Gantt chart (lanes: host read, H2D, kernels, D2H; bars coloured by
slot, tile numbers on the bars); two traces give two panels on a shared time
axis. Only the first `--max-tiles` tiles are drawn and the caption states the
total tile count and tile-loop time.

---

## 3. Log format

Every mode prints the log to standard output and, with `--log-file`, to a
file. Each line is flushed as it is written. Sections start with a line
`--- <name> ---`; other lines are `key: value` pairs, several of which may share
a line separated by ` | `. Numbers are plain (no thousands separators); times
are milliseconds with three decimals. The log below is the real output of a
detect run on the sample cube with `--compare-to cpu --sample-pixels 3` (paths
shortened; `-` marks values that do not apply):

```
=== change_detect 1.0 ===
command: bin/change_detect --mode detect --backend gpu --input data/sample/manifest.txt --output-dir out/sample --compare-to cpu --sample-pixels 3
start_time: 2026-11-24T00:05:54
git_commit: f738f73-dirty
--- device ---
gpu: NVIDIA GeForce RTX 4070 Laptop GPU | cc 8.9 | 36 SMs | 8187 MiB total | 7054 MiB free | driver 13.4 | runtime 12.0
cpu: Intel(R) Core(TM) i9-14900HX | 16 threads
--- input ---
cube: chaco_sample128 (data/sample/manifest.txt)
scenes (dates) T: 70 [2019-01-20 .. 2024-12-29]
image size H x W: 128 x 128
pixels: 16384
pixel time series: 16384
samples (pixel-dates): 1146880
input bytes: 5.73 MB
--- parameters ---
harmonics: 2 (p = 6) | trend: hold | train_years: 2.0 (train dates: 25) | min_train_obs: 18 | threshold: 3.5 | consecutive: 2 | rmse_floor: 0.02 | ridge: 0 | losses_only: no | valid_classes: 4,5,6 | cleanup: no
--- execution ---
backend: gpu (fused kernel) | block_size: 256 | streams: 3 | io_threads: 8 | tile_rows: 128 | tiles: 1 | tile_input_bytes: 5734400
gpu_memory: allocated 5 MiB in 7 ms (pinned host buffers included); cudaMemGetInfo used delta 10 MiB
[tile 1/1] rows 0..127
--- timings (ms) ---
read_input: 3.033 | read_gbps: 1.89 | read_wait: - | setup: 14.633 | h2d: 0.535 | ndvi: - | fit: - | detect: - | fused: 0.145 | d2h: 0.036 | kernel_total: 0.298 | backend_total: 20.773 | post: 3.405 | write_outputs: 2.573 | wall_total: 30.834
throughput: 55.2 Msamples/s | kernel_throughput: 3850.5 Msamples/s
device_memory_peak: 6 MiB (pipeline buffers; CUDA context not included)
--- results ---
stable: 7846 (47.89%) | break: 8538 (52.11%) [loss 7261, gain 1277] | insufficient_data: 0 (0.00%)
reference: data/sample/reference_lossyear.u8
validation: precision 0.995 | recall 0.879 | f1 0.933 | year_exact 0.977 | year_within1 1.000 (tp 7222 fp 39 fn 997)
agreement_vs_cpu: identical_break_index 1.0000 | identical_status 1.0000 | max_coef_diff -
outputs: out/sample
SUMMARY: processed 16384 pixel time series (1146880 samples) in 0.031 seconds (backend gpu)
```

### 3.1 Sections and keys

**Header** (no section line).

| Line | Content |
| --- | --- |
| `=== change_detect 1.0 ===` | program banner |
| `command:` | the command line exactly as typed (the only line that is not rewritten relative to the working directory) |
| `start_time:` | local time, `YYYY-MM-DDTHH:MM:SS` |
| `git_commit:` | short commit hash of the repository that contains the executable, with `-dirty` if it has uncommitted changes, or `unknown` |

**`--- device ---`.** `gpu:` is always printed: name, compute capability, SM
count, total and free memory, driver and CUDA runtime version; or
`gpu: not available (CPU-only build)` / `gpu: none detected`. `cpu:` gives the
CPU model and hardware thread count.

**`--- input ---`.** `cube: <name> (<manifest path or synthetic>)`; the number
of dates and the first and last date; `image size H x W`; `pixels` (`H*W`);
`pixel time series` (the same number: one series per pixel); `samples
(pixel-dates)` (`H*W*T`); `input bytes` (5 bytes per sample: two `u16` bands
and one `u8` SCL, shown in decimal MB or GB). For synthetic inputs a
`synthetic_generation:` line gives the generation time, which is not included
in any timing.

**`--- parameters ---`.** One line with the model parameters actually used:
harmonics K and `p`, the trend mode, `train_years` and the resulting number of
training dates, `min_train_obs` (after resolving `0` to `3p`), `threshold`,
`consecutive`, `rmse_floor`, `ridge`, `losses_only`, the valid SCL classes and
`cleanup`.

On GPU runs a `gpu_memory:` line follows the `backend:` line of the
`--- execution ---` section unless `--quiet` is given: the device memory
allocated for the pipeline buffers (MiB, rounded down), the time the
allocation took (pinned host buffers included) and the change of
`cudaMemGetInfo` free memory measured around the allocation. The two memory
numbers can differ (the CUDA context and other processes also use memory).

**`--- execution ---`.** `backend:` (the backend, with precision, kernel
variant or thread count), `block_size` and `streams` (GPU only), `io_threads`
(parallel disk reads per tile, `--io-threads`; it only matters for raw-file
input, a synthetic cube is read from memory), `tile_rows` (rows per tile),
`tiles`, and `tile_input_bytes` (`tile_rows * W * T * 5`). With
`--trace-csv` a `trace_csv:` line gives the file written (section 2). The GPU backend
allocates `min(3, tiles)` slots with `--streams 3` and one with `--streams 1`.
`note:` lines record automatic decisions: a retry
(`tile allocation failed; retrying with 64 rows`) or an explicit `--tile-rows`
that needed more than half of the free device memory (the rule the automatic
size uses) and was reduced, for example
`--tile-rows 1500 needs 7789 MiB, 7052 MiB free; using 672 rows`
(WSL2 can silently oversubscribe device memory, so an oversized tile would not
fail but thrash). `warning:` lines report non-fatal problems (for example pageable host buffers instead of pinned
ones). One `[tile k/n] rows a..b` line per tile follows unless `--quiet` is
given. A warning about an ignored manifest key is printed earlier, when the
manifest is read (between the `device` and `input` sections).

**`--- timings (ms) ---`.** See 3.2.

**`--- results ---`.** `stable`, `break` (with the split into `loss` and
`gain`) and `insufficient_data`, each as count and percentage of all pixels,
counted after `--cleanup`; `reference:` (path, or `none`); `validation:` with
the metrics of `validation.csv` and `(tp ... fp ... fn ...)` (only with a
reference; ` [some ratios undefined: ...]` is appended when a denominator was
zero); `agreement_vs_<backend>:` (only with `--compare-to`): the fraction of
pixels with identical `break_index` and identical `status`, and the largest
absolute coefficient difference (`-` unless both runs saved coefficients);
`synthetic_recovery:` (synthetic input only):
`recovered R of E eligible (x%) | false_breaks F of S (y%)`, where *eligible*
pixels are injected breaks with enough valid observations before and after the
injection, *recovered* means a break detected at the first valid observation
at or just after the injected date, and false breaks are pixels without an
injected break that report a break; `outputs:` (the output directory).

**`SUMMARY:`** The last line of every run:
`SUMMARY: processed <pixels> pixel time series (<samples> samples) in <s>
seconds (backend <name>)`, where `<s>` is `wall_total` in seconds in detect
mode. In benchmark mode it is the median `backend_total` of the first
backend; quick-diff mode prints `SUMMARY: processed <pixels> pixels (2 dates,
<samples> samples) in <s> seconds (backend <name>)` and generate mode
`SUMMARY: generated <H> x <W> x <T> synthetic cube (<samples> samples) in <s>
seconds`. Scripts parse this line.

**Benchmark mode** prints the header, device, input and parameters sections,
then `benchmark: repeat N | warmup W | csv <path>`, one line
`bench: <backend> repeat i/N backend_total <ms> ms kernel_total <ms> ms` per
timed repetition, a `--- results ---` section with a median table (columns:
backend, median `backend_total` in ms, minimum, maximum, median
`kernel_total`, speedup versus the `cpu-mt` median), `benchmark_csv: <path>`
and `SUMMARY:`. **Quick-diff mode** adds a `--- quick-diff ---` line with the
backend, the two dates and the threshold, a timings section and a results line
`no_data | no_change | loss | gain`. **Generate mode** has a `--- generate ---`
section with the size, seed and noise parameters.

Paths in `cube:`, `reference:`, `outputs:` and `benchmark_csv:` are shown
relative to the working directory when they lie below it, otherwise as given.

### 3.2 Timing fields

All values are wall-clock milliseconds, except that GPU stage times are
measured with CUDA events. A `-` means that the stage does not exist for the
backend (the CPU backends have no `h2d`, `d2h`, `ndvi`, `fit`, `detect` or
`fused`; the fused GPU kernel has no separate `ndvi`, `fit` and `detect`; the
unfused GPU backend has no `fused`; the GPU has no `read_wait`).

| Field | Meaning |
| --- | --- |
| `read_input` | time spent reading tiles from the cube (disk or memory). On the GPU backends the read is part of the host loop; on the CPU backends a background thread prefetches the next tile, so this time overlaps computation. With `--io-threads N` the per-date reads of a tile run on N threads, so this is the wall time of the parallel read. |
| `read_gbps` | input bytes (`samples * 5`) divided by `read_input`, in GB/s (1 GB = 1e9 bytes); `-` when no read time was measured. A value above the disk speed means the cube was served from the page cache. |
| `read_wait` | CPU backends only: time the compute threads spent **waiting** for the prefetcher. `read_input` minus `read_wait` is roughly the read time hidden behind computation. |
| `setup` | work before the first tile is read. GPU: upload of the per-date design table, tile planning, result allocation and all device and pinned host allocations. CPU: result allocation, design table and tile buffers. Included in `backend_total`. |
| `h2d`, `d2h` | GPU only: sum of the CUDA-event times of the host-to-device and device-to-host copies. With `--streams 3` they overlap each other and the kernels. |
| `ndvi`, `fit`, `detect` | unfused GPU backend: sum of the event times of each kernel |
| `fused` | fused GPU backend: sum of the event times of the single kernel |
| `kernel_total` | GPU: sum of all kernel event times; when the validation is counted inside the pipeline (see `post`) the confusion kernel is included here, it is not shown as a separate field. CPU: time of the compute loop (excluding waiting for reads). Compare it with `backend_total` to see how much of the run is not computation. |
| `backend_total` | wall time of the backend from the start of `setup` until all results are in host memory and the pipeline buffers (pinned host and device memory on the GPU, tile buffers on the CPU) have been released (the release time is not shown as a separate field); includes setup, reads, transfers, computation and teardown, but not post-processing or output writing. This is the number to compare between backends. |
| `post` | work done after the backend returned, on the device class of the backend: cleanup (`--cleanup`), the block-priority reduction of the change map to PNG resolution (host) followed by colouring of that small map, and, when the validation was not already counted in the pipeline, the validation counts. GPU backends without `--cleanup` count the validation inside the tile pipeline (the reference rows travel to the device with each tile, the confusion kernel runs on the tile's results before they are copied back, and the counts are read back once at the end), so that cost is in `kernel_total` and `backend_total`, not in `post`. With `--cleanup` the results change after the pipeline and the validation is counted after the cleanup, inside `post`. The counts are identical either way. |
| `write_outputs` | writing all output files, including the FP64 recomputation of the sample pixels and PNG encoding |
| `wall_total` | the whole detect run from just before the reference is loaded (before the backend starts) until the output files are written: it includes `backend_total`, `post`, loading the reference and `write_outputs`. The untimed work of `--compare-to` and of the synthetic recovery scoring is subtracted. Input generation and manifest parsing happen before the clock starts. |

Stage times can sum to more than `backend_total` because stages overlap, and
to less because of untimed host work between stages. `throughput` is
`samples / backend_total` and `kernel_throughput` is `samples / kernel_total`,
both in millions of samples per second. `device_memory_peak: N MiB (pipeline
buffers; CUDA context not included)` is the sum of this run's device
allocations for the tile slots in MiB: it is the size of the pipeline buffers,
not a measured peak of the whole process (the CUDA context, constant memory and
the post-processing buffers are not included). It is `-` on CPU backends. The
CSV column `device_mem_peak_mb` keeps its name for compatibility and holds the
same number.

---

## 4. Benchmark CSV

`--mode benchmark` appends one row per timed repetition (warm-up runs are not
written) to `--benchmark-csv` (default `<output-dir>/benchmark.csv`). The
header line is written when the file is new or empty; an existing file must
already start with exactly this header, otherwise the run stops without
writing anything. Fields containing a comma, a quote or a line break are
quoted. Millisecond and fraction columns have 3 and 6 decimals; `-1` means not
applicable (for example `h2d_ms` on a CPU backend, `streams` on a CPU backend,
`threads` on a GPU backend, the agreement columns without `--compare-to`).
`max_coef_diff` is `-1` unless `--save-coefficients` is given: saving the
coefficients adds a device-to-host copy and a host scatter, so the timed
benchmark rows do not save them. Accuracy numbers (agreement and the largest
coefficient difference) come from a detect run with `--compare-to` and
`--save-coefficients`.
The example column is a row of a real run (GPU, synthetic 512 x 512 x 48 cube,
second repetition).

| # | Column | Unit | Meaning | Example |
| --- | --- | --- | --- | --- |
| 1 | `run_id` | text | `--run-id`, default the start time of the run | `2026-11-23T19:47:02` |
| 2 | `timestamp` | text | local time when the row was produced | `2026-11-23T19:47:08` |
| 3 | `mode` | text | always `benchmark` | `benchmark` |
| 4 | `backend` | text | `gpu`, `gpu-unfused`, `cpu`, `cpu-fp32`, `cpu-mt` | `gpu` |
| 5 | `precision` | text | `fp32` or `fp64` (`cpu` is the only FP64 backend) | `fp32` |
| 6 | `kernels` | text | `fused`, `unfused` or `n/a` (CPU) | `fused` |
| 7 | `input_name` | text | cube name (`synthetic_<H>x<W>x<T>_seed<N>` for synthetic input) | `synthetic_512x512x48_seed42` |
| 8 | `height` | pixels | `H` | `512` |
| 9 | `width` | pixels | `W` | `512` |
| 10 | `dates` | count | `T` | `48` |
| 11 | `pixels` | count | `H * W` | `262144` |
| 12 | `samples` | count | `H * W * T` (pixel-dates) | `12582912` |
| 13 | `tile_rows` | rows | rows per tile actually used | `512` |
| 14 | `num_tiles` | count | number of tiles | `1` |
| 15 | `streams` | count | GPU streaming mode (1 or 3); `-1` on CPU | `3` |
| 16 | `block_size` | threads | GPU threads per block; `-1` on CPU | `256` |
| 17 | `threads` | count | `cpu-mt` worker threads; `-1` on GPU | `-1` |
| 18 | `io_threads` | count | `--io-threads`: parallel disk reads per tile (1 = sequential); recorded for every backend, no effect on a synthetic (in-memory) input | `8` |
| 19 | `repeat_index` | count | 1-based index of the timed repetition | `2` |
| 20 | `input_bytes` | bytes | `samples * 5` | `62914560` |
| 21 | `read_ms` | ms | `read_input` of the log; time reading tiles (wall time of the parallel read), may overlap compute on CPU | `6.433` |
| 22 | `read_wait_ms` | ms | CPU: time compute waited for the prefetcher; `-1` on GPU | `-1` |
| 23 | `setup_ms` | ms | `setup` of the log: work before the first tile is read (allocation, table upload); part of `total_ms` | `59.714` |
| 24 | `h2d_ms` | ms | GPU host-to-device copies (CUDA events) | `4.954` |
| 25 | `ndvi_ms` | ms | unfused GPU NDVI kernel | `-1` |
| 26 | `fit_ms` | ms | unfused GPU fit kernel | `-1` |
| 27 | `detect_ms` | ms | unfused GPU detection kernel | `-1` |
| 28 | `fused_ms` | ms | fused GPU kernel | `0.424` |
| 29 | `d2h_ms` | ms | GPU device-to-host copies | `0.305` |
| 30 | `kernel_ms` | ms | `kernel_total` of the log: all kernels (GPU) or the compute loop (CPU) | `0.424` |
| 31 | `total_ms` | ms | `backend_total` of the log: backend wall time from start of setup to all results in host memory, including the release of the pipeline buffers | `73.210` |
| 32 | `throughput_msamples_s` | Msamples/s | `samples / total_ms / 1e3`: end-to-end rate including setup, disk reads and transfers | `171.875` |
| 33 | `kernel_throughput_msamples_s` | Msamples/s | `samples / kernel_ms / 1e3` | `29681.159` |
| 34 | `device_mem_peak_mb` | MiB | sum of this run's device allocations for the pipeline buffers (the column name is kept for compatibility; it is not a measured peak and excludes the CUDA context); `-1` on CPU | `189.750` |
| 35 | `changed_pixels` | count | pixels with status break in this repetition | `31694` |
| 36 | `compare_to` | text | `--compare-to` backend or `none` | `none` |
| 37 | `agree_break_frac` | fraction | fraction of pixels whose `break_index` equals that of the `compare_to` backend (last repetition); `-1` if none | `-1` |
| 38 | `agree_status_frac` | fraction | same for `status` | `-1` |
| 39 | `max_coef_diff` | | largest absolute coefficient difference (scientific notation); `-1` if not computed (needs `--save-coefficients` on both runs) | `-1` |
| 40 | `gpu_name` | text | GPU model, `none` if no GPU; the same in every row of a run, including CPU rows | `NVIDIA GeForce RTX 4070 Laptop GPU` |
| 41 | `gpu_mem_mb` | MiB | total GPU memory; `-1` if no GPU | `8187.500` |
| 42 | `cpu_model` | text | CPU model name | `Intel(R) Core(TM) i9-14900HX` |
| 43 | `cpu_threads` | count | hardware threads | `16` |
| 44 | `cuda_runtime` | text | CUDA runtime version, `none` if no GPU | `12.0` |
| 45 | `driver_version` | text | CUDA driver version, `none` if no GPU | `13.4` |
| 46 | `git_commit` | text | short commit hash, `-dirty` suffix if the tree had uncommitted changes | `fe50303-dirty` |

Header line (one line in the file):

```
run_id,timestamp,mode,backend,precision,kernels,input_name,height,width,dates,pixels,samples,tile_rows,num_tiles,streams,block_size,threads,io_threads,repeat_index,input_bytes,read_ms,read_wait_ms,setup_ms,h2d_ms,ndvi_ms,fit_ms,detect_ms,fused_ms,d2h_ms,kernel_ms,total_ms,throughput_msamples_s,kernel_throughput_msamples_s,device_mem_peak_mb,changed_pixels,compare_to,agree_break_frac,agree_status_frac,max_coef_diff,gpu_name,gpu_mem_mb,cpu_model,cpu_threads,cuda_runtime,driver_version,git_commit
```

### 4.1 Cache state and fairness of the reads

All backends read the cube with the same reader (`--io-threads` parallel
`pread` calls per tile) and the same tile size rule: the automatic tile height
is capped so that one tile holds at most 64 MiB of input on the CPU and the GPU
backends alike, so both issue requests of the same size. Cubes that fit in the
machine's RAM are read from the operating-system page cache after the warm-up
run (the benchmark discards the warm-up), so the `read_ms` of such rows
measures memory copies, not the disk. A cube larger than RAM (the 40 GB full
tile) is read cold from disk in every repetition. Do not mix or compare the two
regimes. `scripts/run_benchmarks.sh` writes one line per sweep to
`hardware.txt` with the largest raw cube the sweep reads against the total
RAM, stating which of the two cases applies. An explicit `--tile-rows` applies
to every backend and, on the GPU, is reduced when it needs more than half of
the free device memory (see the `note:` line in section 3.1).

`scripts/check_csv.py` validates a benchmark CSV against this header and row
layout, and `scripts/plot_benchmarks.py` draws the figures from it.

---

## 5. Co-registration output

`--mode coreg` (GPU only) writes `coreg_shifts.csv` into `--output-dir` and
prints the usual log (section 3: header, device, input, then the sections
below). The CSV has one row per date, comma separated, with this header:

```
date_index,date,valid_fraction,dy_px,dx_px,shift_px,peak,flag
```

| Column | Meaning |
| --- | --- |
| `date_index`, `date` | index and `YYYY-MM-DD` of the scene |
| `valid_fraction` | fraction of the correlated window that is a valid observation (SCL class in `--valid-classes`, and non-zero `red` and `nir`), 4 decimals |
| `dy_px`, `dx_px` | displacement of the scene relative to the reference scene in pixels (4 decimals): positive `dy` = content moved down (towards larger row numbers), positive `dx` = moved right. The reference row is exactly 0, 0 |
| `shift_px` | `sqrt(dy^2 + dx^2)` |
| `peak` | height of the normalised phase-correlation peak, 1 for a perfect translation, about `6 / window` for unrelated images |
| `flag` | `ok`, `low_peak` or `large_shift` (a date is given one flag; `low_peak` wins) |

Flag thresholds: `low_peak` when `peak < max(0.02, 10 / window)` (the
highest peak of two unrelated images of `window x window` pixels is about
`6 / window`, so the threshold stays above the noise floor for small windows);
`large_shift` when `shift_px > 1.0`.

Log sections of this mode: `--- parameters ---` (window, origin, band,
requested reference, valid classes and the two thresholds), `--- execution ---`
(window position, band, the reference date actually used with its valid
fraction, `dates_per_batch`: the number of dates transformed by one batched
cuFFT call, chosen from the free device memory), `--- timings (ms) ---` and
`--- results ---`. Timing fields: `read_input` (reading the window rows of every
date and cropping them, host), `setup` (device selection, allocations and cuFFT
plans), `h2d` (copying the cropped window to the device), `fft` (window
preparation, forward and inverse transforms, CUDA events), `spectrum`
(normalised cross-power spectrum), `peak` (argmax and sub-pixel refinement),
`d2h` (results), `total` (wall time from the start of the read until the shifts
are on the host; it includes `setup`). The `results` section gives
`median_shift_px`, `max_shift_px` with its date, the number of flagged dates
and one `flagged_date:` line for each. The last line is
`SUMMARY: processed <T> scenes of <N> x <N> pixels (<samples> samples) in <s>
seconds (backend gpu, cuFFT)` with `samples = T * N * N`. Real output of the
demo cube (window 2048, paths shortened):

```
--- parameters ---
coreg_window: 2048 | origin: 1476,1476 (centred) | band: nir | reference_date: auto | valid_classes: 4,5,6 | low_peak_below: 0.020 | large_shift_above_px: 1.00
--- execution ---
backend: gpu (cuFFT phase correlation) | window: 2048 x 2048 at row 1476, col 1476 | band: nir | reference_date: 40 (2022-05-04, valid 1.0000) | dates_per_batch: 70
--- timings (ms) ---
read_input: 955.810 | setup: 149.015 | h2d: 90.555 | fft: 87.662 | spectrum: 16.212 | peak: 5.134 | d2h: 0.361 | total: 1393.794
--- results ---
median_shift_px: 0.1009 | max_shift_px: 0.5077 (date 24, 2021-01-19) | flagged: 0 (low_peak 0, large_shift 0)
outputs: out/coreg_demo/coreg_shifts.csv
SUMMARY: processed 70 scenes of 2048 x 2048 pixels (293601280 samples) in 1.394 seconds (backend gpu, cuFFT)
```
