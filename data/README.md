# Data: Sentinel-2 time series and Hansen forest-loss reference

All scripts mentioned here are non-GPU helper tooling (Python, `project/.venv`, created by
`bash scripts/setup_python.sh`). Run them from the repository root `project/`. Large data lives in the
git-ignored folders `data/cubes/`, `data/staging/` and `data/reference/`; only the scripts, the
manifests (`data/manifests/`) and a small sample cube (`data/sample/`) are committed.

## Sources, licences and attribution

| Data | Source | Licence / credit |
| --- | --- | --- |
| Sentinel-2 L2A bands B04 (red), B08 (near infrared), SCL (scene classification) and the TCI true-colour asset | Earth Search STAC API `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-c1-l2a` (Collection 1 reprocessing, AWS Open Data bucket `e84-earth-search-sentinel-data`), plus `sentinel-2-l2a` for months missing in Collection 1 (see "Data quirks") | Copernicus Sentinel data are free and open. Credit: **Contains modified Copernicus Sentinel data 2019-2024.** |
| Forest-loss reference | Hansen Global Forest Change GFC-2024-v1.12, layers `lossyear` and `treecover2000`, 10 x 10 degree tiles from `https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/` (download page: `.../GFC-2024-v1.12/download.html`, checked 2026-11-23) | Creative Commons Attribution 4.0 International (stated on the download page). Display credit: "Source: Hansen/UMD/Google/USGS/NASA". Citation: Hansen, M. C., et al. 2013. High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science* 342 (15 November): 850-853. doi:10.1126/science.1244693. Data resampled to the 10 m grid (nearest neighbour) and filtered; these are changes made by `scripts/fetch_hansen.py`. |

The Hansen product is a sanity check, not ground truth (30 m Landsat-based, annual, stand-replacement
loss only).

## Study area and windows

Paraguayan Chaco, Sentinel-2 MGRS tile **20KQA** (EPSG:32720, 10980 x 10980 pixels at 10 m, origin
`699960, 7600000`, relative orbit 110). The tile straddles two Hansen tiles (`20S_070W`, `20S_060W`).
The windows were chosen by evidence with `scripts/find_windows.py --tile 20KQA` (Hansen loss in
2021-2024 on pixels with tree cover >= 30 %, window origins on a 64 px grid):

| Cube | Window `r0,c0,h,w` in tile pixels | Loss pixels 2021-2024 | Loss fraction of window |
| --- | --- | --- | --- |
| `chaco_dev2048` | `0,3008,2048,2048` (rank 2 of 2048 windows; rank 1 at `0,8320` has 38 % loss and lies inside the demo window) | 491 321 | 0.117 |
| `chaco_demo5000` | `0,5504,5000,5000` (rank 1 of 5000 windows, disjoint from the dev window) | 1 728 576 | 0.069 |
| `chaco_full` | whole tile | 3 148 725 | 0.026 |
| `chaco_sample128` | 128 x 128 at `864,1824` inside the dev cube (balanced loss / non-loss) | 8 219 of 16 384 | 0.50 |

Tuning uses the dev cube, validation uses the demo cube (they do not overlap).

## Cube format

`data/cubes/<name>/` (specification: `docs/FORMATS.md`). Raw arrays are little-endian and date-major:
element `(t, r, c)` is at index `t * H * W + r * W + c`.

| File | Type | Content |
| --- | --- | --- |
| `manifest.txt` | text | key/value records, one `date` line per scene (index, day, days since 1970, scale, offset, valid fraction, scene id), sha256 lines |
| `red.u16`, `nir.u16` | uint16 | B04 and B08 digital numbers, 0 = no data; reflectance = `DN * scale + offset` per date |
| `scl.u8` | uint8 | Scene classification, 20 m upsampled x2 (nearest) to the 10 m grid |
| `reference_lossyear.u8` | uint8, H x W | Hansen `lossyear` 0..24 (0 = no loss, n = year 2000 + n); 255 = excluded |
| `previews/rgb_<date>.png` | PNG | true colour from the TCI asset, long side <= 1024 px, every date |
| `previews/rgb_full_<date>.png` | PNG | full-resolution true colour of the window, first and last date |

`data/manifests/` holds copies of every manifest plus `<name>_scenes.csv` (date, scene id, cloud %,
nodata %, processing baseline, scale, offset, STAC offset, relative orbit, collection).

### Scene classification (SCL) classes and the mask rule

| SCL | Meaning | SCL | Meaning |
| --- | --- | --- | --- |
| 0 | no data | 6 | water |
| 1 | saturated or defective | 7 | unclassified |
| 2 | dark area pixels | 8 | cloud, medium probability |
| 3 | cloud shadows | 9 | cloud, high probability |
| 4 | vegetation | 10 | thin cirrus |
| 5 | not vegetated (bare soil) | 11 | snow or ice |

Mask rule (default): an observation is valid when SCL is 4, 5 or 6 and the band DN is not 0. The raw SCL
plane is stored; the valid set is a run-time flag of the detector. The manifest `valid_fraction` of a
date is the fraction of window pixels whose SCL is in {4, 5, 6}.

### Reference rule

`reference_lossyear.u8` = Hansen `lossyear` (nearest-neighbour onto the cube grid), except 255 where
`treecover2000 < 30`, where no Hansen data exist, or where loss happened before the monitoring start
year (`lossyear` in 1..20 for the default start year 2021, i.e. lost before or during training).

## Data quirks found (and how the scripts handle them)

* **Collection 1 has a gap in 2022.** For tile 20KQA `sentinel-2-c1-l2a` holds only 7 items in 2022
  (2 below 20 % cloud) against 58-79 in the other years, and none for 2019-01..03. A monitoring series
  with no data from January to November 2022 would date every 2022 clearing to December, so
  `fetch_sentinel2.py` fills the months that have no qualifying Collection 1 scene from the original
  collection `sentinel-2-l2a` (`--fallback-collection`, empty string disables it). 13 of the 70 dates
  come from there. Scenes are on the same MGRS grid (checked) and the same relative orbit 110, but have
  older processing baselines (02.11, 03.01, 04.00).
* **Offset metadata is wrong for some fallback scenes.** Per-date scale and offset are read from each
  item's `raster:bands`. For `sentinel-2-l2a` scenes of 2022-03-05 to 2022-11-15 (baseline 04.00) the
  item says offset -0.1, but the stored DN lack the +1000 shift (first percentile of valid red DN is
  about 350-650 instead of >= 1100 in every other scene, giving mean NDVI of 5440 on one date). The
  script checks this per date (`effective_offset`: first percentile of valid red DN below 800 while the
  metadata claims -0.1) and then records offset 0 for that date (8 dates; the logged warning, the
  `offset`/`stac_offset` columns of the scene CSV and the corrected mean NDVI of 0.32-0.52, in line
  with neighbouring dates, are the evidence).
* The tile-level `eo:cloud_cover < 20` and `s2:nodata_pixel_percentage < 1` filters leave 156
  Collection 1 plus 294 fallback candidate scenes; months without any candidate (2022-07,
  2023-12) have no date, so 70 of 72 months are present.

## Recreate every cube

```
bash scripts/setup_python.sh
# 1. which windows to use (needs ~1.1 GB of Hansen tiles, cached in data/reference/)
.venv/bin/python scripts/find_windows.py --tile 20KQA --sizes 2048,5000
# 2. scene list only
.venv/bin/python scripts/fetch_sentinel2.py --name chaco_full --tile 20KQA --list-only
# 3. development cube (windowed COG reads), then its reference layer
.venv/bin/python scripts/fetch_sentinel2.py --name chaco_dev2048 --tile 20KQA \
    --window 0,3008,2048,2048 --workers 4 --delete-staging
.venv/bin/python scripts/fetch_hansen.py --cube data/cubes/chaco_dev2048/manifest.txt
cp data/cubes/chaco_dev2048/manifest.txt data/manifests/chaco_dev2048_manifest.txt
.venv/bin/python scripts/verify_cube.py --manifest data/cubes/chaco_dev2048/manifest.txt
# 4. committed sample
.venv/bin/python scripts/make_sample.py --input data/cubes/chaco_dev2048/manifest.txt \
    --size 128 --out data/sample
# 5. full tile (resumable: re-run the same command after an interruption), then demo cube
.venv/bin/python scripts/fetch_sentinel2.py --name chaco_full --tile 20KQA --workers 6 --delete-staging
.venv/bin/python scripts/fetch_hansen.py --cube data/cubes/chaco_full/manifest.txt
cp data/cubes/chaco_full/manifest.txt data/manifests/chaco_full_manifest.txt
.venv/bin/python scripts/crop_cube.py --input data/cubes/chaco_full/manifest.txt \
    --window 0,5504,5000,5000 --out data/cubes/chaco_demo5000
.venv/bin/python scripts/verify_cube.py --manifest data/cubes/chaco_demo5000/manifest.txt
```

If the full-tile download is not possible, the demo cube can be fetched directly:
`fetch_sentinel2.py --name chaco_demo5000 --tile 20KQA --window 0,5504,5000,5000 --workers 4`
followed by `fetch_hansen.py --cube data/cubes/chaco_demo5000/manifest.txt`.

`fetch_sentinel2.py` stages each date under `data/staging/<name>/` (`*.part` files renamed when
complete, `<date>.json` written last as the completion marker), so a re-run skips finished dates.
`crop_cube.py --window r0,c0,h,w --dates i0:i1` makes any other crop of an existing cube.

## Sizes and throughput (measured on the development machine)

| Item | Size |
| --- | --- |
| Scene list 20KQA, 2019-2024, best scene per month | 70 dates |
| Hansen tiles `lossyear` 20S_070W + 20S_060W / `treecover2000` | 68.6 + 88.2 MB / 364.6 + 649.4 MB (1.1 GB, cached once) |
| `chaco_dev2048` (70 dates) | red 587 MB, nir 587 MB, scl 294 MB, reference 4 MB, previews 148 MB, 1.6 GB in total |
| `chaco_sample128` | red 2.3 MB, nir 2.3 MB, scl 1.1 MB, reference 16 KB, 72 previews, 6.2 MB in total, no file above 5 MB |
| `chaco_demo5000` (70 dates) | 3.5 GB red + 3.5 GB nir + 1.75 GB scl, 8.75 GB raw |
| `chaco_full` (70 dates) | 16.9 GB red + 16.9 GB nir + 8.4 GB scl, 42.2 GB raw (the staging copy needs the same space again until `--delete-staging` runs) |

Dev cube download: 70 dates in about 4 minutes with 4 workers, 5.6 MB/s average decoded throughput
(window reads from the COGs, about 18 MB per date and band set). Single-connection whole-file download of one
full-tile B04 (231 MB) ran at 3.7 MB/s, a GDAL range-read of the same band at 7.0 MB/s, so the scripts
default to range reads (`--read-mode download` is available). Full-tile run (6 workers, 10980 x 10980
window, about 512 MB decoded per date): 12.2 MB/s average decoded throughput over the first six
dates, which projects to roughly 40 minutes for the 70 dates, followed by about 10 minutes of assembly
into the cube files.
