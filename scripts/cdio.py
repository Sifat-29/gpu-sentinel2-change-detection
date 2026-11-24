"""Non-GPU helper tooling. Shared readers for the change_detect file formats.

This module is imported by render_figures.py, plot_benchmarks.py and
fill_results.py. It has no command line of its own. It reads:

* ``manifest.txt`` (cube) and ``result_manifest.txt`` (detect output), which
  share one key-value grammar (docs/FORMATS.md),
* the raw arrays of a cube (date-major ``red.u16``, ``nir.u16``, ``scl.u8``)
  and the output maps of a detect run, as read-only ``numpy.memmap`` objects
  so a 10980 x 10980 x 72 cube never has to fit in RAM,
* the text log written by ``change_detect`` (docs/FORMATS.md, "Log format"),
* small CSV files (trajectories, validation, benchmark rows).
"""

import csv
import dataclasses
import os
import re

import numpy as np

# Sentinel returned by ndvi_plane() for masked / nodata pixels is NaN.
DEFAULT_VALID_CLASSES = (4, 5, 6)

STATUS_STABLE = 0
STATUS_BREAK = 1
STATUS_INSUFFICIENT = 2


@dataclasses.dataclass
class DateEntry:
    """One ``date`` line of a manifest."""

    index: int
    date: str
    day: int
    scale: float
    offset: float
    valid_fraction: float
    scene_id: str


@dataclasses.dataclass
class Manifest:
    """Parsed ``manifest.txt`` or ``result_manifest.txt``."""

    path: str
    magic: str
    kv: dict
    dates: list
    sha256: dict
    extra: dict

    @property
    def directory(self):
        return os.path.dirname(os.path.abspath(self.path))

    @property
    def height(self):
        return int(self.kv["height"])

    @property
    def width(self):
        return int(self.kv["width"])

    @property
    def num_dates(self):
        return int(self.kv["num_dates"])

    @property
    def name(self):
        return self.kv.get("name", os.path.basename(self.directory))

    @property
    def date_strings(self):
        return [d.date for d in self.dates]

    @property
    def geotransform(self):
        """Returns the six geotransform numbers or None when absent."""
        raw = self.kv.get("geotransform")
        if not raw:
            return None
        try:
            values = tuple(float(v) for v in raw.split())
        except ValueError:
            return None
        return values if len(values) == 6 else None

    @property
    def pixel_size_m(self):
        """Pixel width in metres when the CRS is metric, else None."""
        gt = self.geotransform
        if gt is None or abs(gt[1]) < 0.5:
            return None
        return abs(gt[1])

    def get(self, key, default=None):
        return self.kv.get(key, default)

    def get_float(self, key, default=None):
        try:
            return float(self.kv[key])
        except (KeyError, ValueError):
            return default


# Keys whose value is the rest of the line (free text, may contain spaces).
_TEXT_KEYS = ("name", "source", "crs", "license", "reference_source",
              "cube_manifest", "command")


def parse_manifest(path):
    """Parses a cube or result manifest.

    Grammar: one record per line, ``#`` comments, fields separated by single
    spaces. ``date`` and ``sha256`` lines are collected separately; every
    other line becomes ``kv[key] = rest of line``. Repeated unknown keys keep
    the last value and are also stored as lists in ``extra``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        lines = [ln.rstrip("\n") for ln in handle]
    content = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    if not content:
        raise ValueError("%s: empty manifest" % path)
    magic = content[0].strip()
    if magic not in ("change_detect_cube 1", "change_detect_result 1"):
        raise ValueError("%s: bad first line '%s'" % (path, magic))
    kv, dates, sha, extra = {}, [], {}, {}
    for line in content[1:]:
        key, _, rest = line.partition(" ")
        rest = rest.strip()
        if key == "date":
            fields = rest.split(None, 7)
            if len(fields) < 4:
                raise ValueError("%s: bad date line '%s'" % (path, line))
            dates.append(DateEntry(
                index=int(fields[0]), date=fields[1], day=int(fields[2]),
                scale=float(fields[3]) if len(fields) > 3 else 1.0,
                offset=float(fields[4]) if len(fields) > 4 else 0.0,
                valid_fraction=float(fields[5]) if len(fields) > 5 else -1.0,
                scene_id=fields[6] if len(fields) > 6 else ""))
        elif key == "sha256":
            name, _, digest = rest.partition(" ")
            sha[name] = digest.strip()
        else:
            kv[key] = rest
            extra.setdefault(key, []).append(rest)
    manifest = Manifest(path=path, magic=magic, kv=kv, dates=dates,
                        sha256=sha, extra=extra)
    for key in ("height", "width", "num_dates"):
        if key not in kv:
            raise ValueError("%s: missing required key '%s'" % (path, key))
    if magic.startswith("change_detect_cube"):
        for key in ("red_file", "nir_file", "scl_file"):
            if key not in kv:
                raise ValueError("%s: missing required key '%s'" % (path, key))
    if dates:
        if len(dates) != manifest.num_dates:
            raise ValueError("%s: %d date lines but num_dates %d" % (
                path, len(dates), manifest.num_dates))
        for i, entry in enumerate(dates):
            if entry.index != i:
                raise ValueError("%s: date indices must run 0..T-1" % path)
    return manifest


def _memmap(path, dtype, shape):
    """Opens a raw little-endian array read-only after checking its size."""
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError("%s: size %d bytes, expected %d for shape %s" % (
            path, actual, expected, shape))
    return np.memmap(path, dtype=np.dtype(dtype).newbyteorder("<"),
                     mode="r", shape=shape)


class Cube:
    """Memory-mapped view of a cube directory (arrays are date-major)."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.height = manifest.height
        self.width = manifest.width
        self.num_dates = manifest.num_dates
        self.directory = manifest.directory
        shape = (self.num_dates, self.height, self.width)
        self.red = _memmap(self._file("red_file"), "<u2", shape)
        self.nir = _memmap(self._file("nir_file"), "<u2", shape)
        self.scl = _memmap(self._file("scl_file"), "u1", shape)

    def _file(self, key):
        return os.path.join(self.directory, self.manifest.kv[key])

    def reference(self):
        """Hansen lossyear (H x W uint8) or None when the cube has none."""
        name = self.manifest.get("reference_file")
        if not name:
            return None
        path = os.path.join(self.directory, name)
        if not os.path.exists(path):
            return None
        return _memmap(path, "u1", (self.height, self.width))

    def truth_break_index(self):
        """Injected break index of a synthetic cube (int16) or None."""
        path = os.path.join(self.directory, "truth_break_index.i16")
        if not os.path.exists(path):
            return None
        return _memmap(path, "<i2", (self.height, self.width))

    def preview_path(self, index, full=False):
        """Path of the preview PNG for date index, or None if missing."""
        stem = "rgb_full_" if full else "rgb_"
        date = self.manifest.dates[index].date
        path = os.path.join(self.directory, "previews",
                            "%s%s.png" % (stem, date))
        return path if os.path.exists(path) else None

    def has_previews(self):
        return os.path.isdir(os.path.join(self.directory, "previews"))

    def reflectance(self, plane, index):
        """Converts a DN array of date index to float32 reflectance."""
        entry = self.manifest.dates[index]
        return plane.astype(np.float32) * np.float32(entry.scale) + np.float32(
            entry.offset)

    def ndvi_plane(self, index, rows=slice(None), cols=slice(None),
                   valid_classes=DEFAULT_VALID_CLASSES):
        """Float32 NDVI of date index; NaN where masked or nodata.

        Follows README.md: invalid if the SCL class is not in valid_classes,
        red or nir DN is 0, or nir + red <= 1e-4; otherwise clamped to [-1, 1].
        rows and cols may be slices with a step (strided downsampling).
        """
        red_dn = np.asarray(self.red[index][rows, cols])
        nir_dn = np.asarray(self.nir[index][rows, cols])
        scl = np.asarray(self.scl[index][rows, cols])
        red = self.reflectance(red_dn, index)
        nir = self.reflectance(nir_dn, index)
        denom = nir + red
        ok = np.isin(scl, np.asarray(valid_classes, dtype=np.uint8))
        ok &= (red_dn != 0) & (nir_dn != 0) & (denom > 1e-4)
        out = np.full(red.shape, np.nan, dtype=np.float32)
        out[ok] = np.clip((nir[ok] - red[ok]) / denom[ok], -1.0, 1.0)
        return out


def open_cube(manifest_path):
    """Parses a cube manifest and opens its arrays."""
    return Cube(parse_manifest(manifest_path))


class Result:
    """Memory-mapped view of a detect output directory (docs/FORMATS.md)."""

    def __init__(self, directory):
        self.directory = directory
        path = os.path.join(directory, "result_manifest.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.manifest = parse_manifest(path)
        self.height = self.manifest.height
        self.width = self.manifest.width
        self.num_dates = self.manifest.num_dates
        shape = (self.height, self.width)
        self.break_index = self._map("break_index.i16", "<i2", shape)
        self.status = self._map("status.u8", "u1", shape)
        self.magnitude = self._map("magnitude.f32", "<f4", shape)
        self.rmse = self._map("rmse.f32", "<f4", shape)
        self.train_obs = self._map("train_obs.u16", "<u2", shape)

    def path(self, name):
        return os.path.join(self.directory, name)

    def exists(self, name):
        return os.path.exists(self.path(name))

    def _map(self, name, dtype, shape):
        full = self.path(name)
        return _memmap(full, dtype, shape) if os.path.exists(full) else None

    def coefficients(self, num_params):
        """Coefficient maps (p x H x W float32) or None if not saved."""
        full = self.path("coefficients.f32")
        if not os.path.exists(full):
            return None
        return _memmap(full, "<f4", (num_params, self.height, self.width))

    def cube_manifest_path(self, hint=None):
        """Locates the cube manifest named by the result manifest."""
        candidates = [hint] if hint else []
        recorded = self.manifest.get("cube_manifest")
        if recorded:
            candidates += [recorded,
                           os.path.join(self.directory, recorded),
                           os.path.join(self.directory, "..", recorded)]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None


def open_result(directory):
    return Result(directory)


def _convert(value):
    """Converts a CSV cell: '' -> None, int, float, else the string."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_csv(path, text_columns=()):
    """Loads a CSV into a list of dicts with numeric cells converted.

    Columns named in text_columns stay strings ('' becomes None).
    """
    def cell(key, value):
        if key in text_columns:
            return value if value != "" else None
        return _convert(value)

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{k: cell(k, v) for k, v in row.items()} for row in reader]


def load_key_value_csv(path):
    """Loads a ``metric,value`` CSV (validation.csv) into a dict."""
    return {row["metric"]: row["value"] for row in load_csv(path)
            if row.get("metric") is not None}


# ---------------------------------------------------------------- log parsing

_SECTION_RE = re.compile(r"^---\s*(.+?)\s*---$")
_KEY_RE = re.compile(r"^([A-Za-z_][^:|]*?):\s*(.*)$")
_SUMMARY_RE = re.compile(
    r"^SUMMARY: processed (\d+) pixel time series \((\d+) samples\) in "
    r"([0-9.eE+-]+) seconds(?: \(backend ([^)]*)\))?")


def _split_pairs(text):
    """Splits ``a: 1 | b: 2 | cc 8.9`` into [(key, value)]; keyless parts
    are appended to the previous value."""
    pairs = []
    for part in [p.strip() for p in text.split(" | ")]:
        match = _KEY_RE.match(part)
        if match:
            pairs.append([match.group(1).strip(), match.group(2).strip()])
        elif pairs:
            pairs[-1][1] += " | " + part
    return [(k, v) for k, v in pairs]


class Log:
    """A parsed ``change_detect`` log.

    Attributes:
      header: key/values before the first section (command, start_time, ...).
      sections: ordered dict section name -> dict key -> value (strings).
      summary: dict from the SUMMARY line (pixels, samples, seconds, backend)
        or None.
      bench_lines: raw ``bench:`` lines.
      timings: dict stage -> float ms (``-`` skipped), from the timings
        section.
      lines: all raw lines.
    """

    def __init__(self, path):
        self.path = path
        self.header = {}
        self.sections = {}
        self.summary = None
        self.bench_lines = []
        self.timings = {}
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            self.lines = [ln.rstrip("\n") for ln in handle]
        self._parse()

    def _parse(self):
        current = self.header
        for line in self.lines:
            text = line.strip()
            if not text or text.startswith("==="):
                continue
            match = _SUMMARY_RE.match(text)
            if match:
                self.summary = {
                    "pixels": int(match.group(1)),
                    "samples": int(match.group(2)),
                    "seconds": float(match.group(3)),
                    "backend": (match.group(4) or "").strip(),
                    "text": text,
                }
                continue
            if text.startswith("bench:"):
                self.bench_lines.append(text)
                continue
            section = _SECTION_RE.match(text)
            if section:
                current = self.sections.setdefault(section.group(1), {})
                continue
            if text.startswith("["):  # per-tile progress line
                continue
            if text.startswith("command:"):
                current["command"] = text[len("command:"):].strip()
                continue
            for key, value in _split_pairs(text):
                current.setdefault(key, value)
        for key, value in self.sections.get("timings (ms)", {}).items():
            number = _number(value)
            if number is not None:
                self.timings[key] = number

    def get(self, key, default=None):
        """Finds a key in the header, then every section."""
        if key in self.header:
            return self.header[key]
        for values in self.sections.values():
            if key in values:
                return values[key]
        return default

    @property
    def command(self):
        return self.header.get("command", "")

    @property
    def mode(self):
        match = re.search(r"--mode[= ](\S+)", self.command)
        return match.group(1) if match else "detect"

    def get_int(self, key):
        number = _number(self.get(key))
        return None if number is None else int(number)


def _number(text):
    """Leading number of a string (``4.2``, ``25000000``) or None."""
    if text is None:
        return None
    match = re.match(r"\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", str(text))
    return float(match.group(1)) if match else None


def number(text):
    """Public alias of the leading-number helper."""
    return _number(text)


def parse_log(path):
    return Log(path)


def nice_length(max_value):
    """Largest 1/2/5 x 10^n not above max_value (for scale bars)."""
    if max_value <= 0:
        return 0.0
    exponent = np.floor(np.log10(max_value))
    for factor in (5.0, 2.0, 1.0):
        candidate = factor * 10.0 ** exponent
        if candidate <= max_value:
            return float(candidate)
    return float(10.0 ** exponent)
