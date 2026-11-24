"""Non-GPU helper tooling.

Shared helpers for the data scripts in this directory: reading and writing
the cube ``manifest.txt`` (docs/FORMATS.md), sha256 of large files, window
parsing and the GDAL environment used for remote COG reads. It is imported by
``fetch_sentinel2.py``, ``crop_cube.py``, ``fetch_hansen.py``,
``find_windows.py``, ``make_sample.py`` and ``verify_cube.py``; it is not a
command-line tool.
"""

import dataclasses
import datetime
import hashlib
import os
import pathlib
import threading
import time

import numpy as np

LICENSE_LINE = "Contains modified Copernicus Sentinel data 2019-2024"
SCL_VALID_CLASSES = (4, 5, 6)
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
FALLBACK_COLLECTION = "sentinel-2-l2a"
CHUNK_BYTES = 64 << 20
DOWNLOAD_ATTEMPTS = 6

# Settings that make GDAL read remote COGs with few requests and retries.
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "8",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 << 20),
}


@dataclasses.dataclass
class DateRecord:
    """One ``date`` line of the manifest."""
    index: int
    day: str
    scale: float
    offset: float
    valid_fraction: float
    scene_id: str

    @property
    def days_since_1970(self):
        parsed = datetime.date.fromisoformat(self.day)
        return (parsed - datetime.date(1970, 1, 1)).days


@dataclasses.dataclass
class Manifest:
    """In-memory form of ``manifest.txt``; unknown keys land in ``extra``."""
    directory: pathlib.Path
    name: str = ""
    source: str = ""
    license: str = LICENSE_LINE
    crs: str = ""
    geotransform: tuple = (0.0, 10.0, 0.0, 0.0, 0.0, -10.0)
    height: int = 0
    width: int = 0
    red_file: str = "red.u16"
    nir_file: str = "nir.u16"
    scl_file: str = "scl.u8"
    reference_file: str = ""
    reference_source: str = ""
    dates: list = dataclasses.field(default_factory=list)
    sha256: dict = dataclasses.field(default_factory=dict)
    extra: list = dataclasses.field(default_factory=list)

    @property
    def num_dates(self):
        return len(self.dates)

    def path(self, file_name):
        return self.directory / file_name


def format_number(value):
    """Formats a float without trailing noise (699960 stays '699960')."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def read_manifest(path):
    """Parses a manifest file; raises ValueError on malformed input."""
    path = pathlib.Path(path)
    man = Manifest(directory=path.parent)
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    if not lines or lines[0].strip() != "change_detect_cube 1":
        raise ValueError(f"{path}: first line must be 'change_detect_cube 1'")
    for number, raw in enumerate(lines[1:], start=2):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        if key in ("name", "source", "license", "crs", "reference_source"):
            setattr(man, key, rest)
        elif key == "geotransform":
            man.geotransform = tuple(float(v) for v in rest.split())
        elif key in ("height", "width"):
            setattr(man, key, int(rest))
        elif key == "num_dates":
            pass  # Derived from the date lines.
        elif key in ("red_file", "nir_file", "scl_file", "reference_file"):
            setattr(man, key, rest)
        elif key == "date":
            parts = rest.split()
            if len(parts) != 7:
                raise ValueError(f"{path}:{number}: malformed date line")
            man.dates.append(
                DateRecord(int(parts[0]), parts[1], float(parts[3]),
                           float(parts[4]), float(parts[5]), parts[6]))
        elif key == "sha256":
            file_name, _, digest = rest.partition(" ")
            man.sha256[file_name] = digest
        else:
            man.extra.append(line)
    for index, record in enumerate(man.dates):
        if record.index != index:
            raise ValueError(f"{path}: date indices must run 0..T-1")
    return man


def write_manifest(man, path=None):
    """Writes the manifest in the docs/FORMATS.md grammar (atomic rename)."""
    path = pathlib.Path(path) if path else man.directory / "manifest.txt"
    gt = " ".join(format_number(v) for v in man.geotransform)
    lines = [
        "change_detect_cube 1",
        f"name {man.name}",
        f"source {man.source}",
        f"license {man.license}",
        f"crs {man.crs}",
        f"geotransform {gt}",
        f"height {man.height}",
        f"width {man.width}",
        f"num_dates {man.num_dates}",
        f"red_file {man.red_file}",
        f"nir_file {man.nir_file}",
        f"scl_file {man.scl_file}",
    ]
    if man.reference_file:
        lines.append(f"reference_file {man.reference_file}")
        lines.append(f"reference_source {man.reference_source}")
    lines.extend(man.extra)
    lines.append("# date <index> <YYYY-MM-DD> <days_since_1970> <scale> "
                 "<offset> <valid_fraction> <scene_id>")
    for rec in man.dates:
        lines.append(f"date {rec.index} {rec.day} {rec.days_since_1970} "
                     f"{format_number(rec.scale)} {format_number(rec.offset)} "
                     f"{rec.valid_fraction:.4f} {rec.scene_id}")
    for file_name, digest in man.sha256.items():
        lines.append(f"sha256 {file_name} {digest}")
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path, chunk=CHUNK_BYTES):
    """Streams a file through sha256 and returns the hex digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_window(text, height, width):
    """Parses 'r0,c0,h,w' (None means the full extent) and validates it."""
    if not text:
        return 0, 0, height, width
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--window needs r0,c0,h,w")
    r0, c0, h, w = parts
    if r0 < 0 or c0 < 0 or h <= 0 or w <= 0:
        raise ValueError("--window values must be positive")
    if r0 + h > height or c0 + w > width:
        raise ValueError(f"window exceeds the {height}x{width} grid")
    return r0, c0, h, w


def apply_gdal_options():
    """Exports the GDAL options to the process environment (idempotent)."""
    for key, value in GDAL_OPTIONS.items():
        os.environ.setdefault(key, value)


def scl_valid_fraction(scl):
    """Fraction of pixels whose SCL class is vegetation, bare soil or water."""
    return float(np.isin(scl, SCL_VALID_CLASSES).mean())


class Progress:
    """Thread-safe byte counter used for 'MB/s' progress lines."""

    def __init__(self):
        self.start = time.time()
        self.bytes = 0
        self.lock = threading.Lock()

    def add(self, count):
        with self.lock:
            self.bytes += count

    def rate(self):
        elapsed = max(time.time() - self.start, 1e-6)
        return self.bytes / (1 << 20) / elapsed


def log(message):
    """Prints a timestamped line and flushes (logs are tailed live)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def retry(function, what):
    """Calls function() with exponential back-off; re-raises at the end."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return function()
        except Exception as error:  # Network and GDAL errors are many types.
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            delay = 2 ** attempt
            log(f"retry {attempt} for {what}: {error!r}; sleeping {delay}s")
            time.sleep(delay)


def download_file(url, destination, progress):
    """Streams a URL to destination via a .part file; resumes with Range."""
    import requests
    destination = pathlib.Path(destination)
    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def attempt():
        start = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        with requests.get(url, headers=headers, stream=True,
                          timeout=(30, 60)) as response:
            if response.status_code == 416:
                return
            response.raise_for_status()
            mode = "ab" if response.status_code == 206 else "wb"
            with open(part, mode) as handle:
                for block in response.iter_content(1 << 20):
                    handle.write(block)
                    progress.add(len(block))

    retry(attempt, url)
    os.replace(part, destination)
