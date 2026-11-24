"""Non-GPU helper tooling.

Verifies a cube written by the data scripts: the manifest parses, every
raw file has the size implied by (num_dates, height, width), the sha256 lines
match, the reference layer only holds 0..24 and 255, and the per-date valid
fraction recorded in the manifest matches the SCL plane. It prints the valid
fraction and the mean NDVI of valid pixels per date as a sanity check and
exits 0 when everything passes, 1 otherwise.

This script does not use the GPU. Example:

    .venv/bin/python scripts/verify_cube.py \
        --manifest data/cubes/chaco_dev2048/manifest.txt
"""

import argparse
import sys

import numpy as np

import cube_common as cc

VALID_FRACTION_TOLERANCE = 5.0e-4


def parse_args(argv=None):
    """Builds the command line."""
    parser = argparse.ArgumentParser(
        description="Check sizes, checksums and sanity of a cube "
        "(non-GPU helper tooling).")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skip-sha256", action="store_true",
                        help="skip checksums (very large cubes)")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the per-date table")
    return parser.parse_args(argv)


def check_sizes(man, errors):
    """Compares file sizes with the manifest dimensions."""
    cells = man.height * man.width
    expected = {man.red_file: man.num_dates * cells * 2,
                man.nir_file: man.num_dates * cells * 2,
                man.scl_file: man.num_dates * cells}
    if man.reference_file:
        expected[man.reference_file] = cells
    for file_name, size in expected.items():
        path = man.path(file_name)
        if not path.exists():
            errors.append(f"missing file {file_name}")
        elif path.stat().st_size != size:
            errors.append(f"{file_name}: size {path.stat().st_size} != "
                          f"expected {size}")
    return expected


def check_hashes(man, files, errors):
    """Recomputes sha256 for every file that has a sha256 line."""
    for file_name in files:
        recorded = man.sha256.get(file_name)
        if recorded is None:
            errors.append(f"no sha256 line for {file_name}")
            continue
        actual = cc.sha256_file(man.path(file_name))
        status = "ok" if actual == recorded else "MISMATCH"
        print(f"sha256 {file_name}: {status}")
        if actual != recorded:
            errors.append(f"{file_name}: sha256 mismatch")


def check_dates(man, quiet, errors):
    """Per-date valid fraction and mean NDVI (sanity)."""
    shape = (man.num_dates, man.height, man.width)
    red = np.memmap(man.path(man.red_file), dtype="<u2", mode="r",
                    shape=shape)
    nir = np.memmap(man.path(man.nir_file), dtype="<u2", mode="r",
                    shape=shape)
    scl = np.memmap(man.path(man.scl_file), dtype="u1", mode="r",
                    shape=shape)
    if not quiet:
        print("idx date       valid_manifest valid_scl  mean_ndvi  scene")
    for t, rec in enumerate(man.dates):
        plane = np.asarray(scl[t])
        valid = np.isin(plane, cc.SCL_VALID_CLASSES)
        fraction = float(valid.mean())
        r_dn = np.asarray(red[t])
        n_dn = np.asarray(nir[t])
        usable = valid & (r_dn > 0) & (n_dn > 0)
        if usable.any():
            r = r_dn[usable].astype(np.float64) * rec.scale + rec.offset
            n = n_dn[usable].astype(np.float64) * rec.scale + rec.offset
            ndvi = (n - r) / np.maximum(n + r, 1e-6)
            mean_ndvi = float(ndvi.mean())
        else:
            mean_ndvi = float("nan")
        if not quiet:
            print(f"{t:3d} {rec.day} {rec.valid_fraction:14.4f} "
                  f"{fraction:9.4f}  {mean_ndvi:9.4f}  {rec.scene_id}")
        if abs(fraction - rec.valid_fraction) > VALID_FRACTION_TOLERANCE:
            errors.append(f"date {t}: valid fraction {rec.valid_fraction} "
                          f"vs SCL {fraction:.4f}")
        if usable.any() and not -1.0 <= mean_ndvi <= 1.0:
            errors.append(f"date {t}: mean NDVI {mean_ndvi} out of range")


def check_reference(man, errors):
    """Reference values must be 0..24 or 255."""
    if not man.reference_file:
        print("no reference layer in this cube")
        return
    ref = np.fromfile(man.path(man.reference_file), dtype="u1")
    bad = int(((ref > 24) & (ref != 255)).sum())
    eligible = ref != 255
    print(f"reference: eligible {eligible.mean():.4f}, loss pixels "
          f"{int(((ref >= 1) & eligible).sum())}")
    if bad:
        errors.append(f"reference has {bad} values outside 0..24 and 255")


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    errors = []
    try:
        man = cc.read_manifest(args.manifest)
    except (OSError, ValueError) as error:
        print(f"FAILED: {error}")
        return 1
    print(f"cube {man.name}: {man.num_dates} dates, {man.height}x{man.width}, "
          f"{man.crs}")
    expected = check_sizes(man, errors)
    if not errors:
        if not args.skip_sha256:
            check_hashes(man, expected, errors)
        check_dates(man, args.quiet, errors)
        check_reference(man, errors)
    if errors:
        for message in errors:
            print(f"ERROR: {message}")
        print("FAILED")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
