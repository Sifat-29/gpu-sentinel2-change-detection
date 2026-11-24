"""Non-GPU helper tooling.

Cuts a small, committable sample cube out of a larger cube: a square window
(default 128 x 128, all dates) at the location of the reference layer that
has the most balanced mix of loss and non-loss eligible pixels, so that both
detections and quiet pixels appear in the sample. The result (arrays,
reference, manifest, cropped previews) goes to ``data/sample/`` and every
file stays below 5 MB.

This script does not use the GPU. Example:

    .venv/bin/python scripts/make_sample.py \
        --input data/cubes/chaco_dev2048/manifest.txt --size 128 \
        --out data/sample
"""

import argparse
import pathlib
import sys

import numpy as np

import crop_cube
import cube_common as cc

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 5 * 1000 * 1000


def parse_args(argv=None):
    """Builds the command line."""
    parser = argparse.ArgumentParser(
        description="Crop a small loss-rich sample cube "
        "(non-GPU helper tooling).")
    parser.add_argument("--input", required=True, help="source manifest.txt")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--out", default="", help="default data/sample")
    parser.add_argument("--name", default="chaco_sample128")
    parser.add_argument("--step", type=int, default=16,
                        help="candidate window origin step in pixels")
    return parser.parse_args(argv)


def pick_window(man, size, step):
    """Origin (r0, c0) maximising min(loss, eligible non-loss) pixels."""
    if not man.reference_file:
        raise SystemExit("input cube has no reference layer; run "
                         "fetch_hansen.py first")
    ref = np.fromfile(man.path(man.reference_file), dtype="u1").reshape(
        man.height, man.width)
    loss = ((ref >= 1) & (ref != 255)).astype(np.int64)
    quiet = (ref == 0).astype(np.int64)
    tables = []
    for layer in (loss, quiet):
        table = np.zeros((man.height + 1, man.width + 1), np.int64)
        table[1:, 1:] = layer.cumsum(axis=0).cumsum(axis=1)
        tables.append(table)
    rows = np.arange(0, man.height - size + 1, step)
    cols = np.arange(0, man.width - size + 1, step)
    r, c = np.meshgrid(rows, cols, indexing="ij")
    scores = None
    for table in tables:
        sums = (table[r + size, c + size] - table[r, c + size]
                - table[r + size, c] + table[r, c])
        scores = sums if scores is None else np.minimum(scores, sums)
    index = np.unravel_index(np.argmax(scores), scores.shape)
    return (int(scores[index]), int(rows[index[0]]),
            int(cols[index[1]]))


def main(argv=None):
    """Entry point."""
    args = parse_args(argv)
    man = cc.read_manifest(args.input)
    score, r0, c0 = pick_window(man, args.size, args.step)
    cc.log(f"sample window r0,c0 = {r0},{c0} size {args.size} "
           f"(balanced score {score} pixels)")
    out_dir = (pathlib.Path(args.out) if args.out
               else REPO_DIR / "data" / "sample")
    window = (r0, c0, args.size, args.size)
    sample = crop_cube.crop_cube(man, out_dir, window,
                                 (0, man.num_dates), args.name)
    manifests = REPO_DIR / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    cc.write_manifest(sample, manifests / f"{args.name}_manifest.txt")
    largest = max(path.stat().st_size for path in out_dir.rglob("*")
                  if path.is_file())
    cc.log(f"sample written to {out_dir}; largest file {largest / 1e6:.2f} MB")
    if largest >= MAX_FILE_BYTES:
        print("ERROR: a sample file is 5 MB or larger", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
