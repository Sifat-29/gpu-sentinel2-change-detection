"""Non-GPU helper tooling. Renders the reviewer-facing figures of a run.

Reads the output directory of ``change_detect --mode detect`` and, optionally,
the input cube (formats: docs/FORMATS.md) and writes these PNG/GIF files into
``--out``:

  change_year_map.png          break year as a categorical map (base image)
  before_<date>.png            same crop at the first date
  after_<date>.png             same crop at the last date
  before_after_crop.png        both crops side by side, changed pixels outlined
  ndvi_<date>.png              NDVI maps for 2-3 dates (masked pixels grey)
  ndvi_trajectory_examples.png observed NDVI, fitted model, band and break date
  validation_overlay.png       TP / FP / FN against the reference + metrics
  timelapse.gif                scenes in sequence, detected changes appearing

Large inputs are read through numpy memmaps in row strips or with strided
sampling, so a 10980 x 10980 x 72 cube is never loaded whole. Break maps are
downsampled with block priority (any break pixel in a block wins). A figure
whose inputs are missing is skipped with a printed reason.
"""

import argparse
import math
import os
import sys
import textwrap
import warnings

import numpy as np

import cdio

import matplotlib  # isort: skip

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

# Class codes of the block-priority break map (non-negative = date index of
# the earliest loss break in the block).
CLS_STABLE = -1
CLS_INSUFFICIENT = -2
CLS_GAIN = -3

# Break-year colours, early to late (up to seven classes); gains are green.
RAMP = ["#ffd400", "#ff9100", "#ff4a1c", "#d4002a", "#b00078", "#6a2bd0",
        "#1f6fff"]
GAIN_COLOR = "#22c55e"
HIGHLIGHT = "#00e5ff"
TP_COLOR = "#2ca02c"
FP_COLOR = "#e6194b"
FN_COLOR = "#f58231"

MAX_PNG_BYTES = 2 * 1000 * 1000
MAX_GIF_BYTES = 5 * 1000 * 1000

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
})


class Report:
    """Collects produced files and skip reasons for the final summary."""

    def __init__(self):
        self.produced = []
        self.skipped = []
        self.failed = []

    def skip(self, name, reason):
        print("SKIP %s: %s" % (name, reason))
        self.skipped.append(name)

    def done(self, path):
        self.produced.append(path)
        print("wrote %s (%.0f KB)" % (path, os.path.getsize(path) / 1000.0))


# ------------------------------------------------------------------ helpers

def parse_int_list(text, count=None, name="value"):
    try:
        values = [int(v) for v in text.split(",") if v.strip() != ""]
    except ValueError:
        raise SystemExit("error: --%s expects integers, got '%s'" % (
            name, text))
    if count is not None and len(values) != count:
        raise SystemExit("error: --%s expects %d integers" % (name, count))
    return values


def choose_factor(height, width, max_dim):
    """Smallest integer factor so the long side is <= max_dim."""
    return max(1, int(math.ceil(max(height, width) / float(max_dim))))


def strip_rows(width, factor):
    """Rows per strip (multiple of factor) keeping strips near 4 M pixels."""
    rows = max(factor, (1 << 22) // max(1, width))
    return max(factor, (rows // factor) * factor)


def reduce_blocks(arr, factor, func, pad):
    """Reduces factor x factor blocks of a 2-D array with func over axes."""
    ph = (-arr.shape[0]) % factor
    pw = (-arr.shape[1]) % factor
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw)), constant_values=pad)
    h, w = arr.shape
    return func(arr.reshape(h // factor, factor, w // factor, factor),
                axis=(1, 3))


def block_classes(result, factor):
    """Block-priority class map of the detect result.

    Priority inside a block: loss break (earliest date index wins) > gain >
    stable > insufficient data, so a small clearing never disappears.
    """
    height, width = result.height, result.width
    out_h, out_w = -(-height // factor), -(-width // factor)
    out = np.full((out_h, out_w), CLS_INSUFFICIENT, dtype=np.int16)
    big = np.int16(32767)
    step = strip_rows(width, factor)
    for r0 in range(0, height, step):
        r1 = min(height, r0 + step)
        bi = np.asarray(result.break_index[r0:r1])
        st = np.asarray(result.status[r0:r1])
        if result.magnitude is not None:
            loss_sign = np.asarray(result.magnitude[r0:r1]) < 0
        else:
            loss_sign = np.ones(st.shape, dtype=bool)
        is_break = st == cdio.STATUS_BREAK
        loss = is_break & loss_sign
        loss_min = reduce_blocks(np.where(loss, bi, big).astype(np.int16),
                                 factor, np.min, big)
        any_gain = reduce_blocks(is_break & ~loss_sign, factor, np.any, False)
        any_stable = reduce_blocks(st == cdio.STATUS_STABLE, factor, np.any,
                                   False)
        cls = np.full(loss_min.shape, CLS_INSUFFICIENT, dtype=np.int16)
        cls[any_stable] = CLS_STABLE
        cls[any_gain] = CLS_GAIN
        has_loss = loss_min != big
        cls[has_loss] = loss_min[has_loss]
        out[r0 // factor:r0 // factor + cls.shape[0]] = cls
    return out


def scene_counts(result):
    """Full-resolution pixel counts from the maps, read in strips."""
    counts = {"stable": 0, "break": 0, "insufficient": 0, "loss": 0,
              "gain": 0}
    per_index = np.zeros(result.num_dates, dtype=np.int64)
    step = strip_rows(result.width, 1)
    for r0 in range(0, result.height, step):
        r1 = min(result.height, r0 + step)
        st = np.asarray(result.status[r0:r1])
        counts["stable"] += int(np.count_nonzero(st == cdio.STATUS_STABLE))
        counts["insufficient"] += int(
            np.count_nonzero(st == cdio.STATUS_INSUFFICIENT))
        is_break = st == cdio.STATUS_BREAK
        counts["break"] += int(np.count_nonzero(is_break))
        if result.magnitude is not None:
            loss = is_break & (np.asarray(result.magnitude[r0:r1]) < 0)
        else:
            loss = is_break
        counts["loss"] += int(np.count_nonzero(loss))
        counts["gain"] += int(np.count_nonzero(is_break & ~loss))
        bi = np.asarray(result.break_index[r0:r1])[loss]
        bi = bi[(bi >= 0) & (bi < result.num_dates)]
        per_index += np.bincount(bi, minlength=result.num_dates)
    counts["loss_per_index"] = per_index
    return counts


def densest_loss_crop(result, side_h, side_w):
    """Top-left corner (r0, c0) of the side_h x side_w window with the most
    loss-break pixels (coarse grid search, strip by strip)."""
    height, width = result.height, result.width
    grid = max(1, min(side_h, side_w) // 4)
    gh, gw = -(-height // grid), -(-width // grid)
    counts = np.zeros((gh, gw), dtype=np.int64)
    step = strip_rows(width, grid)
    for r0 in range(0, height, step):
        r1 = min(height, r0 + step)
        st = np.asarray(result.status[r0:r1]) == cdio.STATUS_BREAK
        if result.magnitude is not None:
            st &= np.asarray(result.magnitude[r0:r1]) < 0
        block = reduce_blocks(st.astype(np.int32), grid, np.sum, 0)
        counts[r0 // grid:r0 // grid + block.shape[0]] = block
    wh, ww = max(1, side_h // grid), max(1, side_w // grid)
    wh, ww = min(wh, gh), min(ww, gw)
    cum = np.pad(counts.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    window = (cum[wh:, ww:] - cum[:-wh, ww:] - cum[wh:, :-ww]
              + cum[:-wh, :-ww])
    if window.size == 0 or window.max() <= 0:
        return (height - side_h) // 2, (width - side_w) // 2, 0
    gr, gc = np.unravel_index(int(np.argmax(window)), window.shape)
    r0 = int(np.clip(gr * grid, 0, height - side_h))
    c0 = int(np.clip(gc * grid, 0, width - side_w))
    return r0, c0, int(window.max())


def save_fig(fig, path, max_bytes=MAX_PNG_BYTES, dpi=130, min_dpi=0):
    """Saves a figure, lowering the dpi until the file is <= max_bytes.

    min_dpi stops the reduction (used by maps whose one-array-pixel-per-
    figure-pixel resolution must be kept so small clearings stay visible).
    """
    current = dpi
    for _ in range(8):
        fig.savefig(path, dpi=current, bbox_inches="tight",
                    pil_kwargs={"optimize": True})
        if os.path.getsize(path) <= max_bytes:
            break
        if int(current * 0.8) < min_dpi:
            print("warning: %s is %.1f MB at the minimum dpi %d" % (
                path, os.path.getsize(path) / 1e6, current))
            break
        current = int(current * 0.8)
    plt.close(fig)


def map_dpi(fig, ax, shape, base=130, limit=400):
    """dpi at which the axes show at least one figure pixel per array pixel,
    so single break pixels of a downsampled map are never lost to resampling.
    """
    fig.canvas.draw()
    box = ax.get_window_extent()
    inches_w, inches_h = box.width / fig.dpi, box.height / fig.dpi
    need = max(shape[1] / max(inches_w, 1e-3), shape[0] / max(inches_h, 1e-3))
    return int(min(limit, max(base, math.ceil(need))))


def add_scale_bar(ax, shape, pixel_m, factor, max_fraction=0.22):
    """Draws a km scale bar (lower left) in displayed-array pixel units."""
    if pixel_m is None:
        return
    height, width = shape[:2]
    metres_per_px = pixel_m * factor
    length_m = cdio.nice_length(max_fraction * width * metres_per_px)
    if length_m <= 0:
        return
    length_px = length_m / metres_per_px
    label = ("%g km" % (length_m / 1000.0) if length_m >= 1000
             else "%g m" % length_m)
    x0, y0 = 0.04 * width, 0.94 * height
    pad = 0.012 * height
    ax.add_patch(Rectangle((x0 - 2 * pad, y0 - 5.2 * pad), length_px + 4 * pad,
                           7.2 * pad, facecolor="white", alpha=0.75,
                           edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((x0, y0 - 0.8 * pad), length_px, 1.6 * pad,
                           facecolor="black", edgecolor="none", zorder=6))
    ax.text(x0 + length_px / 2, y0 - 1.8 * pad, label, ha="center",
            va="bottom", fontsize=9, fontweight="bold", zorder=7)


def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def wrap(text, width=110):
    return "\n".join(textwrap.wrap(text, width))


def grey_to_rgb(grey):
    """Lightens a 0..1 grey image so coloured overlays stand out."""
    lifted = 0.28 + 0.62 * np.clip(grey, 0.0, 1.0)
    return np.repeat(lifted[..., None], 3, axis=2)


def strided_ndvi(cube, index, factor, valid_classes):
    return cube.ndvi_plane(index, slice(0, cube.height, factor),
                           slice(0, cube.width, factor), valid_classes)


def base_preview(cube):
    """Index and path of the clearest first-year preview PNG (highest
    manifest valid_fraction, earliest on ties), or (None, None)."""
    if cube is None:
        return None, None
    first_day = cube.manifest.dates[0].day
    best = None
    for i, entry in enumerate(cube.manifest.dates):
        path = cube.preview_path(i)
        if not path or entry.day - first_day >= 365.25:
            continue
        if best is None or entry.valid_fraction > best[0]:
            best = (entry.valid_fraction, i, path)
    if best is None:
        for i in range(cube.num_dates):
            path = cube.preview_path(i)
            if path:
                return i, path
        return None, None
    return best[1], best[2]


def grey_base(cube, result, factor, valid_classes):
    """Greyscale base (0..1) of the displayed shape and a description."""
    out_h, out_w = -(-result.height // factor), -(-result.width // factor)
    index, path = base_preview(cube)
    if path:
        with Image.open(path) as img:
            small = img.convert("L").resize((out_w, out_h), Image.BILINEAR)
        return np.asarray(small, dtype=np.float32) / 255.0, \
            "greyscale of the %s true-colour preview" % cube.manifest.dates[
                index].date
    if cube is not None:
        first_day = cube.manifest.dates[0].day
        picks = [i for i, d in enumerate(cube.manifest.dates)
                 if d.day - first_day < 365.25]
        picks = [picks[int(k)] for k in
                 np.linspace(0, len(picks) - 1, min(6, len(picks)))]
        stack = np.stack([strided_ndvi(cube, i, factor, valid_classes)
                          for i in sorted(set(picks))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(stack, axis=0)
        mean = np.where(np.isnan(mean), np.nanmedian(mean) if np.isfinite(
            mean).any() else 0.3, mean)
        lo, hi = np.percentile(mean, (2, 98))
        return np.clip((mean - lo) / max(hi - lo, 1e-3), 0, 1).astype(
            np.float32), \
            "mean NDVI of the first year (grey)"
    return np.full((out_h, out_w), 0.8, dtype=np.float32), "no base image"


def year_classes(present_indices, date_strings, max_classes=7):
    """Maps break date indices to colour classes.

    One class per calendar year when there are at most max_classes distinct
    years, else equal-width date bins. Returns (lookup, labels) where lookup
    is an int array indexed by date index (-1 = not present).
    """
    lookup = np.full(len(date_strings), -1, dtype=np.int32)
    present = sorted(int(i) for i in present_indices)
    if not present:
        return lookup, []
    years = sorted({int(date_strings[i][:4]) for i in present})
    if len(years) <= max_classes:
        for i in present:
            lookup[i] = years.index(int(date_strings[i][:4]))
        return lookup, [str(y) for y in years]
    edges = np.linspace(present[0], present[-1] + 1, max_classes + 1)
    labels = []
    for k in range(max_classes):
        members = [i for i in present if edges[k] <= i < edges[k + 1]]
        for i in members:
            lookup[i] = k
        lo, hi = int(np.ceil(edges[k])), int(np.ceil(edges[k + 1])) - 1
        lo, hi = min(lo, present[-1]), min(max(hi, lo), present[-1])
        labels.append("%s to %s" % (date_strings[lo][:7],
                                    date_strings[hi][:7]))
    return lookup, labels


def class_colors(count):
    """Picks count colours spread over the ramp (early to late)."""
    if count <= 1:
        return [RAMP[2]] * count
    picks = np.round(np.linspace(0, len(RAMP) - 1, count)).astype(int)
    return [RAMP[i] for i in picks]


def stripe_pattern(shape, display_px=1000, stripe_px=4):
    """Diagonal stripe mask that appears ~stripe_px wide when displayed."""
    scale = display_px / float(max(shape))
    period = max(1, int(round(stripe_px / scale)))
    rows, cols = np.indices(shape)
    return ((rows + cols) // period) % 2 == 0


def hex_to_rgb(color):
    color = color.lstrip("#")
    return np.array([int(color[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


# ------------------------------------------------------------ figure: year map

def fig_change_year_map(ctx):
    result, cube, out = ctx["result"], ctx["cube"], ctx["out"]
    factor = ctx["factor"]
    dates = ctx["dates"]
    cls = block_classes(result, factor)
    base, base_text = grey_base(cube, result, factor, ctx["valid_classes"])
    rgb = (grey_to_rgb(base) * 255).astype(np.uint8)
    lookup, labels = year_classes(np.unique(cls[cls >= 0]), dates)
    colors = [hex_to_rgb(c) for c in class_colors(len(labels))]
    for k, color in enumerate(colors):
        mask = (cls >= 0) & (lookup[np.clip(cls, 0, None)] == k)
        rgb[mask] = color
    gain = cls == CLS_GAIN
    rgb[gain] = hex_to_rgb(GAIN_COLOR)
    insufficient = cls == CLS_INSUFFICIENT
    stripes = stripe_pattern(cls.shape)
    rgb[insufficient & stripes] = (110, 110, 110)
    rgb[insufficient & ~stripes] = (205, 205, 205)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.imshow(rgb, interpolation="nearest")
    hide_ticks(ax)
    add_scale_bar(ax, cls.shape, ctx["pixel_m"], factor)
    handles = [Patch(facecolor=c / 255.0, label=l)
               for c, l in zip(colors, labels)]
    if gain.any():
        handles.append(Patch(facecolor=GAIN_COLOR, label="greenness gain"))
    handles.append(Patch(facecolor="#d0d0d0", edgecolor="#6e6e6e",
                         hatch="////", label="insufficient data"))
    handles.append(Patch(facecolor="#9a9a9a", label="stable (base image)"))
    ax.legend(handles=handles, title="Break year (loss)",
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    counts = ctx["counts"]
    total = result.height * result.width
    ax.set_title("Detected forest-loss break year - %s" % ctx["name"])
    caption = (
        "%s break pixels (%.1f%%: %s loss, %s gain), %s stable, %s "
        "insufficient data. Image %d x %d px, shown downsampled %d x (any "
        "break pixel in a block wins). Base: %s." % (
            "{:,}".format(counts["break"]), 100.0 * counts["break"] / total,
            "{:,}".format(counts["loss"]), "{:,}".format(counts["gain"]),
            "{:,}".format(counts["stable"]),
            "{:,}".format(counts["insufficient"]), result.height, result.width,
            factor, base_text))
    ax.set_xlabel(wrap(caption, 100), fontsize=8, loc="left")
    path = os.path.join(out, "change_year_map.png")
    dpi = map_dpi(fig, ax, cls.shape)
    save_fig(fig, path, dpi=dpi, min_dpi=dpi)
    ctx["report"].done(path)


# ------------------------------------------------ figures: before/after crops

def false_colour_crop(cube, index, r0, c0, h, w):
    """NIR-red-red false colour (R=NIR, G=red, B=red) of a crop, uint8."""
    rows, cols = slice(r0, r0 + h), slice(c0, c0 + w)
    red = cube.reflectance(np.asarray(cube.red[index][rows, cols]), index)
    nir = cube.reflectance(np.asarray(cube.nir[index][rows, cols]), index)

    def stretch(band):
        lo, hi = np.percentile(band, (2, 98))
        return np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)

    img = np.stack([stretch(nir), stretch(red), stretch(red)], axis=-1)
    return (img * 255).astype(np.uint8)


def crop_image(cube, index, r0, c0, h, w):
    """RGB crop (uint8) of a date and a short description, or (None, why)."""
    path = cube.preview_path(index, full=True) if cube else None
    if path:
        with Image.open(path) as img:
            img = img.convert("RGB")
            sx, sy = img.width / float(cube.width), img.height / float(
                cube.height)
            box = (int(c0 * sx), int(r0 * sy),
                   int((c0 + w) * sx), int((r0 + h) * sy))
            return np.asarray(img.crop(box).resize((w, h), Image.BILINEAR)), \
                "true colour (TCI preview)"
    if cube is not None:
        return false_colour_crop(cube, index, r0, c0, h, w), \
            "false colour (R=NIR, G=red, B=red)"
    return None, "no cube available"


def outline(ax, mask, color=HIGHLIGHT):
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color],
                   linewidths=1.2)


def fig_before_after(ctx):
    result, cube, out, dates = (ctx["result"], ctx["cube"], ctx["out"],
                                ctx["dates"])
    report = ctx["report"]
    if cube is None:
        report.skip("before_after", "no --cube (and none found via the "
                    "result manifest), cannot read imagery")
        return
    i_before, i_after = ctx["before_idx"], ctx["after_idx"]
    r0, c0, h, w = ctx["crop"]
    img_b, kind_b = crop_image(cube, i_before, r0, c0, h, w)
    img_a, kind_a = crop_image(cube, i_after, r0, c0, h, w)
    bi = np.asarray(result.break_index[r0:r0 + h, c0:c0 + w])
    st = np.asarray(result.status[r0:r0 + h, c0:c0 + w])
    mg = (np.asarray(result.magnitude[r0:r0 + h, c0:c0 + w])
          if result.magnitude is not None else np.full(st.shape, -1.0))
    changed = ((st == cdio.STATUS_BREAK) & (mg < 0) & (bi > i_before)
               & (bi <= i_after))
    where = "rows %d-%d, cols %d-%d" % (r0, r0 + h - 1, c0, c0 + w - 1)

    for tag, idx, img, kind in (("before", i_before, img_b, kind_b),
                                ("after", i_after, img_a, kind_a)):
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.imshow(img, interpolation="nearest")
        hide_ticks(ax)
        add_scale_bar(ax, img.shape, ctx["pixel_m"], 1)
        ax.set_title("%s: %s, %s" % (tag.capitalize(), dates[idx], kind))
        ax.set_xlabel("Crop of %s (%d x %d px, full resolution)." % (
            where, w, h), fontsize=8, loc="left")
        path = os.path.join(out, "%s_%s.png" % (tag, dates[idx]))
        save_fig(fig, path)
        report.done(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.1,
                        wspace=0.05)
    panels = ((axes[0], "Before", i_before, img_b, kind_b),
              (axes[1], "After", i_after, img_a, kind_a))
    for ax, tag, idx, img, kind in panels:
        ax.imshow(img, interpolation="nearest")
        outline(ax, changed)
        hide_ticks(ax)
        ax.set_title("%s: %s\n%s" % (tag, dates[idx], kind), fontsize=11)
    add_scale_bar(axes[0], img_b.shape, ctx["pixel_m"], 1)
    axes[1].legend(handles=[Line2D([0], [0], color=HIGHLIGHT, lw=2,
                   label="pixels flagged as loss (%s px)" % "{:,}".format(
                       int(changed.sum())))],
                   loc="lower right", framealpha=0.9, fontsize=9)
    fig.suptitle("Before / after crop with detected change outlined",
                 fontsize=13, fontweight="bold", y=0.97)
    fig.text(0.5, 0.09, wrap("Same area at two dates (%s, %d x %d px, full "
                             "resolution). Outlines mark pixels whose break "
                             "date lies between %s and %s." % (
                                 where, w, h, dates[i_before], dates[i_after]),
                             140), ha="center", va="top", fontsize=8)
    path = os.path.join(out, "before_after_crop.png")
    save_fig(fig, path)
    report.done(path)


# --------------------------------------------------------------- figure: NDVI

def fig_ndvi(ctx):
    cube, out, dates = ctx["cube"], ctx["out"], ctx["dates"]
    report = ctx["report"]
    if cube is None:
        report.skip("ndvi_<date>", "no cube available")
        return
    factor = ctx["factor"]
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#b8b8b8")
    for idx in ctx["ndvi_idx"]:
        plane = strided_ndvi(cube, idx, factor, ctx["valid_classes"])
        masked = np.isnan(plane).mean()
        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        im = ax.imshow(np.ma.masked_invalid(plane), cmap=cmap, vmin=-0.2,
                       vmax=0.95, interpolation="nearest")
        hide_ticks(ax)
        add_scale_bar(ax, plane.shape, ctx["pixel_m"], factor)
        bar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        bar.set_label("NDVI (-0.2 water/bare ... 0.95 dense vegetation)")
        ax.set_title("NDVI on %s - %s" % (dates[idx], ctx["name"]))
        ax.set_xlabel(wrap("Grey = masked (cloud, shadow, nodata; %.1f%% of "
                           "the sampled pixels). Image %d x %d px, shown "
                           "strided %d x." % (100 * masked, cube.height,
                                              cube.width, factor), 95),
                      fontsize=8, loc="left")
        path = os.path.join(out, "ndvi_%s.png" % dates[idx])
        save_fig(fig, path)
        report.done(path)


# ------------------------------------------------------- figure: trajectories

def pick_trajectory_pixels(groups, status_of):
    """Chooses 4-6 pixel keys: up to 3 breaks, 2 stable, fill the rest."""
    kinds = {"break": [], "stable": [], "insufficient": []}
    for key in groups:
        kinds[status_of(key)].append(key)

    def spread(items, n):
        if len(items) <= n:
            return list(items)
        return [items[int(i)] for i in np.linspace(0, len(items) - 1, n)]

    chosen = spread(kinds["break"], 3) + spread(kinds["stable"], 2)
    rest = [k for k in kinds["break"] + kinds["stable"] if k not in chosen]
    chosen += rest[:max(0, 6 - len(chosen))]
    if len(chosen) < 4:
        chosen += kinds["insufficient"][:4 - len(chosen)]
    return chosen[:6]


def fig_trajectories(ctx):
    result, out, report = ctx["result"], ctx["out"], ctx["report"]
    path_csv = result.path("trajectories.csv")
    if not os.path.exists(path_csv):
        report.skip("ndvi_trajectory_examples", "trajectories.csv not found")
        return
    rows = cdio.load_csv(path_csv)
    if not rows:
        report.skip("ndvi_trajectory_examples", "trajectories.csv is empty")
        return
    groups = {}
    for row in rows:
        groups.setdefault((row["row"], row["col"]), []).append(row)
    names = {0: "stable", 1: "break", 2: "insufficient"}

    def status_of(key):
        return names.get(int(result.status[key[0], key[1]]), "stable")

    keys = pick_trajectory_pixels(groups, status_of)
    threshold = ctx["threshold"]
    ncols = 3 if len(keys) > 4 else 2
    nrows = int(math.ceil(len(keys) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.5 * nrows),
                             squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax, key in zip(axes.ravel(), keys):
        series = sorted(groups[key], key=lambda r: r["date_index"])
        day = [np.datetime64(r["date"]) for r in series]
        dnum = mdates.date2num(day)
        ndvi = np.array([np.nan if r["ndvi"] is None else r["ndvi"]
                         for r in series])
        valid = np.array([bool(r["valid"]) for r in series])
        train = np.array([bool(r["in_training"]) for r in series])
        pred = np.array([np.nan if r["predicted"] is None else r["predicted"]
                         for r in series])
        run = np.array([bool(r["is_break_run"]) for r in series])
        status = status_of(key)
        rmse = float(result.rmse[key[0], key[1]]) if result.rmse is not None \
            else np.nan
        if train.any():
            ax.axvspan(dnum[train][0], dnum[train][-1], color="#9ecae1",
                       alpha=0.35, lw=0, label="training window")
        have_fit = np.isfinite(pred).any()
        if have_fit and np.isfinite(rmse):
            ax.fill_between(dnum, pred - threshold * rmse,
                            pred + threshold * rmse, color="#bdbdbd",
                            alpha=0.45, lw=0,
                            label="prediction +/- %.3g x RMSE" % threshold)
        if have_fit:
            ax.plot(dnum[train], pred[train], color="#08519c", lw=1.8,
                    label="fitted model")
            ax.plot(dnum[~train], pred[~train], color="#08519c", lw=1.6,
                    ls="--", label="model prediction")
        ok = valid & np.isfinite(ndvi)
        ax.scatter(dnum[ok & ~run], ndvi[ok & ~run], s=22, color="#222222",
                   zorder=4, label="observed NDVI (valid)")
        if (ok & run).any():
            ax.scatter(dnum[ok & run], ndvi[ok & run], s=34, color="#d62728",
                       zorder=5, label="exceedance run")
        title = "pixel row %d, col %d: %s" % (key[0], key[1], status)
        bidx = int(result.break_index[key[0], key[1]])
        if status == "break" and 0 <= bidx < len(series):
            ax.axvline(dnum[bidx], color="#d62728", lw=1.6)
            ax.text(dnum[bidx], 0.03, " break\n %s" % series[bidx]["date"],
                    transform=ax.get_xaxis_transform(), color="#d62728",
                    fontsize=8, va="bottom", ha="left",
                    bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1))
            if result.magnitude is not None:
                title += " (mean residual %.2f)" % float(
                    result.magnitude[key[0], key[1]])
        elif status == "insufficient":
            title += " (too few valid training observations)"
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("NDVI")
        # One label per year (January); July gets an unlabelled minor tick.
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.3, 1.0)
    labels = {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            labels.setdefault(l, h)
    fig.legend(list(labels.values()), list(labels.keys()), loc="lower center",
               ncol=min(4, len(labels)), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-pixel NDVI time series: observations, seasonal model "
                 "and detected break", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = os.path.join(out, "ndvi_trajectory_examples.png")
    save_fig(fig, path)
    report.done(path)


# ------------------------------------------------------- figure: validation

def validation_classes(result, reference, factor, first_year, last_year,
                       years):
    """Block-priority TP/FN/FP map plus exact counts.

    Codes: 0 none, 1 TP, 2 FN, 3 FP (priority TP > FN > FP in a block).
    """
    height, width = result.height, result.width
    out_h, out_w = -(-height // factor), -(-width // factor)
    out = np.zeros((out_h, out_w), dtype=np.uint8)
    counts = {"tp": 0, "fp": 0, "fn": 0}
    step = strip_rows(width, factor)
    for r0 in range(0, height, step):
        r1 = min(height, r0 + step)
        ref = np.asarray(reference[r0:r1]).astype(np.int32)
        bi = np.asarray(result.break_index[r0:r1])
        st = np.asarray(result.status[r0:r1])
        mg = (np.asarray(result.magnitude[r0:r1])
              if result.magnitude is not None else np.full(st.shape, -1.0))
        excluded = ref == 255
        pos_ref = ((ref != 0) & ~excluded & (2000 + ref >= first_year)
                   & (2000 + ref <= last_year))
        year = years[np.clip(bi, 0, len(years) - 1)]
        pos_det = ((st == cdio.STATUS_BREAK) & (mg < 0) & (bi >= 0)
                   & (year >= first_year) & (year <= last_year))
        tp = pos_ref & pos_det & ~excluded
        fp = ~pos_ref & pos_det & ~excluded
        fn = pos_ref & ~pos_det & ~excluded
        counts["tp"] += int(tp.sum())
        counts["fp"] += int(fp.sum())
        counts["fn"] += int(fn.sum())
        cls = np.zeros(reduce_blocks(tp, factor, np.any, False).shape,
                       dtype=np.uint8)
        cls[reduce_blocks(fp, factor, np.any, False)] = 3
        cls[reduce_blocks(fn, factor, np.any, False)] = 2
        cls[reduce_blocks(tp, factor, np.any, False)] = 1
        out[r0 // factor:r0 // factor + cls.shape[0]] = cls
    return out, counts


def fig_validation(ctx):
    result, cube, out, report = (ctx["result"], ctx["cube"], ctx["out"],
                                 ctx["report"])
    reference = cube.reference() if cube is not None else None
    if reference is None:
        report.skip("validation_overlay", "no reference raster (cube has no "
                    "reference_file)")
        return
    dates = ctx["dates"]
    years = np.array([int(d[:4]) for d in dates])
    metrics = {}
    if result.exists("validation.csv"):
        raw = cdio.load_key_value_csv(result.path("validation.csv"))
        metrics = {k: float(v) for k, v in raw.items()}
    first_year = int(metrics.get("first_year", years.min()))
    last_year = int(metrics.get("last_year", years.max()))
    factor = ctx["factor"]
    cls, counts = validation_classes(result, reference, factor, first_year,
                                     last_year, years)
    note = ""
    if metrics:
        if any(int(metrics.get(k, -1)) != counts[k] for k in counts):
            note = " (recomputed counts differ from validation.csv)"
            print("WARNING: tp/fp/fn recomputed from the maps differ from "
                  "validation.csv: %s vs %s" % (counts, {
                      k: metrics.get(k) for k in counts}))
    else:
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        metrics = {"tp": tp, "fp": fp, "fn": fn}
        metrics["precision"] = tp / float(max(1, tp + fp))
        metrics["recall"] = tp / float(max(1, tp + fn))
        p, r = metrics["precision"], metrics["recall"]
        metrics["f1"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
        note = " (computed here: validation.csv missing)"
    base, _ = grey_base(cube, result, factor, ctx["valid_classes"])
    rgb = (grey_to_rgb(base) * 255).astype(np.uint8)
    for code, color in ((3, FP_COLOR), (2, FN_COLOR), (1, TP_COLOR)):
        rgb[cls == code] = hex_to_rgb(color)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.imshow(rgb, interpolation="nearest")
    hide_ticks(ax)
    add_scale_bar(ax, cls.shape, ctx["pixel_m"], factor)
    ax.legend(handles=[
        Patch(facecolor=TP_COLOR, label="TP: detected loss, in reference"),
        Patch(facecolor=FP_COLOR, label="FP: detected loss, not in reference"),
        Patch(facecolor=FN_COLOR, label="FN: reference loss, missed")],
        loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
        title="Detection vs reference")
    lines = ["Precision  %.3f" % metrics.get("precision", float("nan")),
             "Recall     %.3f" % metrics.get("recall", float("nan")),
             "F1         %.3f" % metrics.get("f1", float("nan"))]
    if "year_exact" in metrics:
        lines.append("Year exact %.1f%%" % (100 * metrics["year_exact"]))
    if "year_within1" in metrics:
        lines.append("Year +/-1  %.1f%%" % (100 * metrics["year_within1"]))
    lines.append("TP %s  FP %s  FN %s" % tuple(
        "{:,}".format(int(metrics.get(k, counts[k]))) for k in
        ("tp", "fp", "fn")))
    lines.append("Years %d-%d" % (first_year, last_year))
    ax.text(1.02, 0.45, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, family="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#888"))
    ax.set_title("Validation against reference forest loss - %s" % ctx["name"])
    ref_text = cube.manifest.get("reference_source", "reference raster")
    ax.set_xlabel(wrap(
        "Reference: %s (30 m, resampled to 10 m, so a 3 x 3 pixel blockiness "
        "is expected). Shown downsampled %d x (any flagged pixel in a block "
        "wins; priority TP > FN > FP)%s." % (ref_text, factor, note), 100),
        fontsize=8, loc="left")
    path = os.path.join(out, "validation_overlay.png")
    dpi = map_dpi(fig, ax, cls.shape)
    save_fig(fig, path, dpi=dpi, min_dpi=dpi)
    report.done(path)


# ----------------------------------------------------------- figure: timelapse

def font_for(size, bold=True):
    try:
        from matplotlib import font_manager
        name = "DejaVu Sans:bold" if bold else "DejaVu Sans"
        return ImageFont.truetype(font_manager.findfont(name), size)
    except (OSError, ImportError):
        return ImageFont.load_default()


def build_gif(frames_rgb, path, duration_ms):
    """Writes a GIF with per-frame adaptive palettes (optimised)."""
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96,
                     dither=Image.NONE) for f in frames_rgb]
    durations = [duration_ms] * len(pal)
    durations[-1] = 1800
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations,
                loop=0, optimize=True, disposal=1)


def fig_timelapse(ctx):
    result, cube, out, dates = (ctx["result"], ctx["cube"], ctx["out"],
                                ctx["dates"])
    report = ctx["report"]
    if cube is None:
        report.skip("timelapse", "no cube available (needs previews or raw "
                    "arrays)")
        return
    use_preview = all(cube.preview_path(i) for i in range(cube.num_dates))
    target = 480
    factor = choose_factor(cube.height, cube.width, target)
    cls = block_classes(result, factor)
    out_h, out_w = cls.shape
    cum = np.cumsum(ctx["counts"]["loss_per_index"])
    cmap = plt.get_cmap("RdYlGn")
    label_font = font_for(max(14, out_h // 18))
    small_font = font_for(max(10, out_h // 30), bold=False)
    indices = list(range(cube.num_dates))
    scale = 1.0
    attempt = 0
    path = os.path.join(out, "timelapse.gif")
    while True:
        frame_size = (max(2, int(out_w * scale)), max(2, int(out_h * scale)))
        frames = []
        for i in indices:
            if use_preview:
                with Image.open(cube.preview_path(i)) as img:
                    base = img.convert("RGB").resize(frame_size,
                                                     Image.BILINEAR)
                arr = np.asarray(base).copy()
            else:
                plane = strided_ndvi(cube, i, factor, ctx["valid_classes"])
                col = (cmap(np.clip((plane + 0.2) / 1.15, 0, 1))[..., :3]
                       * 255).astype(np.uint8)
                col[np.isnan(plane)] = (170, 170, 170)
                arr = np.asarray(Image.fromarray(col).resize(
                    frame_size, Image.NEAREST)).copy()
            mask = Image.fromarray(((cls >= 0) & (cls <= i)).astype(np.uint8)
                                   * 255).resize(frame_size, Image.NEAREST)
            m = np.asarray(mask) > 0
            arr[m] = (0.15 * arr[m] + 0.85 * np.array([255, 0, 90])).astype(
                np.uint8)
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, frame_size[0], int(label_font.size * 2.5)],
                           fill=(0, 0, 0))
            draw.text((6, 4), dates[i], fill=(255, 255, 255), font=label_font)
            draw.text((6, 4 + int(label_font.size * 1.25)),
                      "detected loss so far: %s px (magenta)" % "{:,}".format(
                          int(cum[i])), fill=(255, 190, 220), font=small_font)
            frames.append(img)
        build_gif(frames, path, 450)
        size = os.path.getsize(path)
        if size <= MAX_GIF_BYTES or attempt >= 8:
            break
        attempt += 1
        if attempt % 2 == 1 and len(indices) > 12:
            indices = indices[::2] if indices[-1] in indices[::2] \
                else indices[::2] + [indices[-1]]
        else:
            scale *= 0.8
    if size > MAX_GIF_BYTES:
        report.failed.append("timelapse.gif larger than 5 MB")
    report.done(path)


# ------------------------------------------------------------------- main

def build_context(args, report):
    result = cdio.open_result(args.result)
    if result.break_index is None or result.status is None:
        raise SystemExit("error: %s lacks break_index.i16 / status.u8" %
                         args.result)
    if args.cube and not os.path.exists(args.cube):
        raise SystemExit("error: --cube %s not found" % args.cube)
    cube_path = result.cube_manifest_path(args.cube)
    cube = None
    if cube_path:
        cube = cdio.open_cube(cube_path)
        if (cube.height, cube.width) != (result.height, result.width):
            raise SystemExit("error: cube is %dx%d but result is %dx%d" % (
                cube.height, cube.width, result.height, result.width))
    else:
        print("note: no cube manifest found; figures needing imagery are "
              "skipped")
    entries = result.manifest.dates or (cube.manifest.dates if cube else [])
    dates = [d.date for d in entries]
    if len(dates) != result.num_dates:
        raise SystemExit("error: no date table for %d dates" %
                         result.num_dates)
    valid_classes = cdio.DEFAULT_VALID_CLASSES
    raw = result.manifest.get("valid_classes")
    if raw:
        try:
            valid_classes = tuple(int(v) for v in raw.split(","))
        except ValueError:
            pass
    threshold = result.manifest.get_float("threshold", 3.0)
    if result.manifest.get("threshold") is None:
        print("note: threshold not in result_manifest.txt; using 3.0 for the "
              "trajectory band")
    side = min(256, result.height, result.width)
    if args.crop:
        r0, c0, h, w = parse_int_list(args.crop, 4, "crop")
        h, w = min(h, result.height), min(w, result.width)
        r0 = int(np.clip(r0, 0, result.height - h))
        c0 = int(np.clip(c0, 0, result.width - w))
        crop = (r0, c0, h, w)
    else:
        r0, c0, _ = densest_loss_crop(result, side, side)
        crop = (r0, c0, side, side)
    num = result.num_dates
    if args.dates:
        picked = parse_int_list(args.dates, name="dates")
        if not picked or any(not 0 <= i < num for i in picked):
            raise SystemExit("error: --dates out of range 0..%d" % (num - 1))
        before_idx, after_idx = picked[0], picked[-1]
        ndvi_idx = picked if len(picked) <= 3 else [
            picked[0], picked[len(picked) // 2], picked[-1]]
    else:
        before_idx, after_idx = 0, num - 1
        ndvi_idx = sorted({0, num // 2, num - 1})
    return {
        "result": result, "cube": cube, "out": args.out, "dates": dates,
        "name": result.manifest.get("name") or (
            cube.manifest.name if cube else os.path.basename(
                os.path.abspath(args.result))),
        "factor": choose_factor(result.height, result.width, args.max_dim),
        "pixel_m": (cube.manifest.pixel_size_m if cube else None),
        "valid_classes": valid_classes, "threshold": threshold,
        "crop": crop, "before_idx": before_idx, "after_idx": after_idx,
        "ndvi_idx": ndvi_idx, "report": report,
        "counts": scene_counts(result),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the change-detection figures from a detect "
                    "output directory (and optionally its input cube).")
    parser.add_argument("--result", required=True,
                        help="detect output directory (result_manifest.txt)")
    parser.add_argument("--cube", help="cube manifest.txt (default: the "
                        "cube_manifest recorded in the result manifest)")
    parser.add_argument("--out", required=True, help="figure output directory")
    parser.add_argument("--crop", help="r0,c0,h,w for before/after crops "
                        "(default: densest loss cluster, 256 x 256)")
    parser.add_argument("--dates", help="comma list of date indices: first "
                        "and last give the before/after dates, up to 3 are "
                        "used for NDVI maps (default 0 .. T-1)")
    parser.add_argument("--max-dim", type=int, default=2048,
                        help="long side of downsampled maps (default 2048)")
    args = parser.parse_args(argv)
    report = Report()
    ctx = build_context(args, report)
    os.makedirs(args.out, exist_ok=True)
    steps = (("change_year_map", fig_change_year_map),
             ("before_after", fig_before_after), ("ndvi", fig_ndvi),
             ("ndvi_trajectory_examples", fig_trajectories),
             ("validation_overlay", fig_validation),
             ("timelapse", fig_timelapse))
    for name, step in steps:
        try:
            step(ctx)
        except Exception as error:  # report and continue with other figures
            print("FAILED %s: %s: %s" % (name, type(error).__name__, error))
            report.failed.append(name)
    print("figures: %d written, %d skipped, %d failed" % (
        len(report.produced), len(report.skipped), len(report.failed)))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
