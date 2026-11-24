// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "postprocess.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "colormap.h"
#include "pixel_model.h"
#include "timing.h"

namespace cd {
namespace {

// Tile height of the quick-diff reader when RunConfig::tile_rows is 0.
constexpr int kDefaultTileRows = 256;

// True when every per-pixel map of `r` has height * width entries.
bool MapsAreConsistent(const DetectionResult& r) {
  const size_t pixels =
      static_cast<size_t>(r.height) * static_cast<size_t>(r.width);
  return r.height >= 0 && r.width >= 0 && r.break_index.size() == pixels &&
         r.magnitude.size() == pixels && r.status.size() == pixels;
}

}  // namespace

Status CpuColorizeChangeMap(const DetectionResult& r, const DateTable& d,
                            std::vector<uint8_t>* rgba, double* ms) {
  const WallTimer timer;
  if (rgba == nullptr) {
    return Status::Error("CpuColorizeChangeMap: null output");
  }
  if (!MapsAreConsistent(r)) {
    return Status::Error("CpuColorizeChangeMap: inconsistent result maps");
  }
  if (d.day.empty() || d.num_train_dates < 0 ||
      static_cast<size_t>(d.num_train_dates) >= d.day.size()) {
    return Status::Error(
        "CpuColorizeChangeMap: date table has no monitoring "
        "dates");
  }
  const int first_monitor_day = d.day[static_cast<size_t>(d.num_train_dates)];
  const int last_day = d.day.back();
  const size_t pixels = r.break_index.size();
  rgba->assign(pixels * 4, 0);
  for (size_t i = 0; i < pixels; ++i) {
    const int16_t break_index = r.break_index[i];
    // Pixels without a valid break date use the first monitoring day; the
    // colour rule ignores it for them.
    int break_day = first_monitor_day;
    if (break_index >= 0 && static_cast<size_t>(break_index) < d.day.size()) {
      break_day = d.day[static_cast<size_t>(break_index)];
    }
    ChangeColor(break_index, r.magnitude[i], r.status[i], first_monitor_day,
                last_day, break_day, rgba->data() + i * 4);
  }
  if (ms != nullptr) {
    *ms = timer.ElapsedMs();
  }
  return Status::Ok();
}

Status CpuCleanup(DetectionResult* r, double* ms) {
  const WallTimer timer;
  if (r == nullptr) {
    return Status::Error("CpuCleanup: null result");
  }
  if (!MapsAreConsistent(*r)) {
    return Status::Error("CpuCleanup: inconsistent result maps");
  }
  const int height = r->height;
  const int width = r->width;
  constexpr uint8_t kBreak = static_cast<uint8_t>(PixelStatus::kBreak);
  // The rule is evaluated on the unmodified map, so the break mask is taken
  // before any pixel is changed (removing one pixel must not influence the
  // neighbours' counts).
  std::vector<uint8_t> is_break(r->status.size());
  for (size_t i = 0; i < is_break.size(); ++i) {
    is_break[i] = r->status[i] == kBreak ? 1 : 0;
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t index = static_cast<size_t>(y) * width + x;
      if (is_break[index] == 0) {
        continue;
      }
      int neighbours = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int ny = y + dy;
          const int nx = x + dx;
          if ((dy == 0 && dx == 0) || ny < 0 || ny >= height || nx < 0 ||
              nx >= width) {
            continue;  // Out-of-image neighbours count as not-break.
          }
          neighbours += is_break[static_cast<size_t>(ny) * width + nx];
        }
      }
      if (neighbours < 2) {
        r->status[index] = static_cast<uint8_t>(PixelStatus::kStable);
        r->break_index[index] = -1;
        r->magnitude[index] = 0.0f;
      }
    }
  }
  if (ms != nullptr) {
    *ms = timer.ElapsedMs();
  }
  return Status::Ok();
}

Status CpuQuickDiff(CubeSource* source, int date_a, int date_b,
                    float diff_threshold, const ModelParams& p,
                    const RunConfig& c, QuickDiffResult* out, StageTimings* t) {
  if (source == nullptr || out == nullptr) {
    return Status::Error("CpuQuickDiff: null source or output");
  }
  const CubeInfo& info = source->info();
  if (date_a < 0 || date_a >= info.num_dates || date_b < 0 ||
      date_b >= info.num_dates) {
    return Status::Error("quick-diff dates " + std::to_string(date_a) +
                         " and " + std::to_string(date_b) + " must be in 0.." +
                         std::to_string(info.num_dates - 1));
  }
  const DateTable& dates = info.dates;
  if (dates.scale.size() != static_cast<size_t>(info.num_dates) ||
      dates.offset.size() != static_cast<size_t>(info.num_dates)) {
    return Status::Error(
        "quick-diff: calibration arrays do not match the "
        "number of dates");
  }
  const int height = info.height;
  const int width = info.width;
  int tile_rows = c.tile_rows > 0 ? c.tile_rows : kDefaultTileRows;
  tile_rows = std::max(1, std::min(tile_rows, height));

  const WallTimer total_timer;
  const size_t pixels =
      static_cast<size_t>(height) * static_cast<size_t>(width);
  out->height = height;
  out->width = width;
  out->diff.assign(pixels, 0.0f);
  out->flag.assign(pixels, 0);

  const size_t elements =
      2 * static_cast<size_t>(tile_rows) * static_cast<size_t>(width);
  std::vector<uint16_t> red(elements);
  std::vector<uint16_t> nir(elements);
  std::vector<uint8_t> scl(elements);
  const int selected[2] = {date_a, date_b};
  const float scale[2] = {dates.scale[static_cast<size_t>(date_a)],
                          dates.scale[static_cast<size_t>(date_b)]};
  const float offset[2] = {dates.offset[static_cast<size_t>(date_a)],
                           dates.offset[static_cast<size_t>(date_b)]};

  double read_ms = 0.0;
  double compute_ms = 0.0;
  for (int row0 = 0; row0 < height; row0 += tile_rows) {
    const int rows = std::min(tile_rows, height - row0);
    TileView view;
    view.red = red.data();
    view.nir = nir.data();
    view.scl = scl.data();
    view.row0 = row0;
    view.rows = rows;
    const WallTimer read_timer;
    const Status status = source->ReadTileDates(row0, rows, selected, 2, &view);
    read_ms += read_timer.ElapsedMs();
    if (!status.ok()) {
      return status;
    }
    const WallTimer compute_timer;
    const size_t plane = static_cast<size_t>(rows) * width;
    for (size_t i = 0; i < plane; ++i) {
      const float ndvi_a = ComputeNdvi<float>(red[i], nir[i], scl[i], scale[0],
                                              offset[0], p.valid_class_mask);
      const float ndvi_b =
          ComputeNdvi<float>(red[plane + i], nir[plane + i], scl[plane + i],
                             scale[1], offset[1], p.valid_class_mask);
      const size_t pixel = static_cast<size_t>(row0) * width + i;
      QuickDiffPixel(ndvi_a, ndvi_b, diff_threshold, p.losses_only,
                     &out->diff[pixel], &out->flag[pixel]);
    }
    compute_ms += compute_timer.ElapsedMs();
  }
  if (t != nullptr) {
    t->read_ms = read_ms;
    t->kernel_ms = compute_ms;
    t->total_ms = total_timer.ElapsedMs();
  }
  return Status::Ok();
}

}  // namespace cd
