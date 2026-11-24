// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "cpu/cpu_backend.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "pixel_model.h"
#include "system_info.h"
#include "tile_scheduler.h"
#include "timing.h"

namespace cd {
namespace {

// break_index is stored as int16_t, so at most this many dates are supported.
constexpr int kMaxSupportedDates = 32767;

// Checks the model parameters and the cube geometry, copies the date table of
// the source and completes it with BuildDateTable. Used by every entry point
// so that the CPU backend never trusts a caller-built table.
Status PrepareRun(const CubeSource& source, const ModelParams& params,
                  DateTable* dates) {
  const CubeInfo& info = source.info();
  if (info.height < 1 || info.width < 1 || info.num_dates < 1) {
    return Status::Error("cube has an empty dimension");
  }
  if (info.num_dates > kMaxSupportedDates) {
    return Status::Error("cube has " + std::to_string(info.num_dates) +
                         " dates; at most " +
                         std::to_string(kMaxSupportedDates) + " are supported");
  }
  if (params.harmonics < 1 || params.harmonics > kMaxHarmonics) {
    return Status::Error("harmonics must be in 1..3, got " +
                         std::to_string(params.harmonics));
  }
  const int num_params = NumParams(params.harmonics);
  if (params.min_train_obs < num_params + 1) {
    return Status::Error("min_train_obs " +
                         std::to_string(params.min_train_obs) +
                         " is below p + 1 = " + std::to_string(num_params + 1) +
                         " (call ResolveModelParams first)");
  }
  if (params.consecutive < 1) {
    return Status::Error("consecutive must be >= 1");
  }
  *dates = info.dates;
  const size_t num_dates = static_cast<size_t>(info.num_dates);
  if (dates->day.size() != num_dates || dates->scale.size() != num_dates ||
      dates->offset.size() != num_dates) {
    return Status::Error("date table size does not match the cube's " +
                         std::to_string(num_dates) + " dates");
  }
  return BuildDateTable(params, dates);
}

// Converts the FP64 design table to the precision of a backend.
template <typename Real>
std::vector<Real> CastDesign(const std::vector<double>& table) {
  std::vector<Real> out(table.size());
  for (size_t i = 0; i < table.size(); ++i) {
    out[i] = static_cast<Real>(table[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tile reading
// ---------------------------------------------------------------------------

// Uninitialised host buffers for one tile (T planes of rows x W each). They
// are overwritten completely by CubeSource::ReadTile, so zero-filling them
// would only waste time.
struct TileBuffer {
  std::unique_ptr<uint16_t[]> red;
  std::unique_ptr<uint16_t[]> nir;
  std::unique_ptr<uint8_t[]> scl;
  TileView view;
  int row0 = 0;
  int rows = 0;

  void Allocate(size_t elements) {
    red.reset(new uint16_t[elements]);
    nir.reset(new uint16_t[elements]);
    scl.reset(new uint8_t[elements]);
  }
};

// Reads tiles on a helper thread so that tile k + 1 is read from the source
// while tile k is computed. At most one read is in flight.
class TilePrefetcher {
 public:
  explicit TilePrefetcher(CubeSource* source) : source_(source) {}
  TilePrefetcher(const TilePrefetcher&) = delete;
  TilePrefetcher& operator=(const TilePrefetcher&) = delete;
  ~TilePrefetcher() { Join(); }

  // Starts reading rows [row0, row0 + rows) into `buffer`. The previous read
  // must have been collected with Wait().
  void Start(int row0, int rows, TileBuffer* buffer) {
    buffer->row0 = row0;
    buffer->rows = rows;
    buffer->view.red = buffer->red.get();
    buffer->view.nir = buffer->nir.get();
    buffer->view.scl = buffer->scl.get();
    buffer->view.row0 = row0;
    buffer->view.rows = rows;
    status_ = Status::Ok();
    thread_ = std::thread([this, buffer]() {
      const WallTimer timer;
      status_ = source_->ReadTile(buffer->row0, buffer->rows, &buffer->view);
      last_read_ms_ = timer.ElapsedMs();
    });
  }

  // Blocks until the running read has finished and returns its status.
  Status Wait() {
    Join();
    total_read_ms_ += last_read_ms_;
    last_read_ms_ = 0.0;
    return status_;
  }

  // Sum of the durations of all collected reads.
  double total_read_ms() const { return total_read_ms_; }

 private:
  void Join() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  CubeSource* source_;
  std::thread thread_;
  Status status_ = Status::Ok();
  double last_read_ms_ = 0.0;
  double total_read_ms_ = 0.0;
};

// ---------------------------------------------------------------------------
// Per-tile computation
// ---------------------------------------------------------------------------

// Everything a worker needs to process the pixels of one tile. All pointers
// are read-only except `result`, whose maps are written at disjoint pixels.
template <typename Real>
struct TileJob {
  const TileView* tile;
  int width;
  int num_dates;
  int num_train_dates;
  const float* scale;
  const float* offset;
  const Real* design;
  const ModelParams* params;
  bool save_coefficients;
  int harmonics;
  DetectionResult* result;
};

// Number of neighbouring pixels whose NDVI series are computed together.
constexpr int kPixelBlock = 16;

// Analyses the pixels of tile rows [row_begin, row_end).
//
// The pixels of a row are processed in blocks of kPixelBlock columns. For a
// block, the NDVI of every date is first computed into `ndvi` laid out as
// [date][pixel in block]: the inner loop runs over neighbouring pixels, which
// are adjacent in memory in the date-major tile, so the compiler vectorises
// the NDVI formula (including its division) and every date reads kPixelBlock
// consecutive values from each band. The fit, the residual pass and the
// monitoring pass then read the block's NDVI values from the L1 cache instead
// of recomputing the formula three times per observation. This changes no
// value: the gathered numbers are bit-identical to what RawSeries returns.
template <typename Real, int kHarmonics>
void ProcessRows(const TileJob<Real>& job, int row_begin, int row_end,
                 Real* ndvi) {
  constexpr int kParams = 2 + 2 * kHarmonics;
  const TileView& tile = *job.tile;
  const int64_t plane = static_cast<int64_t>(tile.rows) * job.width;
  const int64_t total_pixels =
      static_cast<int64_t>(job.result->height) * job.result->width;
  const uint32_t valid_mask = job.params->valid_class_mask;
  DetectionResult* result = job.result;
  for (int r = row_begin; r < row_end; ++r) {
    for (int c0 = 0; c0 < job.width; c0 += kPixelBlock) {
      const int count = std::min(kPixelBlock, job.width - c0);
      const int64_t block_start = static_cast<int64_t>(r) * job.width + c0;
      for (int t = 0; t < job.num_dates; ++t) {
        const uint16_t* red = tile.red + t * plane + block_start;
        const uint16_t* nir = tile.nir + t * plane + block_start;
        const uint8_t* scl = tile.scl + t * plane + block_start;
        const Real scale = static_cast<Real>(job.scale[t]);
        const Real offset = static_cast<Real>(job.offset[t]);
        Real* out = ndvi + t * kPixelBlock;
        for (int b = 0; b < count; ++b) {
          out[b] = ComputeNdvi<Real>(red[b], nir[b], scl[b], scale, offset,
                                     valid_mask);
        }
      }
      for (int b = 0; b < count; ++b) {
        const NdviArraySeries<Real, Real> series{ndvi + b, kPixelBlock};
        const PixelOutput<Real> out =
            FitAndDetect<Real, kHarmonics>(series, job.design, job.num_dates,
                                           job.num_train_dates, *job.params);

        const int64_t pixel =
            static_cast<int64_t>(tile.row0) * job.width + block_start + b;
        result->break_index[pixel] = out.detect.break_index;
        result->magnitude[pixel] = static_cast<float>(out.detect.magnitude);
        result->status[pixel] = out.detect.status;
        result->rmse[pixel] = static_cast<float>(out.fit.rmse);
        result->train_obs[pixel] = static_cast<uint16_t>(
            std::min(out.fit.n_train, static_cast<int>(UINT16_MAX)));
        if (job.save_coefficients) {
          for (int j = 0; j < kParams; ++j) {
            result
                ->coefficients[static_cast<size_t>(j) * total_pixels + pixel] =
                static_cast<float>(out.fit.coef[j]);
          }
        }
      }
    }
  }
}

// Selects the compile-time harmonic count.
template <typename Real>
void ProcessRowsDispatch(const TileJob<Real>& job, int row_begin, int row_end,
                         Real* ndvi) {
  switch (job.harmonics) {
    case 1:
      ProcessRows<Real, 1>(job, row_begin, row_end, ndvi);
      break;
    case 2:
      ProcessRows<Real, 2>(job, row_begin, row_end, ndvi);
      break;
    default:
      ProcessRows<Real, 3>(job, row_begin, row_end, ndvi);
      break;
  }
}

// Computes one tile with `workers` threads. One worker runs on the calling
// thread. Several workers take chunks of rows from a shared atomic counter
// (dynamic scheduling, so slow rows do not leave other cores idle). Each
// worker owns its NDVI scratch array, the output maps are written at disjoint
// pixels, and the hot loop contains no locks.
template <typename Real>
void ComputeTile(const TileJob<Real>& job, int workers) {
  const int rows = job.tile->rows;
  if (workers <= 1) {
    std::vector<Real> ndvi(static_cast<size_t>(job.num_dates) * kPixelBlock);
    ProcessRowsDispatch<Real>(job, 0, rows, ndvi.data());
    return;
  }
  // About four chunks per worker balance the load without making the counter
  // a contention point.
  const int chunk = std::max(1, rows / (workers * 4));
  std::atomic<int> next_row(0);
  auto work = [&job, &next_row, rows, chunk]() {
    std::vector<Real> ndvi(static_cast<size_t>(job.num_dates) * kPixelBlock);
    while (true) {
      const int begin = next_row.fetch_add(chunk);
      if (begin >= rows) {
        break;
      }
      ProcessRowsDispatch<Real>(job, begin, std::min(rows, begin + chunk),
                                ndvi.data());
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers - 1));
  for (int i = 1; i < workers; ++i) {
    threads.emplace_back(work);
  }
  work();
  for (std::thread& thread : threads) {
    thread.join();
  }
}

// Runs the whole cube in precision `Real` on `workers` threads.
template <typename Real>
Status RunDetection(CubeSource* source, const ModelParams& params,
                    const RunConfig& config, int workers,
                    const DateTable& dates, DetectionResult* result,
                    StageTimings* timings, RunStats* stats) {
  const CubeInfo& info = source->info();
  const int height = info.height;
  const int width = info.width;
  const int num_dates = info.num_dates;
  const int num_params = NumParams(params.harmonics);
  // Automatic tile size: the same input cap as the GPU backend (no device
  // memory limit applies), so that both read the cube with equal requests.
  const TilePlan plan = PlanTiles(height, width, num_dates, config.tile_rows, 0,
                                  0, 0, kMaxTileInputBytes);
  const int tile_rows = plan.tile_rows;
  const int num_tiles = plan.num_tiles;

  WallTimer total_timer;
  AllocateResult(height, width, num_params, config.save_coefficients, result);

  std::vector<double> design_table;
  BuildDesignTable(dates, params.harmonics, params.train_years,
                   params.hold_trend, &design_table);
  const std::vector<Real> design = CastDesign<Real>(design_table);

  const size_t elements = static_cast<size_t>(tile_rows) *
                          static_cast<size_t>(width) *
                          static_cast<size_t>(num_dates);
  TileBuffer buffers[2];
  buffers[0].Allocate(elements);
  if (num_tiles > 1) {
    buffers[1].Allocate(elements);
  }

  TilePrefetcher prefetcher(source);
  double wait_ms = 0.0;
  double kernel_ms = 0.0;
  // Setup ends here: result maps, design table and tile buffers are ready.
  const double setup_ms = total_timer.ElapsedMs();
  prefetcher.Start(0, std::min(tile_rows, height), &buffers[0]);
  for (int k = 0; k < num_tiles; ++k) {
    const WallTimer wait_timer;
    const Status read_status = prefetcher.Wait();
    wait_ms += wait_timer.ElapsedMs();
    if (!read_status.ok()) {
      return Status::Error("reading tile " + std::to_string(k) + ": " +
                           read_status.message());
    }
    if (k + 1 < num_tiles) {
      const int next_row0 = (k + 1) * tile_rows;
      prefetcher.Start(next_row0, std::min(tile_rows, height - next_row0),
                       &buffers[(k + 1) % 2]);
    }
    const WallTimer kernel_timer;
    const TileJob<Real> job{&buffers[k % 2].view,
                            width,
                            num_dates,
                            dates.num_train_dates,
                            dates.scale.data(),
                            dates.offset.data(),
                            design.data(),
                            &params,
                            config.save_coefficients,
                            params.harmonics,
                            result};
    ComputeTile<Real>(job, workers);
    kernel_ms += kernel_timer.ElapsedMs();
  }

  // Releasing the tile buffers is part of the run (counted in total_ms, like
  // the buffer release of the GPU backend).
  buffers[0] = TileBuffer();
  buffers[1] = TileBuffer();
  const double total_ms = total_timer.ElapsedMs();
  if (timings != nullptr) {
    timings->read_ms = prefetcher.total_read_ms();
    timings->read_wait_ms = wait_ms;
    timings->setup_ms = setup_ms;
    timings->kernel_ms = kernel_ms;
    timings->total_ms = total_ms;
  }
  if (stats != nullptr) {
    stats->num_tiles = num_tiles;
    stats->tile_rows_used = tile_rows;
    stats->threads_used = workers;
    stats->streams_used = 0;
    stats->device_name = CpuModel();
  }
  return Status::Ok();
}

// Number of worker threads of a backend: 1 for the single-thread baselines,
// config.threads (0 = all hardware threads) for cpu-mt.
int WorkerCount(const RunConfig& config) {
  if (config.backend != Backend::kCpuMt) {
    return 1;
  }
  return config.threads > 0 ? config.threads : HardwareThreads();
}

// Gives the FP64 trajectory of one pixel for K harmonics.
template <int kHarmonics>
void FillTrajectory(const std::vector<double>& ndvi, const DateTable& dates,
                    const ModelParams& params, PixelTrajectory* out) {
  constexpr int kParams = 2 + 2 * kHarmonics;
  const int num_dates = static_cast<int>(ndvi.size());
  std::vector<double> design;
  BuildDesignTable(dates, kHarmonics, params.train_years, params.hold_trend,
                   &design);

  const NdviArraySeries<double, double> series{ndvi.data(), 1};
  const PixelOutput<double> fit_and_detect = FitAndDetect<double, kHarmonics>(
      series, design.data(), num_dates, dates.num_train_dates, params);
  const FitOutput<double>& fit = fit_and_detect.fit;
  const DetectOutput<double>& detect = fit_and_detect.detect;

  for (int j = 0; j < kParams; ++j) {
    out->coef[j] = fit.coef[j];
  }
  out->rmse = fit.rmse;
  out->n_train = fit.n_train;
  out->break_index = detect.break_index;
  out->magnitude = detect.magnitude;
  out->status = detect.status;

  out->predicted.assign(static_cast<size_t>(num_dates), 0.0);
  out->z.assign(static_cast<size_t>(num_dates), 0.0);
  out->in_break_run.assign(static_cast<size_t>(num_dates), 0);
  if (!fit.ok) {
    return;
  }
  for (int t = 0; t < num_dates; ++t) {
    const double predicted = PredictNdvi<double, kHarmonics>(
        fit.coef, design.data() + static_cast<size_t>(t) * kParams);
    out->predicted[static_cast<size_t>(t)] = predicted;
    if (IsValidObservation(ndvi[static_cast<size_t>(t)])) {
      out->z[static_cast<size_t>(t)] =
          (ndvi[static_cast<size_t>(t)] - predicted) / fit.rmse;
    }
  }
  if (detect.break_index >= 0) {
    // The run is the first `consecutive` valid observations from the break
    // date on (invalid observations inside the run are skipped).
    int marked = 0;
    for (int t = detect.break_index;
         t < num_dates && marked < params.consecutive; ++t) {
      if (IsValidObservation(ndvi[static_cast<size_t>(t)])) {
        out->in_break_run[static_cast<size_t>(t)] = 1;
        ++marked;
      }
    }
  }
}

}  // namespace

Status RunCpuDetection(CubeSource* source, const ModelParams& params,
                       const RunConfig& config, DetectionResult* result,
                       StageTimings* timings, RunStats* stats) {
  if (source == nullptr || result == nullptr) {
    return Status::Error("RunCpuDetection: null source or result");
  }
  if (IsGpuBackend(config.backend)) {
    return Status::Error("RunCpuDetection: backend '" +
                         BackendName(config.backend) +
                         "' is not a CPU backend");
  }
  if (config.tile_rows < 0 || config.threads < 0) {
    return Status::Error("tile_rows and threads must be >= 0");
  }
  DateTable dates;
  const Status prepared = PrepareRun(*source, params, &dates);
  if (!prepared.ok()) {
    return prepared;
  }
  const int workers = WorkerCount(config);
  if (config.backend == Backend::kCpuFp64) {
    return RunDetection<double>(source, params, config, workers, dates, result,
                                timings, stats);
  }
  return RunDetection<float>(source, params, config, workers, dates, result,
                             timings, stats);
}

Status ComputeTrajectory(CubeSource* source, const ModelParams& params, int row,
                         int col, PixelTrajectory* out) {
  if (source == nullptr || out == nullptr) {
    return Status::Error("ComputeTrajectory: null source or output");
  }
  DateTable dates;
  const Status prepared = PrepareRun(*source, params, &dates);
  if (!prepared.ok()) {
    return prepared;
  }
  const CubeInfo& info = source->info();
  if (row < 0 || row >= info.height || col < 0 || col >= info.width) {
    return Status::Error("pixel (" + std::to_string(row) + ", " +
                         std::to_string(col) + ") is outside the " +
                         std::to_string(info.height) + " x " +
                         std::to_string(info.width) + " cube");
  }
  const int width = info.width;
  const int num_dates = info.num_dates;
  const size_t elements =
      static_cast<size_t>(width) * static_cast<size_t>(num_dates);
  std::vector<uint16_t> red(elements);
  std::vector<uint16_t> nir(elements);
  std::vector<uint8_t> scl(elements);
  TileView view;
  view.red = red.data();
  view.nir = nir.data();
  view.scl = scl.data();
  view.row0 = row;
  view.rows = 1;
  const Status read_status = source->ReadTile(row, 1, &view);
  if (!read_status.ok()) {
    return read_status;
  }

  // Plane t of the one-row tile starts at t * W, so the pixel's series has a
  // stride of W elements.
  const RawSeries<double> raw{
      red.data() + col,   nir.data() + col,    scl.data() + col,       width,
      dates.scale.data(), dates.offset.data(), params.valid_class_mask};
  std::vector<double> ndvi(static_cast<size_t>(num_dates));
  for (int t = 0; t < num_dates; ++t) {
    ndvi[static_cast<size_t>(t)] = raw(t);
  }

  *out = PixelTrajectory();
  out->row = row;
  out->col = col;
  out->ndvi = ndvi;
  switch (params.harmonics) {
    case 1:
      FillTrajectory<1>(ndvi, dates, params, out);
      break;
    case 2:
      FillTrajectory<2>(ndvi, dates, params, out);
      break;
    default:
      FillTrajectory<3>(ndvi, dates, params, out);
      break;
  }
  return Status::Ok();
}

}  // namespace cd
