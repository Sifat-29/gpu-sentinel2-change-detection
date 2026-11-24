// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// GPU backend: the streamed tile pipeline (RunGpuDetection, GpuQuickDiff) and
// the chunked post-processing wrappers (colorize, cleanup, confusion).
//
// Pipeline. With --streams 3 there are three
// streams (host-to-device, compute, device-to-host) and three slots; a slot
// owns pinned host input/output buffers and device buffers for one tile.
// For tile k (slot s = k % 3) the host thread:
//   1. waits for the device-to-host copy of the tile that used slot s before
//      (tile k - 3), accumulates its event timings and scatters its pinned
//      output rows into the result;
//   2. reads tile k from the cube into the pinned input of slot s;
//   3. enqueues the H2D copy, then the kernels (the compute stream waits for
//      the copy's event), then the D2H copy (waits for the kernels' event).
// The disk read of tile k therefore overlaps the GPU work of tiles k - 1 and
// k - 2. With --streams 1 there is one stream and one slot and every tile is
// finished before the next one is read: the no-overlap baseline.
//
// In-pipeline validation (GpuValidationRequest). The tile's reference rows
// are copied by the host into a pinned slot buffer next to the cube data and
// travel to the device with the same H2D copies; the confusion kernel then
// runs on the compute stream on the tile's results, which are still on the
// device, and adds to counters that stay on the device until the last tile.
//
// Timeline trace (config.trace_csv). The engine can export one interval per
// (tile, stage). It then gives every tile its own set of CUDA events (instead
// of one set per slot, which later tiles overwrite), records one reference
// event on the first stream when the tile loop starts and, only after the final
// drain, converts every event to milliseconds since that reference with
// cudaEventElapsedTime. The only synchronisation is one wait for the
// reference event before the loop starts. Host disk reads are timed with the
// steady clock from the host time taken right after that wait, so host and
// device times agree to within the wake-up latency of the wait. (Taking the
// host time right after enqueueing the event is not enough: on WSL2 the
// driver batches submissions, and the event could execute milliseconds later.)
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cuda/cuda_check.cuh"
#include "cuda/gpu_backend.h"
#include "cuda/kernels.cuh"
#include "logger.h"
#include "pixel_model.h"
#include "tile_scheduler.h"
#include "timing.h"

namespace cd {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;

// Timed kernel stages per tile: NDVI, fit and detect (unfused), then the
// validation kernel (stage 3, or stage 1 of the fused path); quick-diff uses
// stage 0 only.
constexpr int kMaxStages = 4;

// The largest cube input of one tile (and so of one pinned host slot buffer)
// is kMaxTileInputBytes (tile_scheduler.h): measured on this machine,
// allocating pinned memory costs about 0.6 ms per MiB (WSL2), so three 256 MiB
// slots would spend about half a second in cudaMallocHost before the first
// tile is read. 64 MiB slots cut that to about a tenth and give the pipeline
// more tiles to overlap.

// Test hooks, see kernels.cuh.
size_t g_slot_allocation_limit = 0;
int g_postprocess_chunk_rows = 0;

// Device bytes per pixel of the outputs of every detection path: break index
// (2), magnitude (4), status (1), RMSE (4), training count (2).
constexpr size_t kOutputBytesPerPixel = 13;

// Selects CUDA device `device` and initialises its context, so that context
// creation is not charged to the first timed operation.
Status SelectDevice(int device) {
  int count = 0;
  CD_CUDA_CHECK(cudaGetDeviceCount(&count));
  if (device < 0 || device >= count) {
    return Status::Error("GPU index " + std::to_string(device) +
                         " is out of range (" + std::to_string(count) +
                         " device(s) found)");
  }
  CD_CUDA_CHECK(cudaSetDevice(device));
  CD_CUDA_CHECK(cudaFree(nullptr));
  return Status::Ok();
}

// Validates the cube geometry and model parameters and builds the complete
// date table (tau, year, num_train_dates) from the source's calibration.
Status PrepareDates(const CubeInfo& info, const ModelParams& params,
                    DateTable* dates) {
  if (info.height < 1 || info.width < 1 || info.num_dates < 1) {
    return Status::Error("cube has an empty dimension");
  }
  if (info.num_dates > kMaxDates) {
    return Status::Error("cube has " + std::to_string(info.num_dates) +
                         " dates; the GPU backend supports at most " +
                         std::to_string(kMaxDates));
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

// Checks the execution settings shared by the GPU entry points.
Status CheckRunConfig(const RunConfig& config) {
  if (config.streams != 1 && config.streams != 3) {
    return Status::Error("streams must be 1 or 3, got " +
                         std::to_string(config.streams));
  }
  if (config.tile_rows < 0) {
    return Status::Error("tile_rows must be >= 0");
  }
  if (config.block_size < 32 || config.block_size > 1024 ||
      config.block_size % 32 != 0) {
    return Status::Error(
        "block size must be a multiple of 32 in 32..1024, got " +
        std::to_string(config.block_size));
  }
  return Status::Ok();
}

// ---------------------------------------------------------------------------
// Slot allocation
// ---------------------------------------------------------------------------

// Allocates the buffers of the slots and keeps count of the device bytes.
// Applies the testing limit (SetSlotAllocationLimitForTesting) and remembers
// the first failure and whether any host buffer had to be pageable.
class SlotAllocator {
 public:
  // Allocates `count` elements on the device; false on failure.
  template <typename T>
  bool Device(DeviceBuffer<T>* buffer, size_t count) {
    if (error_ != cudaSuccess) {
      return false;
    }
    const size_t bytes = count * sizeof(T);
    if (g_slot_allocation_limit > 0 &&
        device_bytes_ + bytes > g_slot_allocation_limit) {
      error_ = cudaErrorMemoryAllocation;
      return false;
    }
    const cudaError_t error = buffer->Allocate(count);
    if (error != cudaSuccess) {
      error_ = error;
      return false;
    }
    device_bytes_ += bytes;
    return true;
  }

  // Allocates `count` elements of host memory (pinned when possible).
  template <typename T>
  bool Host(HostBuffer<T>* buffer, size_t count) {
    if (error_ != cudaSuccess) {
      return false;
    }
    if (!buffer->Allocate(count)) {
      error_ = cudaErrorMemoryAllocation;
      return false;
    }
    if (count > 0 && !buffer->pinned()) {
      pageable_ = true;
    }
    return true;
  }

  cudaError_t error() const { return error_; }
  size_t device_bytes() const { return device_bytes_; }
  bool pageable() const { return pageable_; }

 private:
  cudaError_t error_ = cudaSuccess;
  size_t device_bytes_ = 0;
  bool pageable_ = false;
};

// ---------------------------------------------------------------------------
// Generic three-stream tile pipeline
// ---------------------------------------------------------------------------

// Events of one slot. The `*_done` events order the streams; start/stop pairs
// give the stage timings (h2d_start..h2d_done, stage_start[i]..stage_stop[i],
// d2h_start..d2h_done). compute_done follows the last kernel.
struct SlotEvents {
  CudaEvent h2d_start;
  CudaEvent h2d_done;
  CudaEvent compute_done;
  CudaEvent d2h_start;
  CudaEvent d2h_done;
  CudaEvent stage_start[kMaxStages];
  CudaEvent stage_stop[kMaxStages];
  int num_stages = 0;  // set by EnqueueCompute

  Status Create() {
    CD_RETURN_IF_ERROR(h2d_start.Create());
    CD_RETURN_IF_ERROR(h2d_done.Create());
    CD_RETURN_IF_ERROR(compute_done.Create());
    CD_RETURN_IF_ERROR(d2h_start.Create());
    CD_RETURN_IF_ERROR(d2h_done.Create());
    for (int i = 0; i < kMaxStages; ++i) {
      CD_RETURN_IF_ERROR(stage_start[i].Create());
      CD_RETURN_IF_ERROR(stage_stop[i].Create());
    }
    return Status::Ok();
  }
};

// Records start/stop events around the launches made by `launch` (a callable
// returning Status) and notes the stage in `events`.
template <typename Launch>
Status TimedStage(SlotEvents* events, int stage, cudaStream_t stream,
                  Launch&& launch) {
  CD_CUDA_CHECK(cudaEventRecord(events->stage_start[stage].get(), stream));
  CD_RETURN_IF_ERROR(launch());
  CD_CUDA_CHECK(cudaEventRecord(events->stage_stop[stage].get(), stream));
  events->num_stages = std::max(events->num_stages, stage + 1);
  return Status::Ok();
}

// What a pipeline does for one tile; the engine below schedules the calls.
class TileStages {
 public:
  virtual ~TileStages() = default;

  // Allocates the buffers of all slots for tiles of up to `tile_rows` rows;
  // returns the first CUDA error (cudaErrorMemoryAllocation triggers a retry
  // with fewer rows).
  virtual cudaError_t Allocate(int tile_rows, SlotAllocator* allocator) = 0;
  // Frees every buffer.
  virtual void Release() = 0;
  // Host: reads rows [row0, row0 + rows) into the pinned input of `slot`.
  virtual Status Read(int slot, int row0, int rows) = 0;
  virtual Status EnqueueH2D(int slot, int rows, cudaStream_t stream) = 0;
  // Launches the kernels, timing them with TimedStage.
  virtual Status EnqueueCompute(int slot, int rows, cudaStream_t stream,
                                SlotEvents* events) = 0;
  virtual Status EnqueueD2H(int slot, int rows, cudaStream_t stream) = 0;
  // Host: moves the pinned output of `slot` into the final result.
  virtual Status Collect(int slot, int row0, int rows) = 0;
};

// One interval of the exported timeline. `stream` is the index of the CUDA
// stream (0 host-to-device, 1 compute, 2 device-to-host; all 0 with one
// stream) or -1 for the host read. Times are milliseconds since the start of
// the tile loop.
struct TraceRow {
  int tile = 0;
  int slot = 0;
  std::string stage;
  int stream = 0;
  double start_ms = 0.0;
  double end_ms = 0.0;
};

// Times accumulated by the pipeline (milliseconds).
struct PipelineTimes {
  double read_ms = 0.0;
  double h2d_ms = 0.0;
  double d2h_ms = 0.0;
  double stage_ms[kMaxStages] = {};
};

// Milliseconds between two recorded events.
Status ElapsedMs(const CudaEvent& start, const CudaEvent& stop, double* ms) {
  float elapsed = 0.0f;
  CD_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.get(), stop.get()));
  *ms += static_cast<double>(elapsed);
  return Status::Ok();
}

// The streaming engine. Owns the streams and the per-slot events and runs the
// loop described at the top of the file.
class TilePipeline {
 public:
  // `trace` (optional) receives the timeline rows after a successful run;
  // `stage_names` names the compute stages (one per TimedStage index).
  TilePipeline(TileStages* stages, int height, int tile_rows, int num_slots,
               PipelineTimes* times, std::vector<TraceRow>* trace = nullptr,
               std::vector<std::string> stage_names = {})
      : stages_(stages),
        ranges_(TileRanges(height, tile_rows)),
        num_slots_(num_slots),
        times_(times),
        trace_(trace),
        stage_names_(std::move(stage_names)) {}

  Status Run() {
    const Status status = RunLoop();
    if (!status.ok()) {
      // Never free buffers that queued copies or kernels still use.
      cudaDeviceSynchronize();
      cudaGetLastError();
    }
    return status;
  }

 private:
  // A tile that has been enqueued and not yet collected.
  struct Pending {
    bool active = false;
    int tile = 0;
    int row0 = 0;
    int rows = 0;
  };

  Status RunLoop() {
    const int num_streams = num_slots_ == 1 ? 1 : 3;
    std::vector<CudaStream> streams(static_cast<size_t>(num_streams));
    for (CudaStream& stream : streams) {
      CD_RETURN_IF_ERROR(stream.Create());
    }
    h2d_stream_ = streams[0].get();
    compute_stream_ =
        streams[static_cast<size_t>(num_streams > 1 ? 1 : 0)].get();
    d2h_stream_ = streams[static_cast<size_t>(num_streams > 1 ? 2 : 0)].get();
    events_ = std::vector<SlotEvents>(static_cast<size_t>(num_slots_));
    for (SlotEvents& events : events_) {
      CD_RETURN_IF_ERROR(events.Create());
    }
    pending_.assign(static_cast<size_t>(num_slots_), Pending());

    const int num_tiles = static_cast<int>(ranges_.size());
    if (trace_ != nullptr) {
      // Per-tile events, and the reference event of the timeline. Waiting for
      // the reference event pins the host reference time to the instant the
      // event actually executed on the device.
      tile_events_ = std::vector<SlotEvents>(static_cast<size_t>(num_tiles));
      for (SlotEvents& events : tile_events_) {
        CD_RETURN_IF_ERROR(events.Create());
      }
      CD_RETURN_IF_ERROR(reference_event_.Create());
      CD_CUDA_CHECK(cudaEventRecord(reference_event_.get(), h2d_stream_));
      CD_CUDA_CHECK(cudaEventSynchronize(reference_event_.get()));
      host_reference_ = std::chrono::steady_clock::now();
      host_reads_.assign(static_cast<size_t>(num_tiles), {0.0, 0.0});
    }
    for (int k = 0; k < num_tiles; ++k) {
      const int slot = k % num_slots_;
      const int row0 = ranges_[static_cast<size_t>(k)].first;
      const int rows = ranges_[static_cast<size_t>(k)].second;
      // (1) Free the slot: finish the tile that used it last.
      if (pending_[static_cast<size_t>(slot)].active) {
        CD_RETURN_IF_ERROR(Finish(slot));
      }
      // (2) Read the tile from the cube into the slot's pinned input.
      const WallTimer read_timer;
      const std::chrono::steady_clock::time_point read_begin =
          std::chrono::steady_clock::now();
      const Status read_status = stages_->Read(slot, row0, rows);
      times_->read_ms += read_timer.ElapsedMs();
      if (trace_ != nullptr) {
        host_reads_[static_cast<size_t>(k)] = {
            HostMs(read_begin), HostMs(std::chrono::steady_clock::now())};
      }
      if (!read_status.ok()) {
        return Status::Error("reading tile " + std::to_string(k) + ": " +
                             read_status.message());
      }
      // (3) Enqueue the three stages.
      CD_RETURN_IF_ERROR(Enqueue(k, slot, rows));
      Pending& pending = pending_[static_cast<size_t>(slot)];
      pending.active = true;
      pending.tile = k;
      pending.row0 = row0;
      pending.rows = rows;
      if (num_slots_ == 1) {
        CD_RETURN_IF_ERROR(Finish(slot));  // no overlap in the baseline
      }
    }
    // Drain: the oldest unfinished tile first.
    for (int k = std::max(0, num_tiles - num_slots_); k < num_tiles; ++k) {
      const int slot = k % num_slots_;
      if (pending_[static_cast<size_t>(slot)].active) {
        CD_RETURN_IF_ERROR(Finish(slot));
      }
    }
    return trace_ != nullptr ? ResolveTrace() : Status::Ok();
  }

  // The events of tile `tile`, which runs in `slot`: the slot's own set, or
  // the tile's private set when a trace is recorded.
  SlotEvents* EventsOf(int tile, int slot) {
    return trace_ != nullptr ? &tile_events_[static_cast<size_t>(tile)]
                             : &events_[static_cast<size_t>(slot)];
  }

  // Milliseconds from the host reference time to `time`.
  double HostMs(std::chrono::steady_clock::time_point time) const {
    return std::chrono::duration<double, std::milli>(time - host_reference_)
        .count();
  }

  // Converts the recorded events of every tile into trace rows. Called after
  // the drain, when every event has completed.
  Status ResolveTrace() {
    const int compute_stream = compute_stream_ == h2d_stream_ ? 0 : 1;
    const int d2h_stream = d2h_stream_ == h2d_stream_ ? 0 : 2;
    const auto since_reference = [this](const CudaEvent& event, double* ms) {
      float elapsed = 0.0f;
      CD_CUDA_CHECK(
          cudaEventElapsedTime(&elapsed, reference_event_.get(), event.get()));
      *ms = static_cast<double>(elapsed);
      return Status::Ok();
    };
    trace_->clear();
    for (size_t k = 0; k < tile_events_.size(); ++k) {
      const SlotEvents& events = tile_events_[k];
      const int tile = static_cast<int>(k);
      const int slot = tile % num_slots_;
      TraceRow row;
      row.tile = tile;
      row.slot = slot;
      row.stage = "read";
      row.stream = -1;
      row.start_ms = host_reads_[k].first;
      row.end_ms = host_reads_[k].second;
      trace_->push_back(row);

      row.stage = "h2d";
      row.stream = 0;
      CD_RETURN_IF_ERROR(since_reference(events.h2d_start, &row.start_ms));
      CD_RETURN_IF_ERROR(since_reference(events.h2d_done, &row.end_ms));
      trace_->push_back(row);

      row.stream = compute_stream;
      for (int i = 0; i < events.num_stages; ++i) {
        row.stage = stage_names_[static_cast<size_t>(i)];
        CD_RETURN_IF_ERROR(
            since_reference(events.stage_start[i], &row.start_ms));
        CD_RETURN_IF_ERROR(since_reference(events.stage_stop[i], &row.end_ms));
        trace_->push_back(row);
      }

      row.stage = "d2h";
      row.stream = d2h_stream;
      CD_RETURN_IF_ERROR(since_reference(events.d2h_start, &row.start_ms));
      CD_RETURN_IF_ERROR(since_reference(events.d2h_done, &row.end_ms));
      trace_->push_back(row);
    }
    return Status::Ok();
  }

  // Enqueues H2D, kernels and D2H of tile `tile` (in `slot`) with event
  // dependencies.
  Status Enqueue(int tile, int slot, int rows) {
    SlotEvents* events = EventsOf(tile, slot);
    events->num_stages = 0;
    CD_CUDA_CHECK(cudaEventRecord(events->h2d_start.get(), h2d_stream_));
    CD_RETURN_IF_ERROR(stages_->EnqueueH2D(slot, rows, h2d_stream_));
    CD_CUDA_CHECK(cudaEventRecord(events->h2d_done.get(), h2d_stream_));

    CD_CUDA_CHECK(
        cudaStreamWaitEvent(compute_stream_, events->h2d_done.get(), 0));
    CD_RETURN_IF_ERROR(
        stages_->EnqueueCompute(slot, rows, compute_stream_, events));
    CD_CUDA_CHECK(cudaEventRecord(events->compute_done.get(), compute_stream_));

    CD_CUDA_CHECK(
        cudaStreamWaitEvent(d2h_stream_, events->compute_done.get(), 0));
    CD_CUDA_CHECK(cudaEventRecord(events->d2h_start.get(), d2h_stream_));
    CD_RETURN_IF_ERROR(stages_->EnqueueD2H(slot, rows, d2h_stream_));
    CD_CUDA_CHECK(cudaEventRecord(events->d2h_done.get(), d2h_stream_));
    return Status::Ok();
  }

  // Waits for the tile of `slot`, adds its event timings and collects it.
  Status Finish(int slot) {
    Pending* pending = &pending_[static_cast<size_t>(slot)];
    SlotEvents* events = EventsOf(pending->tile, slot);
    CD_CUDA_CHECK(cudaEventSynchronize(events->d2h_done.get()));
    CD_RETURN_IF_ERROR(
        ElapsedMs(events->h2d_start, events->h2d_done, &times_->h2d_ms));
    CD_RETURN_IF_ERROR(
        ElapsedMs(events->d2h_start, events->d2h_done, &times_->d2h_ms));
    for (int i = 0; i < events->num_stages; ++i) {
      CD_RETURN_IF_ERROR(ElapsedMs(
          events->stage_start[i], events->stage_stop[i], &times_->stage_ms[i]));
    }
    pending->active = false;
    return stages_->Collect(slot, pending->row0, pending->rows);
  }

  TileStages* stages_;
  std::vector<std::pair<int, int>> ranges_;
  int num_slots_;
  PipelineTimes* times_;
  std::vector<TraceRow>* trace_;  // null: no timeline is recorded
  std::vector<std::string> stage_names_;
  cudaStream_t h2d_stream_ = nullptr;
  cudaStream_t compute_stream_ = nullptr;
  cudaStream_t d2h_stream_ = nullptr;
  std::vector<SlotEvents> events_;
  std::vector<Pending> pending_;
  // Trace state (used only when trace_ is set): events per tile, the
  // reference event and its host time, and the host read interval per tile.
  std::vector<SlotEvents> tile_events_;
  CudaEvent reference_event_;
  std::chrono::steady_clock::time_point host_reference_;
  std::vector<std::pair<double, double>> host_reads_;
};

// Writes the timeline rows as CSV (columns: tile, slot, stage, stream,
// start_ms, end_ms).
Status WriteTraceCsv(const std::string& path,
                     const std::vector<TraceRow>& rows) {
  std::FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) {
    return Status::Error("cannot write trace file '" + path + "'");
  }
  std::fprintf(file, "tile,slot,stage,stream,start_ms,end_ms\n");
  for (const TraceRow& row : rows) {
    std::fprintf(file, "%d,%d,%s,%d,%.4f,%.4f\n", row.tile, row.slot,
                 row.stage.c_str(), row.stream, row.start_ms, row.end_ms);
  }
  const bool failed = std::ferror(file) != 0;
  if (std::fclose(file) != 0 || failed) {
    return Status::Error("error while writing trace file '" + path + "'");
  }
  return Status::Ok();
}

// Allocates the slots of `stages`. When the device (or the testing limit) runs
// out of memory the tile height is halved, down to one row, and the
// allocation retried; each retry is logged and noted in `stats`.
Status AllocateWithRetry(TileStages* stages, int* tile_rows,
                         SlotAllocator* allocator, RunStats* stats) {
  for (;;) {
    *allocator = SlotAllocator();
    const cudaError_t error = stages->Allocate(*tile_rows, allocator);
    if (error == cudaSuccess) {
      return Status::Ok();
    }
    stages->Release();
    if (error != cudaErrorMemoryAllocation) {
      return CudaStatus(error, "slot buffer allocation", __FILE__, __LINE__);
    }
    if (*tile_rows <= 1) {
      return Status::Error(
          "out of device memory: the buffers of a single-row tile do not fit");
    }
    *tile_rows = std::max(1, *tile_rows / 2);
    const std::string message = "tile allocation failed; retrying with " +
                                std::to_string(*tile_rows) + " rows";
    Log(message);
    if (stats != nullptr) {
      stats->notes.push_back(message);
    }
  }
}

// When PlanTiles reduced an explicit --tile-rows to fit the memory budget,
// records "--tile-rows N needs X MiB, Y MiB free; using R rows" in `stats`
// (the execution section of the log prints it as a "note:" line).
// WSL2 silently oversubscribes device memory, so a failed allocation cannot
// be relied on to catch an oversized tile.
void NoteTileRowsReduced(const TilePlan& plan, int requested_rows,
                         int num_slots, size_t bytes_per_pixel, int width,
                         size_t free_bytes, RunStats* stats) {
  if (!plan.reduced_for_memory) {
    return;
  }
  const double needed_mb = static_cast<double>(num_slots) * requested_rows *
                           width * static_cast<double>(bytes_per_pixel) / kMiB;
  const std::string message =
      "--tile-rows " + std::to_string(requested_rows) + " needs " +
      std::to_string(static_cast<int64_t>(needed_mb)) + " MiB, " +
      std::to_string(static_cast<int64_t>(free_bytes / kMiB)) +
      " MiB free; using " + std::to_string(plan.tile_rows) + " rows";
  if (stats != nullptr) {
    stats->notes.push_back(message);
  }
}

// Logs the warning for pageable fallback buffers.
void WarnIfPageable(const SlotAllocator& allocator, RunStats* stats) {
  if (!allocator.pageable()) {
    return;
  }
  const std::string message =
      "warning: cudaMallocHost failed; using pageable host buffers "
      "(transfers will not overlap with compute)";
  Log(message);
  if (stats != nullptr) {
    stats->notes.push_back(message);
  }
}

// ---------------------------------------------------------------------------
// Detection pipeline
// ---------------------------------------------------------------------------

// Device bytes per pixel of the detection path (inputs, intermediates and
// outputs), used to size the tiles.
size_t DetectionBytesPerPixel(int num_dates, int num_params, bool fused,
                              bool save_coefficients, bool validate) {
  size_t bytes = static_cast<size_t>(num_dates) * 5 + kOutputBytesPerPixel;
  if (validate) {
    bytes += 1;  // the reference byte
  }
  if (fused) {
    if (save_coefficients) {
      bytes += static_cast<size_t>(num_params) * 4;
    }
  } else {
    bytes += static_cast<size_t>(num_dates) * 4 +
             static_cast<size_t>(num_params) * 4;
  }
  return bytes;
}

// Buffers of one slot of the detection pipeline.
struct DetectionSlot {
  HostBuffer<uint16_t> host_red;
  HostBuffer<uint16_t> host_nir;
  HostBuffer<uint8_t> host_scl;
  HostBuffer<int16_t> host_break_index;
  HostBuffer<float> host_magnitude;
  HostBuffer<uint8_t> host_status;
  HostBuffer<float> host_rmse;
  HostBuffer<uint16_t> host_n_train;
  HostBuffer<float> host_coef;
  HostBuffer<uint8_t> host_reference;  // validation only
  DeviceBuffer<uint16_t> dev_red;
  DeviceBuffer<uint16_t> dev_nir;
  DeviceBuffer<uint8_t> dev_scl;
  DeviceBuffer<float> dev_ndvi;  // unfused path only
  DeviceBuffer<float> dev_coef;  // unfused path, or fused with coefficients
  DeviceBuffer<int16_t> dev_break_index;
  DeviceBuffer<float> dev_magnitude;
  DeviceBuffer<uint8_t> dev_status;
  DeviceBuffer<float> dev_rmse;
  DeviceBuffer<uint16_t> dev_n_train;
  DeviceBuffer<uint8_t> dev_reference;  // validation only
};

// The tile stages of change detection (fused or unfused kernels).
class DetectionStages : public TileStages {
 public:
  DetectionStages(CubeSource* source, const DateTable& dates,
                  const ModelParams& params, const RunConfig& config,
                  bool fused, int num_slots, DetectionResult* result,
                  const GpuValidationRequest* validation)
      : source_(source),
        dates_(dates),
        params_(params),
        fused_(fused),
        save_coef_(config.save_coefficients),
        block_size_(config.block_size),
        num_params_(NumParams(params.harmonics)),
        width_(source->info().width),
        height_(source->info().height),
        num_dates_(source->info().num_dates),
        slots_(static_cast<size_t>(num_slots)),
        result_(result),
        validation_(validation),
        validate_stage_(fused ? 1 : 3) {}

  cudaError_t Allocate(int tile_rows, SlotAllocator* a) override {
    const size_t pixels = static_cast<size_t>(tile_rows) * width_;
    const size_t observations = pixels * static_cast<size_t>(num_dates_);
    const size_t coef_count = pixels * static_cast<size_t>(num_params_);
    const bool device_coef = !fused_ || save_coef_;
    const size_t reference_count = validation_ != nullptr ? pixels : 0;
    if (validation_ != nullptr &&
        !(a->Device(&dev_year_, validation_->year->size()) &&
          a->Device(&dev_counters_, kNumConfusionCounters))) {
      return a->error();
    }
    for (DetectionSlot& s : slots_) {
      const bool ok =
          a->Host(&s.host_red, observations) &&
          a->Host(&s.host_nir, observations) &&
          a->Host(&s.host_scl, observations) &&
          a->Host(&s.host_break_index, pixels) &&
          a->Host(&s.host_magnitude, pixels) &&
          a->Host(&s.host_status, pixels) && a->Host(&s.host_rmse, pixels) &&
          a->Host(&s.host_n_train, pixels) &&
          a->Host(&s.host_coef, save_coef_ ? coef_count : 0) &&
          a->Host(&s.host_reference, reference_count) &&
          a->Device(&s.dev_red, observations) &&
          a->Device(&s.dev_nir, observations) &&
          a->Device(&s.dev_scl, observations) &&
          a->Device(&s.dev_ndvi, fused_ ? 0 : observations) &&
          a->Device(&s.dev_coef, device_coef ? coef_count : 0) &&
          a->Device(&s.dev_break_index, pixels) &&
          a->Device(&s.dev_magnitude, pixels) &&
          a->Device(&s.dev_status, pixels) && a->Device(&s.dev_rmse, pixels) &&
          a->Device(&s.dev_n_train, pixels) &&
          a->Device(&s.dev_reference, reference_count);
      if (!ok) {
        return a->error();
      }
    }
    return cudaSuccess;
  }

  void Release() override {
    for (DetectionSlot& slot : slots_) {
      slot = DetectionSlot();
    }
    dev_year_ = DeviceBuffer<int16_t>();
    dev_counters_ = DeviceBuffer<uint64_t>();
  }

  Status Read(int slot, int row0, int rows) override {
    DetectionSlot& s = slots_[static_cast<size_t>(slot)];
    TileView view;
    view.red = s.host_red.get();
    view.nir = s.host_nir.get();
    view.scl = s.host_scl.get();
    view.row0 = row0;
    view.rows = rows;
    CD_RETURN_IF_ERROR(source_->ReadTile(row0, rows, &view));
    if (validation_ != nullptr) {
      // The tile's reference rows are contiguous in the row-major raster.
      std::memcpy(
          s.host_reference.get(),
          validation_->reference->data() + static_cast<size_t>(row0) * width_,
          static_cast<size_t>(rows) * width_);
    }
    return Status::Ok();
  }

  // Uploads the year table and zeroes the device counters; call once after
  // the final allocation and before the first tile. No-op without validation.
  Status InitValidation() {
    if (validation_ == nullptr) {
      return Status::Ok();
    }
    CD_CUDA_CHECK(cudaMemcpy(dev_year_.get(), validation_->year->data(),
                             validation_->year->size() * sizeof(int16_t),
                             cudaMemcpyHostToDevice));
    CD_CUDA_CHECK(cudaMemset(dev_counters_.get(), 0,
                             kNumConfusionCounters * sizeof(uint64_t)));
    return Status::Ok();
  }

  // Copies the accumulated confusion counts to the host; call after the
  // pipeline has drained.
  Status ReadCounts(ConfusionCounts* counts) const {
    uint64_t totals[kNumConfusionCounters] = {0};
    CD_CUDA_CHECK(cudaMemcpy(totals, dev_counters_.get(), sizeof(totals),
                             cudaMemcpyDeviceToHost));
    counts->tp = totals[0];
    counts->fp = totals[1];
    counts->fn = totals[2];
    counts->tn = totals[3];
    counts->excluded = totals[4];
    counts->year_exact = totals[5];
    counts->year_within1 = totals[6];
    return Status::Ok();
  }

  Status EnqueueH2D(int slot, int rows, cudaStream_t stream) override {
    DetectionSlot& s = slots_[static_cast<size_t>(slot)];
    const size_t n = Observations(rows);
    CD_CUDA_CHECK(cudaMemcpyAsync(s.dev_red.get(), s.host_red.get(),
                                  n * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                  stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.dev_nir.get(), s.host_nir.get(),
                                  n * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                  stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.dev_scl.get(), s.host_scl.get(),
                                  n * sizeof(uint8_t), cudaMemcpyHostToDevice,
                                  stream));
    if (validation_ != nullptr) {
      CD_CUDA_CHECK(cudaMemcpyAsync(
          s.dev_reference.get(), s.host_reference.get(),
          static_cast<size_t>(rows) * width_, cudaMemcpyHostToDevice, stream));
    }
    return Status::Ok();
  }

  Status EnqueueCompute(int slot, int rows, cudaStream_t stream,
                        SlotEvents* events) override {
    DetectionSlot& s = slots_[static_cast<size_t>(slot)];
    const int64_t pixels = static_cast<int64_t>(rows) * width_;
    LaunchConfig launch;
    launch.block_size = block_size_;
    launch.stream = stream;
    RawBands bands;
    bands.red = s.dev_red.get();
    bands.nir = s.dev_nir.get();
    bands.scl = s.dev_scl.get();
    DetectionBuffers out;
    out.coef = s.dev_coef.get();  // null when the fused path skips it
    out.rmse = s.dev_rmse.get();
    out.n_train = s.dev_n_train.get();
    out.break_index = s.dev_break_index.get();
    out.magnitude = s.dev_magnitude.get();
    out.status = s.dev_status.get();
    const int train = dates_.num_train_dates;
    if (fused_) {
      CD_RETURN_IF_ERROR(TimedStage(events, 0, stream, [&] {
        return LaunchFusedChangeDetection(bands, pixels, num_dates_, train,
                                          params_, out, launch);
      }));
    } else {
      CD_RETURN_IF_ERROR(TimedStage(events, 0, stream, [&] {
        return LaunchComputeNdvi(bands, num_dates_, pixels,
                                 params_.valid_class_mask, s.dev_ndvi.get(),
                                 launch);
      }));
      CD_RETURN_IF_ERROR(TimedStage(events, 1, stream, [&] {
        return LaunchFitHarmonicModel(s.dev_ndvi.get(), pixels, train, params_,
                                      out, launch);
      }));
      CD_RETURN_IF_ERROR(TimedStage(events, 2, stream, [&] {
        return LaunchDetectBreak(s.dev_ndvi.get(), pixels, num_dates_, train,
                                 params_, out, launch);
      }));
    }
    if (validation_ == nullptr) {
      return Status::Ok();
    }
    // The tile's results are still on the device: count them there.
    return TimedStage(events, validate_stage_, stream, [&] {
      return LaunchConfusion(
          s.dev_status.get(), s.dev_break_index.get(), s.dev_magnitude.get(),
          s.dev_reference.get(), dev_year_.get(),
          static_cast<int>(validation_->year->size()), validation_->window,
          pixels, dev_counters_.get(), stream);
    });
  }

  Status EnqueueD2H(int slot, int rows, cudaStream_t stream) override {
    DetectionSlot& s = slots_[static_cast<size_t>(slot)];
    const size_t n = static_cast<size_t>(rows) * width_;
    CD_CUDA_CHECK(cudaMemcpyAsync(s.host_break_index.get(),
                                  s.dev_break_index.get(), n * sizeof(int16_t),
                                  cudaMemcpyDeviceToHost, stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.host_magnitude.get(), s.dev_magnitude.get(),
                                  n * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.host_status.get(), s.dev_status.get(),
                                  n * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                                  stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.host_rmse.get(), s.dev_rmse.get(),
                                  n * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.host_n_train.get(), s.dev_n_train.get(),
                                  n * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                                  stream));
    if (save_coef_) {
      CD_CUDA_CHECK(
          cudaMemcpyAsync(s.host_coef.get(), s.dev_coef.get(),
                          n * static_cast<size_t>(num_params_) * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    }
    return Status::Ok();
  }

  // Scatters the tile's pinned output rows into the full-image maps.
  Status Collect(int slot, int row0, int rows) override {
    const DetectionSlot& s = slots_[static_cast<size_t>(slot)];
    const size_t n = static_cast<size_t>(rows) * width_;
    const size_t offset = static_cast<size_t>(row0) * width_;
    std::memcpy(result_->break_index.data() + offset, s.host_break_index.get(),
                n * sizeof(int16_t));
    std::memcpy(result_->magnitude.data() + offset, s.host_magnitude.get(),
                n * sizeof(float));
    std::memcpy(result_->status.data() + offset, s.host_status.get(),
                n * sizeof(uint8_t));
    std::memcpy(result_->rmse.data() + offset, s.host_rmse.get(),
                n * sizeof(float));
    std::memcpy(result_->train_obs.data() + offset, s.host_n_train.get(),
                n * sizeof(uint16_t));
    if (save_coef_) {
      // The tile's coefficients are coefficient-major with stride n; the
      // result is coefficient-major with stride height * width.
      const size_t image_pixels = static_cast<size_t>(height_) * width_;
      for (int c = 0; c < num_params_; ++c) {
        std::memcpy(result_->coefficients.data() + c * image_pixels + offset,
                    s.host_coef.get() + static_cast<size_t>(c) * n,
                    n * sizeof(float));
      }
    }
    return Status::Ok();
  }

 private:
  size_t Observations(int rows) const {
    return static_cast<size_t>(rows) * width_ * static_cast<size_t>(num_dates_);
  }

  CubeSource* source_;
  const DateTable& dates_;
  const ModelParams& params_;
  bool fused_;
  bool save_coef_;
  int block_size_;
  int num_params_;
  int width_;
  int height_;
  int num_dates_;
  std::vector<DetectionSlot> slots_;
  DetectionResult* result_;
  const GpuValidationRequest* validation_;  // null: no validation
  int validate_stage_;                      // TimedStage index of the kernel
  DeviceBuffer<int16_t> dev_year_;          // calendar year of each date
  DeviceBuffer<uint64_t> dev_counters_;     // kNumConfusionCounters totals
};

// ---------------------------------------------------------------------------
// Quick-diff pipeline
// ---------------------------------------------------------------------------

// Buffers of one slot of the quick-diff pipeline (two date planes in, diff and
// flag out).
struct QuickDiffSlot {
  HostBuffer<uint16_t> host_red;
  HostBuffer<uint16_t> host_nir;
  HostBuffer<uint8_t> host_scl;
  HostBuffer<float> host_diff;
  HostBuffer<uint8_t> host_flag;
  DeviceBuffer<uint16_t> dev_red;
  DeviceBuffer<uint16_t> dev_nir;
  DeviceBuffer<uint8_t> dev_scl;
  DeviceBuffer<float> dev_diff;
  DeviceBuffer<uint8_t> dev_flag;
};

// Device bytes per pixel of quick-diff: two planes of red, nir and scl in,
// float diff and byte flag out.
constexpr size_t kQuickDiffBytesPerPixel = 2 * 5 + 5;

// The tile stages of the quick two-date difference.
class QuickDiffStages : public TileStages {
 public:
  QuickDiffStages(CubeSource* source, int date_a, int date_b, float threshold,
                  const ModelParams& params, const RunConfig& config,
                  int num_slots, QuickDiffResult* out)
      : source_(source),
        dates_{date_a, date_b},
        threshold_(threshold),
        params_(params),
        block_size_(config.block_size),
        width_(source->info().width),
        slots_(static_cast<size_t>(num_slots)),
        out_(out) {
    const DateTable& table = source->info().dates;
    for (int i = 0; i < 2; ++i) {
      scale_[i] = table.scale[static_cast<size_t>(dates_[i])];
      offset_[i] = table.offset[static_cast<size_t>(dates_[i])];
    }
  }

  cudaError_t Allocate(int tile_rows, SlotAllocator* a) override {
    const size_t pixels = static_cast<size_t>(tile_rows) * width_;
    const size_t observations = 2 * pixels;
    for (QuickDiffSlot& s : slots_) {
      const bool ok =
          a->Host(&s.host_red, observations) &&
          a->Host(&s.host_nir, observations) &&
          a->Host(&s.host_scl, observations) && a->Host(&s.host_diff, pixels) &&
          a->Host(&s.host_flag, pixels) &&
          a->Device(&s.dev_red, observations) &&
          a->Device(&s.dev_nir, observations) &&
          a->Device(&s.dev_scl, observations) &&
          a->Device(&s.dev_diff, pixels) && a->Device(&s.dev_flag, pixels);
      if (!ok) {
        return a->error();
      }
    }
    return cudaSuccess;
  }

  void Release() override {
    for (QuickDiffSlot& slot : slots_) {
      slot = QuickDiffSlot();
    }
  }

  Status Read(int slot, int row0, int rows) override {
    QuickDiffSlot& s = slots_[static_cast<size_t>(slot)];
    TileView view;
    view.red = s.host_red.get();
    view.nir = s.host_nir.get();
    view.scl = s.host_scl.get();
    view.row0 = row0;
    view.rows = rows;
    return source_->ReadTileDates(row0, rows, dates_, 2, &view);
  }

  Status EnqueueH2D(int slot, int rows, cudaStream_t stream) override {
    QuickDiffSlot& s = slots_[static_cast<size_t>(slot)];
    const size_t n = 2 * static_cast<size_t>(rows) * width_;
    CD_CUDA_CHECK(cudaMemcpyAsync(s.dev_red.get(), s.host_red.get(),
                                  n * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                  stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.dev_nir.get(), s.host_nir.get(),
                                  n * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                  stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.dev_scl.get(), s.host_scl.get(),
                                  n * sizeof(uint8_t), cudaMemcpyHostToDevice,
                                  stream));
    return Status::Ok();
  }

  Status EnqueueCompute(int slot, int rows, cudaStream_t stream,
                        SlotEvents* events) override {
    QuickDiffSlot& s = slots_[static_cast<size_t>(slot)];
    LaunchConfig launch;
    launch.block_size = block_size_;
    launch.stream = stream;
    RawBands bands;
    bands.red = s.dev_red.get();
    bands.nir = s.dev_nir.get();
    bands.scl = s.dev_scl.get();
    return TimedStage(events, 0, stream, [&] {
      return LaunchQuickDiff(bands, static_cast<int64_t>(rows) * width_, scale_,
                             offset_, params_.valid_class_mask, threshold_,
                             params_.losses_only, s.dev_diff.get(),
                             s.dev_flag.get(), launch);
    });
  }

  Status EnqueueD2H(int slot, int rows, cudaStream_t stream) override {
    QuickDiffSlot& s = slots_[static_cast<size_t>(slot)];
    const size_t n = static_cast<size_t>(rows) * width_;
    CD_CUDA_CHECK(cudaMemcpyAsync(s.host_diff.get(), s.dev_diff.get(),
                                  n * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream));
    CD_CUDA_CHECK(cudaMemcpyAsync(s.host_flag.get(), s.dev_flag.get(),
                                  n * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                                  stream));
    return Status::Ok();
  }

  Status Collect(int slot, int row0, int rows) override {
    const QuickDiffSlot& s = slots_[static_cast<size_t>(slot)];
    const size_t n = static_cast<size_t>(rows) * width_;
    const size_t offset = static_cast<size_t>(row0) * width_;
    std::memcpy(out_->diff.data() + offset, s.host_diff.get(),
                n * sizeof(float));
    std::memcpy(out_->flag.data() + offset, s.host_flag.get(),
                n * sizeof(uint8_t));
    return Status::Ok();
  }

 private:
  CubeSource* source_;
  int dates_[2];
  float scale_[2];
  float offset_[2];
  float threshold_;
  const ModelParams& params_;
  int block_size_;
  int width_;
  std::vector<QuickDiffSlot> slots_;
  QuickDiffResult* out_;
};

// ---------------------------------------------------------------------------
// Post-processing helpers
// ---------------------------------------------------------------------------

// Rows per post-processing chunk: half of the free device memory, at most the
// whole image (or the testing limit), at least one row.
Status ChunkRows(int height, int width, size_t bytes_per_pixel, int* rows) {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CD_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const size_t row_bytes = static_cast<size_t>(width) * bytes_per_pixel;
  int64_t chunk = static_cast<int64_t>(free_bytes / 2 / row_bytes);
  chunk = std::min<int64_t>(chunk, height);
  if (g_postprocess_chunk_rows > 0) {
    chunk = std::min<int64_t>(chunk, g_postprocess_chunk_rows);
  }
  *rows = static_cast<int>(std::max<int64_t>(chunk, 1));
  return Status::Ok();
}

// Allocates `count` elements, converting failure to a Status.
template <typename T>
Status AllocateDevice(DeviceBuffer<T>* buffer, size_t count) {
  const cudaError_t error = buffer->Allocate(count);
  return CudaStatus(error, "cudaMalloc (post-processing buffer)", __FILE__,
                    __LINE__);
}

// True when every per-pixel map of `r` has height * width entries.
bool MapsAreConsistent(const DetectionResult& r) {
  const size_t pixels =
      static_cast<size_t>(r.height) * static_cast<size_t>(r.width);
  return r.height >= 1 && r.width >= 1 && r.break_index.size() == pixels &&
         r.magnitude.size() == pixels && r.status.size() == pixels;
}

}  // namespace

void SetSlotAllocationLimitForTesting(size_t bytes) {
  g_slot_allocation_limit = bytes;
}

void SetPostprocessChunkRowsForTesting(int rows) {
  g_postprocess_chunk_rows = rows;
}

Status RunGpuDetection(CubeSource* source, const ModelParams& params,
                       const RunConfig& config, DetectionResult* result,
                       StageTimings* timings, RunStats* stats,
                       const GpuValidationRequest* validation,
                       ConfusionCounts* counts) {
  if (source == nullptr || result == nullptr) {
    return Status::Error("RunGpuDetection: null source or result");
  }
  if (!IsGpuBackend(config.backend)) {
    return Status::Error("RunGpuDetection: backend '" +
                         BackendName(config.backend) +
                         "' is not a GPU backend");
  }
  CD_RETURN_IF_ERROR(CheckRunConfig(config));
  const CubeInfo& info = source->info();
  DateTable dates;
  CD_RETURN_IF_ERROR(PrepareDates(info, params, &dates));
  if (validation != nullptr) {
    const size_t pixels =
        static_cast<size_t>(info.height) * static_cast<size_t>(info.width);
    if (counts == nullptr || validation->reference == nullptr ||
        validation->year == nullptr) {
      return Status::Error(
          "RunGpuDetection: a validation request needs a reference, a year "
          "table and a counts output");
    }
    if (validation->reference->size() != pixels) {
      return Status::Error("RunGpuDetection: reference has " +
                           std::to_string(validation->reference->size()) +
                           " pixels but the cube has " +
                           std::to_string(pixels));
    }
    if (validation->year->size() != static_cast<size_t>(info.num_dates)) {
      return Status::Error("RunGpuDetection: year table has " +
                           std::to_string(validation->year->size()) +
                           " entries for " + std::to_string(info.num_dates) +
                           " dates");
    }
  }
  CD_RETURN_IF_ERROR(SelectDevice(config.device));

  // From here on the clock runs: setup, allocation, all tiles.
  const WallTimer total_timer;
  CD_RETURN_IF_ERROR(UploadDateTables(dates, params));

  const bool fused = config.backend == Backend::kGpuFused;
  const int max_slots = config.streams == 3 ? 3 : 1;
  const int num_params = NumParams(params.harmonics);
  const size_t bytes_per_pixel =
      DetectionBytesPerPixel(info.num_dates, num_params, fused,
                             config.save_coefficients, validation != nullptr);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CD_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const TilePlan plan =
      PlanTiles(info.height, info.width, info.num_dates, config.tile_rows,
                free_bytes, max_slots, bytes_per_pixel, kMaxTileInputBytes);
  int tile_rows = plan.tile_rows;
  // No more slots than tiles: a cube with one tile needs one slot.
  const int num_slots = std::min(max_slots, plan.num_tiles);

  RunStats local_stats;
  RunStats* run_stats = stats != nullptr ? stats : &local_stats;
  run_stats->notes.clear();
  NoteTileRowsReduced(plan, config.tile_rows, max_slots, bytes_per_pixel,
                      info.width, free_bytes, run_stats);
  AllocateResult(info.height, info.width, num_params, config.save_coefficients,
                 result);
  DetectionStages stages(source, dates, params, config, fused, num_slots,
                         result, validation);
  SlotAllocator allocator;
  const WallTimer allocation_timer;
  CD_RETURN_IF_ERROR(
      AllocateWithRetry(&stages, &tile_rows, &allocator, run_stats));
  CD_RETURN_IF_ERROR(stages.InitValidation());
  const double allocation_ms = allocation_timer.ElapsedMs();
  WarnIfPageable(allocator, run_stats);
  size_t free_after = 0;
  CD_CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
  // Setup ends here: tables uploaded and all buffers allocated, before the
  // first tile is read.
  const double setup_ms = total_timer.ElapsedMs();

  PipelineTimes times;
  std::vector<TraceRow> trace;
  std::vector<std::string> stage_names =
      fused ? std::vector<std::string>{"fused"}
            : std::vector<std::string>{"ndvi", "fit", "detect"};
  if (validation != nullptr) {
    stage_names.push_back("validate");
  }
  TilePipeline pipeline(&stages, info.height, tile_rows, num_slots, &times,
                        config.trace_csv.empty() ? nullptr : &trace,
                        stage_names);
  CD_RETURN_IF_ERROR(pipeline.Run());
  if (validation != nullptr) {
    CD_RETURN_IF_ERROR(stages.ReadCounts(counts));
  }
  // Releasing the pinned and device buffers is part of the run: timed here so
  // that total_ms is not flattered by deferring the frees to the destructor.
  stages.Release();
  const double total_ms = total_timer.ElapsedMs();
  if (!config.trace_csv.empty()) {
    // Written after total_ms is read: the file is not part of the timed run.
    CD_RETURN_IF_ERROR(WriteTraceCsv(config.trace_csv, trace));
  }

  const int num_tiles =
      static_cast<int>(TileRanges(info.height, tile_rows).size());
  const double allocated_mb =
      static_cast<double>(allocator.device_bytes()) / kMiB;
  run_stats->num_tiles = num_tiles;
  run_stats->tile_rows_used = tile_rows;
  run_stats->threads_used = 0;
  run_stats->streams_used = config.streams;
  run_stats->device_mem_alloc_mb = allocated_mb;
  run_stats->device_mem_peak_mb = allocated_mb;
  {
    GpuInfo gpu;
    run_stats->device_name =
        GpuDeviceInfo(config.device, &gpu).ok() ? gpu.name : "unknown GPU";
  }
  if (!config.quiet) {
    // Free memory is sampled before and after the slot allocation; other
    // processes may change it in between, hence the signed difference.
    const double used_delta_mb =
        (static_cast<double>(free_bytes) - static_cast<double>(free_after)) /
        kMiB;
    // Printed by the caller under "execution" (RunStats::notes).
    run_stats->notes.push_back(
        "gpu_memory: allocated " +
        std::to_string(static_cast<int>(allocated_mb)) + " MiB in " +
        std::to_string(static_cast<int>(allocation_ms)) +
        " ms (pinned host buffers included); cudaMemGetInfo used delta " +
        std::to_string(static_cast<int>(used_delta_mb)) + " MiB");
  }

  if (timings != nullptr) {
    // Every timed kernel stage, including the validation kernel.
    double kernel_ms = 0.0;
    for (int i = 0; i < kMaxStages; ++i) {
      kernel_ms += times.stage_ms[i];
    }
    timings->read_ms = times.read_ms;
    timings->read_wait_ms = -1;
    timings->setup_ms = setup_ms;
    timings->h2d_ms = times.h2d_ms;
    timings->d2h_ms = times.d2h_ms;
    timings->kernel_ms = kernel_ms;
    timings->total_ms = total_ms;
    if (fused) {
      timings->fused_ms = times.stage_ms[0];
    } else {
      timings->ndvi_ms = times.stage_ms[0];
      timings->fit_ms = times.stage_ms[1];
      timings->detect_ms = times.stage_ms[2];
    }
  }
  return Status::Ok();
}

Status GpuQuickDiff(CubeSource* source, int date_a, int date_b,
                    float diff_threshold, const ModelParams& p,
                    const RunConfig& c, QuickDiffResult* out, StageTimings* t) {
  if (source == nullptr || out == nullptr) {
    return Status::Error("GpuQuickDiff: null source or output");
  }
  const CubeInfo& info = source->info();
  if (date_a < 0 || date_a >= info.num_dates || date_b < 0 ||
      date_b >= info.num_dates) {
    return Status::Error("quick-diff dates " + std::to_string(date_a) +
                         " and " + std::to_string(date_b) + " must be in 0.." +
                         std::to_string(info.num_dates - 1));
  }
  const size_t num_dates = static_cast<size_t>(info.num_dates);
  if (info.dates.scale.size() != num_dates ||
      info.dates.offset.size() != num_dates) {
    return Status::Error(
        "quick-diff: calibration arrays do not match the number of dates");
  }
  if (info.height < 1 || info.width < 1) {
    return Status::Error("cube has an empty dimension");
  }
  CD_RETURN_IF_ERROR(CheckRunConfig(c));
  CD_RETURN_IF_ERROR(SelectDevice(c.device));

  const WallTimer total_timer;
  const int max_slots = c.streams == 3 ? 3 : 1;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CD_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const TilePlan plan =
      PlanTiles(info.height, info.width, 2, c.tile_rows, free_bytes, max_slots,
                kQuickDiffBytesPerPixel, kMaxTileInputBytes);
  int tile_rows = plan.tile_rows;
  const int num_slots = std::min(max_slots, plan.num_tiles);

  const size_t pixels =
      static_cast<size_t>(info.height) * static_cast<size_t>(info.width);
  out->height = info.height;
  out->width = info.width;
  out->diff.assign(pixels, 0.0f);
  out->flag.assign(pixels, 0);

  QuickDiffStages stages(source, date_a, date_b, diff_threshold, p, c,
                         num_slots, out);
  RunStats stats;
  NoteTileRowsReduced(plan, c.tile_rows, max_slots, kQuickDiffBytesPerPixel,
                      info.width, free_bytes, &stats);
  SlotAllocator allocator;
  CD_RETURN_IF_ERROR(
      AllocateWithRetry(&stages, &tile_rows, &allocator, &stats));
  WarnIfPageable(allocator, &stats);
  PipelineTimes times;
  TilePipeline pipeline(&stages, info.height, tile_rows, num_slots, &times);
  CD_RETURN_IF_ERROR(pipeline.Run());
  stages.Release();  // counted in total_ms, like the detection run

  if (t != nullptr) {
    t->read_ms = times.read_ms;
    t->h2d_ms = times.h2d_ms;
    t->d2h_ms = times.d2h_ms;
    t->kernel_ms = times.stage_ms[0];
    t->total_ms = total_timer.ElapsedMs();
  }
  return Status::Ok();
}

Status GpuColorizeChangeMap(const DetectionResult& r, const DateTable& d,
                            int device, std::vector<uint8_t>* rgba,
                            double* ms) {
  if (rgba == nullptr) {
    return Status::Error("GpuColorizeChangeMap: null output");
  }
  if (!MapsAreConsistent(r)) {
    return Status::Error("GpuColorizeChangeMap: inconsistent result maps");
  }
  if (d.day.empty() || d.num_train_dates < 0 ||
      static_cast<size_t>(d.num_train_dates) >= d.day.size()) {
    return Status::Error(
        "GpuColorizeChangeMap: date table has no monitoring dates");
  }
  CD_RETURN_IF_ERROR(SelectDevice(device));

  const WallTimer timer;
  const int first_monitor_day = d.day[static_cast<size_t>(d.num_train_dates)];
  const int last_day = d.day.back();
  const int num_dates = static_cast<int>(d.day.size());
  const int width = r.width;
  rgba->assign(static_cast<size_t>(r.height) * width * 4, 0);

  int chunk_rows = 0;
  CD_RETURN_IF_ERROR(ChunkRows(r.height, width, 11, &chunk_rows));
  const size_t max_pixels = static_cast<size_t>(chunk_rows) * width;
  DeviceBuffer<int32_t> dev_day;
  DeviceBuffer<int16_t> dev_break;
  DeviceBuffer<float> dev_magnitude;
  DeviceBuffer<uint8_t> dev_status;
  DeviceBuffer<uint8_t> dev_rgba;
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_day, d.day.size()));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_break, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_magnitude, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_status, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_rgba, max_pixels * 4));
  CD_CUDA_CHECK(cudaMemcpy(dev_day.get(), d.day.data(),
                           d.day.size() * sizeof(int32_t),
                           cudaMemcpyHostToDevice));
  for (int row0 = 0; row0 < r.height; row0 += chunk_rows) {
    const int rows = std::min(chunk_rows, r.height - row0);
    const size_t n = static_cast<size_t>(rows) * width;
    const size_t offset = static_cast<size_t>(row0) * width;
    CD_CUDA_CHECK(cudaMemcpy(dev_break.get(), r.break_index.data() + offset,
                             n * sizeof(int16_t), cudaMemcpyHostToDevice));
    CD_CUDA_CHECK(cudaMemcpy(dev_magnitude.get(), r.magnitude.data() + offset,
                             n * sizeof(float), cudaMemcpyHostToDevice));
    CD_CUDA_CHECK(cudaMemcpy(dev_status.get(), r.status.data() + offset,
                             n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    LaunchConfig launch;
    CD_RETURN_IF_ERROR(LaunchColorizeChangeMap(
        dev_break.get(), dev_magnitude.get(), dev_status.get(), dev_day.get(),
        num_dates, first_monitor_day, last_day, static_cast<int64_t>(n),
        dev_rgba.get(), launch));
    CD_CUDA_CHECK(cudaMemcpy(rgba->data() + offset * 4, dev_rgba.get(), n * 4,
                             cudaMemcpyDeviceToHost));
  }
  if (ms != nullptr) {
    *ms = timer.ElapsedMs();
  }
  return Status::Ok();
}

Status GpuCleanup(DetectionResult* r, int device, double* ms) {
  if (r == nullptr) {
    return Status::Error("GpuCleanup: null result");
  }
  if (!MapsAreConsistent(*r)) {
    return Status::Error("GpuCleanup: inconsistent result maps");
  }
  CD_RETURN_IF_ERROR(SelectDevice(device));

  const WallTimer timer;
  const int width = r->width;
  const int height = r->height;
  int chunk_rows = 0;
  CD_RETURN_IF_ERROR(ChunkRows(height, width, 8, &chunk_rows));
  const size_t max_pixels = static_cast<size_t>(chunk_rows) * width;
  DeviceBuffer<uint8_t> dev_halo;
  DeviceBuffer<uint8_t> dev_status_out;
  DeviceBuffer<int16_t> dev_break;
  DeviceBuffer<float> dev_magnitude;
  CD_RETURN_IF_ERROR(
      AllocateDevice(&dev_halo, max_pixels + 2 * static_cast<size_t>(width)));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_status_out, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_break, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_magnitude, max_pixels));

  // The rule is evaluated on the unmodified map, but the chunks are written
  // back as they finish: the first row of the next chunk's upper halo must
  // therefore be the original last row of this chunk, saved before write-back.
  std::vector<uint8_t> original_above;
  for (int row0 = 0; row0 < height; row0 += chunk_rows) {
    const int rows = std::min(chunk_rows, height - row0);
    const size_t n = static_cast<size_t>(rows) * width;
    const size_t offset = static_cast<size_t>(row0) * width;
    uint8_t* halo = dev_halo.get();
    // Row above the chunk (zeros outside the image).
    if (row0 > 0) {
      CD_CUDA_CHECK(cudaMemcpy(halo, original_above.data(), width,
                               cudaMemcpyHostToDevice));
    } else {
      CD_CUDA_CHECK(cudaMemset(halo, 0, width));
    }
    // The chunk itself.
    CD_CUDA_CHECK(cudaMemcpy(halo + width, r->status.data() + offset, n,
                             cudaMemcpyHostToDevice));
    // Row below the chunk (still unmodified on the host).
    if (row0 + rows < height) {
      CD_CUDA_CHECK(cudaMemcpy(halo + width + n, r->status.data() + offset + n,
                               width, cudaMemcpyHostToDevice));
    } else {
      CD_CUDA_CHECK(cudaMemset(halo + width + n, 0, width));
    }
    original_above.assign(r->status.begin() + (offset + n - width),
                          r->status.begin() + (offset + n));
    CD_CUDA_CHECK(cudaMemcpy(dev_break.get(), r->break_index.data() + offset,
                             n * sizeof(int16_t), cudaMemcpyHostToDevice));
    CD_CUDA_CHECK(cudaMemcpy(dev_magnitude.get(), r->magnitude.data() + offset,
                             n * sizeof(float), cudaMemcpyHostToDevice));
    CD_RETURN_IF_ERROR(LaunchCleanup(halo, rows, width, dev_status_out.get(),
                                     dev_break.get(), dev_magnitude.get(),
                                     nullptr));
    CD_CUDA_CHECK(cudaMemcpy(r->status.data() + offset, dev_status_out.get(), n,
                             cudaMemcpyDeviceToHost));
    CD_CUDA_CHECK(cudaMemcpy(r->break_index.data() + offset, dev_break.get(),
                             n * sizeof(int16_t), cudaMemcpyDeviceToHost));
    CD_CUDA_CHECK(cudaMemcpy(r->magnitude.data() + offset, dev_magnitude.get(),
                             n * sizeof(float), cudaMemcpyDeviceToHost));
  }
  if (ms != nullptr) {
    *ms = timer.ElapsedMs();
  }
  return Status::Ok();
}

Status GpuConfusion(const DetectionResult& r,
                    const std::vector<uint8_t>& reference, const DateTable& d,
                    const ValidationWindow& w, int device,
                    ConfusionCounts* counts, double* ms) {
  if (counts == nullptr) {
    return Status::Error("GpuConfusion: null output");
  }
  if (!MapsAreConsistent(r)) {
    return Status::Error("GpuConfusion: inconsistent result maps");
  }
  if (reference.size() != r.break_index.size()) {
    return Status::Error(
        "GpuConfusion: reference has " + std::to_string(reference.size()) +
        " pixels but the result has " + std::to_string(r.break_index.size()));
  }
  if (d.year.empty() || d.year.size() != d.day.size()) {
    return Status::Error("GpuConfusion: date table has no calendar years");
  }
  CD_RETURN_IF_ERROR(SelectDevice(device));

  const WallTimer timer;
  const int width = r.width;
  int chunk_rows = 0;
  CD_RETURN_IF_ERROR(ChunkRows(r.height, width, 8, &chunk_rows));
  const size_t max_pixels = static_cast<size_t>(chunk_rows) * width;
  DeviceBuffer<uint8_t> dev_status;
  DeviceBuffer<int16_t> dev_break;
  DeviceBuffer<float> dev_magnitude;
  DeviceBuffer<uint8_t> dev_reference;
  DeviceBuffer<int16_t> dev_year;
  DeviceBuffer<uint64_t> dev_counters;
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_status, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_break, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_magnitude, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_reference, max_pixels));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_year, d.year.size()));
  CD_RETURN_IF_ERROR(AllocateDevice(&dev_counters, kNumConfusionCounters));
  CD_CUDA_CHECK(cudaMemcpy(dev_year.get(), d.year.data(),
                           d.year.size() * sizeof(int16_t),
                           cudaMemcpyHostToDevice));
  CD_CUDA_CHECK(cudaMemset(dev_counters.get(), 0,
                           kNumConfusionCounters * sizeof(uint64_t)));
  for (int row0 = 0; row0 < r.height; row0 += chunk_rows) {
    const int rows = std::min(chunk_rows, r.height - row0);
    const size_t n = static_cast<size_t>(rows) * width;
    const size_t offset = static_cast<size_t>(row0) * width;
    CD_CUDA_CHECK(cudaMemcpy(dev_status.get(), r.status.data() + offset, n,
                             cudaMemcpyHostToDevice));
    CD_CUDA_CHECK(cudaMemcpy(dev_break.get(), r.break_index.data() + offset,
                             n * sizeof(int16_t), cudaMemcpyHostToDevice));
    CD_CUDA_CHECK(cudaMemcpy(dev_magnitude.get(), r.magnitude.data() + offset,
                             n * sizeof(float), cudaMemcpyHostToDevice));
    CD_CUDA_CHECK(cudaMemcpy(dev_reference.get(), reference.data() + offset, n,
                             cudaMemcpyHostToDevice));
    CD_RETURN_IF_ERROR(LaunchConfusion(
        dev_status.get(), dev_break.get(), dev_magnitude.get(),
        dev_reference.get(), dev_year.get(), static_cast<int>(d.year.size()), w,
        static_cast<int64_t>(n), dev_counters.get(), nullptr));
  }
  uint64_t totals[kNumConfusionCounters] = {0};
  CD_CUDA_CHECK(cudaMemcpy(totals, dev_counters.get(), sizeof(totals),
                           cudaMemcpyDeviceToHost));
  counts->tp = totals[0];
  counts->fp = totals[1];
  counts->fn = totals[2];
  counts->tn = totals[3];
  counts->excluded = totals[4];
  counts->year_exact = totals[5];
  counts->year_within1 = totals[6];
  if (ms != nullptr) {
    *ms = timer.ElapsedMs();
  }
  return Status::Ok();
}

}  // namespace cd
