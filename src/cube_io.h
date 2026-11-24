// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Cube input/output: manifest reader and writer, raw-file tile reader,
// in-memory cube for synthetic runs and small raw-file helpers.
// Tile planning lives in tile_scheduler.h.
#ifndef SRC_CUBE_IO_H_
#define SRC_CUBE_IO_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cube_source.h"
#include "status.h"
#include "synthetic_cube.h"

namespace cd {

// Reads and validates a manifest (grammar: docs/FORMATS.md, "Input cube").
// File paths in `info` are made absolute (relative to the manifest's
// directory) and `info.dir` is that directory. Errors name the manifest and
// the line.
// Unknown keys are logged as warnings and ignored. tau, year and
// num_train_dates of info->dates are left empty (BuildDateTable fills them).
Status ReadManifest(const std::string& path, CubeInfo* info);

// Writes `info` in the manifest grammar. File names inside the manifest's
// directory are written relative to it; empty file names default to red.u16,
// nir.u16 and scl.u8. The per-date valid_fraction is unknown here and written
// as -1, the scene id as "-"; sha256 lines are not written.
Status WriteManifest(const std::string& path, const CubeInfo& info);

// Same as WriteManifest with a known valid fraction (0..1) per date and an
// optional scene id for every date. `valid_fraction` must have num_dates
// entries (or be empty for "unknown"); `scene_id` must be empty or a single
// word without spaces.
Status WriteManifestWithDateExtras(const std::string& path,
                                   const CubeInfo& info,
                                   const std::vector<double>& valid_fraction,
                                   const std::string& scene_id);

// Opens the three raw files of `info` (sizes are re-checked). Tiles are read
// with one pread per date and band. With io_threads > 1 these reads are
// spread over a persistent pool of that many threads (owned by the source and
// joined when it is destroyed); with 1 they run one after another on the
// calling thread. The result bytes do not depend on io_threads. Returns null
// and sets *status on error.
std::unique_ptr<CubeSource> OpenRawCube(const CubeInfo& info, int io_threads,
                                        Status* status);

// Same with io_threads = 1.
std::unique_ptr<CubeSource> OpenRawCube(const CubeInfo& info, Status* status);

// A cube served from memory (same ReadTile semantics as the raw reader).
std::unique_ptr<CubeSource> MakeMemoryCube(SyntheticCube cube);

// Reads the optional reference raster (H x W, uint8) named by
// info.reference_file.
Status ReadReferenceRaster(const CubeInfo& info,
                           std::vector<uint8_t>* reference);

// Reads exactly `bytes` bytes of the file at `path` into `out`; the file must
// have exactly that size.
Status ReadRawBytes(const std::string& path, size_t bytes, void* out);

// Reads a whole raw array of `count` elements; the file must have exactly
// count * sizeof(T) bytes.
template <typename T>
Status ReadRawArray(const std::string& path, size_t count,
                    std::vector<T>* out) {
  std::vector<T> data(count);
  const Status status = ReadRawBytes(path, count * sizeof(T), data.data());
  if (!status.ok()) {
    return status;
  }
  *out = std::move(data);
  return Status::Ok();
}

// Writes `bytes` bytes to `path` (truncating).
Status WriteRawFile(const std::string& path, const void* data, size_t bytes);

}  // namespace cd

#endif  // SRC_CUBE_IO_H_
