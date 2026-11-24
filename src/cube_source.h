// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Input interface: a satellite time-series cube that delivers row-band tiles.
// Implemented by the raw-file reader and the in-memory synthetic cube.
#ifndef SRC_CUBE_SOURCE_H_
#define SRC_CUBE_SOURCE_H_

#include <cstdint>
#include <string>

#include "status.h"
#include "types.h"

namespace cd {

// Metadata of a cube, as read from manifest.txt (format: docs/FORMATS.md).
struct CubeInfo {
  std::string name;
  std::string dir;
  std::string source;
  std::string license;
  std::string crs;
  std::string reference_file;  // absolute path after ReadManifest, or empty
  std::string reference_source;
  double geotransform[6] = {0};
  bool has_geo = false;
  int height = 0;
  int width = 0;
  int num_dates = 0;
  std::string red_file;  // absolute paths after ReadManifest
  std::string nir_file;
  std::string scl_file;
  DateTable dates;  // iso, day, scale, offset from the manifest
};

// Caller-owned memory for one tile, date-major within the tile.
struct TileView {
  uint16_t* red = nullptr;  // each buffer holds n_dates * rows * W elements
  uint16_t* nir = nullptr;
  uint8_t* scl = nullptr;
  int row0 = 0;
  int rows = 0;
};

// A cube that can be read tile by tile. Thread-compatible: only one reader
// thread may use an instance at a time.
class CubeSource {
 public:
  virtual ~CubeSource() = default;

  virtual const CubeInfo& info() const = 0;

  // Reads all T dates of rows [row0, row0 + rows): plane t starts at offset
  // t * rows * W in each buffer.
  virtual Status ReadTile(int row0, int rows, TileView* out) = 0;

  // Reads only the listed dates (quick-diff, figures): plane i holds
  // dates[i], n entries.
  virtual Status ReadTileDates(int row0, int rows, const int* dates, int n,
                               TileView* out) = 0;
};

}  // namespace cd

#endif  // SRC_CUBE_SOURCE_H_
