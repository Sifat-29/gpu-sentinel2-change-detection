// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): portable paths and mkdir -p
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "cube_io.h"
#include "cube_source.h"
#include "status.h"
#include "synthetic_cube.h"
#include "test_framework.h"
#include "types.h"

namespace cd {
namespace {

namespace fs = std::filesystem;

// A unique temporary directory removed on destruction.
class TempDir {
 public:
  TempDir() {
    std::string pattern =
        (fs::temp_directory_path() / "cd_test_cube_io_XXXXXX").string();
    const char* made = mkdtemp(pattern.data());
    path_ = made != nullptr ? made : "";
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir() {
    if (!path_.empty()) {
      std::error_code error;
      fs::remove_all(path_, error);
    }
  }
  const std::string& path() const { return path_; }
  std::string File(const std::string& name) const {
    return (fs::path(path_) / name).string();
  }

 private:
  std::string path_;
};

// Small deterministic cube: element (t, r, c) has a value derived from its
// position so any misplaced read is detected.
struct TestCube {
  int height = 13;
  int width = 7;
  int num_dates = 5;
  std::vector<uint16_t> red;
  std::vector<uint16_t> nir;
  std::vector<uint8_t> scl;
  CubeInfo info;
};

TestCube MakeTestCube() {
  TestCube cube;
  const size_t cells =
      static_cast<size_t>(cube.height) * cube.width * cube.num_dates;
  for (size_t i = 0; i < cells; ++i) {
    cube.red.push_back(static_cast<uint16_t>(i * 7 + 1));
    cube.nir.push_back(static_cast<uint16_t>(65535 - i * 3));
    cube.scl.push_back(static_cast<uint8_t>(i % 12));
  }
  CubeInfo& info = cube.info;
  info.name = "test cube";
  info.source = "unit test";
  info.license = "none";
  info.crs = "EPSG:32720";
  info.has_geo = true;
  const double geo[6] = {699960.5, 10, 0, 7600000.25, 0, -10};
  for (int i = 0; i < 6; ++i) {
    info.geotransform[i] = geo[i];
  }
  info.height = cube.height;
  info.width = cube.width;
  info.num_dates = cube.num_dates;
  for (int t = 0; t < cube.num_dates; ++t) {
    info.dates.iso.push_back("2019-0" + std::to_string(t + 1) + "-10");
    info.dates.day.push_back(17900 + 31 * t);
    info.dates.scale.push_back(0.0001f);
    info.dates.offset.push_back(t < 3 ? 0.0f : -0.1f);
  }
  return cube;
}

// Writes the cube's raw files and manifest into `dir`.
Status WriteTestCube(const TestCube& cube, const TempDir& dir, CubeInfo* info) {
  *info = cube.info;
  info->dir = dir.path();
  info->red_file = dir.File("red.u16");
  info->nir_file = dir.File("nir.u16");
  info->scl_file = dir.File("scl.u8");
  Status status =
      WriteRawFile(info->red_file, cube.red.data(), cube.red.size() * 2);
  if (!status.ok()) return status;
  status = WriteRawFile(info->nir_file, cube.nir.data(), cube.nir.size() * 2);
  if (!status.ok()) return status;
  status = WriteRawFile(info->scl_file, cube.scl.data(), cube.scl.size());
  if (!status.ok()) return status;
  return WriteManifest(dir.File("manifest.txt"), *info);
}

void WriteText(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

// A minimal valid manifest text for a 2 x 3 x 2 cube (files must exist).
std::string BaseManifest() {
  return "change_detect_cube 1\n"
         "name tiny\n"
         "height 2\n"
         "width 3\n"
         "num_dates 2\n"
         "red_file red.u16\n"
         "nir_file nir.u16\n"
         "scl_file scl.u8\n"
         "date 0 2020-01-01 18262 0.0001 0 0.9 scene_a\n"
         "date 1 2020-01-21 18282 0.0001 -0.1 0.8 scene_b\n";
}

// Creates correctly sized raw files for BaseManifest in `dir`.
void WriteTinyFiles(const TempDir& dir) {
  WriteText(dir.File("red.u16"), std::string(2 * 3 * 2 * 2, '\1'));
  WriteText(dir.File("nir.u16"), std::string(2 * 3 * 2 * 2, '\2'));
  WriteText(dir.File("scl.u8"), std::string(2 * 3 * 2, '\4'));
}

// Replaces the first occurrence of `from` in `text`.
std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const size_t at = text.find(from);
  if (at != std::string::npos) {
    text.replace(at, from.size(), to);
  }
  return text;
}

// Parses `text` as a manifest in a fresh directory with valid raw files and
// returns the status.
Status ParseText(const std::string& text, CubeInfo* info) {
  const TempDir dir;
  WriteTinyFiles(dir);
  WriteText(dir.File("manifest.txt"), text);
  return ReadManifest(dir.File("manifest.txt"), info);
}

void ExpectParseError(const std::string& text, const std::string& needle) {
  CubeInfo info;
  const Status status = ParseText(text, &info);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find(needle), std::string::npos);
  if (status.message().find(needle) == std::string::npos) {
    std::printf("  message was: %s\n", status.message().c_str());
  }
}

TEST(CubeIo, ManifestRoundTrip) {
  const TestCube cube = MakeTestCube();
  const TempDir dir;
  ASSERT_FALSE(dir.path().empty());
  CubeInfo written;
  ASSERT_TRUE(WriteTestCube(cube, dir, &written).ok());

  CubeInfo read;
  const Status status = ReadManifest(dir.File("manifest.txt"), &read);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(read.name, written.name);
  EXPECT_EQ(read.source, written.source);
  EXPECT_EQ(read.license, written.license);
  EXPECT_EQ(read.crs, written.crs);
  EXPECT_TRUE(read.has_geo);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(read.geotransform[i], written.geotransform[i]);
  }
  EXPECT_EQ(read.height, 13);
  EXPECT_EQ(read.width, 7);
  EXPECT_EQ(read.num_dates, 5);
  EXPECT_EQ(read.dir, dir.path());
  EXPECT_EQ(read.red_file, dir.File("red.u16"));
  EXPECT_EQ(read.nir_file, dir.File("nir.u16"));
  EXPECT_EQ(read.scl_file, dir.File("scl.u8"));
  EXPECT_TRUE(read.reference_file.empty());
  ASSERT_EQ(read.dates.day.size(), 5u);
  for (size_t t = 0; t < 5; ++t) {
    EXPECT_EQ(read.dates.iso[t], written.dates.iso[t]);
    EXPECT_EQ(read.dates.day[t], written.dates.day[t]);
    EXPECT_EQ(read.dates.scale[t], written.dates.scale[t]);
    EXPECT_EQ(read.dates.offset[t], written.dates.offset[t]);
  }

  // Writing what was read gives the same text (stable grammar).
  const TempDir second;
  CubeInfo moved = read;
  moved.red_file = second.File("red.u16");
  moved.nir_file = second.File("nir.u16");
  moved.scl_file = second.File("scl.u8");
  ASSERT_TRUE(WriteManifest(second.File("manifest.txt"), moved).ok());
  std::ifstream first_text(dir.File("manifest.txt"));
  std::ifstream second_text(second.File("manifest.txt"));
  const std::string a((std::istreambuf_iterator<char>(first_text)),
                      std::istreambuf_iterator<char>());
  const std::string b((std::istreambuf_iterator<char>(second_text)),
                      std::istreambuf_iterator<char>());
  EXPECT_EQ(a, b);
}

TEST(CubeIo, ManifestAcceptsCommentsCrlfAndUnknownKeys) {
  std::string text = BaseManifest();
  text =
      Replace(text, "name tiny\n", "# a comment\n\nname tiny  \nmystery 5\n");
  std::string crlf;
  for (const char c : text) {
    if (c == '\n') crlf += '\r';
    crlf += c;
  }
  CubeInfo info;
  const Status status = ParseText(crlf, &info);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(info.name, "tiny");
  EXPECT_EQ(info.num_dates, 2);
  // valid_fraction and scene id are optional.
  const std::string short_dates = Replace(BaseManifest(), " 0.9 scene_a", "");
  EXPECT_TRUE(ParseText(short_dates, &info).ok());
}

TEST(CubeIo, ManifestErrors) {
  const std::string good = BaseManifest();
  ExpectParseError(
      Replace(good, "change_detect_cube 1", "change_detect_cube 2"), ":1:");
  ExpectParseError(Replace(good, "name tiny\n", ""),
                   "missing required key "
                   "'name'");
  ExpectParseError(Replace(good, "scl_file scl.u8\n", ""), "'scl_file'");
  ExpectParseError(Replace(good, "height 2", "height 0"), ":3:");
  ExpectParseError(Replace(good, "width 3", "width x"), ":4:");
  ExpectParseError(Replace(good, "name tiny\n", "name tiny\nname again\n"),
                   "duplicate key");
  ExpectParseError(Replace(good, "num_dates 2", "num_dates 3"),
                   "found 2 date lines");
  ExpectParseError(Replace(good, "date 1 ", "date 2 "), ":10:");
  ExpectParseError(Replace(good, "18282", "18262"), "strictly increasing");
  ExpectParseError(Replace(good, "2020-01-21", "2020/01/21"), "malformed date");
  ExpectParseError(Replace(good, "0.0001 -0.1", "0 -0.1"), "scale");
  ExpectParseError(Replace(good,
                           "date 1 2020-01-21 18282 0.0001 -0.1 0.8 "
                           "scene_b",
                           "date 1 2020-01-21"),
                   "date line needs");
  ExpectParseError(Replace(good, "num_dates 2", "num_dates 1025"),
                   "exceeds the supported maximum");
  ExpectParseError(Replace(good, "red_file red.u16", "red_file gone.u16"),
                   "gone.u16");
  ExpectParseError(good + "reference_file ref.u8\n", "ref.u8");
  CubeInfo info;
  EXPECT_FALSE(ReadManifest("/nonexistent/dir/manifest.txt", &info).ok());
}

TEST(CubeIo, ManifestRejectsWrongFileSizes) {
  const TempDir dir;
  WriteTinyFiles(dir);
  WriteText(dir.File("red.u16"), std::string(23, '\1'));  // 24 expected
  WriteText(dir.File("manifest.txt"), BaseManifest());
  CubeInfo info;
  const Status status = ReadManifest(dir.File("manifest.txt"), &info);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("23 bytes, expected 24"), std::string::npos);

  WriteTinyFiles(dir);
  WriteText(dir.File("scl.u8"), std::string(13, '\4'));  // 12 expected
  EXPECT_FALSE(ReadManifest(dir.File("manifest.txt"), &info).ok());

  WriteTinyFiles(dir);
  WriteText(dir.File("ref.u8"), std::string(5, '\0'));  // 6 expected
  WriteText(dir.File("manifest.txt"),
            BaseManifest() + "reference_file ref.u8\n");
  EXPECT_FALSE(ReadManifest(dir.File("manifest.txt"), &info).ok());
  WriteText(dir.File("ref.u8"), std::string(6, '\7'));
  ASSERT_TRUE(ReadManifest(dir.File("manifest.txt"), &info).ok());
  std::vector<uint8_t> reference;
  ASSERT_TRUE(ReadReferenceRaster(info, &reference).ok());
  ASSERT_EQ(reference.size(), 6u);
  EXPECT_EQ(reference[5], 7);
}

TEST(CubeIo, RawArrayHelpers) {
  const TempDir dir;
  const std::vector<int16_t> values = {1, -2, 300, -4000};
  ASSERT_TRUE(
      WriteRawFile(dir.File("a.i16"), values.data(), values.size() * 2).ok());
  std::vector<int16_t> read;
  ASSERT_TRUE(ReadRawArray(dir.File("a.i16"), 4, &read).ok());
  EXPECT_TRUE(read == values);
  EXPECT_FALSE(ReadRawArray(dir.File("a.i16"), 5, &read).ok());
  EXPECT_FALSE(ReadRawArray(dir.File("missing.i16"), 4, &read).ok());
  EXPECT_FALSE(WriteRawFile("/nonexistent/dir/x", values.data(), 8).ok());
}

TEST(CubeIo, ReadTileMatchesRawSlices) {
  const TestCube cube = MakeTestCube();
  const TempDir dir;
  CubeInfo info;
  ASSERT_TRUE(WriteTestCube(cube, dir, &info).ok());
  CubeInfo from_manifest;
  ASSERT_TRUE(ReadManifest(dir.File("manifest.txt"), &from_manifest).ok());
  Status status = Status::Ok();
  std::unique_ptr<CubeSource> source = OpenRawCube(from_manifest, &status);
  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(source != nullptr);
  EXPECT_EQ(source->info().height, 13);

  const int starts[] = {0, 5, 10, 12, 0};
  const int counts[] = {5, 5, 3, 1, 13};  // 10 + 3 is the last partial tile
  for (int k = 0; k < 5; ++k) {
    const int row0 = starts[k];
    const int rows = counts[k];
    const size_t tile = static_cast<size_t>(rows) * cube.width;
    std::vector<uint16_t> red(tile * cube.num_dates);
    std::vector<uint16_t> nir(red.size());
    std::vector<uint8_t> scl(red.size());
    TileView view;
    view.red = red.data();
    view.nir = nir.data();
    view.scl = scl.data();
    ASSERT_TRUE(source->ReadTile(row0, rows, &view).ok());
    EXPECT_EQ(view.row0, row0);
    EXPECT_EQ(view.rows, rows);
    bool same = true;
    for (int t = 0; t < cube.num_dates; ++t) {
      for (size_t i = 0; i < tile; ++i) {
        const size_t source_cell =
            static_cast<size_t>(t) * cube.height * cube.width +
            static_cast<size_t>(row0) * cube.width + i;
        same = same && red[t * tile + i] == cube.red[source_cell] &&
               nir[t * tile + i] == cube.nir[source_cell] &&
               scl[t * tile + i] == cube.scl[source_cell];
      }
    }
    EXPECT_TRUE(same);
  }

  // A date subset: plane i holds dates[i].
  const int dates[] = {4, 1};
  std::vector<uint16_t> red(2 * 3 * cube.width);
  std::vector<uint16_t> nir(red.size());
  std::vector<uint8_t> scl(red.size());
  TileView view;
  view.red = red.data();
  view.nir = nir.data();
  view.scl = scl.data();
  ASSERT_TRUE(source->ReadTileDates(4, 3, dates, 2, &view).ok());
  const size_t tile = 3u * cube.width;
  for (int i = 0; i < 2; ++i) {
    for (size_t j = 0; j < tile; ++j) {
      const size_t cell =
          static_cast<size_t>(dates[i]) * cube.height * cube.width +
          4u * cube.width + j;
      EXPECT_EQ(red[i * tile + j], cube.red[cell]);
      EXPECT_EQ(scl[i * tile + j], cube.scl[cell]);
    }
  }

  // Bounds are validated.
  EXPECT_FALSE(source->ReadTile(11, 3, &view).ok());
  EXPECT_FALSE(source->ReadTile(-1, 2, &view).ok());
  EXPECT_FALSE(source->ReadTile(0, 0, &view).ok());
  const int bad_dates[] = {5};
  EXPECT_FALSE(source->ReadTileDates(0, 1, bad_dates, 1, &view).ok());
  TileView empty;
  EXPECT_FALSE(source->ReadTile(0, 1, &empty).ok());
}

// Reads every tile of `source` (all dates, then a date subset) into `out` as
// one concatenated byte vector per band, so that two readers can be compared.
void ReadEverything(CubeSource* source, int tile_rows,
                    std::vector<uint16_t>* red, std::vector<uint16_t>* nir,
                    std::vector<uint8_t>* scl) {
  const CubeInfo& info = source->info();
  for (int row0 = 0; row0 < info.height; row0 += tile_rows) {
    const int rows = std::min(tile_rows, info.height - row0);
    const size_t cells = static_cast<size_t>(rows) * info.width;
    std::vector<uint16_t> r(cells * info.num_dates);
    std::vector<uint16_t> n(r.size());
    std::vector<uint8_t> c(r.size());
    TileView view = {r.data(), n.data(), c.data(), 0, 0};
    ASSERT_TRUE(source->ReadTile(row0, rows, &view).ok());
    red->insert(red->end(), r.begin(), r.end());
    nir->insert(nir->end(), n.begin(), n.end());
    scl->insert(scl->end(), c.begin(), c.end());
    const int dates[] = {info.num_dates - 1, 0, 2};
    r.assign(cells * 3, 0);
    n.assign(cells * 3, 0);
    c.assign(cells * 3, 0);
    ASSERT_TRUE(source->ReadTileDates(row0, rows, dates, 3, &view).ok());
    red->insert(red->end(), r.begin(), r.begin() + cells * 3);
    nir->insert(nir->end(), n.begin(), n.begin() + cells * 3);
    scl->insert(scl->end(), c.begin(), c.begin() + cells * 3);
  }
}

TEST(CubeIo, ParallelReaderIsByteIdenticalToSequential) {
  const TestCube cube = MakeTestCube();
  const TempDir dir;
  CubeInfo info;
  ASSERT_TRUE(WriteTestCube(cube, dir, &info).ok());
  Status status = Status::Ok();
  std::unique_ptr<CubeSource> sequential = OpenRawCube(info, 1, &status);
  ASSERT_TRUE(status.ok());
  std::vector<uint16_t> red;
  std::vector<uint16_t> nir;
  std::vector<uint8_t> scl;
  // 13 rows in tiles of 4: the last tile has a single row.
  ReadEverything(sequential.get(), 4, &red, &nir, &scl);
  ASSERT_FALSE(red.empty());
  for (const int threads : {3, 8, 64}) {
    std::unique_ptr<CubeSource> parallel = OpenRawCube(info, threads, &status);
    ASSERT_TRUE(status.ok());
    std::vector<uint16_t> red_p;
    std::vector<uint16_t> nir_p;
    std::vector<uint8_t> scl_p;
    ReadEverything(parallel.get(), 4, &red_p, &nir_p, &scl_p);
    EXPECT_TRUE(red == red_p);
    EXPECT_TRUE(nir == nir_p);
    EXPECT_TRUE(scl == scl_p);
  }
}

TEST(CubeIo, ParallelReaderReportsReadErrorsWithoutHanging) {
  const TestCube cube = MakeTestCube();
  const TempDir dir;
  CubeInfo info;
  ASSERT_TRUE(WriteTestCube(cube, dir, &info).ok());
  for (const int threads : {1, 3, 8}) {
    Status status = Status::Ok();
    std::unique_ptr<CubeSource> source = OpenRawCube(info, threads, &status);
    ASSERT_TRUE(status.ok());
    // Cut the NIR file after it was opened: reads of the last dates hit the
    // end of the file.
    std::error_code error;
    fs::resize_file(info.nir_file, 100, error);
    ASSERT_FALSE(static_cast<bool>(error));
    std::vector<uint16_t> red(static_cast<size_t>(cube.height) * cube.width *
                              cube.num_dates);
    std::vector<uint16_t> nir(red.size());
    std::vector<uint8_t> scl(red.size());
    TileView view = {red.data(), nir.data(), scl.data(), 0, 0};
    const Status read = source->ReadTile(0, cube.height, &view);
    EXPECT_FALSE(read.ok());
    EXPECT_TRUE(read.message().find("nir.u16") != std::string::npos);
    // The source stays usable (and joins its threads) after an error.
    EXPECT_FALSE(source->ReadTile(0, cube.height, &view).ok());
    ASSERT_TRUE(WriteTestCube(cube, dir, &info).ok());
  }
}

TEST(CubeIo, OpenRawCubeReportsMissingAndShortFiles) {
  const TestCube cube = MakeTestCube();
  const TempDir dir;
  CubeInfo info;
  ASSERT_TRUE(WriteTestCube(cube, dir, &info).ok());
  Status status = Status::Ok();
  info.nir_file = dir.File("absent.u16");
  EXPECT_TRUE(OpenRawCube(info, &status) == nullptr);
  EXPECT_FALSE(status.ok());
  info.nir_file = dir.File("nir.u16");
  info.num_dates = 6;  // files are one date too short
  EXPECT_TRUE(OpenRawCube(info, &status) == nullptr);
  EXPECT_FALSE(status.ok());
}

TEST(CubeIo, MemoryCubeEqualsRawCube) {
  const TestCube cube = MakeTestCube();
  const TempDir dir;
  CubeInfo info;
  ASSERT_TRUE(WriteTestCube(cube, dir, &info).ok());
  Status status = Status::Ok();
  std::unique_ptr<CubeSource> raw = OpenRawCube(info, &status);
  ASSERT_TRUE(status.ok());

  SyntheticCube memory;
  memory.info = info;
  memory.red = cube.red;
  memory.nir = cube.nir;
  memory.scl = cube.scl;
  std::unique_ptr<CubeSource> mem = MakeMemoryCube(std::move(memory));
  EXPECT_EQ(mem->info().width, 7);

  for (const int row0 : {0, 4, 9}) {
    const int rows = 4;
    const size_t tile = static_cast<size_t>(rows) * cube.width;
    std::vector<uint16_t> red_a(tile * cube.num_dates);
    std::vector<uint16_t> nir_a(red_a.size());
    std::vector<uint8_t> scl_a(red_a.size());
    std::vector<uint16_t> red_b(red_a.size());
    std::vector<uint16_t> nir_b(red_a.size());
    std::vector<uint8_t> scl_b(red_a.size());
    TileView a = {red_a.data(), nir_a.data(), scl_a.data(), 0, 0};
    TileView b = {red_b.data(), nir_b.data(), scl_b.data(), 0, 0};
    ASSERT_TRUE(raw->ReadTile(row0, rows, &a).ok());
    ASSERT_TRUE(mem->ReadTile(row0, rows, &b).ok());
    EXPECT_TRUE(red_a == red_b);
    EXPECT_TRUE(nir_a == nir_b);
    EXPECT_TRUE(scl_a == scl_b);
    const int dates[] = {3, 0, 2};
    ASSERT_TRUE(raw->ReadTileDates(row0, rows, dates, 3, &a).ok());
    ASSERT_TRUE(mem->ReadTileDates(row0, rows, dates, 3, &b).ok());
    EXPECT_TRUE(red_a == red_b);
    EXPECT_TRUE(scl_a == scl_b);
  }
  std::vector<uint16_t> red(10);
  TileView view = {red.data(), red.data(), nullptr, 0, 0};
  EXPECT_FALSE(mem->ReadTile(0, 1, &view).ok());
  std::vector<uint8_t> scl(100);
  view.scl = scl.data();
  EXPECT_FALSE(mem->ReadTile(12, 2, &view).ok());
}

}  // namespace
}  // namespace cd
