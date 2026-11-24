// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Test-only helpers for the CPU backend and post-processing tests: an
// in-memory CubeSource backed by vectors and a deterministic random cube
// generator with known step changes. Independent of src/cube_io.* and
// src/synthetic_cube.*.
#ifndef TESTS_CPU_TEST_CUBE_H_
#define TESTS_CPU_TEST_CUBE_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "cube_source.h"
#include "status.h"
#include "types.h"

namespace cd {
namespace cputest {

// Day number of 2018-01-01 (days since 1970-01-01).
inline constexpr int kFirstDay = 17532;

// "YYYY-MM-DD" of a day number (civil-from-days algorithm).
inline std::string IsoFromDays(int days) {
  const int z = days + 719468;
  const int era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  int year = static_cast<int>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned day = doy - (153 * mp + 2) / 5 + 1;
  const unsigned month = mp < 10 ? mp + 3 : mp - 9;
  if (month <= 2) {
    ++year;
  }
  char text[48];
  std::snprintf(text, sizeof(text), "%04d-%02u-%02u", year, month, day);
  return text;
}

// Date table with the given day numbers (iso, day, scale 1e-4, offset 0).
// tau, year and num_train_dates are left for BuildDateTable.
inline DateTable MakeDateTableFromDays(const std::vector<int>& days) {
  DateTable dates;
  for (const int day : days) {
    dates.day.push_back(day);
    dates.iso.push_back(IsoFromDays(day));
    dates.scale.push_back(1e-4f);
    dates.offset.push_back(0.0f);
  }
  return dates;
}

// Date table of `num_dates` dates `spacing_days` apart starting 2018-01-01.
inline DateTable MakeRegularDates(int num_dates, int spacing_days) {
  std::vector<int> days;
  for (int i = 0; i < num_dates; ++i) {
    days.push_back(kFirstDay + i * spacing_days);
  }
  return MakeDateTableFromDays(days);
}

// An in-memory cube, date-major like the files on disk. Pixel values are set
// with Set()/SetNdvi(); everything starts as red = nir = 0 (no data).
class MemoryCubeSource : public CubeSource {
 public:
  // `dates` needs iso, day, scale and offset (as produced by the helpers
  // above).
  MemoryCubeSource(int height, int width, const DateTable& dates) {
    info_.name = "memory";
    info_.height = height;
    info_.width = width;
    info_.num_dates = static_cast<int>(dates.day.size());
    info_.dates = dates;
    const size_t total = static_cast<size_t>(info_.num_dates) * height * width;
    red_.assign(total, 0);
    nir_.assign(total, 0);
    scl_.assign(total, 0);
  }

  const CubeInfo& info() const override { return info_; }

  Status ReadTile(int row0, int rows, TileView* out) override {
    const std::vector<int> all = AllDates();
    return Copy(row0, rows, all.data(), info_.num_dates, out);
  }

  Status ReadTileDates(int row0, int rows, const int* dates, int n,
                       TileView* out) override {
    return Copy(row0, rows, dates, n, out);
  }

  // Sets the raw values of one pixel on one date.
  void Set(int t, int y, int x, uint16_t red, uint16_t nir, uint8_t scl) {
    const size_t i = Index(t, y, x);
    red_[i] = red;
    nir_[i] = nir;
    scl_[i] = scl;
  }

  // Sets a pixel so that its NDVI is `ndvi` (scale 1e-4, offset 0) with the
  // given SCL class (4 = vegetation).
  void SetNdvi(int t, int y, int x, double ndvi, uint8_t scl = 4) {
    constexpr double kRed = 2000.0;
    const double nir = kRed * (1.0 + ndvi) / (1.0 - ndvi);
    Set(t, y, x, static_cast<uint16_t>(kRed),
        static_cast<uint16_t>(std::min(65535.0, std::round(nir))), scl);
  }

  // After `count` more successful reads every read fails (error-path tests).
  void FailReadsAfter(int count) { reads_until_failure_ = count; }

 private:
  std::vector<int> AllDates() const {
    std::vector<int> all(static_cast<size_t>(info_.num_dates));
    for (int t = 0; t < info_.num_dates; ++t) {
      all[static_cast<size_t>(t)] = t;
    }
    return all;
  }

  size_t Index(int t, int y, int x) const {
    return (static_cast<size_t>(t) * info_.height + y) * info_.width + x;
  }

  Status Copy(int row0, int rows, const int* dates, int n, TileView* out) {
    if (reads_until_failure_ == 0) {
      return Status::Error("injected read failure");
    }
    if (reads_until_failure_ > 0) {
      --reads_until_failure_;
    }
    if (row0 < 0 || rows < 1 || row0 + rows > info_.height) {
      return Status::Error("tile rows out of range");
    }
    const size_t row_elements = static_cast<size_t>(rows) * info_.width;
    for (int i = 0; i < n; ++i) {
      const size_t src = Index(dates[i], row0, 0);
      const size_t dst = static_cast<size_t>(i) * row_elements;
      std::copy(red_.begin() + src, red_.begin() + src + row_elements,
                out->red + dst);
      std::copy(nir_.begin() + src, nir_.begin() + src + row_elements,
                out->nir + dst);
      std::copy(scl_.begin() + src, scl_.begin() + src + row_elements,
                out->scl + dst);
    }
    out->row0 = row0;
    out->rows = rows;
    return Status::Ok();
  }

  CubeInfo info_;
  std::vector<uint16_t> red_;
  std::vector<uint16_t> nir_;
  std::vector<uint8_t> scl_;
  int reads_until_failure_ = -1;
};

// Fills the cube with seasonal NDVI (base 0.45..0.65, amplitude 0.15, random
// phase, noise sd 0.02), about 10 % cloudy observations (SCL 9) and 1 %
// no-data observations. About 20 % of the pixels get a step of -0.35 at a
// random monitoring date; step_date[pixel] receives that date index (-1 for
// pixels without a step). `first_step` is the earliest allowed step index.
inline void FillRandomCube(MemoryCubeSource* cube, uint32_t seed,
                           int first_step, std::vector<int>* step_date) {
  const CubeInfo& info = cube->info();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> noise(0.0, 0.02);
  const double kTwoPi = 6.283185307179586;
  step_date->assign(static_cast<size_t>(info.height) * info.width, -1);
  for (int y = 0; y < info.height; ++y) {
    for (int x = 0; x < info.width; ++x) {
      const double base = 0.45 + 0.2 * uniform(rng);
      const double phase = kTwoPi * uniform(rng);
      int step = -1;
      if (uniform(rng) < 0.2) {
        const int span = info.num_dates - first_step;
        step = first_step + static_cast<int>(uniform(rng) * span) % span;
        (*step_date)[static_cast<size_t>(y) * info.width + x] = step;
      }
      for (int t = 0; t < info.num_dates; ++t) {
        const double tau =
            static_cast<double>(info.dates.day[t] - info.dates.day[0]) / 365.25;
        double ndvi = base + 0.15 * std::sin(kTwoPi * tau + phase) + noise(rng);
        if (step >= 0 && t >= step) {
          ndvi -= 0.35;
        }
        const double draw = uniform(rng);
        if (draw < 0.01) {
          cube->Set(t, y, x, 0, 0, 4);
        } else if (draw < 0.11) {
          cube->SetNdvi(t, y, x, ndvi, 9);
        } else {
          cube->SetNdvi(t, y, x, ndvi, 4);
        }
      }
    }
  }
}

}  // namespace cputest
}  // namespace cd

#endif  // TESTS_CPU_TEST_CUBE_H_
