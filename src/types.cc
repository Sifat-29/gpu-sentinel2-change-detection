// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "types.h"

#include <string>

namespace cd {
namespace {

// Parses the leading "YYYY" of an ISO date string. Returns false when the
// string is shorter than 10 characters or the year is not four digits.
bool ParseIsoYear(const std::string& iso, int16_t* year) {
  if (iso.size() < 10 || iso[4] != '-') {
    return false;
  }
  int value = 0;
  for (int i = 0; i < 4; ++i) {
    if (iso[i] < '0' || iso[i] > '9') {
      return false;
    }
    value = value * 10 + (iso[i] - '0');
  }
  *year = static_cast<int16_t>(value);
  return true;
}

}  // namespace

std::string BackendName(Backend b) {
  switch (b) {
    case Backend::kGpuFused:
      return "gpu";
    case Backend::kGpuUnfused:
      return "gpu-unfused";
    case Backend::kCpuFp64:
      return "cpu";
    case Backend::kCpuFp32:
      return "cpu-fp32";
    case Backend::kCpuMt:
      return "cpu-mt";
  }
  return "unknown";
}

Status ParseBackend(const std::string& s, Backend* out) {
  static const Backend kAll[] = {Backend::kGpuFused, Backend::kGpuUnfused,
                                 Backend::kCpuFp64, Backend::kCpuFp32,
                                 Backend::kCpuMt};
  for (const Backend b : kAll) {
    if (BackendName(b) == s) {
      *out = b;
      return Status::Ok();
    }
  }
  return Status::Error("unknown backend '" + s +
                       "' (expected gpu, gpu-unfused, cpu, cpu-fp32 or "
                       "cpu-mt)");
}

bool IsGpuBackend(Backend b) {
  return b == Backend::kGpuFused || b == Backend::kGpuUnfused;
}

int NumParams(int harmonics) { return 2 + 2 * harmonics; }

Status ResolveModelParams(ModelParams* p) {
  if (p->harmonics < 1 || p->harmonics > kMaxHarmonics) {
    return Status::Error("harmonics must be in 1..3, got " +
                         std::to_string(p->harmonics));
  }
  const int num_params = NumParams(p->harmonics);
  if (p->min_train_obs < 0) {
    return Status::Error("min-train-obs must be >= 0, got " +
                         std::to_string(p->min_train_obs));
  }
  if (p->min_train_obs == 0) {
    p->min_train_obs = 3 * num_params;
  }
  if (p->min_train_obs < num_params + 1) {
    return Status::Error("min-train-obs " + std::to_string(p->min_train_obs) +
                         " is below the minimum " +
                         std::to_string(num_params + 1) +
                         " (p + 1, p = " + std::to_string(num_params) + ")");
  }
  return Status::Ok();
}

void AllocateResult(int height, int width, int num_params,
                    bool with_coefficients, DetectionResult* result) {
  const size_t pixels =
      static_cast<size_t>(height) * static_cast<size_t>(width);
  result->height = height;
  result->width = width;
  result->num_params = num_params;
  result->break_index.assign(pixels, static_cast<int16_t>(-1));
  result->magnitude.assign(pixels, 0.0f);
  result->status.assign(pixels, static_cast<uint8_t>(PixelStatus::kStable));
  result->rmse.assign(pixels, 0.0f);
  result->train_obs.assign(pixels, static_cast<uint16_t>(0));
  if (with_coefficients) {
    result->coefficients.assign(pixels * static_cast<size_t>(num_params), 0.0f);
  } else {
    result->coefficients.clear();
  }
}

Status BuildDateTable(const ModelParams& params, DateTable* dates) {
  const size_t num_dates = dates->day.size();
  if (num_dates == 0) {
    return Status::Error("date table is empty");
  }
  if (dates->iso.size() != num_dates) {
    return Status::Error("date table has " + std::to_string(dates->iso.size()) +
                         " ISO dates but " + std::to_string(num_dates) +
                         " day numbers");
  }
  dates->tau.resize(num_dates);
  dates->year.resize(num_dates);
  for (size_t t = 0; t < num_dates; ++t) {
    dates->tau[t] = static_cast<double>(dates->day[t] - dates->day[0]) / 365.25;
    if (!ParseIsoYear(dates->iso[t], &dates->year[t])) {
      return Status::Error("malformed date '" + dates->iso[t] +
                           "' (expected YYYY-MM-DD) at index " +
                           std::to_string(t));
    }
  }
  // Training dates are the leading dates with tau < train_years.
  int train = 0;
  while (static_cast<size_t>(train) < num_dates &&
         dates->tau[train] < static_cast<double>(params.train_years)) {
    ++train;
  }
  dates->num_train_dates = train;
  if (train == 0) {
    return Status::Error("no training dates: train-years " +
                         std::to_string(params.train_years) +
                         " is too short for the first date");
  }
  if (static_cast<size_t>(train) == num_dates) {
    return Status::Error(
        "no monitoring dates: all " + std::to_string(num_dates) +
        " dates are inside train-years " + std::to_string(params.train_years));
  }
  return Status::Ok();
}

}  // namespace cd
