// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// The per-pixel algorithm (see README.md): NDVI, harmonic regression
// fit, residual statistics and the run-of-exceedances monitoring state machine.
// This header is the single source of truth: the CPU baselines instantiate it
// with double and float, the GPU kernels with float, so "the same algorithm
// on every backend" is true by construction.
//
// Everything that runs per pixel is a CD_HOST_DEVICE template on
// <Real, kHarmonics>, uses only fixed-size local arrays, and has no STL
// containers and no exceptions. The two host-only helpers at the top
// (DesignRow, BuildDesignTable) are not annotated and therefore never
// compiled for the device.
#ifndef SRC_PIXEL_MODEL_H_
#define SRC_PIXEL_MODEL_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "host_device.h"
#include "types.h"

// Asks the compiler to unroll the small fixed-size loop that follows: nvcc in
// device code, GCC when it compiles plain C++ (the CPU backends: the trip
// counts are compile-time constants, and full unrolling keeps the matrices in
// registers, which makes the fit about twice as fast). Empty everywhere else,
// including nvcc's host pass, so that no unknown-pragma warning appears.
#if defined(__CUDA_ARCH__)
#define CD_UNROLL _Pragma("unroll")
#elif defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__)
#define CD_UNROLL _Pragma("GCC unroll 8")
#else
#define CD_UNROLL
#endif

namespace cd {

// Observations below this value are invalid (valid NDVI is in [-1, 1] and the
// sentinel is kInvalidNdvi = -2).
inline constexpr float kInvalidThreshold = -1.5f;

// Smallest allowed relative Cholesky pivot, identical for float and double so
// that both precisions classify near-singular pixels the same way.
inline constexpr float kPivotTolerance = 1e-4f;

// Smallest allowed (nir + red) reflectance sum; below it NDVI is undefined.
inline constexpr float kMinReflectanceSum = 1e-4f;

// Square root usable in host and device code for both precisions.
CD_HOST_DEVICE inline float SqrtOf(float x) { return sqrtf(x); }
CD_HOST_DEVICE inline double SqrtOf(double x) { return sqrt(x); }

// Absolute value usable in host and device code for both precisions.
CD_HOST_DEVICE inline float AbsOf(float x) { return fabsf(x); }
CD_HOST_DEVICE inline double AbsOf(double x) { return fabs(x); }

// ---------------------------------------------------------------------------
// Design matrix (host only)
// ---------------------------------------------------------------------------

// Writes the design row x(tau) = [1, s, cos 2*pi*tau, sin 2*pi*tau, ...,
// cos 2*pi*K*tau, sin 2*pi*K*tau] (2 + 2K values) into `row`, with
// s = (tau - train_years / 2) / (train_years / 2).
//
// Numerics: tau is "years since the first date" and s is the same time
// centred on the middle of the training window and scaled to [-1, 1]. With
// raw day numbers (~19000) the trend column would be ~1e4 times larger than
// the harmonic columns and X'X would be hopelessly ill-conditioned in FP32.
// Centring makes the trend column orthogonal to the intercept over the
// training window and keeps all columns O(1).
inline void DesignRow(double tau, int harmonics, double train_years,
                      double* row) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double half = 0.5 * train_years;
  row[0] = 1.0;
  row[1] = (tau - half) / half;
  for (int k = 1; k <= harmonics; ++k) {
    const double angle = kTwoPi * static_cast<double>(k) * tau;
    row[2 * k] = std::cos(angle);
    row[2 * k + 1] = std::sin(angle);
  }
}

// Fills `table` with the design rows of every date, row-major:
// table[t * (2 + 2K) + j]. The rows depend only on the date, so they are
// computed once per run in FP64 on the host and then cast to the precision of
// each backend. Every backend therefore reads identical design values and
// per-pixel sin/cos differences between host and device libm cannot occur.
// `dates.tau` must be filled (BuildDateTable). With `hold_trend`, every date
// with tau >= train_years gets the trend column of tau = train_years (1.0 in
// the scaled units) while the seasonal columns keep following the real tau;
// the training rows (tau < train_years) are unchanged.
inline void BuildDesignTable(const DateTable& dates, int harmonics,
                             float train_years, bool hold_trend,
                             std::vector<double>* table) {
  const int num_params = 2 + 2 * harmonics;
  const size_t num_dates = dates.tau.size();
  table->assign(num_dates * static_cast<size_t>(num_params), 0.0);
  for (size_t t = 0; t < num_dates; ++t) {
    double* row = table->data() + t * static_cast<size_t>(num_params);
    DesignRow(dates.tau[t], harmonics, static_cast<double>(train_years), row);
    if (hold_trend && dates.tau[t] >= static_cast<double>(train_years)) {
      row[1] = 1.0;
    }
  }
}

// ---------------------------------------------------------------------------
// NDVI and series accessors
// ---------------------------------------------------------------------------

// NDVI of one observation from raw digital numbers. Returns kInvalidNdvi when
// the SCL class is not in `valid_mask` (bit c set = class c is usable), when
// red or nir is 0 (no data), or when the reflectance sum is <= 1e-4. The
// result is clamped to [-1, 1]. Reflectance = DN * scale + offset.
template <typename Real>
CD_HOST_DEVICE inline Real ComputeNdvi(uint16_t red, uint16_t nir, uint8_t scl,
                                       Real scale, Real offset,
                                       uint32_t valid_mask) {
  const Real invalid = static_cast<Real>(kInvalidNdvi);
  if (scl >= 32 || ((valid_mask >> scl) & 1u) == 0 || red == 0 || nir == 0) {
    return invalid;
  }
  const Real red_refl = static_cast<Real>(red) * scale + offset;
  const Real nir_refl = static_cast<Real>(nir) * scale + offset;
  const Real sum = nir_refl + red_refl;
  if (!(sum > static_cast<Real>(kMinReflectanceSum))) {
    return invalid;
  }
  Real ndvi = (nir_refl - red_refl) / sum;
  if (ndvi > Real(1)) {
    ndvi = Real(1);
  }
  if (ndvi < Real(-1)) {
    ndvi = Real(-1);
  }
  return ndvi;
}

// Series accessor that computes NDVI on the fly from the raw bands. The three
// pointers address the pixel's value on date 0; date t is `t * stride`
// elements further (stride = rows * W in a date-major tile). `scale` and
// `offset` are the per-date calibration arrays of the DateTable.
template <typename Real>
struct RawSeries {
  const uint16_t* red;
  const uint16_t* nir;
  const uint8_t* scl;
  int64_t stride;
  const float* scale;
  const float* offset;
  uint32_t valid_mask;

  CD_HOST_DEVICE Real operator()(int t) const {
    const int64_t i = static_cast<int64_t>(t) * stride;
    return ComputeNdvi<Real>(red[i], nir[i], scl[i],
                             static_cast<Real>(scale[t]),
                             static_cast<Real>(offset[t]), valid_mask);
  }
};

// Series accessor over a precomputed NDVI array: element t is at
// ndvi[t * stride]. `Stored` is the element type of the array (float for the
// GPU NDVI cube, double for the FP64 CPU scratch buffer).
template <typename Real, typename Stored = float>
struct NdviArraySeries {
  const Stored* ndvi;
  int64_t stride;

  CD_HOST_DEVICE Real operator()(int t) const {
    return static_cast<Real>(ndvi[static_cast<int64_t>(t) * stride]);
  }
};

// True when an observation returned by a series accessor is usable.
template <typename Real>
CD_HOST_DEVICE inline bool IsValidObservation(Real y) {
  return y >= static_cast<Real>(kInvalidThreshold);
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

// Solves A x = b for a symmetric positive-definite kParams x kParams matrix
// by Cholesky factorisation. `a` is row-major; only its lower triangle is
// read, and it is overwritten with the factor L (A = L L'). `b` is
// overwritten with the solution. Returns false when a pivot is <= 1e-4 times
// the largest diagonal entry of A (singular or nearly singular design, for
// example all training observations on one date) or is not finite. The
// relative tolerance is the same for float and double on purpose.
template <typename Real, int kParams>
CD_HOST_DEVICE inline bool CholeskySolve(Real* a, Real* b) {
  Real max_diag = Real(0);
  CD_UNROLL
  for (int i = 0; i < kParams; ++i) {
    if (a[i * kParams + i] > max_diag) {
      max_diag = a[i * kParams + i];
    }
  }
  if (!(max_diag > Real(0))) {
    return false;
  }
  const Real tolerance = static_cast<Real>(kPivotTolerance) * max_diag;
  CD_UNROLL
  for (int j = 0; j < kParams; ++j) {
    Real pivot = a[j * kParams + j];
    CD_UNROLL
    for (int k = 0; k < j; ++k) {
      pivot -= a[j * kParams + k] * a[j * kParams + k];
    }
    if (!(pivot > tolerance)) {
      return false;
    }
    const Real diag = SqrtOf(pivot);
    a[j * kParams + j] = diag;
    CD_UNROLL
    for (int i = j + 1; i < kParams; ++i) {
      Real sum = a[i * kParams + j];
      CD_UNROLL
      for (int k = 0; k < j; ++k) {
        sum -= a[i * kParams + k] * a[j * kParams + k];
      }
      a[i * kParams + j] = sum / diag;
    }
  }
  // Forward substitution L w = b, then back substitution L' x = w.
  CD_UNROLL
  for (int i = 0; i < kParams; ++i) {
    Real sum = b[i];
    CD_UNROLL
    for (int k = 0; k < i; ++k) {
      sum -= a[i * kParams + k] * b[k];
    }
    b[i] = sum / a[i * kParams + i];
  }
  CD_UNROLL
  for (int i = kParams - 1; i >= 0; --i) {
    Real sum = b[i];
    CD_UNROLL
    for (int k = i + 1; k < kParams; ++k) {
      sum -= a[k * kParams + i] * b[k];
    }
    b[i] = sum / a[i * kParams + i];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Fit, detect, predict
// ---------------------------------------------------------------------------

// Result of fitting one pixel. coef[0 .. 2K+1] are valid when ok; unused
// entries and all entries of a failed fit are 0. rmse is floored and is 0
// when the fit failed. n_train counts the valid training observations.
template <typename Real>
struct FitOutput {
  Real coef[kMaxParams];
  Real rmse;
  int n_train;
  bool ok;
};

// Result of the monitoring pass. break_index is the date index that starts
// the run (-1 = none); magnitude is the signed mean |residual| of the run
// (negative = greenness loss), 0 without a break; status is a PixelStatus.
template <typename Real>
struct DetectOutput {
  int16_t break_index;
  Real magnitude;
  uint8_t status;
};

// Both results of one pixel.
template <typename Real>
struct PixelOutput {
  FitOutput<Real> fit;
  DetectOutput<Real> detect;
};

// Model prediction x(tau)' * coef for one design row of 2 + 2K values.
template <typename Real, int kHarmonics>
CD_HOST_DEVICE inline Real PredictNdvi(const Real* coef,
                                       const Real* design_row) {
  constexpr int kParams = 2 + 2 * kHarmonics;
  Real sum = coef[0] * design_row[0];
  CD_UNROLL
  for (int j = 1; j < kParams; ++j) {
    sum += coef[j] * design_row[j];
  }
  return sum;
}

// Fits the harmonic model to the valid observations among the first
// `num_train_dates` dates of series `s` (steps 2 and 3 of the algorithm).
//
// 1. Accumulate the normal equations X'X (lower triangle only) and X'y over
//    the valid training observations.
// 2. Fewer than max(min_train_obs, p + 1) observations, or a failed Cholesky
//    solve, gives ok = false (status kInsufficientData). `ridge` is added to
//    the diagonal entries 1 .. p-1 (never the intercept).
// 3. The RSS is recomputed from the fitted coefficients in a second pass over
//    the observations instead of using y'y - b'X'y: that identity subtracts
//    two nearly equal large numbers and loses most of its digits in FP32.
// 4. rmse = max(sqrt(RSS / (n - p)), rmse_floor). The floor stops a perfectly
//    fitted (noise-free or very short) series from producing huge z-scores
//    and false breaks on tiny deviations.
template <typename Real, int kHarmonics, typename Series>
CD_HOST_DEVICE inline FitOutput<Real> FitModel(const Series& s,
                                               const Real* design,
                                               int num_train_dates,
                                               const ModelParams& p) {
  constexpr int kParams = 2 + 2 * kHarmonics;
  FitOutput<Real> out;
  CD_UNROLL
  for (int j = 0; j < kMaxParams; ++j) {
    out.coef[j] = Real(0);
  }
  out.rmse = Real(0);
  out.n_train = 0;
  out.ok = false;

  Real xtx[kParams * kParams];
  Real xty[kParams];
  CD_UNROLL
  for (int i = 0; i < kParams * kParams; ++i) {
    xtx[i] = Real(0);
  }
  CD_UNROLL
  for (int i = 0; i < kParams; ++i) {
    xty[i] = Real(0);
  }

  int n = 0;
  for (int t = 0; t < num_train_dates; ++t) {
    const Real y = s(t);
    if (!IsValidObservation(y)) {
      continue;
    }
    const Real* x = design + t * kParams;
    CD_UNROLL
    for (int i = 0; i < kParams; ++i) {
      const Real xi = x[i];
      CD_UNROLL
      for (int j = 0; j <= i; ++j) {
        xtx[i * kParams + j] += xi * x[j];
      }
      xty[i] += xi * y;
    }
    ++n;
  }
  out.n_train = n;
  const int needed =
      p.min_train_obs > kParams + 1 ? p.min_train_obs : kParams + 1;
  if (n < needed) {
    return out;
  }
  const Real ridge = static_cast<Real>(p.ridge);
  CD_UNROLL
  for (int i = 1; i < kParams; ++i) {
    xtx[i * kParams + i] += ridge;
  }
  if (!CholeskySolve<Real, kParams>(xtx, xty)) {
    return out;
  }
  CD_UNROLL
  for (int j = 0; j < kParams; ++j) {
    out.coef[j] = xty[j];
  }

  // Second pass: residual sum of squares from the fitted coefficients.
  Real rss = Real(0);
  for (int t = 0; t < num_train_dates; ++t) {
    const Real y = s(t);
    if (!IsValidObservation(y)) {
      continue;
    }
    const Real r =
        y - PredictNdvi<Real, kHarmonics>(out.coef, design + t * kParams);
    rss += r * r;
  }
  Real rmse = SqrtOf(rss / static_cast<Real>(n - kParams));
  const Real floor_value = static_cast<Real>(p.rmse_floor);
  if (!(rmse >= floor_value)) {
    rmse = floor_value;
  }
  out.rmse = rmse;
  out.ok = true;
  return out;
}

// Monitors the dates t >= num_train_dates of series `s` against a fitted
// model (steps 4 and 5 of the algorithm).
//
// Every valid observation gives r = y - prediction and z = r / rmse. An
// exceedance is |z| > threshold (and r < 0 when losses_only). The state is
// the current run of consecutive exceedances of the same sign: a
// non-exceedance resets the run, an exceedance of the opposite sign starts a
// new run at the current observation, and invalid observations are skipped
// without extending or resetting the run. When the run reaches `consecutive`
// observations the break is the run's first date, the magnitude is
// sign * mean(|r|) over the run, and monitoring stops.
template <typename Real, int kHarmonics, typename Series>
CD_HOST_DEVICE inline DetectOutput<Real> DetectBreak(
    const FitOutput<Real>& fit, const Series& s, const Real* design,
    int num_dates, int num_train_dates, const ModelParams& p) {
  constexpr int kParams = 2 + 2 * kHarmonics;
  DetectOutput<Real> out;
  out.break_index = -1;
  out.magnitude = Real(0);
  out.status = static_cast<uint8_t>(PixelStatus::kStable);
  if (!fit.ok) {
    out.status = static_cast<uint8_t>(PixelStatus::kInsufficientData);
    return out;
  }
  const Real threshold = static_cast<Real>(p.threshold);
  int run_length = 0;
  int run_start = -1;
  bool run_negative = false;
  Real run_sum = Real(0);
  for (int t = num_train_dates; t < num_dates; ++t) {
    const Real y = s(t);
    if (!IsValidObservation(y)) {
      continue;
    }
    const Real r =
        y - PredictNdvi<Real, kHarmonics>(fit.coef, design + t * kParams);
    const Real z = r / fit.rmse;
    const bool negative = r < Real(0);
    const bool exceeds = AbsOf(z) > threshold && (negative || !p.losses_only);
    if (!exceeds) {
      run_length = 0;
      continue;
    }
    if (run_length > 0 && negative == run_negative) {
      ++run_length;
      run_sum += AbsOf(r);
    } else {
      run_length = 1;
      run_start = t;
      run_negative = negative;
      run_sum = AbsOf(r);
    }
    if (run_length >= p.consecutive) {
      const Real mean = run_sum / static_cast<Real>(run_length);
      out.break_index = static_cast<int16_t>(run_start);
      out.magnitude = run_negative ? -mean : mean;
      out.status = static_cast<uint8_t>(PixelStatus::kBreak);
      return out;
    }
  }
  return out;
}

// Fit followed by detection; identical to calling FitModel and DetectBreak.
template <typename Real, int kHarmonics, typename Series>
CD_HOST_DEVICE inline PixelOutput<Real> FitAndDetect(const Series& s,
                                                     const Real* design,
                                                     int num_dates,
                                                     int num_train_dates,
                                                     const ModelParams& p) {
  PixelOutput<Real> out;
  out.fit = FitModel<Real, kHarmonics>(s, design, num_train_dates, p);
  out.detect = DetectBreak<Real, kHarmonics>(out.fit, s, design, num_dates,
                                             num_train_dates, p);
  return out;
}

// ---------------------------------------------------------------------------
// Quick two-date difference
// ---------------------------------------------------------------------------

// Quick-diff rule of one pixel: *diff = ndvi_b - ndvi_a. *flag is 0 when
// either NDVI is invalid (diff is then 0), 2 for a loss (diff < -threshold),
// 3 for a gain (diff > threshold, never when losses_only) and 1 otherwise.
CD_HOST_DEVICE inline void QuickDiffPixel(float ndvi_a, float ndvi_b,
                                          float threshold, bool losses_only,
                                          float* diff, uint8_t* flag) {
  if (!IsValidObservation(ndvi_a) || !IsValidObservation(ndvi_b)) {
    *diff = 0.0f;
    *flag = 0;
    return;
  }
  const float d = ndvi_b - ndvi_a;
  *diff = d;
  if (d < -threshold) {
    *flag = 2;
  } else if (d > threshold && !losses_only) {
    *flag = 3;
  } else {
    *flag = 1;
  }
}

}  // namespace cd

#endif  // SRC_PIXEL_MODEL_H_
