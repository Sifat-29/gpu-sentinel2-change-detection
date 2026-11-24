// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "validation.h"

#include <cstdio>
#include <string>
#include <vector>

#include "timing.h"
#include "validation_rules.h"

namespace cd {
namespace {

// Formats a fraction with a fixed number of decimals.
std::string Fixed(double value, int decimals) {
  char text[64];
  std::snprintf(text, sizeof(text), "%.*f", decimals, value);
  return text;
}

// numerator / denominator, or 0 (and *undefined = true) when the denominator
// is zero.
double SafeRatio(double numerator, double denominator, bool* undefined) {
  if (denominator <= 0.0) {
    *undefined = true;
    return 0.0;
  }
  return numerator / denominator;
}

}  // namespace

ValidationWindow MakeValidationWindow(const DateTable& dates) {
  ValidationWindow window;
  if (dates.year.empty()) {
    return window;
  }
  size_t first = static_cast<size_t>(dates.num_train_dates);
  if (first >= dates.year.size()) {
    first = dates.year.size() - 1;
  }
  window.first_year = dates.year[first];
  window.last_year = dates.year.back();
  return window;
}

Status ComputeConfusionCpu(const DetectionResult& result,
                           const std::vector<uint8_t>& reference,
                           const DateTable& dates,
                           const ValidationWindow& window,
                           ConfusionCounts* counts, double* ms) {
  if (counts == nullptr) {
    return Status::Error("ComputeConfusionCpu: null output");
  }
  const WallTimer timer;
  const size_t pixels =
      static_cast<size_t>(result.height) * static_cast<size_t>(result.width);
  if (reference.size() != pixels || result.status.size() != pixels ||
      result.break_index.size() != pixels ||
      result.magnitude.size() != pixels) {
    return Status::Error(
        "validation: reference raster and detection maps differ in size");
  }
  const int num_dates = static_cast<int>(dates.year.size());
  ConfusionCounts sum;
  for (size_t i = 0; i < pixels; ++i) {
    // The break year is only meaningful for break pixels; ClassifyPixel
    // ignores it otherwise.
    int break_year = 0;
    if (result.status[i] == static_cast<uint8_t>(PixelStatus::kBreak)) {
      const int index = result.break_index[i];
      if (index < 0 || index >= num_dates) {
        return Status::Error("validation: break index out of range");
      }
      break_year = dates.year[static_cast<size_t>(index)];
    }
    const PixelValidation v =
        ClassifyPixel(result.status[i], break_year, result.magnitude[i],
                      reference[i], window);
    switch (v.cls) {
      case ValidationClass::kTruePositive:
        ++sum.tp;
        break;
      case ValidationClass::kFalsePositive:
        ++sum.fp;
        break;
      case ValidationClass::kFalseNegative:
        ++sum.fn;
        break;
      case ValidationClass::kTrueNegative:
        ++sum.tn;
        break;
      case ValidationClass::kExcluded:
        ++sum.excluded;
        break;
    }
    sum.year_exact += v.year_exact ? 1 : 0;
    sum.year_within1 += v.year_within1 ? 1 : 0;
  }
  *counts = sum;
  if (ms != nullptr) {
    *ms = timer.ElapsedMs();
  }
  return Status::Ok();
}

ValidationMetrics MetricsFromCounts(const ConfusionCounts& counts) {
  ValidationMetrics m;
  const double tp = static_cast<double>(counts.tp);
  m.precision =
      SafeRatio(tp, static_cast<double>(counts.tp + counts.fp), &m.undefined);
  m.recall =
      SafeRatio(tp, static_cast<double>(counts.tp + counts.fn), &m.undefined);
  m.f1 = SafeRatio(2.0 * m.precision * m.recall, m.precision + m.recall,
                   &m.undefined);
  m.year_exact_frac =
      SafeRatio(static_cast<double>(counts.year_exact), tp, &m.undefined);
  m.year_within1_frac =
      SafeRatio(static_cast<double>(counts.year_within1), tp, &m.undefined);
  return m;
}

std::string FormatValidationLine(const ConfusionCounts& counts,
                                 const ValidationMetrics& metrics) {
  std::string line =
      "precision " + Fixed(metrics.precision, 3) + " | recall " +
      Fixed(metrics.recall, 3) + " | f1 " + Fixed(metrics.f1, 3) +
      " | year_exact " + Fixed(metrics.year_exact_frac, 3) +
      " | year_within1 " + Fixed(metrics.year_within1_frac, 3) + " (tp " +
      std::to_string(counts.tp) + " fp " + std::to_string(counts.fp) + " fn " +
      std::to_string(counts.fn) + ")";
  if (metrics.undefined) {
    line += " [some ratios undefined: zero denominator, reported as 0]";
  }
  return line;
}

std::string FormatValidationCsv(const ConfusionCounts& counts,
                                const ValidationMetrics& metrics,
                                const ValidationWindow& window) {
  std::string csv = "metric,value\n";
  const auto add = [&csv](const std::string& name, const std::string& value) {
    csv += name + "," + value + "\n";
  };
  add("tp", std::to_string(counts.tp));
  add("fp", std::to_string(counts.fp));
  add("fn", std::to_string(counts.fn));
  add("tn", std::to_string(counts.tn));
  add("excluded", std::to_string(counts.excluded));
  add("precision", Fixed(metrics.precision, 6));
  add("recall", Fixed(metrics.recall, 6));
  add("f1", Fixed(metrics.f1, 6));
  add("year_exact", Fixed(metrics.year_exact_frac, 6));
  add("year_within1", Fixed(metrics.year_within1_frac, 6));
  add("first_year", std::to_string(window.first_year));
  add("last_year", std::to_string(window.last_year));
  return csv;
}

}  // namespace cd
