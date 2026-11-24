// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Validation against a reference loss raster: monitoring window, CPU
// confusion counts (the twin of GpuConfusion), precision/recall/F1 and year
// agreement, and the text forms used by the log and validation.csv.
#ifndef SRC_VALIDATION_H_
#define SRC_VALIDATION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "status.h"
#include "types.h"

namespace cd {

// Derived quality metrics. Ratios whose denominator is zero are reported as 0
// (never NaN) and `undefined` is set.
struct ValidationMetrics {
  double precision = 0.0;          // tp / (tp + fp)
  double recall = 0.0;             // tp / (tp + fn)
  double f1 = 0.0;                 // harmonic mean of precision and recall
  double year_exact_frac = 0.0;    // year_exact / tp
  double year_within1_frac = 0.0;  // year_within1 / tp
  bool undefined = false;          // some ratio had a zero denominator
};

// Monitoring years: the year of the first monitoring date (the first date
// after the training dates) up to the year of the last date. Returns {0, 0}
// when the date table has no years.
ValidationWindow MakeValidationWindow(const DateTable& dates);

// Counts true/false positives and negatives of `result` against `reference`
// (H x W bytes, Hansen lossyear) with the rules of validation_rules.h. Gives
// the same counts as GpuConfusion. `ms` (optional) receives the elapsed time.
Status ComputeConfusionCpu(const DetectionResult& result,
                           const std::vector<uint8_t>& reference,
                           const DateTable& dates,
                           const ValidationWindow& window,
                           ConfusionCounts* counts, double* ms);

// Precision, recall, F1 and year agreement from confusion counts.
ValidationMetrics MetricsFromCounts(const ConfusionCounts& counts);

// The text after "validation: " in the log, for example
// "precision 0.810 | recall 0.660 | f1 0.727 | year_exact 0.700 |
//  year_within1 0.910 (tp 10 fp 2 fn 3)" (on one line).
std::string FormatValidationLine(const ConfusionCounts& counts,
                                 const ValidationMetrics& metrics);

// Contents of validation.csv: a "metric,value" header and the rows tp, fp, fn,
// tn, excluded, precision, recall, f1, year_exact, year_within1 (fractions),
// first_year, last_year.
std::string FormatValidationCsv(const ConfusionCounts& counts,
                                const ValidationMetrics& metrics,
                                const ValidationWindow& window);

}  // namespace cd

#endif  // SRC_VALIDATION_H_
