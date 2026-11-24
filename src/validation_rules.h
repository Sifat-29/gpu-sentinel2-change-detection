// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Per-pixel validation rule, shared by the CPU and GPU confusion counters so
// that both give identical counts.
#ifndef SRC_VALIDATION_RULES_H_
#define SRC_VALIDATION_RULES_H_

#include <cstdint>

#include "host_device.h"
#include "types.h"

namespace cd {

// Outcome of comparing one pixel's detection with the reference.
enum class ValidationClass : uint8_t {
  kExcluded = 0,
  kTruePositive = 1,
  kFalsePositive = 2,
  kFalseNegative = 3,
  kTrueNegative = 4
};

struct PixelValidation {
  ValidationClass cls = ValidationClass::kExcluded;
  bool year_exact = false;    // true positive and break year == 2000 + ref
  bool year_within1 = false;  // true positive and |break year - ref year| <= 1
};

// Classifies one pixel.
//  - reference 255 is excluded (not forest, no data, or loss before the
//    monitoring start).
//  - The reference is positive when ref != 0 and first_year <= 2000 + ref <=
//    last_year.
//  - The detection is positive when status is kBreak, magnitude < 0 (greenness
//    loss) and first_year <= break_year <= last_year.
// `break_year` is the calendar year of the break date; it is ignored unless
// the status is kBreak. year_within1 includes exact matches.
CD_HOST_DEVICE inline PixelValidation ClassifyPixel(
    uint8_t status, int break_year, float magnitude, uint8_t reference,
    const ValidationWindow& window) {
  PixelValidation out;
  if (reference == 255) {
    return out;
  }
  const int reference_year = 2000 + static_cast<int>(reference);
  const bool reference_positive = reference != 0 &&
                                  reference_year >= window.first_year &&
                                  reference_year <= window.last_year;
  const bool detected = status == static_cast<uint8_t>(PixelStatus::kBreak) &&
                        magnitude < 0.0f && break_year >= window.first_year &&
                        break_year <= window.last_year;
  if (detected && reference_positive) {
    out.cls = ValidationClass::kTruePositive;
    const int diff = break_year - reference_year;
    out.year_exact = diff == 0;
    out.year_within1 = diff >= -1 && diff <= 1;
  } else if (detected) {
    out.cls = ValidationClass::kFalsePositive;
  } else if (reference_positive) {
    out.cls = ValidationClass::kFalseNegative;
  } else {
    out.cls = ValidationClass::kTrueNegative;
  }
  return out;
}

}  // namespace cd

#endif  // SRC_VALIDATION_RULES_H_
