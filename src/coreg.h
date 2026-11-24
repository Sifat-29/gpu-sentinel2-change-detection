// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Co-registration check (--mode coreg): estimates the (dy, dx) translation of
// every date of a cube against a reference date with FFT phase correlation
// on the GPU. Method and outputs: README.md ("Co-registration check") and
// docs/FORMATS.md (coreg_shifts.csv).
#ifndef SRC_COREG_H_
#define SRC_COREG_H_

#include <string>

#include "flags.h"
#include "status.h"

namespace cd {

inline constexpr int kMinCoregWindow = 64;
inline constexpr int kMaxCoregWindow = 4096;
// A date is flagged "low_peak" when its correlation peak is below
// LowPeakThreshold(window): the peak is 1 for a perfect translation, and the
// highest peak of unrelated images is about 6 / window (the maximum of
// window^2 noise values of standard deviation 1 / window), so the threshold is
// the larger of kLowPeakFloor and kLowPeakNoiseFactor / window.
inline constexpr float kLowPeakFloor = 0.02f;
inline constexpr float kLowPeakNoiseFactor = 10.0f;
// A date is flagged "large_shift" when its shift magnitude (pixels) exceeds
// this value.
inline constexpr float kLargeShiftPx = 1.0f;

// The square window of the cube that is correlated.
struct CoregWindow {
  int size = 0;
  int row0 = 0;
  int col0 = 0;
};

// Validates the window arguments against the cube size and resolves the
// origin: negative origin values mean a centred window. `size` must be a
// power of two in kMinCoregWindow..kMaxCoregWindow and the window must lie
// inside the cube.
Status ResolveCoregWindow(int size, int origin_row, int origin_col, int height,
                          int width, CoregWindow* window);

// Accepts "nir" and "red".
Status CheckCoregBand(const std::string& band);

// Peak height below which a date of a `window` x `window` correlation is
// flagged "low_peak".
float LowPeakThreshold(int window);

// Returns "low_peak", "large_shift" or "ok" for one date (a weak peak takes
// precedence because the shift is then unreliable).
const char* CoregFlag(float peak, float shift_px, int window);

// Runs --mode coreg and returns the process exit code.
int RunCoregMode(const Flags& flags);

}  // namespace cd

#endif  // SRC_COREG_H_
