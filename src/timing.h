// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Wall-clock timer used for stage timings and benchmarks.
#ifndef SRC_TIMING_H_
#define SRC_TIMING_H_

#include <chrono>

namespace cd {

// Measures elapsed wall time with the monotonic steady clock. The timer starts
// when it is constructed; Reset() restarts it.
class WallTimer {
 public:
  WallTimer() : start_(Clock::now()) {}

  void Reset() { start_ = Clock::now(); }

  // Milliseconds since construction or the last Reset().
  double ElapsedMs() const {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - start_;
    return elapsed.count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

}  // namespace cd

#endif  // SRC_TIMING_H_
