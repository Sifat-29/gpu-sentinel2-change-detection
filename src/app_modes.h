// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Entry points of the run modes dispatched by RunApp. Each returns the
// process exit code and logs its own errors.
#ifndef SRC_APP_MODES_H_
#define SRC_APP_MODES_H_

#include "flags.h"

namespace cd {

// --mode detect (app_detect.cc).
int RunDetectMode(const Flags& flags);

// --mode benchmark (app_benchmark.cc).
int RunBenchmarkMode(const Flags& flags);

// --mode quick-diff (app.cc).
int RunQuickDiffMode(const Flags& flags);

// --mode generate (app.cc).
int RunGenerateMode(const Flags& flags);

}  // namespace cd

#endif  // SRC_APP_MODES_H_
