// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Application entry: dispatches the run mode selected on the command line.
#ifndef SRC_APP_H_
#define SRC_APP_H_

#include "flags.h"

namespace cd {

// Runs the mode in `flags` and returns the process exit code (0 success,
// 1 run-time failure). `argc`/`argv` are only used to log the command line.
int RunApp(const Flags& flags, int argc, char** argv);

}  // namespace cd

#endif  // SRC_APP_H_
