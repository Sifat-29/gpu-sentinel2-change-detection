// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Host and device description helpers used by the log header, the benchmark
// CSV and the test runner.
#ifndef SRC_SYSTEM_INFO_H_
#define SRC_SYSTEM_INFO_H_

#include <string>

namespace cd {

// CPU model name from /proc/cpuinfo, or "unknown".
std::string CpuModel();

// Number of hardware threads (at least 1).
int HardwareThreads();

// Local time as "YYYY-MM-DDTHH:MM:SS".
std::string IsoTimestamp();

// Short git commit of the repository containing the running executable (its
// directory, then the parent of that directory), falling back to the current
// directory. Returns "unknown" when git or the repository is unavailable and
// appends "-dirty" when tracked files have uncommitted changes.
std::string GitShortCommit();

// argv joined with single spaces.
std::string CommandLine(int argc, char** argv);

// "cpu: <model> | <n> threads".
std::string CpuSummaryLine();

// The "gpu: ..." line of the log header (format: docs/FORMATS.md, "Log
// format"): device details, "gpu: not available (CPU-only build)" or "gpu: none
// detected".
std::string GpuSummaryLine(int device);

}  // namespace cd

#endif  // SRC_SYSTEM_INFO_H_
