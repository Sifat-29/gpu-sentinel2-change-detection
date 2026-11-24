// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Test runner: runs all registered tests (optionally filtered) and prints
// PASS/FAIL. Options: --filter <substring>, --list.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "system_info.h"
#include "test_framework.h"
#include "timing.h"

namespace {

std::string FullName(const cd::testing::TestInfo& test) {
  return std::string(test.suite) + "." + test.name;
}

// Suites named Large* are skipped unless the filter selects them.
bool IsLarge(const cd::testing::TestInfo& test) {
  return std::strncmp(test.suite, "Large", 5) == 0;
}

void PrintUsage() {
  std::fputs("usage: run_tests [--filter <substring>] [--list]\n", stderr);
}

}  // namespace

int main(int argc, char** argv) {
  std::string filter;
  bool list_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--list") {
      list_only = true;
    } else if (arg == "--filter" && i + 1 < argc) {
      filter = argv[++i];
    } else {
      PrintUsage();
      return 2;
    }
  }

  std::vector<cd::testing::TestInfo> tests = cd::testing::Registry();
  std::sort(tests.begin(), tests.end(),
            [](const cd::testing::TestInfo& a, const cd::testing::TestInfo& b) {
              return FullName(a) < FullName(b);
            });

  if (list_only) {
    for (const cd::testing::TestInfo& test : tests) {
      std::printf("%s%s\n", FullName(test).c_str(),
                  IsLarge(test) ? " (large, needs --filter)" : "");
    }
    return 0;
  }

  std::printf("%s\n", cd::GpuSummaryLine(0).c_str());
  std::printf("%s\n", cd::CpuSummaryLine().c_str());

  int run = 0;
  int failed = 0;
  int matched = 0;
  for (const cd::testing::TestInfo& test : tests) {
    const std::string name = FullName(test);
    if (!filter.empty() && name.find(filter) == std::string::npos) {
      continue;
    }
    ++matched;
    if (IsLarge(test) && filter.empty()) {
      std::printf("[ SKIP ] %s (large test; select it with --filter)\n",
                  name.c_str());
      continue;
    }
    std::printf("[ RUN  ] %s\n", name.c_str());
    std::fflush(stdout);
    cd::testing::FailureCount() = 0;
    const cd::WallTimer timer;
    test.function();
    const bool ok = cd::testing::FailureCount() == 0;
    std::printf("[ %s ] %s (%.1f ms)\n", ok ? " OK " : "FAIL", name.c_str(),
                timer.ElapsedMs());
    ++run;
    failed += ok ? 0 : 1;
  }

  if (matched == 0) {
    std::printf("FAIL (no test matches filter '%s')\n", filter.c_str());
    return 1;
  }
  if (failed > 0) {
    std::printf("FAIL (%d of %d tests failed)\n", failed, run);
    return 1;
  }
  std::printf("PASS (%d tests)\n", run);
  return 0;
}
