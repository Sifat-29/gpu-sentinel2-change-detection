// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Entry point: parses the command line and hands over to RunApp.
#include <cstdio>
#include <string>

#include "app.h"
#include "flags.h"

int main(int argc, char** argv) {
  cd::Flags flags;
  const cd::Status status = cd::ParseFlags(argc, argv, &flags);
  if (!status.ok()) {
    std::fprintf(stderr, "error: %s; see --help\n", status.message().c_str());
    return 2;
  }
  if (flags.help) {
    std::fputs(cd::UsageText().c_str(), stdout);
    return 0;
  }
  return cd::RunApp(flags, argc, argv);
}
