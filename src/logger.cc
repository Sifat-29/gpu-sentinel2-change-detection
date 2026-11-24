// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "logger.h"

#include <cstdio>
#include <mutex>
#include <string>

namespace cd {
namespace {

// Process-wide logger state. Allocated once and never freed so that logging
// stays valid during static destruction.
struct LogState {
  std::mutex mutex;
  std::FILE* file = nullptr;
};

LogState& State() {
  static LogState* const state = new LogState();
  return *state;
}

}  // namespace

Status InitLog(const std::string& path) {
  LogState& state = State();
  const std::lock_guard<std::mutex> lock(state.mutex);
  if (state.file != nullptr) {
    std::fclose(state.file);
    state.file = nullptr;
  }
  if (path.empty()) {
    return Status::Ok();
  }
  state.file = std::fopen(path.c_str(), "w");
  if (state.file == nullptr) {
    return Status::Error("cannot open log file '" + path + "' for writing");
  }
  return Status::Ok();
}

void CloseLog() {
  LogState& state = State();
  const std::lock_guard<std::mutex> lock(state.mutex);
  if (state.file != nullptr) {
    std::fclose(state.file);
    state.file = nullptr;
  }
}

void Log(const std::string& line) {
  LogState& state = State();
  const std::lock_guard<std::mutex> lock(state.mutex);
  std::fprintf(stdout, "%s\n", line.c_str());
  std::fflush(stdout);
  if (state.file != nullptr) {
    std::fprintf(state.file, "%s\n", line.c_str());
    std::fflush(state.file);
  }
}

void LogKv(const std::string& key, const std::string& value) {
  Log(key + ": " + value);
}

void LogSection(const std::string& name) { Log("--- " + name + " ---"); }

}  // namespace cd
