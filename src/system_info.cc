// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "system_info.h"

#include <unistd.h>

#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <thread>

#include "cuda/gpu_backend.h"
#include "types.h"

namespace cd {
namespace {

// Runs a shell command and returns its standard output without the trailing
// newline. `ok` is set to false when the command cannot be started or exits
// with a non-zero status.
std::string RunCommand(const std::string& command, bool* ok) {
  std::string output;
  std::FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    *ok = false;
    return output;
  }
  char buffer[256];
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  const int exit_code = pclose(pipe);
  *ok = exit_code == 0;
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return output;
}

// Quotes a path for use inside a POSIX shell command.
std::string ShellQuote(const std::string& text) {
  std::string quoted = "'";
  for (const char c : text) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted += c;
    }
  }
  quoted += "'";
  return quoted;
}

std::string DirName(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) {
    return ".";
  }
  return slash == 0 ? "/" : path.substr(0, slash);
}

// Returns the short HEAD commit of the repository at `dir`, "-dirty" appended
// when tracked files changed; empty when `dir` is not in a git repository.
std::string GitCommitIn(const std::string& dir) {
  bool ok = false;
  const std::string base = "git -C " + ShellQuote(dir);
  const std::string head =
      RunCommand(base + " rev-parse --short HEAD 2>/dev/null", &ok);
  if (!ok || head.empty()) {
    return std::string();
  }
  const std::string changes = RunCommand(
      base + " status --porcelain --untracked-files=no 2>/dev/null", &ok);
  return ok && !changes.empty() ? head + "-dirty" : head;
}

}  // namespace

std::string CpuModel() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.compare(0, 10, "model name") == 0) {
      const size_t colon = line.find(':');
      if (colon != std::string::npos && colon + 2 <= line.size()) {
        return line.substr(colon + 2);
      }
    }
  }
  return "unknown";
}

int HardwareThreads() {
  const unsigned int n = std::thread::hardware_concurrency();
  return n == 0 ? 1 : static_cast<int>(n);
}

std::string IsoTimestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm local_time;
  localtime_r(&now, &local_time);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &local_time);
  return buffer;
}

std::string GitShortCommit() {
#ifdef CD_SOURCE_DIR
  // The source tree the binary was built from (set by the Makefile) is the
  // right answer even when the binary was built into another directory.
  const std::string from_source = GitCommitIn(CD_SOURCE_DIR);
  if (!from_source.empty()) {
    return from_source;
  }
#endif
  char exe[4096];
  const ssize_t length = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  if (length > 0) {
    exe[length] = '\0';
    const std::string exe_dir = DirName(exe);
    const std::string from_exe = GitCommitIn(exe_dir);
    if (!from_exe.empty()) {
      return from_exe;
    }
    const std::string from_parent = GitCommitIn(DirName(exe_dir));
    if (!from_parent.empty()) {
      return from_parent;
    }
  }
  const std::string from_cwd = GitCommitIn(".");
  return from_cwd.empty() ? "unknown" : from_cwd;
}

std::string CommandLine(int argc, char** argv) {
  std::string joined;
  for (int i = 0; i < argc; ++i) {
    if (i > 0) {
      joined += " ";
    }
    joined += argv[i];
  }
  return joined;
}

std::string CpuSummaryLine() {
  return "cpu: " + CpuModel() + " | " + std::to_string(HardwareThreads()) +
         " threads";
}

std::string GpuSummaryLine(int device) {
#ifndef CD_WITH_CUDA
  static_cast<void>(device);
  return "gpu: not available (CPU-only build)";
#else
  if (!GpuAvailable()) {
    return "gpu: none detected";
  }
  GpuInfo info;
  const Status status = GpuDeviceInfo(device, &info);
  if (!status.ok()) {
    return "gpu: none detected (" + status.message() + ")";
  }
  const auto version = [](int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
  };
  const size_t mib = static_cast<size_t>(1) << 20;
  return "gpu: " + info.name + " | cc " + std::to_string(info.cc_major) + "." +
         std::to_string(info.cc_minor) + " | " + std::to_string(info.sm_count) +
         " SMs | " + std::to_string(info.total_mem / mib) + " MiB total | " +
         std::to_string(info.free_mem / mib) + " MiB free | driver " +
         version(info.driver_version) + " | runtime " +
         version(info.runtime_version);
#endif
}

}  // namespace cd
