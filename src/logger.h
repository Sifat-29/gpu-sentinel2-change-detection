// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Console and file logger. Every line goes to stdout and, when a log file is
// set, to that file as well; both are flushed per line so logs survive a crash.
#ifndef SRC_LOGGER_H_
#define SRC_LOGGER_H_

#include <string>

#include "status.h"

namespace cd {

// Opens `path` for writing (truncating it) as the secondary log destination.
// An empty path disables file logging. Returns an error if the file cannot be
// opened; console logging keeps working in that case.
Status InitLog(const std::string& path);

// Closes the log file, if any.
void CloseLog();

// Writes one line (a newline is appended).
void Log(const std::string& line);

// Writes "key: value".
void LogKv(const std::string& key, const std::string& value);

// Writes "--- name ---".
void LogSection(const std::string& name);

}  // namespace cd

#endif  // SRC_LOGGER_H_
