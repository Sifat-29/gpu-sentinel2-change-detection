// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Minimal error-reporting value type. The project does not use exceptions;
// every fallible function returns a cd::Status.
#ifndef SRC_STATUS_H_
#define SRC_STATUS_H_

#include <string>
#include <utility>

namespace cd {

// Result of an operation: either success or a human-readable error message.
class Status {
 public:
  // Returns a success status.
  static Status Ok() { return Status(true, std::string()); }

  // Returns a failure status carrying `message`.
  static Status Error(std::string message) {
    return Status(false, std::move(message));
  }

  bool ok() const { return ok_; }

  // The error message; empty for a success status.
  const std::string& message() const { return message_; }

 private:
  Status(bool ok, std::string message)
      : ok_(ok), message_(std::move(message)) {}

  bool ok_;
  std::string message_;
};

}  // namespace cd

#endif  // SRC_STATUS_H_
