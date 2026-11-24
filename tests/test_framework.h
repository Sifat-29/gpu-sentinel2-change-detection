// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Minimal unit-test framework (no external dependency). Usage:
//
//   TEST(Suite, Name) { EXPECT_EQ(1 + 1, 2); }
//
// EXPECT_* records a failure and continues; ASSERT_* records it and returns
// from the test function. Suites whose name starts with "Large" are skipped
// unless --filter selects them. The header also compiles as CUDA (.cu tests).
#ifndef TESTS_TEST_FRAMEWORK_H_
#define TESTS_TEST_FRAMEWORK_H_

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace cd {
namespace testing {

using TestFunction = void (*)();

struct TestInfo {
  const char* suite;
  const char* name;
  TestFunction function;
};

// All registered tests (filled by static registration).
inline std::vector<TestInfo>& Registry() {
  static std::vector<TestInfo> tests;
  return tests;
}

// Constructing a Registrar registers one test.
class Registrar {
 public:
  Registrar(const char* suite, const char* name, TestFunction function) {
    Registry().push_back({suite, name, function});
  }
};

// Number of failures recorded since the last ResetFailures().
inline int& FailureCount() {
  static int count = 0;
  return count;
}

inline void ReportFailure(const char* file, int line,
                          const std::string& message) {
  ++FailureCount();
  std::fprintf(stdout, "%s:%d: failure: %s\n", file, line, message.c_str());
}

// Formats a value for a failure message; small integer types print as numbers.
template <typename T>
std::string ToText(const T& value) {
  std::ostringstream stream;
  if constexpr (std::is_enum<T>::value) {
    stream << static_cast<int64_t>(value);
  } else if constexpr (std::is_arithmetic<T>::value) {
    stream << +value;
  } else {
    stream << value;
  }
  return stream.str();
}

// Equality / ordering that is correct for mixed signed and unsigned integers
// and falls back to the built-in operators for every other type.
template <typename A, typename B>
bool Equal(const A& a, const B& b) {
  if constexpr (std::is_integral<A>::value && std::is_integral<B>::value &&
                std::is_signed<A>::value != std::is_signed<B>::value) {
    if constexpr (std::is_signed<A>::value) {
      return a >= 0 && static_cast<std::make_unsigned_t<A>>(a) == b;
    } else {
      return b >= 0 && a == static_cast<std::make_unsigned_t<B>>(b);
    }
  } else {
    return a == b;
  }
}

template <typename A, typename B>
bool Less(const A& a, const B& b) {
  if constexpr (std::is_integral<A>::value && std::is_integral<B>::value &&
                std::is_signed<A>::value != std::is_signed<B>::value) {
    if constexpr (std::is_signed<A>::value) {
      return a < 0 || static_cast<std::make_unsigned_t<A>>(a) < b;
    } else {
      return b >= 0 && a < static_cast<std::make_unsigned_t<B>>(b);
    }
  } else {
    return a < b;
  }
}

// Failure messages shared by the macros below.
template <typename A, typename B>
std::string BinaryMessage(const char* a_text, const char* op,
                          const char* b_text, const A& a, const B& b) {
  return std::string("expected ") + a_text + " " + op + " " + b_text +
         " (left: " + ToText(a) + ", right: " + ToText(b) + ")";
}

}  // namespace testing
}  // namespace cd

#define TEST(suite, name)                                              \
  static void CdTest_##suite##_##name();                               \
  static const ::cd::testing::Registrar kCdRegistrar_##suite##_##name( \
      #suite, #name, &CdTest_##suite##_##name);                        \
  static void CdTest_##suite##_##name()

#define CD_CHECK_BINARY_(a, b, predicate, op, on_fail)           \
  do {                                                           \
    const auto& cd_a = (a);                                      \
    const auto& cd_b = (b);                                      \
    if (!(predicate)) {                                          \
      ::cd::testing::ReportFailure(                              \
          __FILE__, __LINE__,                                    \
          ::cd::testing::BinaryMessage(#a, op, #b, cd_a, cd_b)); \
      on_fail;                                                   \
    }                                                            \
  } while (0)

#define CD_CHECK_TRUE_(condition, expected, on_fail)                       \
  do {                                                                     \
    if (static_cast<bool>(condition) != (expected)) {                      \
      ::cd::testing::ReportFailure(__FILE__, __LINE__,                     \
                                   std::string("expected ") + #condition + \
                                       " to be " +                         \
                                       ((expected) ? "true" : "false"));   \
      on_fail;                                                             \
    }                                                                      \
  } while (0)

#define CD_CHECK_NEAR_(a, b, tolerance, on_fail)                     \
  do {                                                               \
    const double cd_a = static_cast<double>(a);                      \
    const double cd_b = static_cast<double>(b);                      \
    const double cd_tol = static_cast<double>(tolerance);            \
    if (!(std::fabs(cd_a - cd_b) <= cd_tol)) {                       \
      ::cd::testing::ReportFailure(                                  \
          __FILE__, __LINE__,                                        \
          ::cd::testing::BinaryMessage(#a, "near", #b, cd_a, cd_b) + \
              " within " + ::cd::testing::ToText(cd_tol));           \
      on_fail;                                                       \
    }                                                                \
  } while (0)

#define CD_NOOP_ static_cast<void>(0)

#define EXPECT_TRUE(c) CD_CHECK_TRUE_(c, true, CD_NOOP_)
#define EXPECT_FALSE(c) CD_CHECK_TRUE_(c, false, CD_NOOP_)
#define EXPECT_EQ(a, b) \
  CD_CHECK_BINARY_(a, b, ::cd::testing::Equal(cd_a, cd_b), "==", CD_NOOP_)
#define EXPECT_NE(a, b) \
  CD_CHECK_BINARY_(a, b, !::cd::testing::Equal(cd_a, cd_b), "!=", CD_NOOP_)
#define EXPECT_LT(a, b) \
  CD_CHECK_BINARY_(a, b, ::cd::testing::Less(cd_a, cd_b), "<", CD_NOOP_)
#define EXPECT_LE(a, b) \
  CD_CHECK_BINARY_(a, b, !::cd::testing::Less(cd_b, cd_a), "<=", CD_NOOP_)
#define EXPECT_GT(a, b) \
  CD_CHECK_BINARY_(a, b, ::cd::testing::Less(cd_b, cd_a), ">", CD_NOOP_)
#define EXPECT_GE(a, b) \
  CD_CHECK_BINARY_(a, b, !::cd::testing::Less(cd_a, cd_b), ">=", CD_NOOP_)
#define EXPECT_NEAR(a, b, tol) CD_CHECK_NEAR_(a, b, tol, CD_NOOP_)

#define ASSERT_TRUE(c) CD_CHECK_TRUE_(c, true, return)
#define ASSERT_FALSE(c) CD_CHECK_TRUE_(c, false, return)
#define ASSERT_EQ(a, b) \
  CD_CHECK_BINARY_(a, b, ::cd::testing::Equal(cd_a, cd_b), "==", return)
#define ASSERT_NE(a, b) \
  CD_CHECK_BINARY_(a, b, !::cd::testing::Equal(cd_a, cd_b), "!=", return)
#define ASSERT_LT(a, b) \
  CD_CHECK_BINARY_(a, b, ::cd::testing::Less(cd_a, cd_b), "<", return)
#define ASSERT_LE(a, b) \
  CD_CHECK_BINARY_(a, b, !::cd::testing::Less(cd_b, cd_a), "<=", return)
#define ASSERT_GT(a, b) \
  CD_CHECK_BINARY_(a, b, ::cd::testing::Less(cd_b, cd_a), ">", return)
#define ASSERT_GE(a, b) \
  CD_CHECK_BINARY_(a, b, !::cd::testing::Less(cd_a, cd_b), ">=", return)
#define ASSERT_NEAR(a, b, tol) CD_CHECK_NEAR_(a, b, tol, return)

#endif  // TESTS_TEST_FRAMEWORK_H_
