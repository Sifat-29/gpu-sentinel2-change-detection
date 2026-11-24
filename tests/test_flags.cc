// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <string>
#include <vector>

#include "flags.h"
#include "test_framework.h"
#include "types.h"

namespace cd {
namespace {

// Runs ParseFlags on {"change_detect", args...}.
Status Parse(const std::vector<std::string>& args, Flags* flags) {
  std::vector<std::string> storage = {"change_detect"};
  storage.insert(storage.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (std::string& s : storage) {
    argv.push_back(&s[0]);
  }
  return ParseFlags(static_cast<int>(argv.size()), argv.data(), flags);
}

bool Contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

// Expects a parse failure whose message mentions `part`.
void ExpectError(const std::vector<std::string>& args,
                 const std::string& part) {
  Flags flags;
  const Status status = Parse(args, &flags);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(Contains(status.message(), part));
}

TEST(Flags, DefaultsWithOnlyInput) {
  Flags flags;
  ASSERT_TRUE(Parse({"--input", "synthetic"}, &flags).ok());
  EXPECT_EQ(flags.mode, "detect");
  ASSERT_EQ(flags.backends.size(), 1u);
  EXPECT_EQ(flags.backends[0], "gpu");
  EXPECT_EQ(flags.output_dir, "out");
  EXPECT_EQ(flags.synthetic_height, 1024);
  EXPECT_EQ(flags.synthetic_width, 1024);
  EXPECT_EQ(flags.synthetic_dates, 48);
  EXPECT_NEAR(flags.synthetic_years, 4.0, 1e-9);
  EXPECT_EQ(flags.seed, 42u);
  EXPECT_EQ(flags.tile_rows, 0);
  EXPECT_EQ(flags.streams, 3);
  EXPECT_EQ(flags.block_size, 256);
  EXPECT_EQ(flags.io_threads, 8);
  EXPECT_EQ(flags.harmonics, 2);
  EXPECT_TRUE(flags.hold_trend);
  EXPECT_EQ(flags.consecutive, 2);
  EXPECT_NEAR(flags.threshold, 3.5, 1e-9);
  EXPECT_EQ(flags.repeat, 5);
  EXPECT_EQ(flags.warmup, 1);
  EXPECT_EQ(flags.compare_to, "none");
  EXPECT_EQ(flags.reference, "auto");
  EXPECT_EQ(flags.sample_count, 8);
  EXPECT_EQ(flags.png_max_dim, 2048);
  EXPECT_EQ(flags.date_a, 0);
  EXPECT_EQ(flags.date_b, -1);
  EXPECT_FALSE(flags.quiet);
  EXPECT_FALSE(flags.cleanup);
  EXPECT_EQ(flags.benchmark_csv, "out/benchmark.csv");
  EXPECT_FALSE(flags.run_id.empty());
}

TEST(Flags, ValidFullParse) {
  Flags flags;
  const Status status = Parse({"--mode",
                               "benchmark",
                               "--backend=cpu,gpu",
                               "--input",
                               "a/manifest.txt",
                               "--output-dir=o",
                               "--synthetic-size",
                               "64x96x30",
                               "--synthetic-years",
                               "3.5",
                               "--seed=7",
                               "--noise",
                               "0.05",
                               "--cloud-fraction",
                               "0.4",
                               "--break-fraction",
                               "0.1",
                               "--tile-rows",
                               "32",
                               "--streams",
                               "1",
                               "--block-size",
                               "128",
                               "--threads",
                               "4",
                               "--io-threads",
                               "16",
                               "--device",
                               "0",
                               "--harmonics",
                               "3",
                               "--train-years",
                               "1.5",
                               "--min-train-obs",
                               "20",
                               "--threshold",
                               "2.5",
                               "--consecutive",
                               "4",
                               "--rmse-floor",
                               "0.03",
                               "--ridge",
                               "0.001",
                               "--losses-only",
                               "--cleanup",
                               "--reference",
                               "ref.u8",
                               "--sample-pixels",
                               "5",
                               "--save-coefficients",
                               "--png-max-dim",
                               "512",
                               "--date-a",
                               "2",
                               "--date-b",
                               "9",
                               "--diff-threshold",
                               "0.3",
                               "--repeat",
                               "3",
                               "--warmup",
                               "0",
                               "--benchmark-csv",
                               "b.csv",
                               "--run-id",
                               "r1",
                               "--compare-to",
                               "cpu-fp32",
                               "--log-file",
                               "run.log",
                               "--quiet"},
                              &flags);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(flags.mode, "benchmark");
  ASSERT_EQ(flags.backends.size(), 2u);
  EXPECT_EQ(flags.backends[0], "cpu");
  EXPECT_EQ(flags.backends[1], "gpu");
  EXPECT_EQ(flags.input, "a/manifest.txt");
  EXPECT_EQ(flags.output_dir, "o");
  EXPECT_EQ(flags.synthetic_height, 64);
  EXPECT_EQ(flags.synthetic_width, 96);
  EXPECT_EQ(flags.synthetic_dates, 30);
  EXPECT_EQ(flags.seed, 7u);
  EXPECT_EQ(flags.tile_rows, 32);
  EXPECT_EQ(flags.streams, 1);
  EXPECT_EQ(flags.block_size, 128);
  EXPECT_EQ(flags.threads, 4);
  EXPECT_EQ(flags.io_threads, 16);
  EXPECT_EQ(flags.harmonics, 3);
  EXPECT_NEAR(flags.train_years, 1.5, 1e-6);
  EXPECT_EQ(flags.min_train_obs, 20);
  EXPECT_NEAR(flags.threshold, 2.5, 1e-6);
  EXPECT_TRUE(flags.losses_only);
  EXPECT_TRUE(flags.cleanup);
  EXPECT_TRUE(flags.save_coefficients);
  EXPECT_TRUE(flags.quiet);
  EXPECT_EQ(flags.sample_count, 5);
  EXPECT_EQ(flags.date_a, 2);
  EXPECT_EQ(flags.date_b, 9);
  EXPECT_EQ(flags.benchmark_csv, "b.csv");
  EXPECT_EQ(flags.run_id, "r1");
  EXPECT_EQ(flags.compare_to, "cpu-fp32");
  EXPECT_EQ(flags.log_file, "run.log");
}

TEST(Flags, HelpWinsAndNeedsNoInput) {
  Flags flags;
  ASSERT_TRUE(Parse({"--help"}, &flags).ok());
  EXPECT_TRUE(flags.help);
  Flags other;
  ASSERT_TRUE(Parse({"--mode", "bogus", "--help"}, &other).ok());
  EXPECT_TRUE(other.help);
}

TEST(Flags, UsageListsEveryFlag) {
  const std::string usage = UsageText();
  const char* const kFlags[] = {"--help",
                                "--mode",
                                "--backend",
                                "--input",
                                "--output-dir",
                                "--synthetic-size",
                                "--synthetic-years",
                                "--seed",
                                "--noise",
                                "--cloud-fraction",
                                "--break-fraction",
                                "--tile-rows",
                                "--streams",
                                "--block-size",
                                "--threads",
                                "--io-threads",
                                "--device",
                                "--harmonics",
                                "--train-years",
                                "--min-train-obs",
                                "--threshold",
                                "--consecutive",
                                "--rmse-floor",
                                "--ridge",
                                "--losses-only",
                                "--valid-classes",
                                "--cleanup",
                                "--reference",
                                "--sample-pixels",
                                "--save-coefficients",
                                "--png-max-dim",
                                "--date-a",
                                "--date-b",
                                "--diff-threshold",
                                "--repeat",
                                "--warmup",
                                "--benchmark-csv",
                                "--run-id",
                                "--compare-to",
                                "--log-file",
                                "--quiet"};
  for (const char* flag : kFlags) {
    EXPECT_TRUE(Contains(usage, flag));
  }
  for (const char* group : {"Modes:", "Input:", "Model:", "Detection:",
                            "Streaming", "Outputs:", "Benchmark:"}) {
    EXPECT_TRUE(Contains(usage, group));
  }
}

TEST(Flags, MissingInputIsError) {
  ExpectError({}, "--input is required");
  ExpectError({"--mode", "detect"}, "--input is required");
}

TEST(Flags, UnknownFlagAndStrayArgument) {
  ExpectError({"--input", "x", "--bogus", "1"}, "unknown flag '--bogus'");
  ExpectError({"--input", "x", "extra"}, "unexpected argument 'extra'");
}

TEST(Flags, MissingValue) {
  ExpectError({"--input"}, "requires a value");
  ExpectError({"--input", "x", "--tile-rows"}, "requires a value");
}

TEST(Flags, BadModeAndBackend) {
  ExpectError({"--mode", "bogus", "--input", "x"}, "unknown mode 'bogus'");
  ExpectError({"--backend", "tpu", "--input", "x"}, "unknown backend 'tpu'");
  ExpectError({"--backend", "gpu,cpu", "--input", "x"},
              "only allowed with --mode benchmark");
  ExpectError({"--compare-to", "tpu", "--input", "x"}, "--compare-to");
}

TEST(Flags, NumericRangeErrors) {
  ExpectError({"--input", "synthetic", "--tile-rows", "-3"},
              "--tile-rows must be an integer >= 0");
  ExpectError({"--input", "x", "--tile-rows", "abc"}, "--tile-rows");
  ExpectError({"--input", "x", "--tile-rows", "1.5"}, "--tile-rows");
  ExpectError({"--input", "x", "--streams", "2"}, "--streams must be 1 or 3");
  ExpectError({"--input", "x", "--io-threads", "0"}, "--io-threads");
  ExpectError({"--input", "x", "--io-threads", "65"}, "--io-threads");
  ExpectError({"--input", "x", "--io-threads", "many"}, "--io-threads");
  ExpectError({"--input", "x", "--block-size", "100"}, "multiple of 32");
  ExpectError({"--input", "x", "--block-size", "2048"}, "multiple of 32");
  ExpectError({"--input", "x", "--harmonics", "4"}, "--harmonics");
  ExpectError({"--input", "x", "--train-years", "0"}, "--train-years");
  ExpectError({"--input", "x", "--trend", "linear"}, "--trend must be hold");
  ExpectError({"--input", "x", "--threshold", "nan"}, "--threshold");
  ExpectError({"--input", "x", "--consecutive", "0"}, "--consecutive");
  ExpectError({"--input", "x", "--ridge", "-1"}, "--ridge");
  ExpectError({"--input", "x", "--cloud-fraction", "1.5"}, "--cloud-fraction");
  ExpectError({"--input", "x", "--break-fraction", "-0.1"}, "--break-fraction");
  ExpectError({"--input", "x", "--seed", "-1"}, "--seed");
  ExpectError({"--input", "x", "--repeat", "0"}, "--repeat");
  ExpectError({"--input", "x", "--png-max-dim", "0"}, "--png-max-dim");
  ExpectError({"--input", "x", "--device", "-1"}, "--device");
  ExpectError({"--input", "x", "--date-b", "-2"}, "--date-b");
}

TEST(Flags, MinTrainObsBelowParameterCount) {
  // K = 2 gives p = 6, so at least 7 observations are needed.
  ExpectError({"--input", "x", "--min-train-obs", "5"}, "min-train-obs");
  Flags flags;
  EXPECT_TRUE(Parse({"--input", "x", "--min-train-obs", "7"}, &flags).ok());
}

TEST(Flags, SwitchesRejectValues) {
  ExpectError({"--input", "x", "--quiet=maybe"}, "--quiet");
  Flags flags;
  ASSERT_TRUE(Parse({"--input", "x", "--quiet=false"}, &flags).ok());
  EXPECT_FALSE(flags.quiet);
}

TEST(Flags, SyntheticSize) {
  Flags flags;
  ASSERT_TRUE(
      Parse({"--input", "synthetic", "--synthetic-size=8x16x24"}, &flags).ok());
  EXPECT_EQ(flags.synthetic_height, 8);
  EXPECT_EQ(flags.synthetic_width, 16);
  EXPECT_EQ(flags.synthetic_dates, 24);
  ExpectError({"--input", "x", "--synthetic-size", "8x16"}, "--synthetic-size");
  ExpectError({"--input", "x", "--synthetic-size", "8x0x4"},
              "--synthetic-size");
  ExpectError({"--input", "x", "--synthetic-size", "axbxc"},
              "--synthetic-size");
  ExpectError({"--input", "x", "--synthetic-size", "1x2x3x4"},
              "--synthetic-size");
}

TEST(Flags, ValidClassesMask) {
  Flags flags;
  ASSERT_TRUE(
      Parse({"--input", "x", "--valid-classes", "4,5,6,11"}, &flags).ok());
  ModelParams params;
  ASSERT_TRUE(FlagsToModelParams(flags, &params).ok());
  EXPECT_EQ(params.valid_class_mask,
            (1u << 4) | (1u << 5) | (1u << 6) | (1u << 11));

  Flags defaults;
  ASSERT_TRUE(Parse({"--input", "x"}, &defaults).ok());
  ASSERT_TRUE(FlagsToModelParams(defaults, &params).ok());
  EXPECT_EQ(params.valid_class_mask, (1u << 4) | (1u << 5) | (1u << 6));

  ExpectError({"--input", "x", "--valid-classes", "4,16"}, "--valid-classes");
  ExpectError({"--input", "x", "--valid-classes", "4,,5"}, "--valid-classes");
  ExpectError({"--input", "x", "--valid-classes", ""}, "--valid-classes");
}

TEST(Flags, BackendListsAndAll) {
  Flags flags;
  ASSERT_TRUE(
      Parse({"--mode", "benchmark", "--input", "x", "--backend", "all"}, &flags)
          .ok());
  ASSERT_EQ(flags.backends.size(), 5u);
  const char* const kExpected[] = {"gpu", "gpu-unfused", "cpu", "cpu-fp32",
                                   "cpu-mt"};
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(flags.backends[i], kExpected[i]);
    Backend backend;
    EXPECT_TRUE(ParseBackend(flags.backends[i], &backend).ok());
  }

  Flags listed;
  ASSERT_TRUE(
      Parse({"--mode=benchmark", "--input=x", "--backend=cpu-mt,gpu,cpu-mt"},
            &listed)
          .ok());
  ASSERT_EQ(listed.backends.size(), 2u);
  EXPECT_EQ(listed.backends[0], "cpu-mt");
  EXPECT_EQ(listed.backends[1], "gpu");
}

TEST(Flags, SamplePixelsList) {
  Flags flags;
  ASSERT_TRUE(
      Parse({"--input", "x", "--sample-pixels", "3,4;10,20"}, &flags).ok());
  ASSERT_EQ(flags.sample_pixel_list.size(), 2u);
  EXPECT_EQ(flags.sample_pixel_list[0].first, 3);
  EXPECT_EQ(flags.sample_pixel_list[1].second, 20);
  EXPECT_EQ(flags.sample_count, 2);
  ExpectError({"--input", "x", "--sample-pixels", "3,4;5"}, "--sample-pixels");
  ExpectError({"--input", "x", "--sample-pixels", "-2"}, "--sample-pixels");
}

TEST(Flags, TraceCsvIsForGpuDetectRunsOnly) {
  Flags flags;
  ASSERT_TRUE(Parse({"--input", "x", "--trace-csv", "out/t.csv"}, &flags).ok());
  EXPECT_EQ(flags.trace_csv, "out/t.csv");
  RunConfig config;
  ASSERT_TRUE(FlagsToRunConfig(flags, Backend::kGpuFused, &config).ok());
  EXPECT_EQ(config.trace_csv, "out/t.csv");
  Flags unfused;
  EXPECT_TRUE(
      Parse({"--input", "x", "--backend", "gpu-unfused", "--trace-csv", "t"},
            &unfused)
          .ok());
  Flags off;
  ASSERT_TRUE(Parse({"--input", "x"}, &off).ok());
  EXPECT_TRUE(off.trace_csv.empty());
  ExpectError({"--input", "x", "--backend", "cpu-mt", "--trace-csv", "t"},
              "--trace-csv needs a GPU backend");
  ExpectError({"--input", "x", "--backend", "cpu", "--trace-csv", "t"},
              "--trace-csv needs a GPU backend");
  ExpectError({"--input", "x", "--mode", "quick-diff", "--trace-csv", "t"},
              "--trace-csv is only supported with --mode detect");
}

TEST(Flags, ConversionToConfigs) {
  Flags flags;
  ASSERT_TRUE(
      Parse({"--input", "x", "--harmonics", "1", "--losses-only", "--threshold",
             "4", "--tile-rows", "16", "--streams", "1", "--threads", "3",
             "--io-threads", "5", "--save-coefficients"},
            &flags)
          .ok());
  ModelParams params;
  ASSERT_TRUE(FlagsToModelParams(flags, &params).ok());
  EXPECT_EQ(params.harmonics, 1);
  EXPECT_EQ(params.min_train_obs, 12);  // 3 * (2 + 2)
  EXPECT_TRUE(params.losses_only);
  EXPECT_NEAR(params.threshold, 4.0, 1e-6);
  EXPECT_TRUE(params.hold_trend);  // default
  Flags extrapolate;
  ASSERT_TRUE(
      Parse({"--input", "x", "--trend", "extrapolate"}, &extrapolate).ok());
  ASSERT_TRUE(FlagsToModelParams(extrapolate, &params).ok());
  EXPECT_FALSE(params.hold_trend);
  Flags hold;
  ASSERT_TRUE(Parse({"--input", "x", "--trend=hold"}, &hold).ok());
  EXPECT_TRUE(hold.hold_trend);

  RunConfig config;
  ASSERT_TRUE(FlagsToRunConfig(flags, Backend::kCpuMt, &config).ok());
  EXPECT_EQ(config.backend, Backend::kCpuMt);
  EXPECT_EQ(config.tile_rows, 16);
  EXPECT_EQ(config.streams, 1);
  EXPECT_EQ(config.threads, 3);
  EXPECT_EQ(config.io_threads, 5);
  EXPECT_TRUE(config.save_coefficients);
}

}  // namespace
}  // namespace cd
