// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17): temporary directories
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "app.h"
#include "app_common.h"
#include "cuda/gpu_backend.h"
#include "flags.h"
#include "test_framework.h"

namespace cd {
namespace {

namespace fs = std::filesystem;

// A unique temporary directory removed on destruction.
class TempDir {
 public:
  TempDir() {
    std::string pattern =
        (fs::temp_directory_path() / "cd_test_app_XXXXXX").string();
    const char* made = mkdtemp(pattern.data());
    path_ = made != nullptr ? made : "";
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir() {
    if (!path_.empty()) {
      std::error_code error;
      fs::remove_all(path_, error);
    }
  }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Parses `arguments` like the command line (argv[0] is added) and runs the
// application in-process. Returns 2 when the flags are rejected.
int Run(const std::vector<std::string>& arguments) {
  std::vector<std::string> storage = {"change_detect"};
  storage.insert(storage.end(), arguments.begin(), arguments.end());
  std::vector<char*> argv;
  for (std::string& text : storage) {
    argv.push_back(text.data());
  }
  Flags flags;
  const Status status =
      ParseFlags(static_cast<int>(argv.size()), argv.data(), &flags);
  if (!status.ok()) {
    return 2;
  }
  return RunApp(flags, static_cast<int>(argv.size()), argv.data());
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

int64_t FileSize(const std::string& path) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  return error ? -1 : static_cast<int64_t>(size);
}

bool Contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

size_t CountLines(const std::string& text) {
  size_t lines = 0;
  for (const char c : text) {
    lines += c == '\n' ? 1 : 0;
  }
  return lines;
}

// Checks the detect output files of a 64 x 80 detect run (p = 6, no
// coefficients unless `coefficients`).
void ExpectDetectFiles(const std::string& dir, bool coefficients) {
  const int64_t pixels = 64 * 80;
  EXPECT_EQ(FileSize(dir + "/break_index.i16"), pixels * 2);
  EXPECT_EQ(FileSize(dir + "/magnitude.f32"), pixels * 4);
  EXPECT_EQ(FileSize(dir + "/status.u8"), pixels);
  EXPECT_EQ(FileSize(dir + "/rmse.f32"), pixels * 4);
  EXPECT_EQ(FileSize(dir + "/train_obs.u16"), pixels * 2);
  EXPECT_EQ(FileSize(dir + "/coefficients.f32"),
            coefficients ? pixels * 6 * 4 : -1);
  EXPECT_GT(FileSize(dir + "/change_map.png"), 0);
  EXPECT_GT(FileSize(dir + "/pixel_stats.csv"), 0);
  EXPECT_GT(FileSize(dir + "/trajectories.csv"), 0);
  EXPECT_GT(FileSize(dir + "/result_manifest.txt"), 0);
}

TEST(App, DetectSyntheticCpuMtWritesAllOutputsAndLog) {
  TempDir temp;
  ASSERT_FALSE(temp.path().empty());
  const std::string out = temp.path() + "/out";  // created by the program
  const std::string log = out + "/log.txt";
  ASSERT_EQ(Run({"--mode", "detect", "--input", "synthetic", "--synthetic-size",
                 "64x80x40", "--backend", "cpu-mt", "--output-dir", out,
                 "--log-file", log, "--quiet"}),
            0);
  ExpectDetectFiles(out, false);
  EXPECT_EQ(FileSize(out + "/validation.csv"), -1);  // no reference
  const std::string text = ReadFile(log);
  EXPECT_TRUE(Contains(text, "=== change_detect 1.0 ==="));
  EXPECT_TRUE(Contains(text, "--- device ---\ngpu: "));
  EXPECT_TRUE(Contains(text, "\ncpu: "));
  EXPECT_TRUE(Contains(text, "image size H x W: 64 x 80\n"));
  EXPECT_TRUE(Contains(text, "samples (pixel-dates): 204800\n"));
  EXPECT_TRUE(Contains(text, " | trend: hold | train_years: "));
  EXPECT_TRUE(Contains(text, " | read_wait: "));
  EXPECT_TRUE(Contains(text, " | setup: "));
  EXPECT_LT(text.find(" | read_wait: "), text.find(" | setup: "));
  EXPECT_LT(text.find(" | setup: "), text.find(" | h2d: "));
  EXPECT_TRUE(Contains(text, "\nsynthetic_recovery: recovered "));
  EXPECT_TRUE(
      Contains(text, "SUMMARY: processed 5120 pixel time series (204800 "));
  EXPECT_TRUE(Contains(text, "(backend cpu-mt)\n"));
  // The SUMMARY is the last line.
  EXPECT_TRUE(text.find("SUMMARY:") > text.find("--- results ---"));
  EXPECT_EQ(text.find('\n', text.find("SUMMARY:")), text.size() - 1);
  // --quiet suppresses the per-tile lines.
  EXPECT_FALSE(Contains(text, "[tile "));
  // The result manifest names the synthetic input and the model parameters.
  const std::string manifest = ReadFile(out + "/result_manifest.txt");
  EXPECT_TRUE(Contains(manifest, "\ncube_manifest synthetic\n"));
  EXPECT_TRUE(Contains(manifest, "\nvalid_classes 4,5,6\n"));
  EXPECT_TRUE(Contains(manifest, "\ntrend hold\n"));
  EXPECT_TRUE(Contains(manifest, "\nfile change_map.png png "));
}

TEST(App, DetectWithComparisonCoefficientsAndReference) {
  TempDir temp;
  ASSERT_FALSE(temp.path().empty());
  const std::string out = temp.path() + "/out";
  ASSERT_TRUE(fs::create_directories(out));
  // A reference raster with no loss anywhere (all pixels count as stable).
  const std::string reference = temp.path() + "/reference.u8";
  {
    std::ofstream file(reference, std::ios::binary);
    const std::string zeros(64 * 80, '\0');
    file.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
  }
  ASSERT_EQ(Run({"--input",
                 "synthetic",
                 "--synthetic-size",
                 "64x80x40",
                 "--backend",
                 "cpu-fp32",
                 "--compare-to",
                 "cpu",
                 "--save-coefficients",
                 "--cleanup",
                 "--reference",
                 reference,
                 "--sample-pixels",
                 "3,4;10,20",
                 "--png-max-dim",
                 "32",
                 "--output-dir",
                 out,
                 "--log-file",
                 out + "/log.txt"}),
            0);
  ExpectDetectFiles(out, true);
  EXPECT_GT(FileSize(out + "/validation.csv"), 0);
  const std::string log = ReadFile(out + "/log.txt");
  EXPECT_TRUE(Contains(log, "\nagreement_vs_cpu: identical_break_index "));
  EXPECT_FALSE(Contains(log, "max_coef_diff -"));  // coefficients compared
  EXPECT_TRUE(Contains(log, "\nvalidation: precision "));
  EXPECT_TRUE(Contains(log, "cleanup: yes"));
  EXPECT_TRUE(Contains(log, "[tile 1/1] rows 0..63"));
  // Exactly the two requested pixels, two rows each.
  EXPECT_EQ(CountLines(ReadFile(out + "/pixel_stats.csv")), 1u + 4u);
  EXPECT_TRUE(Contains(ReadFile(out + "/pixel_stats.csv"), "\n3,4,cpu-fp32,"));
}

TEST(App, SamplePixelOutsideTheImageIsAnError) {
  TempDir temp;
  EXPECT_EQ(
      Run({"--input", "synthetic", "--synthetic-size", "64x80x40", "--backend",
           "cpu", "--sample-pixels", "64,0", "--output-dir", temp.path()}),
      1);
}

TEST(App, GenerateThenDetectAndQuickDiffFromDisk) {
  TempDir temp;
  ASSERT_FALSE(temp.path().empty());
  const std::string cube = temp.path() + "/cube";
  ASSERT_EQ(Run({"--mode", "generate", "--input", "synthetic",
                 "--synthetic-size", "64x80x40", "--output-dir", cube}),
            0);
  EXPECT_GT(FileSize(cube + "/manifest.txt"), 0);
  EXPECT_EQ(FileSize(cube + "/red.u16"), 64 * 80 * 40 * 2);
  const std::string out = temp.path() + "/detect";
  ASSERT_EQ(Run({"--input", cube + "/manifest.txt", "--backend", "cpu",
                 "--output-dir", out, "--log-file", out + "/log.txt"}),
            0);
  ExpectDetectFiles(out, false);
  const std::string log = ReadFile(out + "/log.txt");
  EXPECT_TRUE(Contains(log, "pixel time series: 5120\n"));
  EXPECT_FALSE(Contains(log, "synthetic_recovery"));  // only for synthetic
  const std::string quick = temp.path() + "/quick";
  ASSERT_EQ(Run({"--mode", "quick-diff", "--input", cube + "/manifest.txt",
                 "--backend", "cpu-mt", "--output-dir", quick, "--log-file",
                 quick + "/log.txt"}),
            0);
  EXPECT_EQ(FileSize(quick + "/quick_diff.f32"), 64 * 80 * 4);
  EXPECT_EQ(FileSize(quick + "/quick_diff_flag.u8"), 64 * 80);
  EXPECT_GT(FileSize(quick + "/quick_diff.png"), 0);
  const std::string quick_log = ReadFile(quick + "/log.txt");
  EXPECT_TRUE(Contains(quick_log,
                       "SUMMARY: processed 5120 pixels (2 dates, "
                       "10240 samples) in "));
}

// Returns the log line that starts with `prefix` (without the newline), or an
// empty string.
std::string LineStartingWith(const std::string& text,
                             const std::string& prefix) {
  const size_t start = text.find("\n" + prefix);
  if (start == std::string::npos) {
    return "";
  }
  const size_t end = text.find('\n', start + 1);
  return text.substr(start + 1, end - start - 1);
}

TEST(App, ExtrapolateTrendIsLogged) {
  TempDir temp;
  ASSERT_FALSE(temp.path().empty());
  const std::string out = temp.path() + "/out";
  ASSERT_EQ(
      Run({"--input", "synthetic", "--synthetic-size", "32x32x40", "--backend",
           "cpu-mt", "--trend", "extrapolate", "--output-dir", out,
           "--log-file", out + "/log.txt", "--quiet"}),
      0);
  EXPECT_TRUE(Contains(ReadFile(out + "/log.txt"),
                       " | trend: extrapolate | train_years: "));
  EXPECT_TRUE(Contains(ReadFile(out + "/result_manifest.txt"),
                       "\ntrend extrapolate\n"));
}

// Logs and the result manifest name paths relative to the working directory
// when they lie below it, so committed logs contain no personal absolute
// paths; paths outside it are shown as given.
TEST(App, LogShowsPathsRelativeToWorkingDirectory) {
  TempDir temp;
  ASSERT_FALSE(temp.path().empty());
  const fs::path original = fs::current_path();
  fs::current_path(temp.path());
  ASSERT_EQ(Run({"--mode", "generate", "--input", "synthetic",
                 "--synthetic-size", "32x40x40", "--output-dir", "cube"}),
            0);
  {
    std::ofstream file("reference.u8", std::ios::binary);
    const std::string zeros(32 * 40, '\0');
    file.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
  }
  // Absolute spellings of paths below the working directory.
  const std::string absolute_cube = temp.path() + "/cube/manifest.txt";
  const std::string absolute_reference = temp.path() + "/reference.u8";
  const std::string absolute_out = temp.path() + "/out";
  const int code = Run({"--input", absolute_cube, "--backend", "cpu",
                        "--reference", absolute_reference, "--output-dir",
                        absolute_out, "--log-file", "run.log", "--quiet"});
  const std::string log = ReadFile("run.log");
  const std::string manifest = ReadFile("out/result_manifest.txt");
  fs::current_path(original);
  ASSERT_EQ(code, 0);
  EXPECT_TRUE(Contains(LineStartingWith(log, "cube: "), "(cube/manifest.txt)"));
  EXPECT_EQ(LineStartingWith(log, "reference: "), "reference: reference.u8");
  EXPECT_EQ(LineStartingWith(log, "outputs: "), "outputs: out");
  EXPECT_TRUE(Contains(manifest, "\ncube_manifest cube/manifest.txt\n"));
  EXPECT_TRUE(Contains(manifest, "\nreference_file reference.u8\n"));
  // A path outside the working directory is kept as given.
  EXPECT_EQ(DisplayPath("/", "/given/path"), "/given/path");
  EXPECT_EQ(DisplayPath(absolute_cube, "elsewhere"), "elsewhere");
}

TEST(App, QuickDiffRejectsBadDates) {
  TempDir temp;
  EXPECT_EQ(Run({"--mode", "quick-diff", "--input", "synthetic",
                 "--synthetic-size", "16x16x40", "--backend", "cpu", "--date-a",
                 "40", "--output-dir", temp.path()}),
            1);
}

TEST(App, BenchmarkWritesOneRowPerTimedRun) {
  TempDir temp;
  ASSERT_FALSE(temp.path().empty());
  const std::string out = temp.path() + "/bench";
  ASSERT_EQ(Run({"--mode", "benchmark", "--input", "synthetic",
                 "--synthetic-size", "64x80x40", "--repeat", "2", "--warmup",
                 "0", "--backend", "cpu,cpu-mt", "--compare-to", "cpu",
                 "--output-dir", out, "--log-file", out + "/log.txt"}),
            0);
  const std::string csv = ReadFile(out + "/benchmark.csv");
  EXPECT_EQ(CountLines(csv), 1u + 4u);  // header and four timed runs
  const std::string log = ReadFile(out + "/log.txt");
  EXPECT_TRUE(Contains(log, "bench: cpu repeat 2/2 backend_total "));
  EXPECT_TRUE(Contains(log, "bench: cpu-mt repeat 1/2 backend_total "));
  EXPECT_TRUE(Contains(log, "| speedup vs cpu-mt"));
  EXPECT_TRUE(Contains(log, "(backend cpu)\n"));  // first backend in the list
  // Running again appends rows under the same header.
  ASSERT_EQ(Run({"--mode", "benchmark", "--input", "synthetic",
                 "--synthetic-size", "32x32x40", "--repeat", "1", "--warmup",
                 "0", "--backend", "cpu-mt", "--output-dir", out}),
            0);
  EXPECT_EQ(CountLines(ReadFile(out + "/benchmark.csv")), 1u + 4u + 1u);
}

TEST(App, GpuBackendWithoutGpuIsAClearError) {
  if (GpuAvailable()) {
    return;  // covered by DetectSyntheticGpu
  }
  TempDir temp;
  const std::string log = temp.path() + "/log.txt";
  EXPECT_EQ(
      Run({"--input", "synthetic", "--synthetic-size", "16x16x40", "--backend",
           "gpu", "--output-dir", temp.path(), "--log-file", log}),
      1);
  EXPECT_TRUE(Contains(
      ReadFile(log),
      "error: backend 'gpu' requires a CUDA build and an NVIDIA GPU (see "
      "INSTALL)\n"));
}

TEST(App, CoregNeedsAGpuBackend) {
  TempDir temp;
  const std::string log = temp.path() + "/log.txt";
  EXPECT_EQ(Run({"--mode", "coreg", "--input", "synthetic", "--backend", "cpu",
                 "--log-file", log, "--output-dir", temp.path()}),
            1);
  EXPECT_TRUE(Contains(
      ReadFile(log),
      "error: --mode coreg needs a GPU backend (gpu or gpu-unfused), got "
      "'cpu'"));
}

TEST(App, MissingManifestIsAnError) {
  TempDir temp;
  EXPECT_EQ(Run({"--input", temp.path() + "/nope/manifest.txt", "--backend",
                 "cpu", "--output-dir", temp.path()}),
            1);
}

// GPU variant: skipped when no GPU (or a CPU-only build) is present.
TEST(App, DetectSyntheticGpuAgreesWithCpu) {
  if (!GpuAvailable()) {
    return;
  }
  TempDir temp;
  ASSERT_FALSE(temp.path().empty());
  const std::string out = temp.path() + "/out";
  ASSERT_EQ(Run({"--input", "synthetic", "--synthetic-size", "64x80x40",
                 "--backend", "gpu", "--compare-to", "cpu", "--cleanup",
                 "--output-dir", out, "--log-file", out + "/log.txt"}),
            0);
  ExpectDetectFiles(out, false);
  const std::string log = ReadFile(out + "/log.txt");
  EXPECT_TRUE(Contains(log, "SUMMARY: processed 5120 pixel time series"));
  EXPECT_TRUE(Contains(log, "(backend gpu)\n"));
  EXPECT_TRUE(Contains(log, "\nagreement_vs_cpu: identical_break_index "));
  const std::string bench = temp.path() + "/bench";
  ASSERT_EQ(
      Run({"--mode", "benchmark", "--input", "synthetic", "--synthetic-size",
           "64x80x40", "--repeat", "2", "--warmup", "1", "--backend",
           "gpu,gpu-unfused,cpu-mt", "--output-dir", bench}),
      0);
  EXPECT_EQ(CountLines(ReadFile(bench + "/benchmark.csv")), 1u + 6u);
}

// Writes a 64 x 80 reference raster with losses in 2021 and 2022 (and some
// excluded pixels) and returns its path.
std::string WriteReference(const std::string& dir) {
  const std::string path = dir + "/reference.u8";
  std::string bytes(64 * 80, '\0');
  for (size_t i = 0; i < bytes.size(); ++i) {
    const size_t draw = (i * 7 + i / 80) % 10;
    bytes[i] = draw == 0 ? static_cast<char>(255)
                         : (draw < 4 ? '\0' : static_cast<char>(21 + draw % 2));
  }
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return path;
}

// The text of the "validation:" line of a log.
std::string ValidationLine(const std::string& log) {
  const size_t begin = log.find("\nvalidation: ");
  if (begin == std::string::npos) {
    return "";
  }
  return log.substr(begin + 1, log.find('\n', begin + 1) - begin - 1);
}

// The GPU counts the validation inside the tile pipeline (no cleanup) or
// after it (cleanup); either way the outputs and the log lines exist, and
// the in-pipeline counts agree with the CPU backend on this small cube.
TEST(App, GpuValidationInPipelineAgreesWithCpu) {
  if (!GpuAvailable()) {
    return;
  }
  TempDir temp;
  ASSERT_FALSE(temp.path().empty());
  const std::string reference = WriteReference(temp.path());
  const std::string gpu_out = temp.path() + "/gpu";
  const std::string cpu_out = temp.path() + "/cpu";
  const std::vector<std::string> common = {
      "--input",     "synthetic", "--synthetic-size", "64x80x40",
      "--reference", reference,   "--png-max-dim",    "16"};
  std::vector<std::string> gpu_args = common;
  gpu_args.insert(gpu_args.end(), {"--backend", "gpu-unfused", "--tile-rows",
                                   "16", "--output-dir", gpu_out, "--log-file",
                                   gpu_out + "/log.txt"});
  ASSERT_EQ(Run(gpu_args), 0);
  std::vector<std::string> cpu_args = common;
  cpu_args.insert(cpu_args.end(),
                  {"--backend", "cpu-fp32", "--output-dir", cpu_out,
                   "--log-file", cpu_out + "/log.txt"});
  ASSERT_EQ(Run(cpu_args), 0);
  const std::string gpu_log = ReadFile(gpu_out + "/log.txt");
  const std::string cpu_log = ReadFile(cpu_out + "/log.txt");
  ASSERT_FALSE(ValidationLine(gpu_log).empty());
  EXPECT_EQ(ValidationLine(gpu_log), ValidationLine(cpu_log));
  EXPECT_EQ(ReadFile(gpu_out + "/validation.csv"),
            ReadFile(cpu_out + "/validation.csv"));
  // The memory line of the GPU backend is printed in the execution section.
  EXPECT_LT(gpu_log.find("--- execution ---"), gpu_log.find("\ngpu_memory: "));
  EXPECT_LT(gpu_log.find("\ngpu_memory: "), gpu_log.find("--- timings"));
  // The PNG (3 x 3 blocks of 5 x 5 pixels at most) is written for both.
  EXPECT_GT(FileSize(gpu_out + "/change_map.png"), 0);
  EXPECT_EQ(ReadFile(gpu_out + "/change_map.png").substr(0, 8),
            ReadFile(cpu_out + "/change_map.png").substr(0, 8));
  // With --cleanup the validation is counted after the pipeline.
  const std::string cleaned = temp.path() + "/cleaned";
  gpu_args = common;
  gpu_args.insert(gpu_args.end(),
                  {"--backend", "gpu", "--cleanup", "--output-dir", cleaned,
                   "--log-file", cleaned + "/log.txt"});
  ASSERT_EQ(Run(gpu_args), 0);
  EXPECT_FALSE(ValidationLine(ReadFile(cleaned + "/log.txt")).empty());
  EXPECT_GT(FileSize(cleaned + "/validation.csv"), 0);
}

}  // namespace
}  // namespace cd
