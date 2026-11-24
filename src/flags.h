// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
// Command-line interface: the Flags struct, parsing, validation and usage text
// (flag table: README.md).
#ifndef SRC_FLAGS_H_
#define SRC_FLAGS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "status.h"
#include "types.h"

namespace cd {

// One member per command-line flag, initialised to the documented default.
struct Flags {
  bool help = false;

  // Modes and input.
  std::string mode =
      "detect";  // detect, quick-diff, benchmark, generate, coreg
  // Backend names after expansion of "all"; one entry unless benchmarking.
  std::vector<std::string> backends = {"gpu"};
  std::string input;  // manifest path or "synthetic"; required
  std::string output_dir = "out";

  // Synthetic cube.
  int synthetic_height = 1024;
  int synthetic_width = 1024;
  int synthetic_dates = 48;
  float synthetic_years = 4.0f;
  uint32_t seed = 42;
  float noise = 0.03f;
  float cloud_fraction = 0.2f;
  float break_fraction = 0.15f;

  // Streaming and launch tuning.
  int tile_rows = 0;  // 0 = auto
  int streams = 3;    // 1 or 3
  int block_size = 256;
  int threads = 0;     // 0 = all hardware threads
  int io_threads = 8;  // parallel disk reads per tile; 1 = sequential
  int device = 0;

  // Model.
  int harmonics = 2;
  float train_years = 2.0f;
  bool hold_trend = true;  // --trend hold (true) | extrapolate (false)
  int min_train_obs = 0;   // 0 = auto (3p)

  // Detection.
  float threshold = 3.5f;
  int consecutive = 2;
  float rmse_floor = 0.02f;
  float ridge = 0.0f;
  bool losses_only = false;
  bool cleanup = false;                        // 3x3 isolated-change removal
  std::vector<int> valid_classes = {4, 5, 6};  // SCL codes 0..15

  // Validation and exported pixels.
  std::string reference = "auto";
  // --sample-pixels N sets sample_count (auto-pick); an explicit list
  // "r,c;r,c" fills sample_pixel_list and sets sample_count to its size.
  int sample_count = 8;
  std::vector<std::pair<int, int>> sample_pixel_list;
  bool save_coefficients = false;
  int png_max_dim = 2048;
  std::string trace_csv;  // GPU pipeline timeline CSV; empty = off

  // Quick-diff.
  int date_a = 0;
  int date_b = -1;  // -1 = last date (T - 1)
  float diff_threshold = 0.2f;

  // Benchmark.
  int repeat = 5;
  int warmup = 1;
  std::string benchmark_csv;  // resolved to <output-dir>/benchmark.csv
  std::string run_id;         // resolved to a timestamp
  std::string compare_to = "none";

  // Co-registration check (--mode coreg).
  int coreg_window = 1024;    // power of two, 64..4096
  int coreg_origin_row = -1;  // -1 = centred window
  int coreg_origin_col = -1;
  std::string coreg_band = "nir";  // nir or red
  int reference_date = -1;         // -1 = best valid fraction in the window

  // Logging.
  std::string log_file;
  bool quiet = false;
};

// Parses argv into *flags and validates every value. When --help appears
// anywhere, sets flags->help and returns Ok without further validation.
// On failure returns an error whose message is the "<what>" part of
// "error: <what>; see --help" (the caller adds the prefix and suffix).
Status ParseFlags(int argc, char** argv, Flags* flags);

// Returns the grouped usage text printed by --help, with defaults.
std::string UsageText();

// Builds the resolved model parameters (calls ResolveModelParams).
Status FlagsToModelParams(const Flags& flags, ModelParams* params);

// Builds the run configuration for one backend of flags.backends.
Status FlagsToRunConfig(const Flags& flags, Backend backend, RunConfig* config);

}  // namespace cd

#endif  // SRC_FLAGS_H_
