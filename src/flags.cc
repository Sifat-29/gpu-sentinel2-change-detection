// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "flags.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "system_info.h"

namespace cd {
namespace {

constexpr int64_t kIntMax = std::numeric_limits<int>::max();

// One command-line flag: its name (without "--"), whether it is a boolean
// switch without a value, and the function that stores a parsed value.
struct FlagSpec {
  std::string name;
  bool is_switch;
  std::function<Status(Flags*, const std::string&)> set;
};

std::string FormatNumber(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%g", value);
  return buffer;
}

// Describes an allowed numeric range for error messages.
std::string RangeText(double lo, bool lo_inclusive, double hi) {
  if (std::isinf(hi)) {
    return std::string(lo_inclusive ? ">= " : "> ") + FormatNumber(lo);
  }
  return std::string("in ") + (lo_inclusive ? "[" : "(") + FormatNumber(lo) +
         ", " + FormatNumber(hi) + "]";
}

std::vector<std::string> Split(const std::string& text, char separator) {
  std::vector<std::string> parts;
  size_t begin = 0;
  while (true) {
    const size_t end = text.find(separator, begin);
    if (end == std::string::npos) {
      parts.push_back(text.substr(begin));
      return parts;
    }
    parts.push_back(text.substr(begin, end - begin));
    begin = end + 1;
  }
}

bool IsBlank(const std::string& text) {
  return text.empty() || text[0] == ' ' || text[0] == '\t';
}

// Parses a whole-string integer in [lo, hi].
Status ParseInteger(const std::string& name, const std::string& text,
                    int64_t lo, int64_t hi, int64_t* out) {
  const std::string expected =
      hi >= kIntMax
          ? RangeText(static_cast<double>(lo), true,
                      std::numeric_limits<double>::infinity())
          : RangeText(static_cast<double>(lo), true, static_cast<double>(hi));
  const Status bad = Status::Error("--" + name + " must be an integer " +
                                   expected + ", got '" + text + "'");
  if (IsBlank(text)) {
    return bad;
  }
  errno = 0;
  char* end = nullptr;
  const int64_t value = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || value < lo ||
      value > hi) {
    return bad;
  }
  *out = value;
  return Status::Ok();
}

// Parses a whole-string finite number with the given lower bound.
Status ParseNumber(const std::string& name, const std::string& text, double lo,
                   bool lo_inclusive, double hi, float* out) {
  const Status bad =
      Status::Error("--" + name + " must be a number " +
                    RangeText(lo, lo_inclusive, hi) + ", got '" + text + "'");
  if (IsBlank(text)) {
    return bad;
  }
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      !std::isfinite(value)) {
    return bad;
  }
  if (value < lo || (!lo_inclusive && value == lo) || value > hi) {
    return bad;
  }
  *out = static_cast<float>(value);
  return Status::Ok();
}

FlagSpec IntFlag(const std::string& name, int Flags::*member, int64_t lo,
                 int64_t hi = kIntMax) {
  return {name, false, [=](Flags* flags, const std::string& text) {
            int64_t value = 0;
            const Status status = ParseInteger(name, text, lo, hi, &value);
            if (status.ok()) {
              flags->*member = static_cast<int>(value);
            }
            return status;
          }};
}

FlagSpec FloatFlag(const std::string& name, float Flags::*member, double lo,
                   bool lo_inclusive,
                   double hi = std::numeric_limits<double>::infinity()) {
  return {name, false, [=](Flags* flags, const std::string& text) {
            float value = 0;
            const Status status =
                ParseNumber(name, text, lo, lo_inclusive, hi, &value);
            if (status.ok()) {
              flags->*member = value;
            }
            return status;
          }};
}

FlagSpec StringFlag(const std::string& name, std::string Flags::*member) {
  return {name, false, [=](Flags* flags, const std::string& text) {
            if (text.empty()) {
              return Status::Error("--" + name + " must not be empty");
            }
            flags->*member = text;
            return Status::Ok();
          }};
}

FlagSpec SwitchFlag(const std::string& name, bool Flags::*member) {
  return {name, true, [=](Flags* flags, const std::string& text) {
            if (text == "true" || text == "1" || text.empty()) {
              flags->*member = true;
            } else if (text == "false" || text == "0") {
              flags->*member = false;
            } else {
              return Status::Error("--" + name +
                                   " is a switch; use it without a value, got "
                                   "'" +
                                   text + "'");
            }
            return Status::Ok();
          }};
}

// Parses "HxWxT" (also accepts 'X') into three positive integers.
Status ParseSyntheticSize(const std::string& text, Flags* flags) {
  const Status bad = Status::Error(
      "--synthetic-size must look like HxWxT with positive integers, got '" +
      text + "'");
  std::string normalized = text;
  for (char& c : normalized) {
    if (c == 'X') {
      c = 'x';
    }
  }
  const std::vector<std::string> parts = Split(normalized, 'x');
  if (parts.size() != 3) {
    return bad;
  }
  int64_t values[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    if (!ParseInteger("synthetic-size", parts[i], 1, 1000000, &values[i])
             .ok()) {
      return bad;
    }
  }
  flags->synthetic_height = static_cast<int>(values[0]);
  flags->synthetic_width = static_cast<int>(values[1]);
  flags->synthetic_dates = static_cast<int>(values[2]);
  return Status::Ok();
}

Status ParseMode(const std::string& text, Flags* flags) {
  static const char* const kModes[] = {"detect", "quick-diff", "benchmark",
                                       "generate", "coreg"};
  for (const char* mode : kModes) {
    if (text == mode) {
      flags->mode = text;
      return Status::Ok();
    }
  }
  return Status::Error("unknown mode '" + text +
                       "' (expected detect, quick-diff, benchmark, generate "
                       "or coreg)");
}

// Parses a comma list of backend names; "all" expands to the five backends.
Status ParseBackends(const std::string& text, Flags* flags) {
  static const char* const kAll[] = {"gpu", "gpu-unfused", "cpu", "cpu-fp32",
                                     "cpu-mt"};
  std::vector<std::string> names;
  for (const std::string& token : Split(text, ',')) {
    std::vector<std::string> expanded;
    if (token == "all") {
      expanded.assign(std::begin(kAll), std::end(kAll));
    } else {
      Backend backend;
      const Status status = ParseBackend(token, &backend);
      if (!status.ok()) {
        return Status::Error("--backend: " + status.message());
      }
      expanded.push_back(token);
    }
    for (const std::string& name : expanded) {
      bool seen = false;
      for (const std::string& existing : names) {
        seen = seen || existing == name;
      }
      if (!seen) {
        names.push_back(name);
      }
    }
  }
  flags->backends = names;
  return Status::Ok();
}

// Parses a comma list of Sentinel-2 SCL codes 0..15.
Status ParseValidClasses(const std::string& text, Flags* flags) {
  std::vector<int> classes;
  for (const std::string& token : Split(text, ',')) {
    int64_t value = 0;
    if (!ParseInteger("valid-classes", token, 0, 15, &value).ok()) {
      return Status::Error(
          "--valid-classes must be a comma list of SCL codes 0..15, got '" +
          text + "'");
    }
    classes.push_back(static_cast<int>(value));
  }
  flags->valid_classes = classes;
  return Status::Ok();
}

// Parses "N" (auto-pick N pixels) or "r,c;r,c;..." (explicit pixels).
Status ParseSamplePixels(const std::string& text, Flags* flags) {
  const Status bad = Status::Error(
      "--sample-pixels must be a count N or a list r,c;r,c;..., got '" + text +
      "'");
  if (text.find(',') == std::string::npos) {
    int64_t count = 0;
    if (!ParseInteger("sample-pixels", text, 0, 1000000, &count).ok()) {
      return bad;
    }
    flags->sample_count = static_cast<int>(count);
    flags->sample_pixel_list.clear();
    return Status::Ok();
  }
  std::vector<std::pair<int, int>> pixels;
  for (const std::string& pair : Split(text, ';')) {
    const std::vector<std::string> rc = Split(pair, ',');
    int64_t row = 0;
    int64_t col = 0;
    if (rc.size() != 2 ||
        !ParseInteger("sample-pixels", rc[0], 0, kIntMax, &row).ok() ||
        !ParseInteger("sample-pixels", rc[1], 0, kIntMax, &col).ok()) {
      return bad;
    }
    pixels.emplace_back(static_cast<int>(row), static_cast<int>(col));
  }
  flags->sample_pixel_list = pixels;
  flags->sample_count = static_cast<int>(pixels.size());
  return Status::Ok();
}

// Parses --coreg-window: a power of two in 64..4096.
Status ParseCoregWindow(const std::string& text, Flags* flags) {
  int64_t value = 0;
  const Status bad = Status::Error(
      "--coreg-window must be a power of two in 64..4096, got '" + text + "'");
  if (!ParseInteger("coreg-window", text, 64, 4096, &value).ok() ||
      (value & (value - 1)) != 0) {
    return bad;
  }
  flags->coreg_window = static_cast<int>(value);
  return Status::Ok();
}

// Parses --coreg-origin "r,c" (top-left pixel of the window).
Status ParseCoregOrigin(const std::string& text, Flags* flags) {
  const std::vector<std::string> rc = Split(text, ',');
  int64_t row = 0;
  int64_t col = 0;
  if (rc.size() != 2 ||
      !ParseInteger("coreg-origin", rc[0], 0, kIntMax, &row).ok() ||
      !ParseInteger("coreg-origin", rc[1], 0, kIntMax, &col).ok()) {
    return Status::Error(
        "--coreg-origin must look like r,c with non-negative integers, got '" +
        text + "'");
  }
  flags->coreg_origin_row = static_cast<int>(row);
  flags->coreg_origin_col = static_cast<int>(col);
  return Status::Ok();
}

// Builds the table of every flag with its parser and valid range.
std::vector<FlagSpec> BuildFlagTable() {
  std::vector<FlagSpec> table;
  table.push_back({"mode", false, [](Flags* f, const std::string& v) {
                     return ParseMode(v, f);
                   }});
  table.push_back({"backend", false, [](Flags* f, const std::string& v) {
                     return ParseBackends(v, f);
                   }});
  table.push_back(StringFlag("input", &Flags::input));
  table.push_back(StringFlag("output-dir", &Flags::output_dir));
  table.push_back({"synthetic-size", false, [](Flags* f, const std::string& v) {
                     return ParseSyntheticSize(v, f);
                   }});
  table.push_back(
      FloatFlag("synthetic-years", &Flags::synthetic_years, 0, false));
  table.push_back({"seed", false, [](Flags* f, const std::string& v) {
                     int64_t value = 0;
                     const Status status = ParseInteger(
                         "seed", v, 0, std::numeric_limits<uint32_t>::max(),
                         &value);
                     if (status.ok()) {
                       f->seed = static_cast<uint32_t>(value);
                     }
                     return status;
                   }});
  table.push_back(FloatFlag("noise", &Flags::noise, 0, true));
  table.push_back(
      FloatFlag("cloud-fraction", &Flags::cloud_fraction, 0, true, 1));
  table.push_back(
      FloatFlag("break-fraction", &Flags::break_fraction, 0, true, 1));
  table.push_back(IntFlag("tile-rows", &Flags::tile_rows, 0));
  table.push_back(
      {"streams", false, [](Flags* f, const std::string& v) {
         int64_t value = 0;
         if (!ParseInteger("streams", v, 1, 3, &value).ok() || value == 2) {
           return Status::Error("--streams must be 1 or 3, got '" + v + "'");
         }
         f->streams = static_cast<int>(value);
         return Status::Ok();
       }});
  table.push_back(
      {"block-size", false, [](Flags* f, const std::string& v) {
         int64_t value = 0;
         if (!ParseInteger("block-size", v, 32, 1024, &value).ok() ||
             value % 32 != 0) {
           return Status::Error(
               "--block-size must be a multiple of 32 in "
               "32..1024, got '" +
               v + "'");
         }
         f->block_size = static_cast<int>(value);
         return Status::Ok();
       }});
  table.push_back(IntFlag("threads", &Flags::threads, 0));
  table.push_back(IntFlag("io-threads", &Flags::io_threads, 1, 64));
  table.push_back(IntFlag("device", &Flags::device, 0));
  table.push_back(IntFlag("harmonics", &Flags::harmonics, 1, kMaxHarmonics));
  table.push_back(FloatFlag("train-years", &Flags::train_years, 0, false));
  table.push_back({"trend", false, [](Flags* f, const std::string& v) {
                     if (v != "hold" && v != "extrapolate") {
                       return Status::Error(
                           "--trend must be hold or extrapolate, got '" + v +
                           "'");
                     }
                     f->hold_trend = (v == "hold");
                     return Status::Ok();
                   }});
  table.push_back(IntFlag("min-train-obs", &Flags::min_train_obs, 0));
  table.push_back(FloatFlag("threshold", &Flags::threshold, 0, false));
  table.push_back(IntFlag("consecutive", &Flags::consecutive, 1));
  table.push_back(FloatFlag("rmse-floor", &Flags::rmse_floor, 0, false));
  table.push_back(FloatFlag("ridge", &Flags::ridge, 0, true));
  table.push_back(SwitchFlag("losses-only", &Flags::losses_only));
  table.push_back({"valid-classes", false, [](Flags* f, const std::string& v) {
                     return ParseValidClasses(v, f);
                   }});
  table.push_back(SwitchFlag("cleanup", &Flags::cleanup));
  table.push_back(StringFlag("reference", &Flags::reference));
  table.push_back({"sample-pixels", false, [](Flags* f, const std::string& v) {
                     return ParseSamplePixels(v, f);
                   }});
  table.push_back(SwitchFlag("save-coefficients", &Flags::save_coefficients));
  table.push_back(IntFlag("png-max-dim", &Flags::png_max_dim, 1));
  table.push_back(IntFlag("date-a", &Flags::date_a, 0));
  table.push_back(IntFlag("date-b", &Flags::date_b, -1));
  table.push_back(
      FloatFlag("diff-threshold", &Flags::diff_threshold, 0, false));
  table.push_back(IntFlag("repeat", &Flags::repeat, 1));
  table.push_back(IntFlag("warmup", &Flags::warmup, 0));
  table.push_back(StringFlag("benchmark-csv", &Flags::benchmark_csv));
  table.push_back(StringFlag("run-id", &Flags::run_id));
  table.push_back(StringFlag("compare-to", &Flags::compare_to));
  table.push_back({"coreg-window", false, [](Flags* f, const std::string& v) {
                     return ParseCoregWindow(v, f);
                   }});
  table.push_back({"coreg-origin", false, [](Flags* f, const std::string& v) {
                     return ParseCoregOrigin(v, f);
                   }});
  table.push_back({"coreg-band", false, [](Flags* f, const std::string& v) {
                     if (v != "nir" && v != "red") {
                       return Status::Error(
                           "--coreg-band must be nir or red, got '" + v + "'");
                     }
                     f->coreg_band = v;
                     return Status::Ok();
                   }});
  table.push_back(IntFlag("reference-date", &Flags::reference_date, 0));
  table.push_back(StringFlag("trace-csv", &Flags::trace_csv));
  table.push_back(StringFlag("log-file", &Flags::log_file));
  table.push_back(SwitchFlag("quiet", &Flags::quiet));
  return table;
}

// Checks cross-flag constraints and fills derived defaults.
Status FinalizeFlags(Flags* flags) {
  if (flags->input.empty() && flags->mode != "generate") {
    return Status::Error("--input is required");
  }
  if (flags->backends.size() > 1 && flags->mode != "benchmark") {
    return Status::Error(
        "several backends are only allowed with --mode benchmark");
  }
  if (flags->compare_to != "none") {
    Backend backend;
    const Status status = ParseBackend(flags->compare_to, &backend);
    if (!status.ok()) {
      return Status::Error("--compare-to: " + status.message() + " or none");
    }
  }
  if (!flags->trace_csv.empty()) {
    if (flags->mode != "detect") {
      return Status::Error("--trace-csv is only supported with --mode detect");
    }
    Backend backend;
    if (!ParseBackend(flags->backends.front(), &backend).ok() ||
        !IsGpuBackend(backend)) {
      return Status::Error(
          "--trace-csv needs a GPU backend (gpu or gpu-unfused), got '" +
          flags->backends.front() + "'");
    }
  }
  if (flags->benchmark_csv.empty()) {
    flags->benchmark_csv = flags->output_dir + "/benchmark.csv";
  }
  if (flags->run_id.empty()) {
    flags->run_id = IsoTimestamp();
  }
  ModelParams params;
  return FlagsToModelParams(*flags, &params);
}

}  // namespace

Status ParseFlags(int argc, char** argv, Flags* flags) {
  // --help wins over everything else, including invalid values.
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--help") {
      flags->help = true;
      return Status::Ok();
    }
  }
  const std::vector<FlagSpec> table = BuildFlagTable();
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.size() < 3 || arg.compare(0, 2, "--") != 0) {
      return Status::Error("unexpected argument '" + arg + "'");
    }
    std::string name = arg.substr(2);
    std::string value;
    bool has_value = false;
    const size_t equals = name.find('=');
    if (equals != std::string::npos) {
      value = name.substr(equals + 1);
      name = name.substr(0, equals);
      has_value = true;
    }
    const FlagSpec* spec = nullptr;
    for (const FlagSpec& candidate : table) {
      if (candidate.name == name) {
        spec = &candidate;
      }
    }
    if (spec == nullptr) {
      return Status::Error("unknown flag '--" + name + "'");
    }
    if (!has_value && !spec->is_switch) {
      // Values may start with '-' (for example --tile-rows -3), so the next
      // argument is always consumed as the value.
      if (i + 1 >= argc) {
        return Status::Error("flag '--" + name + "' requires a value");
      }
      value = argv[++i];
    }
    const Status status = spec->set(flags, value);
    if (!status.ok()) {
      return status;
    }
  }
  return FinalizeFlags(flags);
}

Status FlagsToModelParams(const Flags& flags, ModelParams* params) {
  ModelParams p;
  p.harmonics = flags.harmonics;
  p.train_years = flags.train_years;
  p.hold_trend = flags.hold_trend;
  p.min_train_obs = flags.min_train_obs;
  p.threshold = flags.threshold;
  p.consecutive = flags.consecutive;
  p.rmse_floor = flags.rmse_floor;
  p.ridge = flags.ridge;
  p.losses_only = flags.losses_only;
  p.valid_class_mask = 0;
  for (const int code : flags.valid_classes) {
    p.valid_class_mask |= 1u << code;
  }
  const Status status = ResolveModelParams(&p);
  if (!status.ok()) {
    return status;
  }
  *params = p;
  return Status::Ok();
}

Status FlagsToRunConfig(const Flags& flags, Backend backend,
                        RunConfig* config) {
  RunConfig c;
  c.backend = backend;
  c.tile_rows = flags.tile_rows;
  c.streams = flags.streams;
  c.block_size = flags.block_size;
  c.threads = flags.threads;
  c.io_threads = flags.io_threads;
  c.device = flags.device;
  c.save_coefficients = flags.save_coefficients;
  c.quiet = flags.quiet;
  c.trace_csv = flags.trace_csv;
  *config = c;
  return Status::Ok();
}

std::string UsageText() {
  return R"(Usage: change_detect --input <manifest.txt|synthetic> [flags]

GPU time-series change detection for satellite imagery. Flags take
--name value or --name=value; switches take no value.

Modes:
  --help                  print this text and exit
  --mode M                detect | quick-diff | benchmark | generate | coreg
                          (default detect)
  --backend B[,B...]      gpu | gpu-unfused | cpu | cpu-fp32 | cpu-mt
                          (default gpu); benchmark mode accepts a comma
                          list or 'all'. cpu = 1 thread FP64 reference,
                          cpu-fp32 = 1 thread FP32, cpu-mt = threaded FP32

Input:
  --input PATH            manifest.txt of a cube, or 'synthetic' (required except --mode generate)
  --synthetic-size HxWxT  synthetic cube size (default 1024x1024x48)
  --synthetic-years Y     time span of synthetic dates (default 4.0)
  --seed N                generator seed (default 42)
  --noise S               synthetic NDVI noise sd (default 0.03)
  --cloud-fraction F      synthetic masked fraction 0..1 (default 0.2)
  --break-fraction F      synthetic pixels with a break 0..1 (default 0.15)
  --reference PATH|auto   validation raster, auto = manifest reference_file
                          (default auto)

Model:
  --harmonics K           seasonal harmonics 1..3, p = 2 + 2K (default 2)
  --train-years Y         history window, dates with tau < Y (default 2.0)
  --trend hold|extrapolate
                          hold = trend frozen at its end-of-history value
                          for monitoring dates (default); extrapolate =
                          linear trend continued
  --min-train-obs N       minimum valid training observations, 0 = 3p
                          (default 0); must be >= p + 1
  --ridge R               ridge added to non-intercept terms (default 0.0)
  --valid-classes LIST    SCL codes kept, comma list 0..15 (default 4,5,6)

Detection:
  --threshold Z           abs(z) threshold (default 3.5)
  --consecutive M         consecutive same-sign exceedances (default 2)
  --rmse-floor X          lower bound of the fit RMSE (default 0.02)
  --losses-only           only greenness-loss breaks count
  --cleanup               remove isolated change pixels (3x3 rule)
  --date-a I, --date-b J  quick-diff date indices (default 0 and T-1;
                          -1 means the last date)
  --diff-threshold X      quick-diff NDVI drop threshold (default 0.2)

Streaming and execution:
  --tile-rows N           rows per tile, 0 = auto (default 0)
  --streams S             1 or 3 CUDA streams (default 3)
  --block-size N          threads per block, 32..1024, multiple of 32
                          (default 256)
  --threads N             cpu-mt worker threads, 0 = all (default 0)
  --io-threads N          parallel disk reads per tile; 1 = sequential
                          (1..64, default 8)
  --device N              GPU index (default 0)

Outputs:
  --output-dir DIR        output directory, created if missing (default out)
  --sample-pixels N|LIST  N auto-picked pixels or 'r,c;r,c;...' whose
                          trajectories are exported (default 8)
  --save-coefficients     also write coefficients.f32
  --png-max-dim N         PNG long side limit (default 2048)
  --trace-csv PATH        detect mode, GPU backends: write the per-tile
                          stage timeline (read, h2d, kernels, d2h) as CSV
  --log-file PATH         also write the log to this file
  --quiet                 suppress per-tile lines

Benchmark:
  --repeat N              timed repetitions (default 5)
  --warmup N              warm-up runs (default 1)
  --benchmark-csv PATH    appended CSV (default <output-dir>/benchmark.csv)
  --run-id ID             tag in the CSV and log (default timestamp)
  --compare-to B|none     also run backend B once and report agreement
                          (default none)

Co-registration check (--mode coreg, GPU only, cuFFT phase correlation):
  --coreg-window N        square window side, power of two 64..4096
                          (default 1024)
  --coreg-origin R,C      top-left pixel of the window (default: centred)
  --coreg-band nir|red    band correlated (default nir)
  --reference-date I      reference date index (default: the date with the
                          highest valid fraction in the window)

Exit codes: 0 success, 1 run-time failure, 2 invalid command line.
)";
}

}  // namespace cd
