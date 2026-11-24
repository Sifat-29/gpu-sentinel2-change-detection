// Copyright 2026 The change_detect authors. MIT licence, see LICENSE.
#include "benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17): portable paths and mkdir -p
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "cuda/gpu_backend.h"
#include "system_info.h"

namespace cd {
namespace {

namespace fs = std::filesystem;

std::string Printf(const char* format, double value) {
  char text[64];
  std::snprintf(text, sizeof(text), format, value);
  return text;
}

// Milliseconds (or any "-1 = n/a" quantity) with 3 decimals.
std::string Ms(double value) {
  return value < 0 ? "-1" : Printf("%.3f", value);
}

std::string Fraction(double value) {
  return value < 0 ? "-1" : Printf("%.6f", value);
}

// Quotes a field that contains a comma, a quote or a line break.
std::string CsvField(const std::string& text) {
  if (text.find_first_of(",\"\n\r") == std::string::npos) {
    return text;
  }
  std::string quoted = "\"";
  for (const char c : text) {
    if (c == '"') {
      quoted += '"';
    }
    quoted += c;
  }
  quoted += '"';
  return quoted;
}

// "12.0" from a CUDA version integer such as 12000.
std::string VersionText(int version) {
  return std::to_string(version / 1000) + "." +
         std::to_string((version % 1000) / 10);
}

}  // namespace

Status ComputeAgreement(const DetectionResult& result,
                        const DetectionResult& reference, Agreement* out) {
  if (out == nullptr) {
    return Status::Error("ComputeAgreement: null output");
  }
  const size_t pixels =
      static_cast<size_t>(result.height) * static_cast<size_t>(result.width);
  if (result.height != reference.height || result.width != reference.width ||
      result.break_index.size() != pixels ||
      reference.break_index.size() != pixels ||
      result.status.size() != pixels || reference.status.size() != pixels) {
    return Status::Error("agreement: the two results differ in size");
  }
  Agreement agreement;
  if (pixels == 0) {
    *out = agreement;
    return Status::Ok();
  }
  uint64_t same_break = 0;
  uint64_t same_status = 0;
  for (size_t i = 0; i < pixels; ++i) {
    same_break += result.break_index[i] == reference.break_index[i] ? 1 : 0;
    same_status += result.status[i] == reference.status[i] ? 1 : 0;
  }
  agreement.break_frac =
      static_cast<double>(same_break) / static_cast<double>(pixels);
  agreement.status_frac =
      static_cast<double>(same_status) / static_cast<double>(pixels);
  if (!result.coefficients.empty() &&
      result.coefficients.size() == reference.coefficients.size()) {
    double largest = 0.0;
    for (size_t i = 0; i < result.coefficients.size(); ++i) {
      largest = std::max(
          largest, std::fabs(static_cast<double>(result.coefficients[i]) -
                             static_cast<double>(reference.coefficients[i])));
    }
    agreement.max_coef_diff = largest;
  }
  *out = agreement;
  return Status::Ok();
}

SystemColumns CollectSystemColumns(int device) {
  SystemColumns columns;
  columns.cpu_model = CpuModel();
  columns.cpu_threads = HardwareThreads();
  columns.git_commit = GitShortCommit();
  GpuInfo info;
  if (GpuAvailable() && GpuDeviceInfo(device, &info).ok()) {
    columns.gpu_name = info.name;
    columns.gpu_mem_mb =
        static_cast<double>(info.total_mem) / static_cast<double>(1 << 20);
    columns.cuda_runtime = VersionText(info.runtime_version);
    columns.driver_version = VersionText(info.driver_version);
  }
  return columns;
}

const std::string& BenchmarkCsvHeader() {
  static const std::string header =
      "run_id,timestamp,mode,backend,precision,kernels,input_name,height,"
      "width,dates,pixels,samples,tile_rows,num_tiles,streams,block_size,"
      "threads,io_threads,repeat_index,input_bytes,read_ms,read_wait_ms,"
      "setup_ms,h2d_ms,ndvi_ms,fit_ms,detect_ms,fused_ms,d2h_ms,kernel_ms,"
      "total_ms,throughput_msamples_s,kernel_throughput_msamples_s,device_mem_"
      "peak_mb,"
      "changed_pixels,compare_to,agree_break_frac,agree_status_frac,"
      "max_coef_diff,gpu_name,gpu_mem_mb,cpu_model,cpu_threads,cuda_runtime,"
      "driver_version,git_commit";
  return header;
}

BenchmarkRow MakeBenchmarkRow(const std::string& run_id, const CubeInfo& info,
                              const RunConfig& config,
                              const StageTimings& timings,
                              const RunStats& stats, int repeat_index,
                              int64_t changed_pixels,
                              const SystemColumns& system) {
  BenchmarkRow row;
  row.run_id = run_id;
  row.timestamp = IsoTimestamp();
  row.backend = BackendName(config.backend);
  const bool gpu = IsGpuBackend(config.backend);
  row.precision = config.backend == Backend::kCpuFp64 ? "fp64" : "fp32";
  row.kernels =
      !gpu ? "n/a"
           : (config.backend == Backend::kGpuFused ? "fused" : "unfused");
  row.input_name = info.name;
  row.height = info.height;
  row.width = info.width;
  row.dates = info.num_dates;
  row.pixels = static_cast<int64_t>(info.height) * info.width;
  row.samples = row.pixels * info.num_dates;
  row.tile_rows = stats.tile_rows_used;
  row.num_tiles = stats.num_tiles;
  row.streams = gpu ? stats.streams_used : -1;
  row.block_size = gpu ? config.block_size : -1;
  row.threads = gpu ? -1 : stats.threads_used;
  row.io_threads = config.io_threads;
  row.repeat_index = repeat_index;
  row.input_bytes = row.samples * 5;  // 2 + 2 + 1 bytes per sample
  row.timings = timings;
  row.device_mem_peak_mb = stats.device_mem_peak_mb;
  row.changed_pixels = changed_pixels;
  row.system = system;
  return row;
}

std::string FormatBenchmarkRow(const BenchmarkRow& row) {
  const StageTimings& t = row.timings;
  // Msamples/s = samples / (ms * 1e3); n/a when the time is unknown or zero.
  const auto throughput = [&row](double ms) {
    return ms > 0 ? static_cast<double>(row.samples) / (ms * 1e3) : -1.0;
  };
  const std::vector<std::string> fields = {
      CsvField(row.run_id),
      row.timestamp,
      row.mode,
      row.backend,
      row.precision,
      row.kernels,
      CsvField(row.input_name),
      std::to_string(row.height),
      std::to_string(row.width),
      std::to_string(row.dates),
      std::to_string(row.pixels),
      std::to_string(row.samples),
      std::to_string(row.tile_rows),
      std::to_string(row.num_tiles),
      std::to_string(row.streams),
      std::to_string(row.block_size),
      std::to_string(row.threads),
      std::to_string(row.io_threads),
      std::to_string(row.repeat_index),
      std::to_string(row.input_bytes),
      Ms(t.read_ms),
      Ms(t.read_wait_ms),
      Ms(t.setup_ms),
      Ms(t.h2d_ms),
      Ms(t.ndvi_ms),
      Ms(t.fit_ms),
      Ms(t.detect_ms),
      Ms(t.fused_ms),
      Ms(t.d2h_ms),
      Ms(t.kernel_ms),
      Ms(t.total_ms),
      Ms(throughput(t.total_ms)),
      Ms(throughput(t.kernel_ms)),
      Ms(row.device_mem_peak_mb),
      std::to_string(row.changed_pixels),
      CsvField(row.compare_to),
      Fraction(row.agreement.break_frac),
      Fraction(row.agreement.status_frac),
      row.agreement.max_coef_diff < 0
          ? "-1"
          : Printf("%.6e", row.agreement.max_coef_diff),
      CsvField(row.system.gpu_name),
      Ms(row.system.gpu_mem_mb),
      CsvField(row.system.cpu_model),
      std::to_string(row.system.cpu_threads),
      row.system.cuda_runtime,
      row.system.driver_version,
      CsvField(row.system.git_commit)};
  std::string line;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i > 0) {
      line += ",";
    }
    line += fields[i];
  }
  return line;
}

Status AppendBenchmarkRows(const std::string& path,
                           const std::vector<BenchmarkRow>& rows) {
  std::error_code error;
  const fs::path parent = fs::path(path).parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent, error);
    if (error) {
      return Status::Error("cannot create directory '" + parent.string() +
                           "': " + error.message());
    }
  }
  bool write_header = true;
  {
    std::ifstream existing(path);
    std::string first_line;
    if (existing.good() && std::getline(existing, first_line)) {
      if (!first_line.empty() && first_line.back() == '\r') {
        first_line.pop_back();
      }
      if (first_line != BenchmarkCsvHeader()) {
        return Status::Error("benchmark CSV '" + path +
                             "' exists with a different header; refusing to "
                             "append (use another --benchmark-csv)");
      }
      write_header = false;
    }
  }
  std::ofstream out(path, std::ios::app);
  if (!out.good()) {
    return Status::Error("cannot open '" + path + "' for writing");
  }
  if (write_header) {
    out << BenchmarkCsvHeader() << "\n";
  }
  for (const BenchmarkRow& row : rows) {
    out << FormatBenchmarkRow(row) << "\n";
  }
  out.flush();
  if (!out.good()) {
    return Status::Error("write to '" + path + "' failed");
  }
  return Status::Ok();
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

std::vector<std::string> FormatMedianTable(
    const std::vector<BackendSamples>& samples) {
  // Median total time of cpu-mt, the headline CPU comparison.
  double cpu_mt_median = -1;
  for (const BackendSamples& s : samples) {
    if (s.backend == "cpu-mt" && !s.total_ms.empty()) {
      cpu_mt_median = Median(s.total_ms);
    }
  }
  std::vector<std::string> lines;
  lines.push_back(
      "backend | median backend_total ms | min | max | median kernel ms | "
      "speedup vs cpu-mt");
  for (const BackendSamples& s : samples) {
    if (s.total_ms.empty()) {
      continue;
    }
    const double median = Median(s.total_ms);
    const double low = *std::min_element(s.total_ms.begin(), s.total_ms.end());
    const double high = *std::max_element(s.total_ms.begin(), s.total_ms.end());
    const double kernel = Median(s.kernel_ms);
    const bool has_kernel = !s.kernel_ms.empty() && s.kernel_ms.front() >= 0;
    const std::string speedup =
        cpu_mt_median > 0 && median > 0
            ? Printf("%.2f", cpu_mt_median / median) + "x"
            : "-";
    lines.push_back(s.backend + " | " + Printf("%.3f", median) + " | " +
                    Printf("%.3f", low) + " | " + Printf("%.3f", high) + " | " +
                    (has_kernel ? Printf("%.3f", kernel) : "-") + " | " +
                    speedup);
  }
  return lines;
}

}  // namespace cd
