#!/usr/bin/env bash
# One-command installation check (called by `make smoke`).
#
# Steps: build the program (GPU build when nvcc is available, else the CPU
# build), print the device lines once, run the unit tests, then run a small
# synthetic detect (512 x 512 x 48 time series, no downloads) and check its
# log and output files. One line per check: [PASS] or [FAIL] <reason>; the
# last line is "SMOKE: PASS" or "SMOKE: FAIL" and the exit code is 0 or 1.
#
# The complete log (including the compiler and test output) is written to
# out/smoke/smoke_log.txt; paste it into a bug report if a check fails.
#
# Environment (all optional): NVCC (compiler used to decide between the GPU
# and CPU build, default "nvcc"), BUILD_DIR and BIN_DIR (passed to make).
set -euo pipefail

# Work from the repository root, wherever the script is called from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

readonly nvcc_cmd="${NVCC:-nvcc}"
readonly bin_dir="${BIN_DIR:-bin}"
readonly out_dir="out/smoke"
readonly log_file="${out_dir}/smoke_log.txt"
readonly size="512x512x48"
readonly pixels=262144
readonly min_recovery=95      # percent of injected breaks recovered
readonly min_agreement=0.995  # identical break index, backend vs cpu

# BUILD_DIR and BIN_DIR reach make through the environment (the Makefile
# uses `?=`), so private build directories work for this script as well.
export BIN_DIR="${bin_dir}"
if [[ -n "${BUILD_DIR:-}" ]]; then
  export BUILD_DIR
fi

failures=0

# Prints one line to the terminal and to the log.
say() {
  printf '%s\n' "$*" | tee -a "${log_file}"
}

# check <description> <ok|fail> [reason]: prints a [PASS]/[FAIL] line.
check() {
  if [[ "$2" == "ok" ]]; then
    say "[PASS] $1"
  else
    say "[FAIL] $1${3:+: $3}"
    failures=$((failures + 1))
  fi
}

# Elapsed seconds since the timestamp passed in (seconds with decimals).
elapsed_since() {
  awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }'
}

# Runs a make target, appending its output to the log only. On failure the
# last lines are shown on the terminal as well. Returns make's exit status.
run_make() {
  local status=0
  {
    echo "+ make $*"
    make --no-print-directory "$@"
  } >>"${log_file}" 2>&1 || status=$?
  if [[ ${status} -ne 0 ]]; then
    tail -n 15 "${log_file}"
  fi
  return "${status}"
}

# Prints the value of "key: ..." from the detect log (first match).
log_line() {
  grep -m 1 -E "^$1:" "$2" || true
}

main() {
  local build_target test_target program backend started
  say "=== change_detect smoke test ==="

  # --- Build ------------------------------------------------------------
  if command -v "${nvcc_cmd}" >/dev/null 2>&1; then
    build_target="gpu"
    test_target="test"
    program="${bin_dir}/change_detect"
    say "build: GPU program (compiler ${nvcc_cmd})"
  else
    build_target="cpu"
    test_target="test-cpu"
    program="${bin_dir}/change_detect_cpu"
    say "build: CPU-only program (${nvcc_cmd} not found)"
  fi
  started="$(date +%s.%N)"
  if run_make -j "$(nproc)" "${build_target}"; then
    check "build make ${build_target} ($(elapsed_since "${started}") s)" ok
  else
    check "build make ${build_target}" fail "see ${log_file}"
    return
  fi

  # --- Unit tests (their first lines are the device section) --------------
  local test_out="${out_dir}/tests_output.txt"
  local test_status=0
  started="$(date +%s.%N)"
  make --no-print-directory -j "$(nproc)" "${test_target}" \
    >"${test_out}" 2>&1 || test_status=$?
  {
    echo "+ make ${test_target}"
    cat "${test_out}"
  } >>"${log_file}"
  # The runner prints the device lines before the first test.
  local device_lines
  device_lines="$(grep -E -m 2 '^(gpu|cpu):' "${test_out}" || true)"
  if [[ -n "${device_lines}" ]]; then
    printf '%s\n' "${device_lines}" | tee -a "${log_file}"
  fi
  local test_summary
  test_summary="$(grep -E '^(PASS|FAIL)' "${test_out}" | tail -n 1 || true)"
  if [[ ${test_status} -eq 0 && "${test_summary}" == PASS* ]]; then
    check "unit tests: ${test_summary} ($(elapsed_since "${started}") s)" ok
  else
    check "unit tests" fail \
      "${test_summary:-no result line}; see ${test_out}"
    return
  fi

  # --- Synthetic detect ---------------------------------------------------
  # Use the GPU backend when the build has one and a GPU was detected.
  backend="cpu-mt"
  if [[ "${build_target}" == "gpu" ]] &&
     printf '%s\n' "${device_lines}" | grep -q -E '^gpu: [^n]'; then
    backend="gpu"
  fi
  local detect_log="${out_dir}/log_detect_${backend}_${size}.txt"
  local detect_out="${out_dir}/detect_stdout.txt"
  local cmd=("${program}" --mode detect --input synthetic
             --synthetic-size "${size}" --backend "${backend}"
             --compare-to cpu --quiet --output-dir "${out_dir}"
             --log-file "${detect_log}")
  say "run: ${cmd[*]}"
  local detect_status=0
  started="$(date +%s.%N)"
  "${cmd[@]}" >"${detect_out}" 2>&1 || detect_status=$?
  {
    echo "+ ${cmd[*]}"
    cat "${detect_out}"
  } >>"${log_file}"
  local seconds
  seconds="$(elapsed_since "${started}")"

  if [[ ${detect_status} -eq 0 ]]; then
    check "detect exit code 0 (backend ${backend}, ${seconds} s)" ok
  else
    check "detect exit code" fail \
      "exit ${detect_status}; $(tail -n 1 "${detect_out}")"
    return
  fi

  local summary
  summary="$(log_line SUMMARY "${detect_log}")"
  if [[ "${summary}" == "SUMMARY: processed ${pixels} pixel time series"* ]]
  then
    check "SUMMARY line processed ${pixels} pixel time series" ok
  else
    check "SUMMARY line" fail "got '${summary:-nothing}'"
  fi

  # Percentage inside "recovered R of E eligible (97.0%)".
  local recovery_line recovery
  recovery_line="$(log_line synthetic_recovery "${detect_log}")"
  recovery="$(sed -n -E 's/.*eligible \(([0-9.]+)%\).*/\1/p' \
    <<<"${recovery_line}")"
  if [[ -n "${recovery}" ]] &&
     awk -v r="${recovery}" -v m="${min_recovery}" \
       'BEGIN { exit !(r >= m) }'; then
    check "synthetic_recovery ${recovery}% >= ${min_recovery}%" ok
  else
    check "synthetic_recovery >= ${min_recovery}%" fail \
      "got '${recovery_line:-nothing}'"
  fi

  local agree_line agree
  agree_line="$(log_line agreement_vs_cpu "${detect_log}")"
  agree="$(sed -n -E 's/.*identical_break_index ([0-9.]+).*/\1/p' \
    <<<"${agree_line}")"
  if [[ -n "${agree}" ]] &&
     awk -v a="${agree}" -v m="${min_agreement}" \
       'BEGIN { exit !(a >= m) }'; then
    check "agreement_vs_cpu identical break index ${agree} >= ${min_agreement}" \
      ok
  else
    check "agreement_vs_cpu identical break index >= ${min_agreement}" fail \
      "got '${agree_line:-nothing}'"
  fi

  # The files every detect run writes (a synthetic input has no reference,
  # so there is no validation.csv).
  local missing=() name
  for name in result_manifest.txt break_index.i16 magnitude.f32 status.u8 \
              rmse.f32 train_obs.u16 change_map.png pixel_stats.csv \
              trajectories.csv; do
    if [[ ! -s "${out_dir}/${name}" ]]; then
      missing+=("${name}")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    check "output files present in ${out_dir}" ok
  else
    check "output files present in ${out_dir}" fail \
      "missing or empty: ${missing[*]}"
  fi
}

# Start from an empty output directory so stale files cannot pass a check.
rm -rf "${out_dir}"
mkdir -p "${out_dir}"
: >"${log_file}"

main
if [[ ${failures} -eq 0 ]]; then
  say "SMOKE: PASS"
  exit 0
fi
say "SMOKE: FAIL (${failures} check(s) failed; log: ${log_file})"
exit 1
