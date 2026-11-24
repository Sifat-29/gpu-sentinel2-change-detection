#!/usr/bin/env bash
# Runs every benchmark sweep of the project and checks each result CSV.
#
# Usage: scripts/run_benchmarks.sh [--dry-run] [--only SWEEP[,SWEEP]]
#                                  [--out DIR] [--quick]
#   --dry-run  print every command with the expected CSV row counts and check
#              that the inputs exist (missing inputs are reported, not fatal)
#   --only L   run only the listed sweeps (default: all)
#   --out DIR  output directory (default out/bench); logs go to DIR/logs,
#              CSVs to DIR/csv, the hardware record to DIR/hardware.txt
#   --quick    replace every input by a small synthetic cube and use 2
#              repeats; only for testing this script, never for reported
#              numbers
#
# Sweeps: backends_demo backends_full streams tiles blocks io scaling_dates
#         scaling_pixels
# Environment overrides: BIN, DEMO_CUBE, FULL_CUBE, FULL_REPEAT, PYTHON.
# Run it with the GPU otherwise idle: it refuses to start while another
# change_detect process is running. Each sweep CSV is checked with
# scripts/check_csv.py and the script stops at the first failure.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "${script_dir}")"

# ---- Configuration (override through the environment) ----------------------
BIN="${BIN:-bin/change_detect}"
DEMO_CUBE="${DEMO_CUBE:-data/cubes/chaco_demo5000/manifest.txt}"
FULL_CUBE="${FULL_CUBE:-data/cubes/chaco_full/manifest.txt}"
FULL_REPEAT="${FULL_REPEAT:-5}"  # repeats of the full-cube multi-run backends
if [[ -x .venv/bin/python ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi
REPEAT=5
WARMUP=1
ALL_SWEEPS="backends_demo backends_full streams tiles blocks io scaling_dates scaling_pixels"
# Synthetic stand-ins used by --quick (HxWxT).
QUICK_DEMO="512x512x24"
QUICK_FULL="640x640x24"

dry_run=0
quick=0
only=""
out_dir="out/bench"

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --quick) quick=1; shift ;;
    --only | --out)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 needs a value" >&2
        exit 2
      fi
      if [[ "$1" == "--only" ]]; then only="$2"; else out_dir="$2"; fi
      shift 2
      ;;
    -h | --help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${only}" ]]; then
  selected="${ALL_SWEEPS}"
else
  selected="${only//,/ }"
  for name in ${selected}; do
    if [[ " ${ALL_SWEEPS} " != *" ${name} "* ]]; then
      echo "error: unknown sweep '${name}' (known: ${ALL_SWEEPS})" >&2
      exit 2
    fi
  done
fi

if [[ ${quick} -eq 1 ]]; then
  REPEAT=2
  FULL_REPEAT=2
fi

log_dir="${out_dir}/logs"
csv_dir="${out_dir}/csv"
work_dir="${out_dir}/work"

# ---- Per-sweep bookkeeping --------------------------------------------------
sweep=""            # name of the sweep being planned or run
run_id=""           # run id shared by the runs of the sweep
csv_file=""         # result CSV of the sweep
sweep_rows=0        # expected CSV rows
sweep_configs=0     # expected configurations
sweep_min_rep=999   # smallest repeat count of a multi-repeat run
sweep_slack=0       # configurations allowed to merge (auto tile size)
sweep_singles=""    # backends allowed to have single-row configurations
sweep_max_bytes=0   # input bytes of the largest raw cube read by the sweep
sweep_file_input="" # name of that input when it is read from disk
total_rows=0
missing_inputs=0
summary=""          # dry-run matrix summary, one line per sweep
missing_list=" "    # inputs already reported as missing

# Prints H, W and T of an input ("synthetic:HxWxT" or a manifest path).
input_dims() {
  local input="$1"
  if [[ "${input}" == synthetic:* ]]; then
    echo "${input#synthetic:}"
  elif [[ -f "${input}" ]]; then
    awk '$1 == "height" { h = $2 } $1 == "width" { w = $2 }
         $1 == "num_dates" { t = $2 }
         END { printf "%sx%sx%s", h, w, t }' "${input}"
  else
    echo "?x?x?"
  fi
}

# Number of backends in a --backend value ("all" means five).
backend_count() {
  if [[ "$1" == "all" ]]; then
    echo 5
  else
    local only_commas="${1//[^,]/}"
    echo $((${#only_commas} + 1))
  fi
}

# Starts a sweep: resets the counters, names the CSV.
begin_sweep() {
  sweep="$1"
  run_id="${sweep}_$(date +%Y%m%dT%H%M%S)"
  csv_file="${csv_dir}/benchmark_${sweep}.csv"
  sweep_rows=0
  sweep_configs=0
  sweep_min_rep=999
  sweep_slack=0
  sweep_singles=""
  sweep_max_bytes=0
  sweep_file_input=""
}

# Registers (dry run) or executes one benchmark invocation.
#   job VARIANT BACKENDS INPUT WARMUP REPEAT [extra change_detect flags...]
# INPUT is a manifest path or synthetic:HxWxT. VARIANT may be empty.
job() {
  local variant="$1" backends="$2" input="$3" warmup="$4" repeat="$5"
  shift 5
  local count dims label log_file
  count="$(backend_count "${backends}")"
  dims="$(input_dims "${input}")"
  label="${backends//,/+}"
  log_file="${log_dir}/log_${sweep}_${label}_${dims}"
  if [[ -n "${variant}" ]]; then log_file+="_${variant}"; fi
  log_file+=".txt"

  local cmd=("${BIN}" --mode benchmark)
  if [[ "${input}" == synthetic:* ]]; then
    cmd+=(--input synthetic --synthetic-size "${input#synthetic:}")
  else
    cmd+=(--input "${input}")
  fi
  cmd+=(--backend "${backends}" --warmup "${warmup}" --repeat "${repeat}"
    "$@" --quiet --run-id "${run_id}" --benchmark-csv "${csv_file}"
    --output-dir "${work_dir}" --log-file "${log_file}")

  # Input bytes (5 per sample) of the largest cube of the sweep, for the
  # page-cache note in hardware.txt.
  local h w t bytes
  IFS=x read -r h w t <<< "${dims}"
  if [[ "${h}" =~ ^[0-9]+$ && "${w}" =~ ^[0-9]+$ && "${t}" =~ ^[0-9]+$ ]]; then
    bytes=$((h * w * t * 5))
    if [[ "${input}" != synthetic:* && ${bytes} -gt ${sweep_max_bytes} ]]; then
      sweep_max_bytes=${bytes}
      sweep_file_input="${input}"
    fi
  fi

  sweep_rows=$((sweep_rows + count * repeat))
  sweep_configs=$((sweep_configs + count))
  if [[ ${repeat} -gt 1 && ${repeat} -lt ${sweep_min_rep} ]]; then
    sweep_min_rep=${repeat}
  fi

  if [[ ${dry_run} -eq 1 ]]; then
    echo "  ${cmd[*]}"
    return 0
  fi
  echo "  run: ${label} ${dims} ${variant}"
  local stderr_file="${work_dir}/stderr.txt"
  if ! "${cmd[@]}" > /dev/null 2> "${stderr_file}"; then
    echo "error: benchmark run failed: ${cmd[*]}" >&2
    cat "${stderr_file}" >&2
    echo "run_benchmarks: FAILED" >&2
    exit 1
  fi
}

# ---- Sweep definitions ------------------------------------------------------
demo_input() {
  if [[ ${quick} -eq 1 ]]; then echo "synthetic:${QUICK_DEMO}"; else echo "${DEMO_CUBE}"; fi
}

full_input() {
  if [[ ${quick} -eq 1 ]]; then echo "synthetic:${QUICK_FULL}"; else echo "${FULL_CUBE}"; fi
}

# Inputs that must exist before a sweep can run.
sweep_inputs() {
  case "$1" in
    backends_demo | tiles | blocks | io) demo_input ;;
    backends_full) full_input ;;
    streams) demo_input; full_input ;;
    *) ;;
  esac
}

plan_backends_demo() {
  job "" all "$(demo_input)" "${WARMUP}" "${REPEAT}" --compare-to cpu
}

plan_backends_full() {
  local input
  input="$(full_input)"
  job "" gpu,gpu-unfused,cpu-mt "${input}" "${WARMUP}" "${FULL_REPEAT}"
  job single_run cpu-fp32 "${input}" 0 1
  job single_run cpu "${input}" 0 1
  sweep_singles="cpu,cpu-fp32"
}

plan_streams() {
  local input stream
  for input in "$(demo_input)" "$(full_input)"; do
    for stream in 1 3; do
      job "streams${stream}" gpu "${input}" "${WARMUP}" "${REPEAT}" \
        --streams "${stream}"
    done
  done
}

plan_tiles() {
  local input rows
  input="$(demo_input)"
  for rows in 16 32 64 128 256 512 0; do
    if [[ ${rows} -eq 0 ]]; then
      job tile_auto gpu "${input}" "${WARMUP}" "${REPEAT}" --tile-rows 0
    else
      job "tile${rows}" gpu "${input}" "${WARMUP}" "${REPEAT}" \
        --tile-rows "${rows}"
    fi
  done
  # The automatic tile size can equal one of the explicit sizes, which merges
  # two configurations in the CSV.
  sweep_slack=1
}

plan_blocks() {
  local input backend block
  input="$(demo_input)"
  for backend in gpu gpu-unfused; do
    for block in 64 128 256 512; do
      job "block${block}" "${backend}" "${input}" "${WARMUP}" "${REPEAT}" \
        --block-size "${block}"
    done
  done
}

# Parallel disk reads per tile on the demo cube; every backend uses the same
# reader, so the CPU and GPU rows show the same effect.
plan_io() {
  local input threads
  input="$(demo_input)"
  for threads in 1 2 4 8 16; do
    job "io${threads}" gpu,cpu-mt "${input}" "${WARMUP}" "${REPEAT}" \
      --io-threads "${threads}"
  done
}

plan_scaling_dates() {
  local dates size
  size="2048x2048"
  if [[ ${quick} -eq 1 ]]; then size="256x256"; fi
  # Monthly dates (T / 12 years) keep the 2-year training window at 24 dates,
  # so T changes only the length of the monitoring period. With a fixed time
  # span, small T would leave fewer training observations than the model
  # needs and most pixels would skip monitoring, which is not a fair scaling.
  for dates in 36 48 72 96 120; do
    job "" gpu,cpu-mt "synthetic:${size}x${dates}" "${WARMUP}" "${REPEAT}" \
      --synthetic-years "$((dates / 12))"
  done
}

plan_scaling_pixels() {
  local side sides
  sides="1024 2048 3072 4096"
  if [[ ${quick} -eq 1 ]]; then sides="128 256 384 512"; fi
  for side in ${sides}; do
    job "" gpu,cpu-mt "synthetic:${side}x${side}x48" "${WARMUP}" "${REPEAT}"
  done
}

# ---- Hardware record --------------------------------------------------------
record_hardware() {
  local file="${out_dir}/hardware.txt"
  {
    echo "date: $(date -Is)"
    echo "note: snapshot taken before the timed runs (the GPU is idle here)."
    echo "note: WSL2 exposes 16 of the 32 hardware threads of this CPU; every"
    echo "      CPU baseline uses at most those 16."
    echo
    echo "--- nvidia-smi (name, memory, driver, clocks, power, state) ---"
    nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.draw,power.limit,pstate --format=csv 2>&1 || true
    echo
    echo "--- nvidia-smi -q -d PERFORMANCE,POWER ---"
    nvidia-smi -q -d PERFORMANCE,POWER 2>&1 || true
    echo
    echo "--- lscpu ---"
    lscpu 2>&1 || true
    echo
    echo "--- free -h ---"
    free -h 2>&1 || true
    echo
    echo "--- df -h . ---"
    df -h . 2>&1 || true
    echo
    echo "--- kernel ---"
    uname -a
    echo
    echo "--- nproc ---"
    nproc
  } > "${file}"
  echo "hardware record: ${file}"
}

# Appends one line per sweep to hardware.txt: the largest raw cube the sweep
# reads against the total RAM, so readers can tell whether the cube fits in
# the page cache (warm reads after the warm-up) or is read cold from disk.
record_cache_note() {
  local file="${out_dir}/hardware.txt" ram_kb ram_bytes verdict
  ram_kb="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
  ram_bytes=$((ram_kb * 1024))
  if [[ ${sweep_max_bytes} -eq 0 ]]; then
    echo "sweep ${sweep}: synthetic cubes generated in memory, no disk reads" >> "${file}"
    return 0
  fi
  if [[ ${sweep_max_bytes} -lt ${ram_bytes} ]]; then
    verdict="fits in RAM: warm page cache after the warm-up run"
  else
    verdict="larger than RAM: read cold from disk"
  fi
  echo "sweep ${sweep}: input ${sweep_max_bytes} bytes ($(basename "$(dirname "${sweep_file_input}")")) vs RAM ${ram_bytes} bytes; ${verdict}" >> "${file}"
}

# ---- Main -------------------------------------------------------------------
mode_text="run"
if [[ ${dry_run} -eq 1 ]]; then mode_text="dry run"; fi
if [[ ${quick} -eq 1 ]]; then mode_text+=", quick (synthetic inputs, not for reporting)"; fi
echo "run_benchmarks: ${mode_text}; sweeps: ${selected}; output: ${out_dir}"

# Inputs must exist (a dry run only lists the missing ones).
for name in ${selected}; do
  for input in $(sweep_inputs "${name}"); do
    if [[ "${input}" != synthetic:* && ! -f "${input}" ]]; then
      echo "MISSING input for ${name}: ${input}"
      if [[ "${missing_list}" != *" ${input} "* ]]; then
        missing_list+="${input} "
        missing_inputs=$((missing_inputs + 1))
      fi
    fi
  done
done

if [[ ${dry_run} -eq 0 ]]; then
  if [[ ${missing_inputs} -gt 0 ]]; then
    echo "error: ${missing_inputs} input(s) missing; nothing was run" >&2
    exit 1
  fi
  if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not found or not executable (run make first)" >&2
    exit 1
  fi
  # pgrep -x compares the process name, which Linux truncates to 15
  # characters (change_detect_cpu becomes change_detect_c).
  if pgrep -x change_detect > /dev/null || pgrep -x change_detect_c > /dev/null; then
    echo "error: another change_detect process is running; refusing to start" >&2
    exit 1
  fi
  mkdir -p "${log_dir}" "${csv_dir}" "${work_dir}"
  record_hardware
fi

failed=0
for name in ${selected}; do
  begin_sweep "${name}"
  echo "== sweep ${name} =="
  if [[ ${dry_run} -eq 0 ]]; then
    # A CSV left from an earlier run would be appended to; set it aside.
    if [[ -f "${csv_file}" ]]; then
      mv "${csv_file}" "${csv_file}.old-$(date +%Y%m%dT%H%M%S)"
    fi
  fi
  sweep_start=${SECONDS}
  "plan_${name}"
  total_rows=$((total_rows + sweep_rows))
  if [[ ${dry_run} -eq 0 ]]; then
    record_cache_note
  fi
  if [[ ${dry_run} -eq 1 ]]; then
    echo "  expected: ${sweep_rows} CSV rows, ${sweep_configs} configurations -> ${csv_file}"
    summary+="$(printf '  %-15s %5d rows %3d configs' "${name}" "${sweep_rows}" "${sweep_configs}")"$'\n'
    continue
  fi
  check=("${PYTHON}" scripts/check_csv.py --csv "${csv_file}"
    --min-repeats "$((sweep_min_rep == 999 ? 1 : sweep_min_rep))"
    --expect-configs "${sweep_configs}" --config-slack "${sweep_slack}"
    --expect-rows "${sweep_rows}")
  if [[ -n "${sweep_singles}" ]]; then check+=(--allow-single "${sweep_singles}"); fi
  if ! "${check[@]}"; then
    echo "error: CSV check failed for sweep ${name}" >&2
    failed=1
    break
  fi
  echo "sweep ${name}: $((SECONDS - sweep_start)) s"
done

if [[ ${dry_run} -eq 1 ]]; then
  echo "== dry-run summary =="
  printf '%s' "${summary}"
  echo "TOTAL expected CSV rows: ${total_rows}; missing inputs: ${missing_inputs}"
  exit 0
fi
if [[ ${failed} -ne 0 ]]; then
  echo "run_benchmarks: FAILED" >&2
  exit 1
fi
echo "run_benchmarks: all sweeps passed (${total_rows} rows); CSVs in ${csv_dir}, logs in ${log_dir}"
