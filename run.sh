#!/usr/bin/env bash
# Documented example runs of change_detect.
#
# Usage: ./run.sh <example> [extra change_detect flags...]
#
# Every example prints the exact command before it runs and writes its
# results and a log (log_<mode>_<backend>_<H>x<W>x<T>.txt) to out/<example>/.
# Extra flags are appended to the command, so they override the defaults of
# the example (for example `./run.sh dev --threshold 4`).
#
# Environment: BIN_DIR (directory of the built programs, default bin),
# PYTHON (default .venv/bin/python, used for figures).
#
# The real-data examples use the deforestation setting (--losses-only: only
# greenness losses count as changes); the model parameters otherwise keep the
# program defaults. They need data cubes created by scripts/fetch_sentinel2.py
# (see data/README.md); a missing cube prints the commands that create it.
set -euo pipefail

# Work from the repository root, wherever the script is called from.
cd "$(dirname "${BASH_SOURCE[0]}")"

readonly bin_dir="${BIN_DIR:-bin}"
readonly python_bin="${PYTHON:-.venv/bin/python}"
export BIN_DIR="${bin_dir}"

# Cube manifests of the documented data sets.
readonly sample_manifest="data/sample/manifest.txt"
readonly dev_manifest="data/cubes/chaco_dev2048/manifest.txt"
readonly demo_manifest="data/cubes/chaco_demo5000/manifest.txt"
readonly full_manifest="data/cubes/chaco_full/manifest.txt"

# Flags of every real-data detect run.
readonly real_flags=(--losses-only)

# Set by resolve_program: the program, the backend that stands for "the GPU
# run" and the backend that the synthetic example compares it with.
program=""
gpu_backend="gpu"
compare_backend="cpu-mt"

# Extra command-line flags of the example (appended to every command).
extra=()

# Output directory of the last run_cd call (the figures read it).
last_outdir=""

usage() {
  cat <<'EOF'
Usage: ./run.sh <example> [extra change_detect flags...]

Examples (results and logs go to out/<example>/):
  synthetic          2048x2048x48 synthetic cube on the GPU, compared with
                     cpu-mt. Needs no data.
  sample             committed 128x128 real cube (data/sample), GPU detect
                     with validation, then figures. Needs no download.
  dev                2048x2048 development cube, GPU detect with validation.
                     Needs data/cubes/chaco_dev2048.
  demo               5000x5000 demo cube, GPU detect with validation.
                     Needs data/cubes/chaco_demo5000.
  full               full 10980x10980 tile streamed from disk, GPU detect with
                     validation. Needs data/cubes/chaco_full.
  cpu-baseline <dev|demo|full>
                     same input with --backend cpu-mt (and cpu, the
                     single-thread reference, for dev). Needs that cube.
  quick-diff         NDVI difference of two dates on the dev cube (sample if
                     dev is missing).
  live-demo          what the video shows live: GPU detect plus figures on
                     the dev cube (sample if dev is missing); prints the
                     elapsed time (target <= 30 s).
  benchmark [args]   the benchmark suite (scripts/run_benchmarks.sh).
  help               this text (default).

Environment: BIN_DIR (built programs, default bin), PYTHON (figures).
Without bin/change_detect (no CUDA build) the CPU program is used with
--backend cpu-mt instead of the GPU.
EOF
}

# Finds the program to run. Prefers the GPU build, falls back to the CPU one.
resolve_program() {
  if [[ -x "${bin_dir}/change_detect" ]]; then
    program="${bin_dir}/change_detect"
  elif [[ -x "${bin_dir}/change_detect_cpu" ]]; then
    echo "note: ${bin_dir}/change_detect not built; using" \
      "${bin_dir}/change_detect_cpu with --backend cpu-mt"
    program="${bin_dir}/change_detect_cpu"
    gpu_backend="cpu-mt"
    compare_backend="cpu"
  else
    echo "error: no program in ${bin_dir}/ (run: make)" >&2
    exit 1
  fi
}

# flag_value <flag> <default>: value of the last "--flag v" or "--flag=v" in
# the extra flags, else the default.
flag_value() {
  local flag="$1" value="$2" i
  for ((i = 0; i < ${#extra[@]}; i++)); do
    if [[ "${extra[i]}" == "${flag}" && $((i + 1)) -lt ${#extra[@]} ]]; then
      value="${extra[i + 1]}"
    elif [[ "${extra[i]}" == "${flag}="* ]]; then
      value="${extra[i]#"${flag}="}"
    fi
  done
  printf '%s' "${value}"
}

# manifest_label <manifest>: "HxWxT" of a cube, for the log file name.
manifest_label() {
  awk '$1 == "height" { h = $2 } $1 == "width" { w = $2 }
       $1 == "num_dates" { t = $2 }
       END { printf "%sx%sx%s", h, w, t }' "$1"
}

# Prints the commands that create a missing cube (from data/README.md) and
# returns 1.
missing_cube() {
  local name="$1"
  echo "error: cube data/cubes/${name} not found; create it first" \
    "(details: data/README.md):" >&2
  echo "  bash scripts/setup_python.sh" >&2
  case "${name}" in
    chaco_dev2048)
      cat >&2 <<'EOF'
  .venv/bin/python scripts/fetch_sentinel2.py --name chaco_dev2048 --tile 20KQA \
      --window 0,3008,2048,2048 --workers 4 --delete-staging
  .venv/bin/python scripts/fetch_hansen.py --cube data/cubes/chaco_dev2048/manifest.txt
EOF
      ;;
    chaco_demo5000)
      cat >&2 <<'EOF'
  .venv/bin/python scripts/fetch_sentinel2.py --name chaco_demo5000 --tile 20KQA \
      --window 0,5504,5000,5000 --workers 4
  .venv/bin/python scripts/fetch_hansen.py --cube data/cubes/chaco_demo5000/manifest.txt
  (or crop it from the full cube:
  .venv/bin/python scripts/crop_cube.py --input data/cubes/chaco_full/manifest.txt \
      --window 0,5504,5000,5000 --out data/cubes/chaco_demo5000)
EOF
      ;;
    chaco_full)
      cat >&2 <<'EOF'
  .venv/bin/python scripts/fetch_sentinel2.py --name chaco_full --tile 20KQA \
      --workers 6 --delete-staging
  .venv/bin/python scripts/fetch_hansen.py --cube data/cubes/chaco_full/manifest.txt
EOF
      ;;
  esac
  return 1
}

# run_cd <outdir> <mode> <backend> <input> <label> [default flags...]
# Prints and runs one change_detect command; the extra flags come last so
# they win. The log goes to <outdir>/log_<mode>_<backend>_<label>.txt.
run_cd() {
  local outdir="$1" mode="$2" backend="$3" input="$4" label="$5"
  shift 5
  backend="$(flag_value --backend "${backend}")"
  outdir="$(flag_value --output-dir "${outdir}")"
  mkdir -p "${outdir}"
  last_outdir="${outdir}"
  local cmd=("${program}" --mode "${mode}" --input "${input}"
             --backend "${backend}" --output-dir "${outdir}"
             --log-file "${outdir}/log_${mode}_${backend}_${label}.txt"
             "$@" "${extra[@]+"${extra[@]}"}")
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

# render_figures <result dir> <cube manifest> [render_figures.py flags...]:
# figures into <result>/figures.
render_figures() {
  if [[ ! -x "${python_bin}" ]]; then
    echo "error: ${python_bin} not found (run: bash scripts/setup_python.sh)" \
      >&2
    return 1
  fi
  local result="$1" cube="$2"
  shift 2
  local cmd=("${python_bin}" scripts/render_figures.py --result "${result}"
             --cube "${cube}" --out "${result}/figures" "$@")
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

# Detect on a real cube, GPU (or cpu-mt without a CUDA build).
# real_detect <name> <manifest> <cube name> [flags...]
real_detect() {
  local name="$1" manifest="$2" cube="$3"
  shift 3
  [[ -f "${manifest}" ]] || missing_cube "${cube}" || return 1
  resolve_program
  run_cd "out/${name}" detect "${gpu_backend}" "${manifest}" \
    "$(manifest_label "${manifest}")" "${real_flags[@]}" "$@"
}

example_synthetic() {
  resolve_program
  local size
  size="$(flag_value --synthetic-size 2048x2048x48)"
  run_cd out/synthetic detect "${gpu_backend}" synthetic "${size}" \
    --synthetic-size 2048x2048x48 --compare-to "${compare_backend}"
}

example_sample() {
  real_detect sample "${sample_manifest}" chaco_sample128 || return 1
  render_figures "${last_outdir}" "${sample_manifest}"
}

example_cpu_baseline() {
  local which="${1:-}"
  [[ $# -gt 0 ]] && shift
  extra=("$@")
  local manifest cube backends
  case "${which}" in
    dev) manifest="${dev_manifest}"; cube=chaco_dev2048
         backends=(cpu-mt cpu) ;;
    demo) manifest="${demo_manifest}"; cube=chaco_demo5000
          backends=(cpu-mt) ;;
    full) manifest="${full_manifest}"; cube=chaco_full
          backends=(cpu-mt) ;;
    *) echo "usage: ./run.sh cpu-baseline <dev|demo|full> [flags...]" >&2
       return 2 ;;
  esac
  [[ -f "${manifest}" ]] || missing_cube "${cube}" || return 1
  resolve_program
  local backend
  for backend in "${backends[@]}"; do
    run_cd "out/cpu-baseline/${which}_${backend}" detect "${backend}" "${manifest}" \
      "$(manifest_label "${manifest}")" "${real_flags[@]}"
  done
}

example_quick_diff() {
  local manifest="${dev_manifest}"
  if [[ ! -f "${manifest}" ]]; then
    echo "note: dev cube not found; using the committed sample cube"
    manifest="${sample_manifest}"
  fi
  resolve_program
  run_cd out/quick-diff quick-diff "${gpu_backend}" "${manifest}" \
    "$(manifest_label "${manifest}")"
}

example_live_demo() {
  local manifest="${dev_manifest}" cube=chaco_dev2048
  if [[ ! -f "${manifest}" ]]; then
    echo "note: dev cube not found; using the committed sample cube"
    manifest="${sample_manifest}"
    cube=chaco_sample128
  fi
  local started
  started="$(date +%s.%N)"
  real_detect live-demo "${manifest}" "${cube}" || return 1
  # Maps downsampled to 1024 px keep the figure step short enough for a
  # live run (the full-size figures take about twice as long).
  render_figures "${last_outdir}" "${manifest}" --max-dim 1024
  local elapsed
  elapsed="$(awk -v a="${started}" -v b="$(date +%s.%N)" \
    'BEGIN { printf "%.1f", b - a }')"
  echo "live-demo elapsed: ${elapsed} s (detect and figures, target <= 30 s)"
  if awk -v e="${elapsed}" 'BEGIN { exit !(e > 30) }'; then
    echo "warning: slower than the 30 s target for a live demo" >&2
  fi
}

example_benchmark() {
  if [[ ! -f scripts/run_benchmarks.sh ]]; then
    echo "error: scripts/run_benchmarks.sh not found" >&2
    return 1
  fi
  bash scripts/run_benchmarks.sh "$@"
}

example="${1:-help}"
[[ $# -gt 0 ]] && shift

# cpu-baseline and benchmark take their own arguments; the other examples
# pass every remaining argument to change_detect.
if [[ "${example}" != "cpu-baseline" && "${example}" != "benchmark" ]]; then
  extra=("$@")
fi

case "${example}" in
  synthetic) example_synthetic ;;
  sample) example_sample ;;
  dev) real_detect dev "${dev_manifest}" chaco_dev2048 ;;
  demo) real_detect demo "${demo_manifest}" chaco_demo5000 ;;
  full) real_detect full "${full_manifest}" chaco_full --quiet ;;
  cpu-baseline) example_cpu_baseline "$@" ;;
  quick-diff) example_quick_diff ;;
  live-demo) example_live_demo ;;
  benchmark) example_benchmark "$@" ;;
  help | -h | --help) usage ;;
  *)
    echo "error: unknown example '${example}'" >&2
    usage >&2
    exit 2
    ;;
esac
