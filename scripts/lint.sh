#!/usr/bin/env bash
# Non-GPU helper tooling. Checks the C++/CUDA sources in src/ and tests/ with
# clang-format (Google style, --dry-run --Werror) and cpplint, and prints
# "LINT: PASS" (exit 0) or "LINT: FAIL" (exit 1). lib/ and scripts/ are not
# checked.
#
# Usage: scripts/lint.sh [--save DIR]
#   --save DIR   also write clang_format.txt and cpplint.txt into DIR
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "${script_dir}")"
save_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --save)
      if [[ $# -lt 2 ]]; then
        echo "error: --save needs a directory" >&2
        exit 2
      fi
      save_dir="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

cd "${repo_dir}"

# Locate cpplint on PATH or in the project virtual environment.
if command -v cpplint > /dev/null 2>&1; then
  cpplint_bin="$(command -v cpplint)"
elif [[ -x "${repo_dir}/.venv/bin/cpplint" ]]; then
  cpplint_bin="${repo_dir}/.venv/bin/cpplint"
else
  echo "error: cpplint not found; run 'make venv' first" >&2
  echo "LINT: FAIL"
  exit 1
fi

mapfile -t sources < <(find src tests -type f \( -name '*.cc' -o -name '*.h' \
  -o -name '*.cu' -o -name '*.cuh' \) | sort)

clang_out="$(mktemp)"
cpplint_out="$(mktemp)"
trap 'rm -f "${clang_out}" "${cpplint_out}"' EXIT

status=0
clang_status=0

echo "clang-format: checking ${#sources[@]} files"
if ! clang-format --style=file --dry-run --Werror "${sources[@]}" \
  > "${clang_out}" 2>&1; then
  status=1
  clang_status=1
fi
cat "${clang_out}"

echo "cpplint: checking src and tests"
if ! "${cpplint_bin}" --recursive --extensions=cc,h,cu,cuh src tests \
  > "${cpplint_out}" 2>&1; then
  status=1
fi
cat "${cpplint_out}"

if [[ -n "${save_dir}" ]]; then
  mkdir -p "${save_dir}"
  {
    echo "# clang-format --style=file --dry-run --Werror (${#sources[@]} files)"
    cat "${clang_out}"
    echo "clang-format check exit status: ${clang_status}"
  } > "${save_dir}/clang_format.txt"
  cp "${cpplint_out}" "${save_dir}/cpplint.txt"
fi

if [[ ${status} -eq 0 ]]; then
  echo "LINT: PASS"
  exit 0
fi
echo "LINT: FAIL"
exit 1
