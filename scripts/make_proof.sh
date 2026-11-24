#!/usr/bin/env bash
# Non-GPU helper tooling. Builds the proof-of-execution bundle from results/.
#
# Usage: scripts/make_proof.sh [--out FILE] [--check] [--root DIR] [--skip-lint]
#   --out FILE    archive to write (default: proof_of_execution.tar.gz in the
#                 project directory)
#   --check       do every check but do not write the archive
#   --root DIR    project directory to work on (default: parent of scripts/)
#   --skip-lint   do not regenerate results/lint (testing only)
#
# Steps: (1) regenerate lint evidence with scripts/lint.sh --save results/lint,
# (2) require results/MANIFEST.md and results/RESULTS.md, (3) check that every
# file under results/ is listed exactly once in the MANIFEST.md table
# (| File | Produced by (command) | What it shows |, File = path relative to
# results/ in backticks) and every listed file exists, (4) warn for files over
# 5 MB and a total over 25 MB, (5) write a gzip tar whose top folder is
# proof_of_execution/, list it, and print "PROOF: PASS <size>".
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "${script_dir}")"
invocation_dir="$(pwd)"
out=""
check_only=0
skip_lint=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      [[ $# -ge 2 ]] || { echo "error: --out needs a file" >&2; exit 2; }
      out="$2"
      shift 2
      ;;
    --root)
      [[ $# -ge 2 ]] || { echo "error: --root needs a directory" >&2; exit 2; }
      root="$(cd "$2" && pwd)"
      shift 2
      ;;
    --check)
      check_only=1
      shift
      ;;
    --skip-lint)
      skip_lint=1
      shift
      ;;
    -h | --help)
      sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

# A relative --out is relative to where the command was started.
if [[ -z "${out}" ]]; then
  out="${root}/proof_of_execution.tar.gz"
elif [[ "${out}" != /* ]]; then
  out="${invocation_dir}/${out}"
fi

fail() {
  echo "PROOF: FAIL ($1)"
  exit 1
}

cd "${root}"

# Step 1: lint evidence.
if [[ ${skip_lint} -eq 0 ]]; then
  echo "== regenerating lint evidence"
  if ! bash "${root}/scripts/lint.sh" --save results/lint; then
    fail "lint is not clean; fix it before bundling"
  fi
fi

# Step 2: required documents.
[[ -d results ]] || fail "results/ does not exist"
for required in MANIFEST.md RESULTS.md; do
  [[ -f "results/${required}" ]] || fail "results/${required} is missing"
done

# Step 3: MANIFEST.md versus the tree.
listed="$(mktemp)"
actual="$(mktemp)"
trap 'rm -f "${listed}" "${actual}"' EXIT

# First cell of every table row that holds a backticked path.
grep -E '^\|[[:space:]]*`[^`]+`[[:space:]]*\|' results/MANIFEST.md |
  sed -E 's/^\|[[:space:]]*`([^`]+)`.*/\1/' | LC_ALL=C sort > "${listed}" || true
(cd results && find . -type f | sed 's|^\./||' | LC_ALL=C sort) > "${actual}"

problems=0
if ! grep -qE '^\|[[:space:]]*File[[:space:]]*\|[[:space:]]*Produced by' \
  results/MANIFEST.md; then
  echo "MISMATCH: MANIFEST.md has no '| File | Produced by (command) | What it shows |' header"
  problems=$((problems + 1))
fi
while IFS= read -r name; do
  echo "MISMATCH: listed more than once: ${name}"
  problems=$((problems + 1))
done < <(uniq -d "${listed}")
while IFS= read -r name; do
  echo "MISMATCH: listed in MANIFEST.md but missing from results/: ${name}"
  problems=$((problems + 1))
done < <(comm -23 <(uniq "${listed}") "${actual}")
while IFS= read -r name; do
  echo "MISMATCH: in results/ but not listed in MANIFEST.md: ${name}"
  problems=$((problems + 1))
done < <(comm -13 <(uniq "${listed}") "${actual}")
if [[ ${problems} -gt 0 ]]; then
  fail "${problems} manifest mismatch(es)"
fi
echo "== MANIFEST.md lists all $(wc -l < "${actual}") files exactly once"

# Step 4: size warnings.
while IFS= read -r name; do
  size=$(stat -c %s "results/${name}")
  if [[ ${size} -gt 5000000 ]]; then
    echo "WARNING: ${name} is $((size / 1000000)) MB (> 5 MB)"
  fi
done < "${actual}"
total=$(du -sb results | cut -f1)
if [[ ${total} -gt 25000000 ]]; then
  echo "WARNING: results/ totals $((total / 1000000)) MB (> 25 MB)"
fi

if [[ ${check_only} -eq 1 ]]; then
  echo "PROOF: PASS (check only, no archive written)"
  exit 0
fi

# Step 5: archive with top folder proof_of_execution/.
mkdir -p "$(dirname "${out}")"
tar --sort=name --owner=0 --group=0 --numeric-owner \
  --transform 's,^results,proof_of_execution,' -czf "${out}" results
listing="$(tar -tzf "${out}")"
echo "${listing}"
for required in MANIFEST.md RESULTS.md; do
  grep -qx "proof_of_execution/${required}" <<< "${listing}" ||
    fail "archive lacks ${required}"
done
echo "PROOF: PASS $(du -h "${out}" | cut -f1)"
