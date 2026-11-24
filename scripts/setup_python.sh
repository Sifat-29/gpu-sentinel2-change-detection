#!/usr/bin/env bash
# Non-GPU helper tooling. Creates project/.venv and installs the pinned
# requirements from scripts/requirements.txt. Safe to run repeatedly: an
# existing venv is reused and pip only installs what is missing.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "${script_dir}")"
venv_dir="${repo_dir}/.venv"
python_bin="${PYTHON:-python3}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  echo "creating virtual environment in ${venv_dir}"
  "${python_bin}" -m venv "${venv_dir}"
else
  echo "reusing virtual environment in ${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install --quiet --upgrade pip
"${venv_dir}/bin/python" -m pip install --quiet \
  -r "${script_dir}/requirements.txt"

# Fail loudly if the lint tool is not importable.
"${venv_dir}/bin/python" -c "import cpplint; print('cpplint importable')"
echo "python environment ready: ${venv_dir}"
