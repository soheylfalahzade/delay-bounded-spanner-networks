#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! conda env list | grep -q "spanner_research_env"; then
  echo "[setup] creating conda env spanner_research_env ..."
  conda env create -f environment.yml
fi
conda run -n spanner_research_env --no-capture-output python evaluate.py "$@"
