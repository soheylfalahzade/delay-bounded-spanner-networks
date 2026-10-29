#!/usr/bin/env bash
# Reproducible environment (WSL2 / Ubuntu). Usage:  bash scripts/setup_wsl_env.sh
set -euo pipefail
ENV_NAME=spanner_research_env
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found: install Miniconda first (https://docs.conda.io/en/latest/miniconda.html)"; exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^${ENV_NAME} " || conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"
conda install -c conda-forge networkx osmnx geopandas shapely scipy numpy pandas matplotlib seaborn tqdm tabulate -y
pip install -r requirements.txt
python -m unittest discover -s tests -v
echo "OK: environment '${ENV_NAME}' ready. Activate with: conda activate ${ENV_NAME}"
