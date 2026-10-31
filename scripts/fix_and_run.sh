#!/usr/bin/env bash
# Stops the invalid (synthetic-network) runs, installs osmnx, restarts the real-data runs, prints a status report.
cd "$(dirname "$0")/.." || exit 1
source ~/spanner_venv/bin/activate

pkill -f large_scale_run 2>/dev/null
pkill -f run_spanner_benchmark 2>/dev/null
sleep 1
rm -f large_scale*.log od_big.log
git checkout -- results/ 2>/dev/null
git clean -fdq results/benchmarks/large_scale_ 2>/dev/null

pip install -q osmnx || { echo "[FAIL] pip install osmnx"; exit 1; }
python -c "import osmnx, geopandas; print('[ok] osmnx', osmnx.__version__)" || { echo "[FAIL] osmnx import"; exit 1; }

nohup python benchmarks/large_scale_run.py --cities Isfahan Tokyo Manhattan --dist 3500 --t 1.5 2 3 > large_scale_3500.log 2>&1 &
echo "[started] large_scale_run (log: large_scale_3500.log)"

rm -rf ~/spanner_bigrun
cp -r "$(pwd)" ~/spanner_bigrun
( cd ~/spanner_bigrun && source ~/spanner_venv/bin/activate && \
  nohup python run_spanner_benchmark.py --city-sources 30 --city-targets 30 --repeats 5 > od_big.log 2>&1 & )
echo "[started] OD run in ~/spanner_bigrun (log: ~/spanner_bigrun/od_big.log)"

sleep 90
bash scripts/status.sh
