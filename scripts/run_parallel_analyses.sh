#!/usr/bin/env bash
# Run the added analyses in parallel on the real OpenStreetMap city cores, then refresh the README and run the tests.
#
#   bash scripts/run_parallel_analyses.sh                 # all jobs, at most 5 at a time
#   JOBS=3 bash scripts/run_parallel_analyses.sh          # fewer parallel jobs
#   QUICK=1 bash scripts/run_parallel_analyses.sh         # Yazd and Isfahan only, few cities/windows (first look)
#   SKIP_METRO=1 bash scripts/run_parallel_analyses.sh    # skip the metropolitan-scale run (largest memory/time)
#   ONLY="directedness lightness edge_order minimality" bash scripts/run_parallel_analyses.sh   # run a subset of the jobs
#
# Logs: logs/<job>.log.  A job that fails is reported at the end; the README is refreshed from the jobs that succeeded.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
JOBS="${JOBS:-5}"
echo "cpus: $(nproc 2>/dev/null || echo ?)   parallel jobs: $JOBS"

if [ "${QUICK:-0}" = "1" ]; then
  CITIES="Yazd Isfahan"; REGIONS=10; HELDOUT=20; WINDOWS=12; ADDED=4; METRO_CITIES="Yazd"; METRO_DIST=5000
else
  CITIES="Yazd Isfahan Manhattan Tokyo"; REGIONS=20; HELDOUT=40; WINDOWS=30; ADDED=0; METRO_CITIES="Isfahan Tokyo"; METRO_DIST=7000
fi

rm -f logs/.failed
run() {
  local name="$1"; shift
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then return; fi
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  ( if "$@" > "logs/$name.log" 2>&1; then echo "[done]   $name"; else echo "[FAILED] $name (logs/$name.log)"; echo "$name" >> logs/.failed; fi ) &
}

python benchmarks/slack_capture.py > logs/slack_capture.log 2>&1 || echo slack_capture >> logs/.failed

run minimality        python benchmarks/minimality.py --cities $CITIES --t 1.5 2 3
run road_exact        python benchmarks/road_exact.py --cities $CITIES --t 2 3 5 --windows "$WINDOWS" --max-free 18
run emergency         python benchmarks/emergency_response.py --cities $CITIES --t 2 --regions "$REGIONS" --heldout "$HELDOUT"
if [ "$ADDED" = "0" ]; then
  run more_cities     python benchmarks/more_cities.py --t 2 3
else
  run more_cities     python benchmarks/more_cities.py --t 2 3 --max-cities "$ADDED"
fi
if [ "${SKIP_METRO:-0}" != "1" ]; then
  run metro           python benchmarks/large_scale_run.py --cities $METRO_CITIES --dist "$METRO_DIST" --t 2 --max-edges 400000
fi
if [ "$ADDED" = "0" ]; then
  run directedness    python benchmarks/directedness.py --t 2 3
  run lightness       python benchmarks/lightness.py --t 2 3
else
  run directedness    python benchmarks/directedness.py --t 2 3 --max-cities "$ADDED"
  run lightness       python benchmarks/lightness.py --t 2 3 --max-cities "$ADDED"
fi
run edge_order        python benchmarks/edge_order.py --cities $CITIES --t 2 3 --seeds 5
run all_discs         python benchmarks/all_discs.py --cities $CITIES --t 2
run cut_structure     python benchmarks/cut_structure.py --cities $CITIES
if [ "$ADDED" = "0" ]; then
  run oneway_dose     python benchmarks/oneway_dose.py --t 2 3
else
  run oneway_dose     python benchmarks/oneway_dose.py --t 2 3 --max-cities "$ADDED"
fi
run region_fault      python benchmarks/region_fault.py --cities $CITIES --t 2 --regions "$REGIONS" --heldout "$HELDOUT"
run forced_by_hour    python benchmarks/forced_by_hour.py --t 1.5 2 3
wait

echo "== README"
python benchmarks/render_readme_tables.py
python scripts/update_readme_new_sections.py
echo "== tests"
python -m unittest discover -s tests 2>&1 | tail -4
python benchmarks/render_readme_tables.py --check
python scripts/update_readme_new_sections.py --check
python verify_results.py | tail -2
if [ -f logs/.failed ]; then echo "failed jobs:"; cat logs/.failed; exit 1; fi
echo "all jobs finished; review: git status && git diff README.md"
