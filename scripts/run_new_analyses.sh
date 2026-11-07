#!/usr/bin/env bash
# Run the three added analyses on the real OpenStreetMap city cores, refresh the README sections and re-run the tests.
# Intended for WSL / Linux in the same environment used for the earlier runs (conda env spanner_research_env).
#
#   bash scripts/run_new_analyses.sh            # all steps
#   QUICK=1 bash scripts/run_new_analyses.sh    # Yazd and Isfahan only, 10 regions (first look, a few minutes)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${QUICK:-0}" = "1" ]; then
  CITIES="Yazd Isfahan"; REGIONS=10; HELDOUT=20
else
  CITIES="Yazd Isfahan Manhattan Tokyo"; REGIONS=20; HELDOUT=40
fi

echo "== 1/5 slack captured (reads existing result files)"
python benchmarks/slack_capture.py

echo "== 2/5 forced edges by hour (real city cores; the graph cache is used when present)"
if [ "${QUICK:-0}" = "1" ]; then
  python benchmarks/forced_by_hour.py --cities $CITIES --t 1.5 2 3
else
  python benchmarks/forced_by_hour.py --t 1.5 2 3
fi

echo "== 3/5 region faults: ${CITIES}, ${REGIONS} training regions, ${HELDOUT} held-out"
python benchmarks/region_fault.py --cities $CITIES --t 2 --regions "$REGIONS" --heldout "$HELDOUT"

echo "== 4/5 refresh README sections"
python benchmarks/render_readme_tables.py
python scripts/update_readme_new_sections.py

echo "== 5/5 tests and consistency checks"
python -m unittest discover -s tests
python benchmarks/render_readme_tables.py --check
python scripts/update_readme_new_sections.py --check
python verify_results.py | tail -3
echo "done. Review with: git status && git diff README.md"
