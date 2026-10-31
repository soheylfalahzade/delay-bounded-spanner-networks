#!/usr/bin/env bash
# usage: [OVERPASS_URL=https://.../api] bash scripts/restart_large_scale.sh
cd "$(dirname "$0")/.." || exit 1
source ~/spanner_venv/bin/activate
pkill -f large_scale_run 2>/dev/null; sleep 1
rm -f large_scale_3500.log
EXTRA=""
[ -n "${OVERPASS_URL:-}" ] && EXTRA="--overpass-url ${OVERPASS_URL}"
nohup python benchmarks/large_scale_run.py --cities Isfahan Tokyo Manhattan --dist 3500 --t 1.5 2 3 $EXTRA > large_scale_3500.log 2>&1 &
echo "[started] large_scale_run $EXTRA (log: large_scale_3500.log)"
sleep 120
bash scripts/status.sh
