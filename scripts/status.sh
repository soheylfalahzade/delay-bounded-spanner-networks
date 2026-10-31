#!/usr/bin/env bash
# What is running, and is it on real OSM data?
cd "$(dirname "$0")/.." || exit 1
echo "=== running processes ==="
ps aux | grep -E "large_scale_run|run_spanner_benchmark" | grep -v grep | awk '{printf "PID %s  cpu-time %s  %s %s %s\n",$2,$10,$12,$13,$14}'
[ -z "$(pgrep -f 'large_scale_run|run_spanner_benchmark')" ] && echo "(nothing running)"
for f in large_scale_3500.log ~/spanner_bigrun/od_big.log; do
  [ -f "$f" ] || continue
  echo; echo "=== $f  (last modified: $(date -r "$f" +%H:%M:%S), now: $(date +%H:%M:%S)) ==="
  echo "osmnx-missing / synthetic-fallback lines: $(grep -c -i 'osmnx unavailable\|Synthetic grid\|FAIL' "$f")"
  grep -i "real OSM\|FAIL\|osmnx unavailable" "$f" | head -4
  tail -n 3 "$f"
done
