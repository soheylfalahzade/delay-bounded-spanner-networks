#!/usr/bin/env bash
# Waits for the NYC DOT CSV downloaded in the browser, then runs the congestion-shape validation.
cd "$(dirname "$0")/.." || exit 1
source ~/spanner_venv/bin/activate
D=/mnt/c/Users/Admin/Downloads
echo "waiting for a CSV named i4gi-tjb9*.csv, DOT_Traffic_Speeds*.csv or nyc_speeds*.csv in $D (Ctrl+C to stop)"
while true; do
  F=$(ls -t "$D"/i4gi-tjb9*.csv "$D"/DOT_Traffic_Speeds*.csv "$D"/nyc_speeds*.csv 2>/dev/null | head -1)
  if [ -n "$F" ] && [ -z "$(ls "$D"/*.crdownload "$D"/*.part 2>/dev/null)" ]; then break; fi
  sleep 10
done
echo "found: $F"
head -2 "$F" | cut -c1-300
python tools/validate_congestion_shape.py --start 2026-09-01 --end 2026-10-01 --borough Manhattan \
  --csv "$F" --model-json results/model_speed_profile.json --out results/congestion_shape_validation_Manhattan.json
