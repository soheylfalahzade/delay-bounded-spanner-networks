#!/usr/bin/env bash
# usage: bash scripts/stash_nyc.sh Brooklyn   -> moves the newest i4gi-tjb9*.csv from Downloads to ~/data/nyc_Brooklyn.csv
B="${1:?usage: stash_nyc.sh <Borough>}"
D=/mnt/c/Users/Admin/Downloads
F=$(ls -t "$D"/i4gi-tjb9*.csv 2>/dev/null | head -1)
[ -n "$F" ] || { echo "no i4gi-tjb9*.csv in $D"; exit 1; }
mkdir -p ~/data && mv "$F" ~/data/nyc_${B}.csv && echo "stored: ~/data/nyc_${B}.csv" && head -2 ~/data/nyc_${B}.csv && wc -l ~/data/nyc_${B}.csv
