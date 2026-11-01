# Runbook for the larger-scale runs

```bash
cd ~/delay-bounded-spanner-networks && source ~/spanner_venv/bin/activate

# larger scale (3 cities, 3.5 km) and larger OD samples
python benchmarks/large_scale_run.py --cities Isfahan Tokyo Manhattan --dist 3500 --t 1.5 2 3
python run_spanner_benchmark.py --city-sources 30 --city-targets 30 --repeats 5

# data archive for a DOI + locked environment
python scripts/package_for_zenodo.py
pip freeze > requirements-lock.txt

# congestion model vs measured street speeds (NYC DOT), four boroughs
python scripts/make_model_profile.py
for B in Manhattan Brooklyn Queens Bronx; do
  python tools/validate_congestion_shape.py --start 2026-09-01 --end 2026-10-01 --borough "$B" \
    --model-json results/model_speed_profile.json --out "results/congestion_shape_validation_${B}.json"
done
```
