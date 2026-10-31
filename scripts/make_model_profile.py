#!/usr/bin/env python3
"""Write the 24-point MEAN SPEED profile of the parametric congestion model (km/h) for tools/validate_congestion_shape.py.
Representative edge: inbound 0.5, CBD weight 0.6, jitter 0.5, free flow 100 km/h, alpha 0.35 (same as benchmarks/literature_benchmarks.py)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run_spanner_benchmark as core
cong = core.edge_congestion_curve(0.5, 0.6, 0.5)
speeds = [float(v) for v in core.speed_curve(100.0, 0.35, cong)]
out = os.path.join(ROOT, "results", "model_speed_profile.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(speeds, open(out, "w"))
print("wrote", out, "min/max speed:", round(min(speeds), 1), round(max(speeds), 1))
