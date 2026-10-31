#!/usr/bin/env python3
"""Delete cached graphs whose meta says source == 'synthetic' (created when osmnx was missing or Overpass failed).
A synthetic graph cached under a real-city key is silently re-loaded on every later run."""
import glob, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n = 0
for m in sorted(glob.glob(os.path.join(ROOT, "results", "graph_cache", "*.meta.json"))):
    try:
        d = json.load(open(m))
    except Exception:
        continue
    if d.get("source") == "synthetic":
        print("delete", os.path.basename(m), "dist", d.get("dist"), "nodes", d.get("nodes"))
        os.remove(m)
        pkl = m.replace(".meta.json", ".pkl")
        if os.path.exists(pkl):
            os.remove(pkl)
        n += 1
print(f"{n} synthetic cache entries removed")
real = [json.load(open(m)) for m in glob.glob(os.path.join(ROOT, "results", "graph_cache", "*.meta.json"))]
print(f"{sum(1 for d in real if d.get('source') == 'osmnx')} real-OSM cache entries remain")
