#!/usr/bin/env python3
"""Try a small real Overpass request (Isfahan, 500 m) against several public endpoints and report which one works."""
import sys, time
import osmnx as ox

CANDIDATES = [
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://overpass.kumi.systems/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]
ox.settings.requests_timeout = 90
ox.settings.use_cache = False
ok = []
for url in CANDIDATES:
    ox.settings.overpass_url = url
    t = time.time()
    try:
        G = ox.graph_from_point((31.8974, 54.3569), dist=500, network_type="drive")
        print(f"OK    {url}  ({len(G.nodes)} nodes / {len(G.edges)} edges, {time.time() - t:.0f}s)", flush=True)
        ok.append(url)
    except Exception as e:
        print(f"FAIL  {url}  ({type(e).__name__}: {str(e)[:90].splitlines()[0]})", flush=True)
print("WORKING:", ok[0] if ok else "none")
sys.exit(0 if ok else 1)
