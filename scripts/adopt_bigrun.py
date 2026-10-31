#!/usr/bin/env python3
"""Copy the results of the large-OD run (~/spanner_bigrun/results) into this repository's results/ (overwrite),
leaving graph_cache/, benchmarks/ and figures/ untouched."""
import os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/spanner_bigrun/results")
DST = os.path.join(ROOT, "results")
SKIP = {"graph_cache", "benchmarks", "figures"}
if not os.path.exists(os.path.join(SRC, "spanner_report.json")):
    sys.exit(f"no spanner_report.json in {SRC}")
n = 0
for name in sorted(os.listdir(SRC)):
    if name in SKIP:
        continue
    s, d = os.path.join(SRC, name), os.path.join(DST, name)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)
    n += 1
    print("copied", name)
print(f"{n} entries copied from {SRC}")
