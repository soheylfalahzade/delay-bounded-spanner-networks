#!/usr/bin/env python3
"""Citation corrections and scope wording (idempotent)."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def edit(path, subs):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        print("[skip]", path); return
    s = open(p, encoding="utf-8").read(); o = s
    for old, new in subs:
        s = s.replace(old, new)
    if s != o:
        open(p, "w", encoding="utf-8").write(s); print("[edited]", path)
    else:
        print("[no change]", path)
edit("README.md", [
    ("Angrick et al. (arXiv:2107.02018)", "Chimani & Stutzenstein (arXiv:2107.02018)"),
    ("Angrick et al. (2021)", "Chimani & Stutzenstein (arXiv:2107.02018)"),
    ("at every hour of the day,", "at every hour of the day (and, through a per-edge interpolation-robust variant, at every time of day),"),
])
edit("benchmarks/literature_benchmarks.py", [("Angrick et al. 2021 style", "Chimani-Stutzenstein style"), ("Angrick et al. (arXiv:2107.02018)", "Chimani & Stutzenstein (arXiv:2107.02018)")])
edit("docs/REFEREE_AUDIT.md", [
    ("| **open**: run `tools/validate_congestion_shape.py`", "| **open (commands in docs/FINAL_RUNBOOK.md)**: run `tools/validate_congestion_shape.py`"),
])
