"""Global configuration constants for the delay-bounded spanner pipeline."""
from __future__ import annotations

import os

# --- Study area -----------------------------------------------------------------
CENTER_POINT = (31.8974, 54.3569)     # Yazd city centre (lat, lon)
DEFAULT_DIST = 1500                   # metres (bounding radius)
NETWORK_TYPE = "drive"

# --- Temporal model -------------------------------------------------------------
HOURS = list(range(24))               # tau in {0, ..., 23}
MORNING_PEAK = 8.0
EVENING_PEAK = 17.5
MORNING_SIGMA = 1.35
EVENING_SIGMA = 1.75

# --- Experiment grid ------------------------------------------------------------
T_VALUES = [1.2, 1.4, 1.6, 1.8]
T_FOCUS = 1.4                         # t used for the time-of-day and map panels
N_SOURCES = 16
N_TARGETS_PER_SOURCE = 12
SEED = 42

# --- Numerics -------------------------------------------------------------------
EPS = 1e-9
DILATION_CAP = 25.0                   # plotting cap for unreachable pairs

# --- Paths ----------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
CACHE_DIR = os.path.join(ROOT, "cache")
FIGURE_PATH = os.path.join(RESULTS_DIR, "spanner_benchmark.png")
REPORT_PATH = os.path.join(RESULTS_DIR, "spanner_report.json")
TABLE_PATH = os.path.join(RESULTS_DIR, "benchmark_table.md")
README_PATH = os.path.join(ROOT, "README.md")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
