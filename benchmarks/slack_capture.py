#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How much of the removable slack does the construction actually remove?

The forced-edge bound F(t) says at most 100 - F(t) percent of the edges can be removed by ANY edge-subset
t-spanner (Theorem 2).  When F(t) is close to 100, retention is close to F(t) trivially: keeping every edge is
already within 100 - F(t) points of the bound.  The informative quantity is therefore not the gap to the bound
but the share of the removable slack that the construction removes:

    capture(t) = (100 - retention(t)) / (100 - F(t))          in [0, 1]

capture = 1 means the construction removes every edge that the bound allows to be removed (hence is optimal on
that instance); capture = 0 means it removes nothing.  The slack is an upper bound on what any algorithm can
remove (the removable edges need not all be removable together), so capture is a LOWER bound on the share of the
optimal removal that the construction achieves, whenever the bound is the binding limit.

Reads existing result files; nothing is recomputed.

    python benchmarks/slack_capture.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "results", "benchmarks")


def capture(retention_pct: pd.Series, forced_pct: pd.Series) -> pd.Series:
    slack = 100.0 - forced_pct
    removed = 100.0 - retention_pct
    return (removed / slack.where(slack > 1e-9)).clip(lower=0.0, upper=1.0)


def main() -> None:
    parts = []
    lt = pd.read_csv(os.path.join(B, "retention_large_t.csv"))
    lt = lt.rename(columns={"forced_pct": "forced_lower_bound_pct"})
    lt["dataset"] = "12 OSM city cores (1.5 km)"
    parts.append(lt[["dataset", "city", "t", "retention_pct", "forced_lower_bound_pct"]])
    ls = pd.read_csv(os.path.join(B, "large_scale_3500m.csv"))
    ls = ls.rename(columns={"delay_bounded_retention_pct": "retention_pct"})
    ls["dataset"] = "12 OSM cities (3.5 km)"
    parts.append(ls[["dataset", "city", "t", "retention_pct", "forced_lower_bound_pct"]])
    df = pd.concat(parts, ignore_index=True)
    df["removable_slack_pct"] = 100.0 - df.forced_lower_bound_pct
    df["removed_pct"] = 100.0 - df.retention_pct
    df["capture"] = capture(df.retention_pct, df.forced_lower_bound_pct)
    assert (df.removed_pct <= df.removable_slack_pct + 1e-6).all(), "removed more than the bound allows: bug"
    df.round(4).to_csv(os.path.join(B, "slack_capture.csv"), index=False)

    summ = (df.groupby(["dataset", "t"])
              .agg(cities=("city", "nunique"), removable_slack_pct=("removable_slack_pct", "mean"),
                   removed_pct=("removed_pct", "mean"), capture_mean=("capture", "mean"),
                   capture_min=("capture", "min"), capture_max=("capture", "max"))
              .reset_index())
    summ["capture_of_means"] = summ.removed_pct / summ.removable_slack_pct
    with open(os.path.join(B, "slack_capture.md"), "w", encoding="utf-8") as f:
        f.write(summ.round(3).to_markdown(index=False))
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
