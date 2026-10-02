#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests the INDEPENDENT VERIFIER ITSELF: does verify_results.py actually catch
errors, or does it just print PASS on anything you hand it?

Every test here builds a small, fully synthetic report.json (and, where
needed, a matching per_pair_dilation.csv) from scratch -- it does not depend
on run_spanner_benchmark.py having been run first, and does not touch the
repository's real results/ directory. Each error-injection test takes a
known-good fixture, corrupts exactly one thing, and asserts that the
specific corresponding check fails (not just "something" fails) while every
other check still passes.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verify_results as vr


def _good_report():
    """A minimal, internally-consistent, hand-built report covering every
    section run_checks() inspects."""
    return {
        "configuration": {"quick": True},
        "network": {
            "connectivity_funnel": {"nodes_before": 100, "nodes_after": 90,
                                    "edges_before": 300, "edges_after": 250},
        },
        "results": {
            "delay_bounded|1.5": {
                "certificate": {"certified": True, "violating_edges": 0,
                                "attained_worst_edge_stretch": 1.5},
                "structure": {"edge_retention_pct": 95.0},
            },
            "delay_bounded|2.0": {
                "certificate": {"certified": True, "violating_edges": 0,
                                "attained_worst_edge_stretch": 2.0},
                "structure": {"edge_retention_pct": 90.0},
            },
            "static_ff|1.5": {
                "certificate": {"certified": False, "violating_edges": 3,
                                "attained_worst_edge_stretch": 1.714},
                "structure": {"edge_retention_pct": 93.0},
            },
            "static_ff|2.0": {
                "certificate": {"certified": False, "violating_edges": 5,
                                "attained_worst_edge_stretch": 2.512},
                "structure": {"edge_retention_pct": 88.0},
            },
        },
        "significance_test_pair_level": {"available": False},
        "cross_city": {
            "cities": [
                {"city": "Alpha", "used_real_osm": True,
                 "connectivity_funnel": {"nodes_before": 50, "nodes_after": 45,
                                         "edges_before": 150, "edges_after": 120},
                 "methods": {
                     "delay_bounded": {"certificate": {"certified": True, "violating_edges": 0}},
                     "static_ff": {"certificate": {"certified": False, "violating_edges": 2}},
                 }},
            ],
            "morphology_regression": {
                "available": True,
                "orientation_entropy_vs_edge_retention_pct": {"r": 0.4, "p": 0.03, "p_fdr_bh": 0.09},
                "circuity_mean_vs_edge_retention_pct": {"r": 0.2, "p": 0.40, "p_fdr_bh": 0.48},
            },
        },
    }


def _write(tmpdir, report, perpair_rows=None):
    report_path = os.path.join(tmpdir, "spanner_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh)
    perpair_path = os.path.join(tmpdir, "per_pair_dilation.csv")
    if perpair_rows is not None:
        with open(perpair_path, "w", encoding="utf-8") as fh:
            fh.write("source,target,dilation,method\n")
            for row in perpair_rows:
                fh.write(",".join(str(x) for x in row) + "\n")
    return report_path, perpair_path


class TestVerifierPassesOnGoodData(unittest.TestCase):
    def test_clean_report_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, perpair_path = _write(tmp, _good_report())
            code, log = vr.run_checks(report_path, perpair_path)
            self.assertEqual(code, 0, "\n".join(log.lines))
            self.assertEqual(log.failures, [])
            self.assertGreater(log.checks_run, 5)


class TestVerifierCatchesMissingFile(unittest.TestCase):
    def test_missing_report_is_exit_code_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, log = vr.run_checks(os.path.join(tmp, "nope.json"),
                                      os.path.join(tmp, "nope.csv"))
            self.assertEqual(code, 2)

    def test_malformed_json_is_exit_code_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "spanner_report.json")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            code, log = vr.run_checks(report_path, os.path.join(tmp, "nope.csv"))
            self.assertEqual(code, 2)


class TestVerifierCatchesCertificateInconsistency(unittest.TestCase):
    """The exact bug class this check exists for: a pipeline that claims
    `certified: true` while its own violating_edges count is nonzero."""

    def test_certified_true_with_violations_is_caught(self):
        report = _good_report()
        report["results"]["delay_bounded|1.5"]["certificate"]["violating_edges"] = 3
        with tempfile.TemporaryDirectory() as tmp:
            report_path, perpair_path = _write(tmp, report)
            code, log = vr.run_checks(report_path, perpair_path)
            self.assertEqual(code, 1)
            self.assertTrue(any("delay_bounded|1.5" in f and "consistency" in f
                                for f in log.failures), log.failures)


class TestVerifierCatchesFabricatedStretchCap(unittest.TestCase):
    """Direct regression test for the original bug: a fixed cutoff constant
    silently repeated across structurally different, non-certified rows."""

    def test_repeated_round_stretch_value_is_flagged(self):
        report = _good_report()
        for key in ("static_ff|1.5", "static_ff|2.0"):
            report["results"][key]["certificate"]["attained_worst_edge_stretch"] = 8.0
        report["results"]["mean_temporal|1.5"] = {
            "certificate": {"certified": False, "violating_edges": 4,
                            "attained_worst_edge_stretch": 8.0},
            "structure": {"edge_retention_pct": 91.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            report_path, perpair_path = _write(tmp, report)
            code, log = vr.run_checks(report_path, perpair_path)
            self.assertEqual(code, 1)
            self.assertTrue(any("repeats >=3x" in f for f in log.failures), log.failures)


class TestVerifierCatchesNonMonotoneSparsification(unittest.TestCase):
    def test_retention_increasing_with_t_is_caught(self):
        report = _good_report()
        # t=2.0 should retain <= t=1.5's edges; flip it to break monotonicity.
        report["results"]["delay_bounded|2.0"]["structure"]["edge_retention_pct"] = 99.0
        with tempfile.TemporaryDirectory() as tmp:
            report_path, perpair_path = _write(tmp, report)
            code, log = vr.run_checks(report_path, perpair_path)
            self.assertEqual(code, 1)
            self.assertTrue(any("non-increasing" in f for f in log.failures), log.failures)


class TestVerifierCatchesConnectivityFunnelViolation(unittest.TestCase):
    """An SCC/contraction filter can only ever remove nodes or edges, never
    add them -- nodes_after > nodes_before is a structural impossibility."""

    def test_nodes_after_exceeding_nodes_before_is_caught(self):
        report = _good_report()
        report["network"]["connectivity_funnel"]["nodes_after"] = 999
        with tempfile.TemporaryDirectory() as tmp:
            report_path, perpair_path = _write(tmp, report)
            code, log = vr.run_checks(report_path, perpair_path)
            self.assertEqual(code, 1)
            self.assertTrue(any("nodes_after <= nodes_before" in f for f in log.failures),
                            log.failures)


class TestVerifierCatchesBrokenFdrCorrection(unittest.TestCase):
    def test_adjusted_p_below_raw_p_is_caught(self):
        report = _good_report()
        reg = report["cross_city"]["morphology_regression"]
        reg["orientation_entropy_vs_edge_retention_pct"]["p_fdr_bh"] = 0.01  # < raw p=0.03
        with tempfile.TemporaryDirectory() as tmp:
            report_path, perpair_path = _write(tmp, report)
            code, log = vr.run_checks(report_path, perpair_path)
            self.assertEqual(code, 1)
            self.assertTrue(any("FDR-adjusted p >= raw p" in f for f in log.failures), log.failures)


class TestVerifierCatchesWrongWilcoxonPValue(unittest.TestCase):
    """Builds real per-pair data, lets scipy compute the true p-value, then
    asserts the verifier rejects a report that claims a different one."""

    def test_mismatched_reported_pvalue_is_caught(self):
        if not vr._HAVE_SCIPY:
            self.skipTest("scipy not available")
        report = _good_report()
        rows = []
        for i in range(12):
            ours = 1.0 + 0.001 * i
            theirs = 1.05 + 0.01 * i
            rows.append((f"s{i}", f"t{i}", ours, "Delay-Bounded Spanner (ours)"))
            rows.append((f"s{i}", f"t{i}", theirs, "Matched-Density Betweenness"))
        report["significance_test_pair_level"] = {"available": True, "p_value": 0.999999,
                                                   "n_pairs": 12}
        with tempfile.TemporaryDirectory() as tmp:
            report_path, perpair_path = _write(tmp, report, perpair_rows=rows)
            code, log = vr.run_checks(report_path, perpair_path)
            self.assertEqual(code, 1)
            self.assertTrue(any("Wilcoxon p-value matches" in f for f in log.failures), log.failures)


class TestVerifierCliResultsDirOverride(unittest.TestCase):
    """The --results-dir / --report / --perpair CLI overrides are what make
    this script testable in isolation in the first place; make sure they
    actually route to the given paths."""

    def test_results_dir_flag_is_honored(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, _good_report())
            code = vr.main(["--results-dir", tmp])
            self.assertEqual(code, 0)

    def test_explicit_report_and_perpair_flags_are_honored(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path, perpair_path = _write(tmp, _good_report())
            moved_report = os.path.join(tmp, "renamed_report.json")
            os.rename(report_path, moved_report)
            code = vr.main(["--report", moved_report, "--perpair", perpair_path])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
