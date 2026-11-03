"""Banner, pipeline diagram and animation: assets stay in step with the data and refuse unsupported claims."""
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import make_visuals as mv  # noqa: E402


class TestVisuals(unittest.TestCase):
    def test_committed_svg_assets_match_the_generator(self):
        for name, fn in (("banner.svg", mv.banner_svg), ("pipeline.svg", mv.pipeline_svg)):
            with open(os.path.join(ROOT, "docs", "assets", name), encoding="utf-8", newline="") as f:
                self.assertEqual(f.read(), fn(), f"{name} is stale: run python benchmarks/make_visuals.py")

    def test_banner_numbers_are_the_computed_ones(self):
        m = mv.headline_metrics()
        svg = mv.banner_svg()
        for key in ("cities", "gap", "optimal"):
            self.assertIn(m[key], svg)

    def test_banner_refuses_to_overstate_certification(self):
        tmp = tempfile.mkdtemp()
        try:
            for f in os.listdir(mv.RES):
                if f.endswith(".csv"):
                    shutil.copy(os.path.join(mv.RES, f), tmp)
            d = pd.read_csv(os.path.join(tmp, "large_scale_3500m.csv"))
            d.loc[0, "delay_bounded_certified"] = False
            d.to_csv(os.path.join(tmp, "large_scale_3500m.csv"), index=False)
            old = mv.RES
            mv.RES = tmp
            try:
                with self.assertRaises(AssertionError):
                    mv.banner_svg()
            finally:
                mv.RES = old
        finally:
            shutil.rmtree(tmp)

    def test_animation_replays_the_library_decisions(self):
        tmp = tempfile.mkdtemp()
        try:
            out = os.path.join(tmp, "a.gif")
            mv.animation(out)
            self.assertGreater(os.path.getsize(out), 20000)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
