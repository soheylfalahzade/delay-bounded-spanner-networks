"""README.md result tables (sections 5b-5d) must equal what the raw files in results/benchmarks/ produce."""
import importlib.util
import os
import shutil
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load():
    spec = importlib.util.spec_from_file_location("render_readme_tables", os.path.join(ROOT, "benchmarks", "render_readme_tables.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestReadmeTables(unittest.TestCase):
    def test_readme_matches_raw_data(self):
        mod = load()
        text = open(mod.README, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        self.assertEqual(mod.render(text), text)

    def test_edited_readme_number_is_detected(self):
        mod = load()
        text = open(mod.README, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        tampered = text.replace("| 1.5 | 96.2 | 96.6 | 0.4 |", "| 1.5 | 96.2 | 99.9 | 0.4 |", 1)
        self.assertNotEqual(tampered, text)
        self.assertNotEqual(mod.render(tampered), tampered)

    def test_changed_certification_result_stops_the_renderer(self):
        mod = load()
        tmp = tempfile.mkdtemp()
        try:
            shutil.copytree(mod.BEN, os.path.join(tmp, "b"))
            path = os.path.join(tmp, "b", "metrla_table.csv")
            text = open(path, encoding="utf-8").read().replace("DelayBounded,", "DelayBounded,", 1)
            lines = text.splitlines()
            i = next(k for k, ln in enumerate(lines) if ",DelayBounded," in ln)
            lines[i] = lines[i].replace(",True,", ",False,", 1)
            open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
            mod.BEN = os.path.join(tmp, "b")
            with self.assertRaises(AssertionError):
                mod.table_literature()
        finally:
            shutil.rmtree(tmp)

    def test_every_block_has_markers(self):
        mod = load()
        text = open(mod.README, encoding="utf-8").read()
        for name in mod.RENDERERS:
            self.assertIn(f"<!-- BEGIN:{name} -->", text)
            self.assertIn(f"<!-- END:{name} -->", text)


if __name__ == "__main__":
    unittest.main()
