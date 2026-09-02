from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ANALYZER = HERE / "analyze_guarded_paired5.py"


def assert_numeric_tree(test: unittest.TestCase, expected, actual, path="root"):
    if isinstance(expected, dict):
        test.assertIsInstance(actual, dict, path)
        for key, value in expected.items():
            if key in {"input_files", "analysis_role", "limitations", "schema_version"}:
                continue
            test.assertIn(key, actual, f"{path}.{key}")
            assert_numeric_tree(test, value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        test.assertEqual(len(expected), len(actual), path)
        for index, value in enumerate(expected):
            assert_numeric_tree(test, value, actual[index], f"{path}[{index}]")
    elif isinstance(expected, float):
        if math.isnan(expected):
            test.assertTrue(math.isnan(actual), path)
        else:
            test.assertTrue(math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-12), path)
    else:
        test.assertEqual(expected, actual, path)


class ReleaseRecomputeTest(unittest.TestCase):
    def test_public_package_recomputes_registered_statistics(self):
        registered = json.loads(
            (PACKAGE / "job426307_independent_analysis.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "analysis.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    "--baseline-root",
                    str(PACKAGE / "baseline"),
                    "--candidate-root",
                    str(PACKAGE / "per_seed"),
                    "--output",
                    str(output),
                ],
                check=True,
            )
            recomputed = json.loads(output.read_text(encoding="utf-8"))
        # The public independent-analysis artifact is the direct output of this
        # analyzer. Recompute it from the packaged seed records and compare all
        # numerical and categorical claims, excluding only provenance paths.
        assert_numeric_tree(self, registered, recomputed)


if __name__ == "__main__":
    unittest.main()
