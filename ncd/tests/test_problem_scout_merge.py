"""Portable publication checks; no training or data preparation."""
import csv
import importlib.util
import io
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("merge", Path(__file__).resolve().parents[1] / "merge_problem_scout.py")
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


class PortableMergeTests(unittest.TestCase):
    def test_fixed_first_valid_selection(self):
        selected = merge.selected_packages(Path("local"), Path("hpc"))
        self.assertEqual(len(selected), 11)
        self.assertEqual(sum(remote for _, _, remote, _, _ in selected), 7)
        names = {x[0] for x in selected}
        self.assertNotIn("hpc_cic18_seed42", names)
        self.assertNotIn("hpc_cic17_seed4", names)
        self.assertNotIn("hpc_cic17_seed42", names)
        self.assertIn("local_tabular", names)

    def test_redaction_retains_lines_and_urls(self):
        original = "C:/Users/example/work/file.py\nC:\\Users\\example\\work\\file.py\n/home/user/example/code\n/scr/user/example/data\n../../scr/user/example/data\naccount=example_user\nhttps://doi.org/10.1/example\n"
        with patch.dict(os.environ, {"NCD_PUBLICATION_REDACT_USERS": "example_user"}):
            clean = merge.redact_log(original)
            merge.scan_portable(clean)
        self.assertEqual(original.count("\n"), clean.count("\n"))
        self.assertIn("https://doi.org/10.1/example", clean)
        self.assertNotIn("example_user", clean)

    def test_personal_paths_fail_closed(self):
        for path in ("C:/Users/example/file", "C:\\Users\\example\\file", "/home/user/name/file", "/scr/user/name/file"):
            with self.assertRaises(ValueError):
                merge.scan_portable(path)

    def test_relative_path_escape_fails(self):
        for path in ("../data", "a/../../b", "C:/Users/name", "/absolute"):
            with self.assertRaises(ValueError):
                merge.safe_relative(path)

    def test_nonfinite_json_fails(self):
        with self.assertRaises(ValueError):
            merge.json_bytes({"value": float("nan")})

    def test_csv_preserves_null_and_refuses_duplicates(self):
        row = dict(dataset="CIC-IDS-2017", method="joint", seed=1, split="shift", class_id=4, recall=None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            path.write_text(stream.getvalue(), encoding="utf-8")
            merge.check_csv(path, [row])
            path.write_text(stream.getvalue() + stream.getvalue().splitlines(keepends=True)[1], encoding="utf-8")
            with self.assertRaises(ValueError):
                merge.check_csv(path, [row, row])


if __name__ == "__main__":
    unittest.main()
