from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ncd.verify_exp3_evidence import verify


def write_summary(root: Path, records: list[dict[str, str]]) -> Path:
    summary = root / "summary.json"
    summary.write_text(
        json.dumps({"protected_repository_artifacts": {"files": records}}),
        encoding="utf-8",
    )
    return summary


def record(path: str, content: bytes) -> dict[str, str]:
    return {"path": path, "sha256": hashlib.sha256(content).hexdigest()}


class VerifyExp3EvidenceTests(unittest.TestCase):
    def test_verify_accepts_matching_repository_relative_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "results" / "table.csv"
            artifact.parent.mkdir()
            content = b"metric,value\nrate,0.5\n"
            artifact.write_bytes(content)
            summary = write_summary(root, [record("results/table.csv", content)])

            result = verify(summary, root)

            self.assertEqual(result["status"], "VERIFIED_REPOSITORY_SAFE_EXP3_EVIDENCE")
            self.assertEqual(result["verified_files"], 1)

    def test_verify_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "table.csv"
            artifact.write_bytes(b"changed")
            summary = write_summary(root, [record("table.csv", b"original")])

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify(summary, root)

    def test_verify_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_summary(root, [record("../outside.csv", b"unused")])

            with self.assertRaisesRegex(ValueError, "unsafe component"):
                verify(summary, root)

    def test_verify_rejects_internal_host_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "receipt.json"
            content = b'{"path":"/scr/user/example/private/result.json"}'
            artifact.write_bytes(content)
            summary = write_summary(root, [record("receipt.json", content)])

            with self.assertRaisesRegex(ValueError, "forbidden host path"):
                verify(summary, root)


if __name__ == "__main__":
    unittest.main()
