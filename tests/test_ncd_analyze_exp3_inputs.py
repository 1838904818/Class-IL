from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ncd.analyze_exp3 import resolve_training_path, state_inputs


class NcdAnalyzeExp3InputsTest(unittest.TestCase):
    def test_relocated_training_and_state_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_repo = Path(directory)
            run_dir = run_repo / "ncd" / "results" / "exp3" / "runs" / "one"
            run_dir.mkdir(parents=True)
            training = run_dir / "training_result.json"
            explanation = run_dir / "explanation_result.json"
            training.write_text("{}\n", encoding="utf-8")
            explanation.write_text("{}\n", encoding="utf-8")
            state_path = run_repo / "ncd" / "results" / "exp3" / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "expected_runs": 1,
                        "completed_runs": [
                            {"explanation_result": str(explanation.relative_to(run_repo))}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(state_inputs(state_path, run_repo), [str(explanation.resolve())])
            relocated = resolve_training_path(
                "ncd/results/original/training_result.json", explanation
            )
            self.assertEqual(relocated, training.resolve())


if __name__ == "__main__":
    unittest.main()
