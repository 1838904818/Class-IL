from __future__ import annotations

import json
import unittest
from pathlib import Path

from ncd.generate_exp2_validation_configs import CELLS, P2_FRACTIONS, RHO_VALUES, SEEDS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "ncd" / "configs"


class Exp2ValidationConfigTest(unittest.TestCase):
    def test_frozen_config_matrix_is_complete_and_unique(self) -> None:
        outputs = set()
        for dataset_short, dataset_dir, method, _ in CELLS:
            for seed in SEEDS:
                path = CONFIG_DIR / f"exp2_validation_v1_{dataset_short}_{method}_seed{seed}.json"
                self.assertTrue(path.is_file(), path)
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["seed"], seed)
                self.assertEqual(config["rho_values"], list(RHO_VALUES))
                self.assertEqual(
                    config["output_dir"],
                    f"ncd/results/exp2_validation_v1/{dataset_dir}/{method}/seed{seed}",
                )
                self.assertNotIn(config["output_dir"], outputs)
                outputs.add(config["output_dir"])
                expected_p2 = list(P2_FRACTIONS) if method == "replay" else []
                self.assertEqual(config["p2_mask_fractions"], expected_p2)
        self.assertEqual(len(outputs), 20)

if __name__ == "__main__":
    unittest.main()
