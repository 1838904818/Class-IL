import unittest
from pathlib import Path

from ncd.run_section448_confirm import (
    ARMS, DATASETS, DATASET_DIRS, SEEDS, THREADS, planned_cells, result_directory,
)


class TestSection448Confirm(unittest.TestCase):
    def test_fixed_confirmation_scope_is_45_cells(self):
        self.assertEqual(DATASETS, ("UNSW-NB15", "CIC-IDS-2017", "CSE-CIC-IDS2018"))
        self.assertEqual(ARMS, ("AR-Stateless", "V1a", "V1b"))
        self.assertEqual(SEEDS, (15, 16, 17, 18, 19))
        self.assertEqual(THREADS, 2)
        self.assertEqual(sum(len(planned_cells(dataset)) for dataset in DATASETS), 45)
        self.assertEqual(len(set((dataset, arm, seed)
                                 for dataset in DATASETS
                                 for arm, seed in planned_cells(dataset))), 45)

    def test_dataset_cell_order_and_output_isolation(self):
        self.assertEqual(planned_cells("UNSW-NB15")[:6], (
            ("AR-Stateless", 15), ("AR-Stateless", 16), ("AR-Stateless", 17),
            ("AR-Stateless", 18), ("AR-Stateless", 19), ("V1a", 15),
        ))
        root = Path("root")
        self.assertEqual(result_directory(root, "CSE-CIC-IDS2018", "V1b", 19),
                         root / "confirm" / DATASET_DIRS["CSE-CIC-IDS2018"] / "V1b" / "seed19")
        with self.assertRaises(ValueError):
            result_directory(root, "CIC-IDS-2017", "V2", 19)
        with self.assertRaises(ValueError):
            planned_cells("NSL-KDD")


if __name__ == "__main__":
    unittest.main()
