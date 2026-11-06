import unittest

from ncd.export_section448_screening import calibrated_forgetting, fixed_rule


class TestSection448ScreeningExport(unittest.TestCase):
    def test_calibrated_forgetting_matches_signed_class_mean_definition(self):
        result = {
            "tasks": [[0], [1], [2]],
            "checkpoints": [
                {"operating_points": {"0.01": {"per_class": [
                    {"class_id": 0, "recall": .9},
                ]}}},
                {"operating_points": {"0.01": {"per_class": [
                    {"class_id": 0, "recall": .8}, {"class_id": 1, "recall": .7},
                ]}}},
                {"operating_points": {"0.01": {"per_class": [
                    {"class_id": 0, "recall": .85}, {"class_id": 1, "recall": .6},
                    {"class_id": 2, "recall": .5},
                ]}}},
            ],
        }
        self.assertAlmostEqual(calibrated_forgetting(result), .075)

    def test_fixed_rule_requires_complete_two_dataset_evidence_and_caps_at_two(self):
        def row(bacc, forget=0, n=2):
            return {"n_complete": n, "mean_final_bacc_pp": bacc,
                    "mean_forgetting_pp": forget}

        aggregates = {
            "UNSW-NB15": {
                "AR-Stateless": row(65, 20), "V1a": row(64.5, 14),
                "V1b": row(66, 15), "V2": row(64, 10), "V3": row(64, 10),
            },
            "CIC-IDS-2017": {
                "AR-Stateless": row(99), "V1a": row(99), "V1b": row(98.5),
                "V2": row(99, .2, 1), "V3": row(99, .2, 1),
            },
        }
        decisions = fixed_rule(aggregates)
        self.assertEqual(decisions["V1a"], "ADVANCE_1")
        self.assertEqual(decisions["V1b"], "ADVANCE_2")
        self.assertEqual(decisions["V2"], "INCOMPLETE")
        self.assertEqual(decisions["V3"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
