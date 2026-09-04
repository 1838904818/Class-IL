import copy
import unittest

from formal_v2_explanation_etg.attribution_scope import ATTRIBUTION_TARGET_SCOPE
from formal_v2_explanation_etg.verify_multiseed_bindings import validate_binding_shape


def binding():
    return {
        "schema_version": "ofra_attribution_multiseed_bindings_v2",
        "dataset": "malaya-network-gt",
        "analysis_seeds": [1, 2, 3, 4, 42],
        "computed_seeds": [1, 2, 3, 4, 42],
        "analysis_contract": {
            "score_target": "joint_cap3000 class margin",
            "explainer_primary": "shap.GradientExplainer Expected Gradients",
            "robustness_methods": [
                "expected_gradients", "feature_ablation", "gradient_x_input"
            ],
            "nsamples": 64,
            "primary_top_k": 15,
            "primary_jaccard_threshold": 0.7,
            "primary_allowed_recall_drop": 0.05,
            "etg_scope": "offline_post_hoc_non_suppressing_ledger_only",
            "attribution_target_scope": ATTRIBUTION_TARGET_SCOPE,
            "cross_device_equivalence_claimed": False,
            "batch_partition_equivalence_claimed": False,
            "selection_policy": "fixed protocol; no test-driven model or threshold selection",
        },
        "seeds": [
            {"seed": 1, "upstream_job_id": 388991},
            {"seed": 2, "upstream_job_id": 394503},
            {"seed": 3, "upstream_job_id": 394646},
            {"seed": 4, "upstream_job_id": 394745},
            {"seed": 42, "upstream_job_id": 412039},
        ],
    }


class BindingShapeTests(unittest.TestCase):
    def test_exact_registry_passes(self):
        validate_binding_shape(binding())

    def test_missing_seed_fails(self):
        value = binding()
        value["computed_seeds"] = [1, 2, 3, 4]
        with self.assertRaisesRegex(RuntimeError, "computed seed registry"):
            validate_binding_shape(value)

    def test_wrong_upstream_job_fails(self):
        value = copy.deepcopy(binding())
        value["seeds"][0]["upstream_job_id"] = 1
        with self.assertRaisesRegex(RuntimeError, "upstream training job"):
            validate_binding_shape(value)

    def test_equivalence_claim_fails(self):
        value = copy.deepcopy(binding())
        value["analysis_contract"]["cross_device_equivalence_claimed"] = True
        with self.assertRaisesRegex(RuntimeError, "equivalence claim"):
            validate_binding_shape(value)


if __name__ == "__main__":
    unittest.main()
