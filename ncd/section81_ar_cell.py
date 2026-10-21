"""B1 fixed-budget AR-TabT adapter over the unchanged AR cell implementation.

Factories are replaced only in a private function-global dictionary. Neither
the imported runner nor any on-disk dependency is mutated, including while an
ER cell is active. The original AR losses, optimizer, replay and WA are reused.
"""
from __future__ import annotations

import types

import numpy as np

from ncd import run_aacil as aa
from ncd import run_aacil_compact as original
from ncd.run_problem_scout import save_json, section26_pools
from ncd.section81_tabtransformer import TabularClassifier, TabularFamily, fit_feature_spec


class ARFamily(TabularFamily):
    def metadata(self):
        result = super().metadata()
        count = self.assert_capacity()
        result.update(training_parameters=count, inference_parameters=count,
                      optimizer_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))
        return result


def run_ar_cell(data, config, identity, output, device):
    """Only registered fixed200 AR-TabT; percent1 needs independent accounting."""
    if (identity.get("method") != "ar_tabt" or identity.get("section") != 81
            or identity.get("memory_setting") != "fixed200"
            or identity.get("effective_config") != config):
        raise ValueError("B1 AR requires its own bound fixed-budget identity")
    if (config["lambda_g"] != 0 or config["derpp_alpha"] != 1
            or config["derpp_beta"] != 1 or config["selection_only"]):
        raise ValueError("AR means no gate KD, DER alpha=beta=1, and no selection")
    if not config["smoke"] and (config["epochs_per_task"] != 8
            or config["min_steps_per_task"] != 2000 or config["batch_size"] != 256
            or config["learning_rate"] != 0.001 or config["normal_training_rows"] != 50000):
        raise ValueError("Registered AR schedule changed")
    train, valid = section26_pools(data, identity["seed"])
    normal = data.class_order.index("Normal")
    rows = [aa.select_rows(train[normal], int(config["normal_training_rows"]),
                          identity["seed"] * 10000 * 10007 + normal)]
    rows.extend(train[c] for c in data.tasks[0] if c != normal)
    ids = np.concatenate(rows)
    if np.intersect1d(ids, valid[normal]).size:
        raise ValueError("Task-0 vocabulary accessed permanent calibration")
    spec = fit_feature_spec(identity["dataset"], data.feature_names, data.x_train[ids])
    if identity.get("feature_spec") != spec:
        raise ValueError("Task-0 training vocabulary must be bound before execution")

    def gate_factory(_input_features, _hidden, outputs):
        return TabularClassifier(spec, outputs)

    def family_factory(_input_features, _hidden, outputs, kind, **kwargs):
        if kind != "aacil_kd_wa":
            raise ValueError("Unexpected AR family dispatch")
        return ARFamily(spec, outputs)

    namespace = dict(original.run_cell.__globals__)
    namespace.update(MLP=gate_factory, CompactFamily=family_factory,
                     NOGKD_WA="ar_tabt", command=lambda: identity["command"])
    runner = types.FunctionType(original.run_cell.__code__, namespace,
                                "section81_fixed_budget_ar", original.run_cell.__defaults__)
    result = runner(data, config, identity, output, device)
    result.update(section=81, feature_spec=spec,
                  selection_rule="Section81 registered fixed-budget AR-TabT; no tuning",
                  limitations=[original.ADAPTATION,
                    "Same registered TabTransformer backbone in independent G and F networks.",
                    "Original AR loss and schedule, not ER optimization or compute matching.",
                    "Bounded RQ2 comparison; no RQ1/RQ3 or substantive-innovation completion."])
    save_json(output / "result.json", result)
    return result
