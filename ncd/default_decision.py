"""Section118 evaluation primitives; no calibration or training operations.

Callers must restore the original inference model (including its saved head
semantics), bind the checkpoint and TEST split, and supply global class IDs.
Calibrated confusion matrices are deliberately not an input to this module.
"""
from __future__ import annotations

import numpy as np
import torch


def evaluate_models(x_test, y_test, seen, names, model, *, family=None,
                    attack_ids=None, batch_size=4096):
    """Bounded CPU TEST-only forward using already restored inference models.

    Feature arrays may be memory mapped. Models are switched to evaluation
    mode; no optimizer, TRAIN arrays, calibration pool or thresholds are used.
    Architecture restoration and source binding remain the caller's duty.
    """
    if batch_size <= 0 or len(x_test) != len(y_test):
        raise ValueError("Invalid TEST arrays or batch size")
    models = [model] if family is None else [model, family]
    for component in models:
        if any(p.device.type != "cpu" for p in component.parameters()):
            raise ValueError("Default-decision worker requires CPU models")
        component.eval()
    rows = np.flatnonzero(np.isin(y_test, seen))
    predicted = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            ids = rows[start:start+batch_size]
            x = torch.from_numpy(np.array(x_test[ids], dtype=np.float32, copy=True))
            values = model(x)
            if family is None:
                labels = predict_single(values, seen)
            else:
                if attack_ids is None:
                    raise ValueError("Gate-family evaluation requires saved attack mapping")
                labels = predict_gate_family(values, family(x), attack_ids, seen,
                                             names.index("Normal"))
            predicted.append(labels.cpu().numpy())
    if not predicted:
        raise ValueError("No seen-class TEST rows")
    return metrics_from_predictions(np.asarray(y_test)[rows],
                                    np.concatenate(predicted), seen, names)


def predict_single(logits, seen):
    """Argmax in the explicitly ordered seen-class columns of a global head."""
    if logits.ndim != 2 or not seen or len(set(seen)) != len(seen):
        raise ValueError("Invalid single-head shape or seen-class mapping")
    if min(seen) < 0 or max(seen) >= logits.shape[1]:
        raise ValueError("Seen class outside global output head")
    values = logits[:, seen]
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite seen-class logits")
    return torch.as_tensor(seen, device=logits.device)[values.argmax(dim=1)]


def predict_gate_family(gate_logits, family_logits, attack_ids, seen, normal):
    """Strict P(attack)>0.5; F columns follow the saved attack_ids mapping."""
    if (gate_logits.ndim != 2 or gate_logits.shape[1] != 2
            or family_logits.ndim != 2
            or len(gate_logits) != len(family_logits)
            or len(attack_ids) != family_logits.shape[1]
            or len(set(attack_ids)) != len(attack_ids)
            or normal in attack_ids or normal not in seen):
        raise ValueError("Invalid gate/family shape or class mapping")
    attacks = [c for c in seen if c != normal]
    if not attacks or any(c not in attack_ids for c in attacks):
        raise ValueError("Seen attacks missing from family head")
    columns = [attack_ids.index(c) for c in attacks]
    values = family_logits[:, columns]
    if not torch.isfinite(gate_logits).all() or not torch.isfinite(values).all():
        raise ValueError("Nonfinite inference logits")
    family = torch.as_tensor(attacks, device=values.device)[values.argmax(dim=1)]
    # Use softmax explicitly to preserve the specified strict probability rule.
    alert = gate_logits.softmax(dim=1)[:, 1] > 0.5
    return torch.where(alert, family, torch.full_like(family, normal))


def metrics_from_predictions(truth, predicted, seen, names):
    """Compute default-decision metrics from predictions, not fixed-FPR CMs."""
    y, pred = np.asarray(truth), np.asarray(predicted)
    if (y.ndim != 1 or y.shape != pred.shape or not len(y)
            or y.dtype.kind not in "iu" or pred.dtype.kind not in "iu"
            or len(set(seen)) != len(seen) or not seen
            or min(seen) < 0 or max(seen) >= len(names)
            or not np.isin(y, seen).all() or not np.isin(pred, seen).all()):
        raise ValueError("Invalid TEST predictions or seen-class scope")
    n, normal = len(names), names.index("Normal")
    if normal not in seen:
        raise ValueError("Normal must be seen")
    cm = np.bincount(y * n + pred, minlength=n*n).reshape(n, n)
    support, count, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    if any(support[c] == 0 for c in seen):
        raise ValueError("Seen class absent from TEST")
    f1 = {c: float(2*tp[c]/(support[c]+count[c])) for c in seen}
    attacks = [c for c in seen if c != normal]
    return {
        "confusion_matrix": cm.tolist(), "seen_classes": list(seen),
        "accuracy": float(tp.sum()/support.sum()),
        "macro_f1": float(np.mean(list(f1.values()))),
        "weighted_f1": float(sum(support[c]*f1[c] for c in seen)/support.sum()),
        "attack_detection": float(1-cm[attacks, normal].sum()/support[attacks].sum()) if attacks else None,
        "normal_fpr": float(1-tp[normal]/support[normal]),
        "per_class_recall": {str(c): float(tp[c]/support[c]) for c in seen},
        "evaluated_test_rows": int(support.sum()),
    }


def summarize(stages, tasks):
    """Class-mean BWT: final recall minus recall at class introduction.

    All old classes, including Normal, have equal weight. Signed forgetting
    uses the best preceding recall minus final recall, without clipping.
    """
    flat = [c for task in tasks for c in task]
    if not tasks or any(not task for task in tasks) or len(set(flat)) != len(flat):
        raise ValueError("Tasks must partition class introductions")
    if len(stages) != len(tasks):
        raise ValueError("Incomplete default-decision trajectory")
    for i, stage in enumerate(stages):
        if stage["seen_classes"] != [c for task in tasks[:i+1] for c in task]:
            raise ValueError("Stage order or class scope mismatch")
    final = stages[-1]["per_class_recall"]
    rows = []
    for i, task in enumerate(tasks[:-1]):
        for c in task:
            key = str(c)
            initial = stages[i]["per_class_recall"][key]
            best = max(s["per_class_recall"][key] for s in stages[i:-1])
            rows.append({"class_id": c, "introduction_task": i,
                         "bwt": final[key]-initial,
                         "forgetting": best-final[key]})
    return {"acc": float(np.mean([s["accuracy"] for s in stages])),
            "bwt_per_class_recall": float(np.mean([r["bwt"] for r in rows])) if rows else None,
            "forgetting_all_old_classes": float(np.mean([r["forgetting"] for r in rows])) if rows else None,
            "final_macro_f1": stages[-1]["macro_f1"], "per_old_class": rows}
