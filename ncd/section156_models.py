"""Section156 model primitives; trajectory integration is required before use."""
import math
import time
import pickle

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier


class HistGBModule(torch.nn.Module):
    """CPU estimator with torch inference interface; checkpoint via extra state.

    Only trusted locally produced checkpoints may be loaded: extra state holds
    the fitted sklearn estimator, not merely tensors.
    """
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.estimator = None
        self.constant_class = None

    def set_seen(self, columns):
        if any(c < 0 or c >= self.outputs for c in columns):
            raise ValueError("Invalid seen HistGB class")

    def get_extra_state(self):
        return dict(outputs=self.outputs, estimator_pickle=pickle.dumps(self.estimator),
                    constant_class=self.constant_class)

    def set_extra_state(self, state):
        if state["outputs"] != self.outputs:
            raise ValueError("HistGB checkpoint output mismatch")
        self.estimator = pickle.loads(state["estimator_pickle"])
        self.constant_class = state["constant_class"]

    def forward(self, x):
        if self.constant_class is not None:
            scores = torch.full((len(x), self.outputs), -1e6, device=x.device)
            scores[:, self.constant_class] = 0.
            return scores
        if self.estimator is None:
            raise RuntimeError("HistGB estimator is not fitted")
        raw = self.estimator.decision_function(x.detach().cpu().numpy())
        classes = self.estimator.classes_.astype(int)
        if np.ndim(raw) == 1:
            raw = np.stack((-raw / 2, raw / 2), axis=1)
        scores = np.full((len(x), self.outputs), -1e6, dtype=np.float32)
        scores[:, classes] = raw
        return torch.as_tensor(scores, device=x.device)


class ReplayGBDTGate(HistGBModule):
    """One joint classifier: internal column zero is benign."""
    def joint_logits(self, x):
        return super().forward(x)

    def forward(self, x):
        logits = self.joint_logits(x)
        margin = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
        return torch.stack((-margin / 2, margin / 2), dim=1)


class ReplayGBDTFamily(torch.nn.Module):
    """Attack view of the same estimator, not a second trained classifier."""
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def set_seen(self, columns):
        self.joint.set_seen([0, *[c + 1 for c in columns]])

    def forward(self, x):
        return self.joint.joint_logits(x)[:, 1:]


class ReplayGBDTGlobal(torch.nn.Module):
    """Original single-model argmax view in dataset class order."""
    def __init__(self, joint, normal, attacks):
        super().__init__()
        self.joint = joint
        self.order = [([normal, *attacks]).index(c) for c in range(len(attacks) + 1)]

    def forward(self, x):
        return self.joint.joint_logits(x)[:, self.order]


def train_gate(model, x, y, *, kind, config, seed, device, training_ids,
               stopping_ids, calibration_ids, stopping_x, stopping_y):
    """Stateless gate dispatch; caller creates a fresh model at every task."""
    from ncd.section123_es_training import assert_disjoint_roles, train_es
    from ncd.crr_benchmarks import _class_balanced_cross_entropy
    from ncd.train_cil import seed_everything
    assert_disjoint_roles(training_ids, stopping_ids, calibration_ids)
    started = time.perf_counter()
    if kind == "histgb":
        model.estimator = fit_histgb(x, y, stopping_x=stopping_x,
            stopping_y=stopping_y, seed=seed, training_ids=training_ids,
            stopping_ids=stopping_ids, calibration_ids=calibration_ids)
        stats = dict(iterations=int(model.estimator.n_iter_), backend="histgb")
    elif kind in ("embedding", "ensemble5", "partial_auc"):
        members = model.members if kind == "ensemble5" else [model]
        histories = []
        for k, member in enumerate(members):
            member_seed = seed + 7919 * k
            seed_everything(member_seed)
            # Reset each independent network after setting its own init seed.
            for module in member.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            def objective(m, bx, by):
                logits = m(bx)
                ce = _class_balanced_cross_entropy(logits, by)
                return ce + partial_auc_penalty(logits, by) if kind == "partial_auc" else ce
            histories.append(train_es(member, x, y, stopping_x, stopping_y,
                objective=objective, logits=lambda m, bx: m(bx),
                seen_classes=(0, 1), seed=member_seed, device=device,
                config=config, smoke=bool(config.get("smoke", False))))
        stats = dict(members=histories, backend=kind)
    else:
        raise NotImplementedError(f"Gate backend not integrated: {kind}")
    elapsed = time.perf_counter() - started
    stats.update(elapsed_seconds=elapsed, gate_retention=False)
    return elapsed, stats


def build_models(configuration, dataset, feature_names, task0_training_x, attack_classes):
    """Factorial gate variants with f1/f2; other trajectories fail explicitly."""
    from ncd.section138_models import Classifier, Family
    from ncd.section81_tabtransformer import fit_feature_spec
    if configuration.special in ("C1", "C2"):
        gate = ReplayGBDTGate(attack_classes + 1)
        return gate, ReplayGBDTFamily(gate), None
    if configuration.special or configuration.family == 3:
        raise NotImplementedError("f3 and special trajectories are not integrated yet")
    spec = fit_feature_spec(dataset, feature_names, task0_training_x)
    def member():
        return Classifier(len(feature_names), 2, 128, 2, spec=spec)
    kind = configuration.switches()["gate"]
    gate = HistGBModule(2) if kind == "histgb" else (
        EnsembleGate([member() for _ in range(5)]) if kind == "ensemble5" else member())
    family = (HistGBModule(attack_classes) if configuration.family == 2 else
              Family(len(feature_names), attack_classes, 128, 2, spec=spec))
    return gate, family, spec


def train_histgb_family(model, x, y, *, stopping_x, stopping_y, seed,
                        training_ids, stopping_ids, calibration_ids):
    from ncd.section123_es_training import assert_disjoint_roles
    assert_disjoint_roles(training_ids, stopping_ids, calibration_ids)
    started = time.perf_counter()
    classes = np.unique(y)
    model.estimator, model.constant_class = None, None
    if len(classes) == 1:
        if not np.isin(stopping_y, classes).all():
            raise ValueError("Stopping class absent from family training")
        model.constant_class = int(classes[0])
        iterations = 0
    else:
        model.estimator = fit_histgb(x, y, stopping_x=stopping_x,
            stopping_y=stopping_y, seed=seed, training_ids=training_ids,
            stopping_ids=stopping_ids, calibration_ids=calibration_ids)
        iterations = int(model.estimator.n_iter_)
    elapsed = time.perf_counter() - started
    return elapsed, dict(backend="histgb", iterations=iterations,
        constant_class=model.constant_class, elapsed_seconds=elapsed,
        logits_stored=False, weight_alignment=False, cbf=False)


def fit_histgb(x, y, *, stopping_x, stopping_y, seed, training_ids,
               stopping_ids, calibration_ids):
    """Fit only against the explicit registered stopping split (sklearn >=1.7)."""
    training_ids, stopping_ids, calibration_ids = map(np.asarray,
        (training_ids, stopping_ids, calibration_ids))
    for a, b in ((training_ids, stopping_ids), (training_ids, calibration_ids),
                 (stopping_ids, calibration_ids)):
        if np.intersect1d(a, b).size:
            raise ValueError("HistGB data roles overlap")
    if len(x) != len(training_ids) or len(stopping_x) != len(stopping_ids):
        raise ValueError("HistGB row identities do not match arrays")
    if not len(stopping_ids) or len(np.unique(y)) < 2:
        raise ValueError("HistGB needs stopping rows and at least two classes")
    model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1,
        max_leaf_nodes=63, l2_regularization=1.0, class_weight="balanced",
        early_stopping=True, random_state=seed)
    model.fit(x, y, X_val=stopping_x, y_val=stopping_y)
    return model


def partial_auc_penalty(logits, labels):
    """Registered batch pair surrogate; CE balancing belongs to the caller."""
    scores = logits[:, 1] - logits[:, 0]
    benign, attack = scores[labels == 0], scores[labels == 1]
    if not benign.numel() or not attack.numel():
        return scores.sum() * 0.0
    hard = benign.topk(math.ceil(0.05 * benign.numel())).values
    return torch.nn.functional.softplus(hard[:, None] - attack[None, :]).mean()


class EnsembleGate(torch.nn.Module):
    """Five independent gates; average margins, not average probabilities."""
    def __init__(self, members):
        super().__init__()
        if len(members) != 5:
            raise ValueError("The registered ensemble has five members")
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        margins = [m(x).diff(dim=1).squeeze(1) for m in self.members]
        score = torch.stack(margins).mean(0)
        return torch.stack((-score / 2, score / 2), dim=1)


def select_benign_hard(ids, scores, *, stopping_ids, calibration_ids, limit=2000):
    """Select from the actual fresh draw only, with deterministic tie handling."""
    ids, scores = np.asarray(ids), np.asarray(scores)
    if ids.ndim != 1 or scores.shape != ids.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid benign scores")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate fresh benign identities")
    if np.intersect1d(ids, stopping_ids).size or np.intersect1d(ids, calibration_ids).size:
        raise ValueError("Benign-hard pool overlaps held-out roles")
    return ids[np.lexsort((ids, -scores))[:limit]]


def two_key_log_score(gate_attack_probability, family_benign_probability):
    p, q = np.asarray(gate_attack_probability), np.asarray(family_benign_probability)
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Invalid two-key probabilities")
    if ((p < 0) | (p > 1) | (q < 0) | (q > 1)).any():
        raise ValueError("Probabilities outside [0,1]")
    with np.errstate(divide="ignore"):
        return np.log(p) + np.log1p(-q)
