"""Durable stage accounting and native-anchored offline experiment orchestration.

Library only. Callbacks must come from the independently reviewed input loader.
No CLI, network, automatic review, deployment or checkpoint modification.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import numpy as np
from pilot_core import ARMS, ROLES, choose, digest, empirical_accept, explanation_signal, require
from native_r1 import adjust, fit, validate_settings
from native_target import validate_native

STAGES = ("CLAIMED", "CANDIDATES_LOCKED", "FIT_ATTEMPTS_SPENT", "FITS_LOCKED",
          "ACCEPTANCE_SPENT", "DECISIONS_LOCKED", "EVALUATION_SPENT", "COMPLETE")


def no_links(path):
    for p in (Path(path).absolute(), *Path(path).absolute().parents):
        require(not p.is_symlink() and (not p.exists() or
                not getattr(p.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                "linked ledger path forbidden")


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Ledger:
    def __init__(self, path):
        path = Path(path).absolute(); no_links(path)
        require(path.parent.is_dir(), "ledger parent must already exist")
        if path.exists():
            require(path.is_file() and path.stat().st_size > 0, "invalid existing ledger")
        self.db = sqlite3.connect(path, isolation_level=None, timeout=5)
        try:
            self.db.execute("PRAGMA synchronous=FULL")
            with self.transaction():
                tables = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                expected = {"etg_meta", "studies", "claims", "events"}
                require(not tables or tables == expected, "unrecognized ledger database")
                if not tables:
                    self.db.execute("CREATE TABLE etg_meta(schema TEXT NOT NULL)")
                    self.db.execute("INSERT INTO etg_meta VALUES ('etg-early-one-use-v1')")
                    self.db.execute("CREATE TABLE studies(id TEXT PRIMARY KEY, binding TEXT, stage TEXT)")
                    self.db.execute("CREATE TABLE claims(row_id TEXT PRIMARY KEY, study TEXT, role TEXT)")
                    self.db.execute("CREATE TABLE events(study TEXT, stage TEXT, payload TEXT, sha TEXT, PRIMARY KEY(study,stage))")
                require(self.db.execute("SELECT schema FROM etg_meta").fetchall() == [("etg-early-one-use-v1",)], "ledger schema mismatch")
        except BaseException:
            self.db.close(); raise

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def close(self):
        self.db.close()

    def stage(self, study):
        row = self.db.execute("SELECT stage FROM studies WHERE id=?", (study,)).fetchone()
        require(row is not None, "unknown study")
        return row[0]

    def claim(self, study, binding, roles):
        require(isinstance(study, str) and study and valid_sha(binding) and tuple(roles) == ROLES, "claim contract")
        ids = [v for name in ROLES for v in roles[name]]
        require(all(roles[r] for r in ROLES) and all(valid_sha(i) for i in ids)
                and len(ids) == len(set(ids)), "duplicate/invalid row claims")
        with self.transaction():
            self.db.execute("INSERT INTO studies VALUES (?,?,?)", (study, binding, "CLAIMED"))
            self.db.executemany("INSERT INTO claims VALUES (?,?,?)",
                                [(i, study, role) for role in ROLES for i in roles[role]])
            self._event(study, "CLAIMED", {"binding": binding, "roles": roles})

    def _event(self, study, stage, payload):
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.db.execute("INSERT INTO events VALUES (?,?,?,?)",
                        (study, stage, text, hashlib.sha256(text.encode()).hexdigest()))

    def advance(self, study, before, after, payload):
        require(before in STAGES and after in STAGES and STAGES.index(after) == STAGES.index(before)+1,
                "invalid stage transition")
        with self.transaction():
            require(self.stage(study) == before, "stage already spent or out of order")
            self._event(study, after, payload)
            self.db.execute("UPDATE studies SET stage=? WHERE id=?", (after, study))

    def fail(self, study, exception):
        with self.transaction():
            current = self.stage(study)
            require(current not in ("COMPLETE", "FAILED"), "terminal state cannot be replaced")
            self._event(study, "FAILED", {"from": current, "error_type": type(exception).__name__})
            self.db.execute("UPDATE studies SET stage='FAILED' WHERE id=?", (study,))

    def events(self, study):
        return [json.loads(v[0]) for v in self.db.execute("SELECT payload FROM events WHERE study=? ORDER BY rowid", (study,))]


def metrics(labels, predictions, axis):
    y, pred = np.asarray(labels), np.asarray(predictions)
    require(y.ndim == 1 and y.shape == pred.shape and set(y.tolist()) == set(axis)
            and set(pred.tolist()) <= set(axis), "metric population mismatch")
    cm = np.asarray([[np.sum((y == a) & (pred == b)) for b in axis] for a in axis], dtype=np.int64)
    support = cm.sum(1); tp = cm.diagonal()
    recall = tp / support
    precision = np.divide(tp, cm.sum(0), out=np.zeros(len(axis)), where=cm.sum(0) > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(len(axis)), where=(precision+recall)>0)
    attack = y != 0
    return {"confusion_counts": cm.tolist(), "class_axis": axis, "support": support.tolist(),
            "recall": recall.tolist(), "precision": precision.tolist(), "f1": f1.tolist(),
            "accuracy": float(tp.sum()/len(y)), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall.mean()), "attack_recall": float((pred[attack] != 0).mean()),
            "benign_fpr": float((pred[~attack] != 0).mean()),
            "balanced_old_class_error": float(1-recall[:2].mean())}


def validate_stage(data, role_ids, axis):
    native, y, ids = data["native"], np.asarray(data["labels"]), list(data["row_ids"])
    require(ids == role_ids and y.ndim == 1 and len(y) == len(ids), "stage identity/order mismatch")
    validate_native(native, len(y), axis)
    require(set(y.tolist()) == set(axis), "all four class labels required")
    return native, y


def derive_candidates(data, role_ids):
    new, y = validate_stage(data, role_ids, [0, 1, 2, 3])
    old = validate_native(data["old_native"], len(y), [0, 1])
    candidates = []
    for c in [0, 1]:
        mask = y == c
        harm = float(np.mean(new["predicted_class_id"][mask] != c) -
                     np.mean(old["predicted_class_id"][mask] != c))
        record = {"class_id": c, "error_increase": harm, "attribution_validated": False}
        attribution = data.get("attributions", {}).get(c)
        if attribution is not None:
            expected_ids = [i for i, selected in zip(role_ids, mask) if selected]
            require(attribution["row_ids"] == expected_ids, "attribution row alignment changed")
            require(attribution["old"].shape[1] == len(expected_ids), "attribution support mismatch")
            try:
                signal = explanation_signal(attribution["old"], attribution["new"])
            except ValueError:
                record["attribution_unavailable_reason"] = "invalid-or-degenerate-vector"
            else:
                record.update(signal, attribution_validated=True)
        candidates.append(record)
    return candidates


def run(ledger, study, binding, roles, loader, settings, salt, evidence_kind):
    """Single-use pipeline. Loader exposes trigger(), fit(), acceptance(), evaluation().

    This does not turn a caller's evidence_kind or binding string into verified
    provenance. Real materialization, source closure and independent authorization
    belong to the launcher, which is NOT supplied by this library.
    """
    require(evidence_kind in ("synthetic", "real-inputs"), "explicit source kind required")
    # Protect the claimed identities and optimizer recipe from mutable loader state.
    roles = {key: list(value) for key, value in roles.items()}
    settings = dict(settings)
    validate_settings(settings)
    axis = [0, 1, 2, 3]
    ledger.claim(study, binding, roles)
    try:
        candidates = derive_candidates(loader.trigger(), roles["trigger"])
        choices = {arm: choose(candidates, arm, salt) for arm in ARMS}
        ledger.advance(study, "CLAIMED", "CANDIDATES_LOCKED", {"choices": choices, "candidates": candidates,
                       "settings_sha256": digest(settings), "evidence_kind": evidence_kind})
        active = [arm for arm in ARMS if choices[arm]["status"] == "attempt"]
        # Reserve all active attempts before reading FIT labels or doing work.
        ledger.advance(study, "CANDIDATES_LOCKED", "FIT_ATTEMPTS_SPENT", {
            "arms": active, "reserved_steps_per_arm": settings["steps"],
            "reserved_seconds_per_arm": settings["max_seconds"]})
        fitted = {}
        if active:
            native_fit, yfit = validate_stage(loader.fit(), roles["fit"], axis)
            for arm in active:
                fitted[arm] = fit(native_fit, yfit, choices[arm]["target"], settings, "fit")
        ledger.advance(study, "FIT_ATTEMPTS_SPENT", "FITS_LOCKED", {"states": fitted})
        ledger.advance(study, "FITS_LOCKED", "ACCEPTANCE_SPENT", {"attempts": len(active)})
        decisions = {arm: dict(choices[arm], accepted_empirically=False) for arm in ARMS}
        if active:
            native_accept, ya = validate_stage(loader.acceptance(), roles["acceptance"], axis)
            baseline = native_accept["predicted_class_id"]
            for arm in active:
                state = fitted[arm]
                scores = adjust(native_accept, state["target"], state["params"])
                proposed = np.asarray(axis)[scores.argmax(1)]
                decisions[arm].update(empirical_accept(ya, baseline, proposed, state["target"], axis, [1, 2, 3]))
        ledger.advance(study, "ACCEPTANCE_SPENT", "DECISIONS_LOCKED", decisions)
        # The label callback is inaccessible before this persistent spend.
        ledger.advance(study, "DECISIONS_LOCKED", "EVALUATION_SPENT", {"decisions_sha256": digest(decisions)})
        native_eval, ye = validate_stage(loader.evaluation(), roles["evaluation"], axis)
        output = {}
        for arm in ARMS:
            baseline = native_eval["predicted_class_id"]
            proposed = baseline
            if arm in fitted:
                s = fitted[arm]
                proposed = np.asarray(axis)[adjust(native_eval, s["target"], s["params"]).argmax(1)]
            effective = proposed if decisions[arm]["accepted_empirically"] else baseline
            output[arm] = {"decision": decisions[arm], "proposed": metrics(ye, proposed, axis),
                           "effective": metrics(ye, effective, axis),
                           "attempts_spent": int(arm in active),
                           "optimizer_steps": fitted[arm]["steps"] if arm in fitted else 0,
                           "fit_wall_seconds": fitted[arm]["wall_seconds"] if arm in fitted else 0,
                           "fit_label_accesses": len(roles["fit"]) if arm in active else 0,
                           "acceptance_label_accesses": len(roles["acceptance"]) if arm in active else 0}
        available = choices["noise-aware"]["status"] != "unavailable-attribution"
        delta = (output["noise-aware"]["effective"]["balanced_old_class_error"] -
                 output["error-only"]["effective"]["balanced_old_class_error"]) if available else None
        result = {"schema": "etg-early-one-use-output-v1", "evidence_kind": evidence_kind, "arms": output,
                  "noise_aware_minus_error_only_old_error": delta, "rows": len(ye),
                  "primary_contrast_status": "DESCRIPTIVE_AVAILABLE" if available else "UNAVAILABLE_ATTRIBUTION",
                  "statistical_certificate": False, "full_test_population_evaluated": False,
                  "external_execution_review_verified_by_this_library": False,
                  "interpretation": "one descriptive checkpoint pair, not general efficacy or novelty",
                  "upstream_attribution_cost_included": False}
        ledger.advance(study, "EVALUATION_SPENT", "COMPLETE", result)
        return result
    except BaseException as error:
        if ledger.stage(study) not in ("COMPLETE", "FAILED"):
            ledger.fail(study, error)
        raise
