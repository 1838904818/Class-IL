"""Durable, single-stage CPU score-level repair study. No remote operations."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sqlite3
import sys
import time

import numpy as np

from score_core import (ARMS, Blocked, candidate_pool, canonical, check, classification_report,
                        digest, evaluate_pair, finite_matrix, fit_r1, fused)


ROLES = ("trigger", "fit", "acceptance", "subsequent")


def file_hash(path):
    reject_links(path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    return digest({"source": {name: file_hash(Path(__file__).with_name(name)) for name in ("runner.py", "score_core.py")},
                   "python": sys.version, "numpy": np.__version__, "arithmetic": "float64"})


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            check(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def number(value):
        parsed = float(value)
        check(math.isfinite(parsed), "nonfinite or overflowed JSON number")
        return parsed
    def constant(_):
        raise Blocked("nonstandard JSON numeric constant")
    reject_links(path)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=constant)


def reject_links(path):
    absolute = Path(path).absolute()
    for component in (absolute, *absolute.parents):
        check(not component.is_symlink(), "symlink input/output forbidden")
        if component.exists():
            attributes = getattr(os.lstat(component), "st_file_attributes", 0)
            check(not attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024), "Windows reparse point forbidden")


def save_json(path, value):
    """Atomic output creation; never overwrite an existing research artifact."""
    path = Path(path)
    reject_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def bound_path(root, relative):
    relative = Path(relative)
    check(not relative.is_absolute() and not relative.drive and ".." not in relative.parts and
          all(":" not in p for p in relative.parts), "non-relative or traversal bundle path")
    root = Path(root).absolute()
    reject_links(root)
    path = root / relative
    reject_links(path)
    root, path = root.resolve(), path.resolve()
    check(path.is_relative_to(root) and path != root, "bundle path escapes its registered root")
    return path


def validate_manifest(manifest):
    check(manifest["schema"] == "score-repair-bundle-v1", "unknown bundle schema")
    check(manifest["source_kind"] in ("synthetic-software-test", "score-bundle-pending-fidelity", "externally-verified-score-bundle"), "unknown score source")
    classes, old = manifest["classes"], manifest["old_classes"]
    check(len(classes) >= 3 and len(old) >= 2 and len(set(classes)) == len(classes) and len(set(old)) == len(old)
          and set(old) < set(classes) and all(isinstance(c, str) and c for c in classes + old), "explicit old and new class sets required")
    check(type(manifest["increment"]) is int and manifest["increment"] >= 0, "integer increment required")
    check(set(manifest["partitions"]) == set(ROLES), "four partitions required")
    row_seen, group_seen = set(), set()
    for role in ROLES:
        metadata = manifest["partitions"][role]
        check(isinstance(metadata["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]), "partition file SHA-256 required")
        ids, groups = metadata["row_ids"], metadata["group_ids"]
        check(bool(ids) and len(ids) == len(groups) and len(set(ids)) == len(ids), "invalid partition IDs")
        check(all(isinstance(x, str) and x for x in ids + groups), "nonempty opaque row/group identities required")
        check(not row_seen.intersection(ids) and not group_seen.intersection(groups), "cross-partition row/group leakage")
        row_seen.update(ids)
        group_seen.update(groups)
        observed, available = metadata["observed_increment"], metadata["labels_available_increment"]
        check(type(observed) is int and observed >= 0, "integer observation increment required")
        if role == "subsequent":
            check(observed > manifest["increment"] and available is None, "subsequent labels must remain sealed")
        else:
            check(type(available) is int and 0 <= observed <= available <= manifest["increment"], "future/missing fitting or trigger labels")
    check("labels" not in manifest and isinstance(manifest["sealed_labels_sha256"], str) and
          re.fullmatch(r"[0-9a-f]{64}", manifest["sealed_labels_sha256"]), "sealed evaluation-label digest required")


def source_binding(manifest):
    value = {key: manifest[key] for key in ("schema", "increment", "classes", "old_classes", "partitions", "sealed_labels_sha256")}
    value["native_export_receipts"] = manifest.get("native_export_receipts")
    return digest(value)


def verify_real_evidence(root, manifest, policy):
    if manifest["source_kind"] == "synthetic-software-test":
        return
    check(manifest["source_kind"] == "externally-verified-score-bundle", "real score bundle is pending external numerical/governance verification")
    requirements = [("score_fidelity_evidence", "score-fidelity-evidence-v1", "PASSED"),
                    ("data_governance_evidence", "data-governance-evidence-v1", "APPROVED")]
    if manifest.get("attribution_fidelity_evidence") is not None:
        requirements.append(("attribution_fidelity_evidence", "attribution-fidelity-evidence-v1", "PASSED"))
    for key, schema, status in requirements:
        reference = manifest.get(key)
        check(isinstance(reference, dict) and set(reference) == {"file", "sha256"}, "external evidence must reference a real bound file")
        path = bound_path(root, reference["file"])
        check(file_hash(path) == reference["sha256"], "external evidence file hash mismatch")
        evidence = read_json(path)
        check(evidence.get("schema") == schema and evidence.get("status") == status and
              evidence.get("source_binding_sha256") == source_binding(manifest), "external evidence schema/status/source binding mismatch")
        report_ref = evidence.get("report")
        check(isinstance(report_ref, dict) and set(report_ref) == {"file", "sha256"}, "underlying external review report must be file-bound")
        report_path = bound_path(root, report_ref["file"])
        check(file_hash(report_path) == report_ref["sha256"], "underlying review report hash mismatch")
        report = read_json(report_path)
        check(report.get("schema") == "external-research-review-report-v1" and report.get("status") == status and
              report.get("source_binding_sha256") == source_binding(manifest) and report.get("review_id"),
              "review report verdict or scope is not bound")
        if key == "score_fidelity_evidence":
            check(report.get("fusion_sha256") == digest(policy["fusion"]) and report.get("numerical_fidelity_verified") is True and
                  report.get("training_lineage_verified") is True,
                  "registered numerical target/fusion verification is missing")
        elif key == "data_governance_evidence":
            check(report.get("semantics_sha256") == digest(policy["semantics"]) and
                  report.get("decision_partition_source") == "increment-available-training" and
                  report.get("official_test_used_for_decisions") is False, "label governance or no-test-fitting evidence missing")
        else:
            check(report.get("method") == policy["trigger"]["explanation_method"] and report.get("faithful_target_verified") is True,
                  "method-specific attribution verification is missing")


def validate_policy(policy, manifest):
    check(policy["schema"] == "score-repair-policy-v1" and policy["scientific_status"] == "prospective-not-validated", "prospective policy required")
    check(tuple(policy["arms"]) == ARMS, "freeze all six controls and registered ablations before running")
    required = {
        "trigger": {"min_rows_per_class", "min_error_increase", "min_disagreement", "min_drift_excess", "state_priority_weight",
                    "periodic_every", "periodic_phase", "random_seed", "explanation_method"},
        "fusion": {"router_weight", "epsilon", "identity_atol"},
        "r1": {"scale_min", "scale_max", "offset_min", "offset_max", "steps", "learning_rate", "identity_l2", "max_fit_seconds"},
        "acceptance": {"min_rows_per_class", "min_target_improvement", "max_old_error_increase", "max_new_error_increase",
                       "max_missed_attack_increase", "max_fpr_increase"},
        "uncertainty": {"min_groups", "trigger_alpha", "acceptance_alpha", "subsequent_alpha", "trigger_family_size",
                        "acceptance_family_size", "subsequent_family_size"},
        "budget": {"unique_labels_per_arm", "max_repairs_per_arm", "fit_steps_per_arm", "acceptance_score_rows_per_arm",
                   "diagnostic_cells_per_arm"},
        "semantics": {"kind", "attack_classes"},
    }
    for key, expected in required.items():
        check(set(policy[key]) == expected, f"all {key} fields must be explicit; unknown fields forbidden")
        for value in policy[key].values():
            if isinstance(value, (float, int)):
                check(not isinstance(value, bool) and np.isfinite(value), "finite policy values required")
    r1, t, a, ci, b, fusion = (policy[x] for x in ("r1", "trigger", "acceptance", "uncertainty", "budget", "fusion"))
    check(all(type(x) is int for x in (t["min_rows_per_class"], t["random_seed"], a["min_rows_per_class"], ci["min_groups"],
                                      ci["trigger_family_size"], ci["acceptance_family_size"], ci["subsequent_family_size"])),
          "support/seed/family counts must be integers")
    check(0 < r1["scale_min"] <= 1 <= r1["scale_max"] and r1["offset_min"] <= 0 <= r1["offset_max"], "R1 bounds must include identity and preserve monotonicity")
    check(isinstance(r1["steps"], int) and r1["steps"] > 0 and r1["learning_rate"] > 0 and r1["identity_l2"] >= 0 and r1["max_fit_seconds"] > 0, "invalid optimizer recipe")
    check(fusion["router_weight"] >= 0 and fusion["epsilon"] > 0 and fusion["identity_atol"] > 0, "invalid frozen fusion")
    check(isinstance(t["periodic_every"], int) and t["periodic_every"] > 0 and isinstance(t["periodic_phase"], int) and
          0 <= t["periodic_phase"] < t["periodic_every"] and t["random_seed"] >= 0, "invalid trigger schedule")
    check(t["min_rows_per_class"] > 0 and a["min_rows_per_class"] > 0 and ci["min_groups"] >= 2, "explicit positive support minima required")
    check(all(t[k] >= 0 for k in ("min_error_increase", "min_disagreement", "min_drift_excess", "state_priority_weight")), "negative trigger threshold")
    check(all(0 <= a[k] <= 1 for k in ("min_target_improvement", "max_old_error_increase", "max_new_error_increase")), "invalid error tolerances")
    check(all(isinstance(x, int) and x >= 0 for x in b.values()), "budgets must be nonnegative integer counts")
    check(b["max_repairs_per_arm"] == 1 and b["fit_steps_per_arm"] >= r1["steps"], "one fixed repair attempt required")
    semantics = policy["semantics"]
    check(semantics["kind"] in ("application", "attack-binary"), "authorized label semantics required")
    if semantics["kind"] == "application":
        check(not semantics["attack_classes"] and a["max_missed_attack_increase"] is None and a["max_fpr_increase"] is None,
              "application labels cannot produce attack recall/FPR")
    else:
        check(bool(semantics["attack_classes"]) and set(semantics["attack_classes"]) < set(manifest["classes"]), "explicit attack and benign classes required")
        check(all(isinstance(a[k], (int, float)) and 0 <= a[k] <= 1 for k in ("max_missed_attack_increase", "max_fpr_increase")), "binary safety tolerances required")
    metric_count = len(manifest["classes"]) + 3 + (2 if semantics["attack_classes"] else 0)
    check(ci["trigger_family_size"] >= 4 * len(manifest["old_classes"]) * len(ARMS), "trigger family must cover all arms/candidates")
    check(ci["acceptance_family_size"] >= metric_count * len(ARMS) and
          ci["subsequent_family_size"] >= 2 * metric_count * len(ARMS),
          "CI family must cover all registered arm/metric contrasts, including proposed/effective evaluation")
    check(all(0 < ci[f"{stage}_alpha"] < 1 for stage in ("trigger", "acceptance", "subsequent")), "invalid alpha")


def load_partition(root, manifest, role):
    metadata = manifest["partitions"][role]
    path = bound_path(root, metadata["file"])
    check(file_hash(path) == metadata["sha256"], f"{role} file hash mismatch")
    part = read_json(path)
    check(part["role"] == role and part["row_ids"] == metadata["row_ids"] and part["group_ids"] == metadata["group_ids"], "partition metadata mismatch")
    n = len(part["row_ids"])
    finite_matrix(part["new_head_logits"], (n, len(manifest["classes"])))
    finite_matrix(part["new_router_raw"], (n, len(manifest["classes"])))
    if role == "subsequent":
        check("labels" not in part, "sealed evaluation labels appeared in score partition")
    else:
        check(len(part["labels"]) == n and set(part["labels"]) == set(manifest["classes"]), "complete authorized class support required")
    return part


class Ledger:
    """SQLite durable single-spend ledger; all arms share one immutable study.

    Do not delete/reset the ledger to reuse an acceptance set. A crash leaves a
    spent attempt; there is intentionally no resume-until-success command.
    """
    def __init__(self, path):
        path = Path(path)
        reject_links(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS studies(id TEXT PRIMARY KEY, manifest TEXT NOT NULL, policy TEXT NOT NULL,
          manifest_sha TEXT NOT NULL, policy_sha TEXT NOT NULL, code_sha TEXT NOT NULL, stage TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS claims(kind TEXT NOT NULL, identity TEXT NOT NULL, study TEXT NOT NULL,
          role TEXT NOT NULL, PRIMARY KEY(kind,identity), FOREIGN KEY(study) REFERENCES studies(id));
        CREATE TABLE IF NOT EXISTS arms(study TEXT NOT NULL, arm TEXT NOT NULL, stage TEXT NOT NULL,
          pool TEXT, pool_sha TEXT, reservation TEXT, fit TEXT, decision TEXT, progress TEXT,
          PRIMARY KEY(study,arm), FOREIGN KEY(study) REFERENCES studies(id));
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, study TEXT NOT NULL,
          arm TEXT, kind TEXT NOT NULL, payload TEXT NOT NULL, timestamp REAL NOT NULL);
        """)

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

    def event(self, study, arm, kind, payload):
        self.db.execute("INSERT INTO events(study,arm,kind,payload,timestamp) VALUES(?,?,?,?,?)",
                        (study, arm, kind, canonical(payload).decode(), time.time()))

    def register(self, study, manifest, policy):
        check(isinstance(study, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", study), "safe opaque study ID required")
        validate_manifest(manifest)
        validate_policy(policy, manifest)
        with self.transaction():
            check(self.db.execute("SELECT 1 FROM studies WHERE id=?", (study,)).fetchone() is None, "study already spent; no retries")
            for role, p in manifest["partitions"].items():
                for kind, values in (("row", p["row_ids"]), ("group", set(p["group_ids"]))):
                    for value in values:
                        check(self.db.execute("SELECT 1 FROM claims WHERE kind=? AND identity=?", (kind, value)).fetchone() is None,
                              "cross-increment row/group reuse forbidden by durable ledger")
            self.db.execute("INSERT INTO studies VALUES(?,?,?,?,?,?,?)", (study, canonical(manifest).decode(), canonical(policy).decode(),
                            digest(manifest), digest(policy), code_hash(), "REGISTERED"))
            for role, p in manifest["partitions"].items():
                for kind, values in (("row", p["row_ids"]), ("group", set(p["group_ids"]))):
                    self.db.executemany("INSERT INTO claims VALUES(?,?,?,?)", [(kind, value, study, role) for value in values])
            for arm in policy["arms"]:
                self.db.execute("INSERT INTO arms(study,arm,stage) VALUES(?,?,?)", (study, arm, "REGISTERED"))
            self.event(study, None, "REGISTERED_NO_EVALUATION_LABELS_READ", {"manifest": digest(manifest), "policy": digest(policy), "code": code_hash()})

    def study(self, study):
        record = self.db.execute("SELECT * FROM studies WHERE id=?", (study,)).fetchone()
        check(record is not None, "unknown study")
        return dict(record)

    def arm(self, study, arm):
        record = self.db.execute("SELECT * FROM arms WHERE study=? AND arm=?", (study, arm)).fetchone()
        check(record is not None, "unregistered arm")
        return dict(record)

    def save_pool(self, study, arm, pool):
        with self.transaction():
            check(self.arm(study, arm)["stage"] == "REGISTERED", "candidate pool already frozen")
            self.db.execute("UPDATE arms SET pool=?,pool_sha=?,stage='DIAGNOSED' WHERE study=? AND arm=?",
                            (canonical(pool).decode(), digest(pool), study, arm))
            self.event(study, arm, "CANDIDATE_POOL_FROZEN", {"pool_sha256": digest(pool)})

    def reserve(self, study, arm, pool, manifest, policy, bundle_root):
        verify_runtime_binding(self, study, manifest, policy)
        verify_real_evidence(bundle_root, manifest, policy)
        check(arm != "audit-only" and pool["ordered_targets"], "no eligible active target")
        trigger = load_partition(bundle_root, manifest, "trigger")
        recomputed = candidate_pool(trigger, manifest, policy, arm)
        check(pool["arm"] == arm and pool["candidates"] == recomputed["candidates"] and
              pool["ordered_targets"] == recomputed["ordered_targets"], "candidate-pool eligibility/ordering cannot be bypassed")
        b = policy["budget"]
        total_labels = sum(len(manifest["partitions"][role]["row_ids"]) for role in ("trigger", "fit", "acceptance"))
        acceptance_rows = 2 * len(manifest["partitions"]["acceptance"]["row_ids"])
        check(total_labels <= b["unique_labels_per_arm"] and acceptance_rows <= b["acceptance_score_rows_per_arm"] and
              pool["diagnostic_cells_processed"] + recomputed["diagnostic_cells_processed"] <= b["diagnostic_cells_per_arm"],
              "fixed intervention or diagnosis budget exceeded")
        reservation = {"arm": arm, "target": pool["ordered_targets"][0], "pool_sha256": digest(pool),
                       "manifest_sha256": digest(manifest), "policy_sha256": digest(policy),
                       "trigger_policy_sha256": digest(policy["trigger"]), "acceptance_policy_sha256": digest(policy["acceptance"]),
                       "r1_sha256": digest(policy["r1"]), "unique_labels_reserved": total_labels,
                       "fit_steps_reserved": policy["r1"]["steps"], "acceptance_score_rows_reserved": acceptance_rows,
                       "diagnostic_cells_spent": pool["diagnostic_cells_processed"] + recomputed["diagnostic_cells_processed"],
                       "diagnosis_wall_seconds_spent": pool["diagnosis_wall_seconds"] + recomputed["diagnosis_wall_seconds"],
                       "attempts_reserved": 1, "refund_permitted": False}
        with self.transaction():
            row = self.arm(study, arm)
            check(row["stage"] == "DIAGNOSED" and row["pool_sha"] == digest(pool), "changed pool or repeated reservation")
            self.db.execute("UPDATE arms SET stage='FITTING', reservation=? WHERE study=? AND arm=?",
                            (canonical(reservation).decode(), study, arm))
            self.event(study, arm, "PACKAGE_SPENT_BEFORE_FIT", reservation)
        return reservation

    def progress(self, study, arm, steps, wall, cpu):
        value = {"steps_completed": steps, "fit_wall_seconds": wall, "fit_cpu_seconds": cpu}
        self.db.execute("UPDATE arms SET progress=? WHERE study=? AND arm=?", (canonical(value).decode(), study, arm))

    def fitted(self, study, arm, state):
        with self.transaction():
            check(self.arm(study, arm)["stage"] == "FITTING", "fit state cannot be replaced")
            self.db.execute("UPDATE arms SET stage='ACCEPTING', fit=? WHERE study=? AND arm=?", (canonical(state).decode(), study, arm))
            self.event(study, arm, "FIT_STATE_FROZEN_BEFORE_ACCEPTANCE", {"state_sha256": digest(state)})

    def final(self, study, arm, decision):
        with self.transaction():
            check(self.arm(study, arm)["stage"] in ("REGISTERED", "DIAGNOSED", "FITTING", "ACCEPTING"), "decision already locked")
            self.db.execute("UPDATE arms SET stage='FINAL', decision=? WHERE study=? AND arm=?", (canonical(decision).decode(), study, arm))
            self.event(study, arm, "DECISION_LOCKED", decision)

    def lock(self, study):
        with self.transaction():
            check(all(r[0] == "FINAL" for r in self.db.execute("SELECT stage FROM arms WHERE study=?", (study,))), "all arms must finish before evaluation")
            self.db.execute("UPDATE studies SET stage='DECISIONS_LOCKED' WHERE id=?", (study,))
            self.event(study, None, "ALL_DECISIONS_LOCKED", {"evaluation_labels_read": False})


def verify_runtime_binding(ledger, study, manifest, policy):
    current = ledger.study(study)
    check(current["manifest_sha"] == digest(manifest) and current["policy_sha"] == digest(policy) and current["code_sha"] == code_hash(),
          "immutable manifest/policy/code binding changed")


def run_study(manifest_path, policy_path, ledger_path, output, study):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest, policy = read_json(manifest_path), read_json(policy_path)
    validate_manifest(manifest)
    validate_policy(policy, manifest)
    verify_real_evidence(manifest_path.parent, manifest, policy)
    ledger = Ledger(ledger_path)
    registered_here = False
    try:
        ledger.register(study, manifest, policy)  # Freeze policy and all row/group claims BEFORE loading labels.
        registered_here = True
        trigger = load_partition(manifest_path.parent, manifest, "trigger")
        if manifest["source_kind"] != "synthetic-software-test" and trigger.get("explanations"):
            check(manifest.get("attribution_fidelity_evidence") is not None, "real attribution arrays need separately verified method evidence")
        pools = {}
        # Every full candidate pool is frozen before any fitting/acceptance.
        for arm in policy["arms"]:
            pool = candidate_pool(trigger, manifest, policy, arm)
            if pool["diagnostic_cells_processed"] > policy["budget"]["diagnostic_cells_per_arm"]:
                pool["ordered_targets"] = []
                pool["budget_blocked"] = True
            pools[arm] = pool
            ledger.save_pool(study, arm, pool)
        decisions = {}
        for arm in policy["arms"]:
            pool = pools[arm]
            try:
                if arm == "audit-only" or not pool["ordered_targets"]:
                    decision = {"decision": "AUDIT_ONLY" if arm == "audit-only" else "ABSTAIN", "effective_state": "baseline",
                                "repairs_attempted": 0, "unused_repair_steps": policy["r1"]["steps"], "pool_sha256": digest(pool),
                                "acquired_trigger_labels": len(trigger["labels"]), "diagnosis": pool}
                else:
                    reservation = ledger.reserve(study, arm, pool, manifest, policy, manifest_path.parent)
                    fit = load_partition(manifest_path.parent, manifest, "fit")
                    check(all(fit["labels"].count(c) >= policy["acceptance"]["min_rows_per_class"] for c in manifest["classes"]), "insufficient fit support")
                    state = fit_r1(fit, manifest["classes"], reservation["target"], policy["fusion"], policy["r1"],
                                   lambda steps, wall, cpu: ledger.progress(study, arm, steps, wall, cpu))
                    state.update(fit_partition_sha256=manifest["partitions"]["fit"]["sha256"], reservation_sha256=digest(reservation),
                                 baseline_definition_sha256=digest({"scores": manifest["partitions"], "fusion": policy["fusion"]}))
                    ledger.fitted(study, arm, state)
                    acceptance = load_partition(manifest_path.parent, manifest, "acceptance")
                    check(all(acceptance["labels"].count(c) >= policy["acceptance"]["min_rows_per_class"] for c in manifest["classes"]), "insufficient acceptance support")
                    verify_runtime_binding(ledger, study, manifest, policy)
                    verify_real_evidence(manifest_path.parent, manifest, policy)
                    check(ledger.arm(study, arm)["pool_sha"] == digest(pool), "frozen candidate pool changed")
                    paired = evaluate_pair(acceptance, manifest["classes"], manifest["old_classes"], policy["fusion"], state, policy, "acceptance")
                    score_path = output / f"{study}.{arm}.acceptance.json"
                    save_json(score_path, paired)
                    decision = {"decision": paired["decision"], "effective_state": paired["effective_state"], "repairs_attempted": 1,
                                "state": state, "failed_constraints": paired["failed_constraints"], "reservation": reservation,
                                "acceptance_output_sha256": file_hash(score_path), "diagnosis": pool,
                                "raw_alerts_preserved": True, "subsequent_labels_read": False}
                ledger.final(study, arm, decision)
                decisions[arm] = decision
            except Exception as exc:
                row = ledger.arm(study, arm)
                decision = {"decision": "FAILED_ROLLBACK", "effective_state": "baseline", "error_type": type(exc).__name__,
                            "reason": str(exc), "reserved_budget_retained": row["reservation"] is not None,
                            "last_durable_progress": json.loads(row["progress"]) if row["progress"] else None,
                            "raw_alerts_preserved": True, "subsequent_labels_read": False}
                ledger.final(study, arm, decision)
                decisions[arm] = decision
        ledger.lock(study)
        report = {"study": study, "stage": "DECISIONS_LOCKED", "source_kind": manifest["source_kind"], "code_sha256": code_hash(),
                  "manifest_sha256": digest(manifest), "policy_sha256": digest(policy), "decisions": decisions,
                  "real_data_utility_established": False, "total_compute_matched": False, "audit_only_active_compute_matched": False}
        save_json(output / f"{study}.decisions.json", report)
        return report
    except Exception as exc:
        if registered_here:
            with ledger.transaction():
                ledger.db.execute("UPDATE studies SET stage='FAILED_CLOSED' WHERE id=?", (study,))
                ledger.event(study, None, "STUDY_FAILED_CLOSED_CLAIMS_REMAIN_SPENT", {"error_type": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        ledger.close()


def evaluate_study(manifest_path, policy_path, labels_path, ledger_path, output, study):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest, policy = read_json(manifest_path), read_json(policy_path)
    ledger = Ledger(ledger_path)
    evaluation_started = False
    try:
        # Do not open or hash labels_path until all decisions are durably locked.
        verify_runtime_binding(ledger, study, manifest, policy)
        verify_real_evidence(manifest_path.parent, manifest, policy)
        with ledger.transaction():
            check(ledger.study(study)["stage"] == "DECISIONS_LOCKED", "evaluation labels sealed until all arm decisions lock; no repeated evaluation")
            ledger.db.execute("UPDATE studies SET stage='EVALUATING' WHERE id=?", (study,))
            ledger.event(study, None, "EVALUATION_SPENT_BEFORE_LABEL_READ", {})
        evaluation_started = True
        check(file_hash(labels_path) == manifest["sealed_labels_sha256"], "sealed label hash mismatch")
        labels = read_json(labels_path)
        subsequent = load_partition(manifest_path.parent, manifest, "subsequent")
        check(labels["row_ids"] == subsequent["row_ids"] and labels["group_ids"] == subsequent["group_ids"] and
              labels["observed_increment"] == manifest["partitions"]["subsequent"]["observed_increment"] and
              len(labels["labels"]) == len(subsequent["row_ids"]) and set(labels["labels"]) == set(manifest["classes"]), "sealed evaluation support mismatch")
        subsequent["labels"] = labels["labels"]
        reports = {}
        for arm in policy["arms"]:
            stored = ledger.arm(study, arm)
            decision = json.loads(stored["decision"])
            state = json.loads(stored["fit"]) if stored["fit"] else {"target": manifest["old_classes"][0], "scale": 1.0, "offset": 0.0}
            paired = evaluate_pair(subsequent, manifest["classes"], manifest["old_classes"], policy["fusion"], state, policy, "subsequent")
            paired["effective_state"] = decision["effective_state"]
            paired["effective_report"] = paired["proposed"] if decision["effective_state"] == "proposed" else paired["baseline"]
            effective = paired if decision["effective_state"] == "proposed" else evaluate_pair(
                subsequent, manifest["classes"], manifest["old_classes"], policy["fusion"],
                {"target": manifest["old_classes"][0], "scale": 1.0, "offset": 0.0}, policy, "subsequent")
            paired["paired_group_intervals_for_effective"] = effective["paired_group_intervals"]
            paired["decision_was_locked"] = True
            paired["proposed_was_accepted"] = decision["effective_state"] == "proposed"
            save_json(output / f"{study}.{arm}.subsequent.json", paired)
            reports[arm] = {"decision": decision["decision"], "effective_report": paired["effective_report"],
                            "proposed_report_is_counterfactual_if_rejected": paired["proposed"],
                            "paired_group_intervals_for_proposed": paired["paired_group_intervals"],
                            "paired_group_intervals_for_effective": effective["paired_group_intervals"],
                            "source_kind": manifest["source_kind"]}
        report = {"study": study, "stage": "EVALUATED_ONCE", "source_kind": manifest["source_kind"], "arms": reports,
                  "real_data_utility_established": False, "subsequent_data_used_for_selection": False,
                  "shared_evaluation_labels_used_after_lock": len(labels["labels"])}
        save_json(output / f"{study}.evaluation.json", report)
        with ledger.transaction():
            ledger.db.execute("UPDATE studies SET stage='EVALUATED_ONCE' WHERE id=?", (study,))
            ledger.event(study, None, "EVALUATION_LOCKED", {"report_sha256": digest(report)})
        return report
    except Exception as exc:
        if evaluation_started:
            with ledger.transaction():
                ledger.db.execute("UPDATE studies SET stage='EVALUATION_FAILED_SPENT' WHERE id=?", (study,))
                ledger.event(study, None, "EVALUATION_FAILED_SPENT", {"error_type": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("decide", "evaluate"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--labels")
    args = parser.parse_args()
    if args.stage == "decide":
        check(args.labels is None, "decide must not receive a labels file")
        result = run_study(args.manifest, args.policy, args.ledger, args.output, args.study)
    else:
        check(args.labels is not None, "explicit sealed-label file required after decision lock")
        result = evaluate_study(args.manifest, args.policy, args.labels, args.ledger, args.output, args.study)
    print(json.dumps({"stage": result["stage"], "study": result["study"], "source_kind": result["source_kind"]}))


if __name__ == "__main__":
    main()
