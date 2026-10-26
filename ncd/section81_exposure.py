"""TRAIN-only B1 exposure construction; never fetch discarded old attacks."""
from __future__ import annotations

import math

import numpy as np

from ncd.run_aacil import select_rows
from ncd.run_problem_scout import section26_pools


class ERExposure:
    """Keep the original benign stream, with stable ER stopping-validation roles.

    The union of benign optimization and stopping rows is exactly AR's current
    stream. ER reserves a fixed TRAIN-only 10% subset for stopping; it never
    turns an earlier optimization row into a validation row. This is exposure
    matching, not identical optimization counts or a shared training schedule.
    Permanent Normal calibration is excluded from both roles.
    """

    def __init__(self, data, seed):
        self.data, self.seed = data, int(seed)
        self.normal = data.class_order.index("Normal")
        self.train, self.valid = section26_pools(data, self.seed)
        self.calibration_ids = self.valid[self.normal].copy()
        pool = self.train[self.normal]
        if len(pool) < 2:
            raise ValueError("Not enough non-calibration benign TRAIN rows")
        count = min(len(pool) - 1, max(5, math.ceil(len(pool) * 0.1)))
        self.normal_stopping_ids = select_rows(
            pool, count, self.seed * 100003 + self.normal * 1009 + 7919)
        self.admission_training_counts = {
            c: len(self.train[c]) for c in range(len(data.class_order))}
        self.original_counts = {
            c: int(np.count_nonzero(data.y_train == c))
            for c in range(len(data.class_order))}

    def __call__(self, task_index, *, include_normal=True):
        if not 0 <= task_index < len(self.data.tasks):
            raise ValueError("Invalid B1 task index")
        task = self.data.tasks[task_index]
        earlier = {c for part in self.data.tasks[:task_index] for c in part}
        new = set(task) - {self.normal}
        if new & earlier:
            raise ValueError("B1 task would reread an old attack class")
        task_seed = self.seed * 10000 + task_index
        pairs = {}
        if include_normal:
            normal_ids = select_rows(self.train[self.normal], 50000,
                                     task_seed * 10007 + self.normal)
            stopping = np.isin(normal_ids, self.normal_stopping_ids)
            pairs[self.normal] = (normal_ids[~stopping], normal_ids[stopping])
        pairs.update({c: (self.train[c], self.valid[c]) for c in sorted(new)})
        slots = {}
        for c, (train_ids, validation_ids) in pairs.items():
            slots[c] = {}
            for split, ids in (("train", train_ids), ("validation", validation_ids)):
                if (not np.all(self.data.y_train[ids] == c)
                        or np.intersect1d(ids, self.calibration_ids).size):
                    raise ValueError("B1 exposure label or calibration isolation differs")
                slots[c][split + "_ids"] = ids.copy()
                slots[c][split + "_x"] = self.data.x_train[ids].copy()
        return slots

    def description(self):
        return {
            "benign_stream": "Original AR select_rows, union of optimization and stopping rows",
            "benign_stopping": "Fixed 10% of non-calibration TRAIN, stable across tasks",
            "attack_stopping": "Original section26 TRAIN validation split; old rows only from bounded memory",
            "calibration_used_for_stopping": False,
            "test_used_for_stopping": False,
            "optimization_counts_identical_to_ar": False,
            "percent1_denominator": "Class optimization-eligible TRAIN pool before ER stopping allocation",
        }
