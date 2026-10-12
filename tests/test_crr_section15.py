from __future__ import annotations

import json

import numpy as np
import pytest

from ncd.collect_crr_section15_lambdas import collect
from ncd.merge_crr_section15 import METHODS, SEEDS, merge
from ncd.run_crr_real_sequence import rankdata, spearman


def test_rankdata_and_spearman_handle_ties_and_constants():
    assert np.allclose(rankdata(np.array([4.0, 1.0, 1.0])), [3.0, 1.5, 1.5])
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])) == pytest.approx(1.0)
    assert spearman(np.ones(3), np.ones(3)) == 1.0
    assert spearman(np.ones(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_collect_requires_all_methods(tmp_path):
    paths = []
    for index, method in enumerate(METHODS[1:], start=1):
        path = tmp_path / f"{method}.json"
        path.write_text(json.dumps({"method": method, "selected_lambda": index / 4}), encoding="utf-8")
        paths.append(path)
    result = collect(paths, "NSL-KDD", "a" * 40)
    assert set(result["lambdas"]) == set(METHODS)
    assert result["lambdas"]["replay"] == 0.0


def test_merge_registered_necessity_matrix(tmp_path):
    commit = "b" * 40
    rows = [
        {"dataset": "NSL-KDD", "seed": seed, "method": method, "k": k}
        for seed in SEEDS
        for method in METHODS
        for k in (2.0, 3.0)
    ]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"git_commit": commit, "rows": rows}), encoding="utf-8")
    result = merge([path], "necessity", "NSL-KDD", commit)
    assert len(result["rows"]) == 80


def test_merge_fails_on_duplicate_cell(tmp_path):
    commit = "c" * 40
    row = {"dataset": "NSL-KDD", "seed": 1, "method": "replay", "k": 2.0}
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"git_commit": commit, "rows": [row, row]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        merge([path], "necessity", "NSL-KDD", commit)
