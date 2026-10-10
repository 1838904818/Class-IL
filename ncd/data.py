"""Task-0-only preprocessing for the NCD experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSL_ATTACK_FAMILY = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

CACHE_DATASETS = {
    "CIC-IDS-2017": {
        "slug": "cic-ids-2017",
        "class_order": [
            "Normal", "DoS", "DDoS", "Bruteforce", "PortScan",
            "WebAttack", "Botnet", "Infiltration",
        ],
        "train_cap": {"Normal": 160_000},
        "test_cap": {"Normal": 40_000},
    },
    "CSE-CIC-IDS2018": {
        "slug": "cic-ids-2018",
        "class_order": [
            "Normal", "DoS", "DDoS", "Bruteforce",
            "WebAttack", "Infiltration", "Botnet",
        ],
        "train_cap": 40_000,
        "test_cap": 10_000,
    },
}


@dataclass(frozen=True)
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_order: list[str]
    tasks: list[list[int]]
    feature_names: list[str]
    source_files: list[str]


def make_tasks(n_classes: int, classes_per_task: int) -> list[list[int]]:
    tasks: list[list[int]] = []
    start = 0
    while start < n_classes:
        size = classes_per_task
        if start == 0 and size < 2:
            size = 2
        end = min(start + size, n_classes)
        tasks.append(list(range(start, end)))
        start = end
    return tasks


def _sample_shards(files: list[Path], cap: int | None, seed: int) -> np.ndarray:
    """Load all rows or an exact deterministic global sample from NPY shards."""
    if not files:
        raise FileNotFoundError("no NPY shards supplied")
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in files]
    feature_count = arrays[0].shape[1]
    if any(array.ndim != 2 or array.shape[1] != feature_count for array in arrays):
        raise ValueError("inconsistent cache shard shapes")
    total = sum(len(array) for array in arrays)
    if total == 0:
        raise ValueError("empty cache class")
    if cap is None or total <= cap:
        pieces = [np.asarray(array, dtype=np.float32) for array in arrays]
    else:
        selected = np.sort(np.random.default_rng(seed).choice(total, cap, replace=False))
        pieces = []
        offset = 0
        for array in arrays:
            stop = offset + len(array)
            left = np.searchsorted(selected, offset, side="left")
            right = np.searchsorted(selected, stop, side="left")
            if right > left:
                pieces.append(np.asarray(array[selected[left:right] - offset], dtype=np.float32))
            offset = stop
    result = np.concatenate(pieces, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("cache contains non-finite feature values")
    return result


def _cap_for(configured: int | dict[str, int], class_name: str) -> int | None:
    if isinstance(configured, int):
        return configured
    return configured.get(class_name)


def _load_frozen_cache(
    dataset: str,
    data_root: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str], list[str], list[str]]:
    specification = CACHE_DATASETS[dataset]
    cache_root = data_root / str(specification["slug"])
    manifest_path = cache_root / "manifest.json"
    schema_path = cache_root / "feature_schema.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    class_order = list(specification["class_order"])
    if manifest.get("dataset") != specification["slug"]:
        raise ValueError(f"cache dataset mismatch: {manifest.get('dataset')}")
    if schema.get("class_order") != class_order:
        raise ValueError("cache class order does not match the frozen NCD order")
    feature_names = list(schema.get("feature_columns", []))
    if len(feature_names) != int(schema.get("feature_count", -1)):
        raise ValueError("invalid cache feature schema")

    split_arrays: dict[str, list[np.ndarray]] = {"train": [], "test": []}
    split_labels: dict[str, list[np.ndarray]] = {"train": [], "test": []}
    for split_index, split in enumerate(("train", "test")):
        configured_cap = specification[f"{split}_cap"]
        for class_id, class_name in enumerate(class_order):
            files = sorted((cache_root / split / class_name).glob("part-*.npy"))
            values = _sample_shards(
                files,
                _cap_for(configured_cap, class_name),
                seed=42_000 + split_index * 1_000 + class_id,
            )
            if values.shape[1] != len(feature_names):
                raise ValueError(f"feature count mismatch for {split}/{class_name}")
            split_arrays[split].append(values)
            split_labels[split].append(np.full(len(values), class_id, dtype=np.int64))

    train_array = np.concatenate(split_arrays["train"], axis=0)
    test_array = np.concatenate(split_arrays["test"], axis=0)
    return (
        pd.DataFrame(train_array, columns=feature_names),
        np.concatenate(split_labels["train"]),
        pd.DataFrame(test_array, columns=feature_names),
        np.concatenate(split_labels["test"]),
        class_order,
        [],
        [str(manifest_path), str(schema_path)],
    )


def _load_raw(dataset: str, data_root: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str], list[str], list[str]]:
    if dataset in CACHE_DATASETS:
        return _load_frozen_cache(dataset, data_root)
    if dataset == "NSL-KDD":
        train_path = data_root / "nsl-kdd" / "KDDTrain+.txt"
        test_path = data_root / "nsl-kdd" / "KDDTest+.txt"
        train = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
        test = pd.read_csv(test_path, header=None, names=NSL_COLUMNS)
        for frame in (train, test):
            frame["family"] = frame["label"].astype(str).str.strip().map(NSL_ATTACK_FAMILY).fillna("R2L")
            frame.drop(columns=["label", "difficulty"], inplace=True)
        class_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        categorical = ["protocol_type", "service", "flag"]
        classes_per_task = 1
        source_files = [str(train_path), str(test_path)]
    elif dataset == "UNSW-NB15":
        train_path = data_root / "unsw-nb15" / "UNSW_NB15_training-set.csv"
        test_path = data_root / "unsw-nb15" / "UNSW_NB15_testing-set.csv"
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        for frame in (train, test):
            frame.drop(columns=[c for c in ("id", "label") if c in frame], inplace=True)
            frame["family"] = frame.pop("attack_cat").astype(str).str.strip()
        class_order = [
            "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
            "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
        ]
        categorical = ["proto", "service", "state"]
        classes_per_task = 2
        source_files = [str(train_path), str(test_path)]
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    cls_to_idx = {name: index for index, name in enumerate(class_order)}
    unknown_train = sorted(set(train["family"]) - set(cls_to_idx))
    unknown_test = sorted(set(test["family"]) - set(cls_to_idx))
    if unknown_train or unknown_test:
        raise ValueError(f"unknown labels: train={unknown_train}, test={unknown_test}")
    y_train = train.pop("family").map(cls_to_idx).to_numpy(dtype=np.int64)
    y_test = test.pop("family").map(cls_to_idx).to_numpy(dtype=np.int64)
    tasks = make_tasks(len(class_order), classes_per_task)
    return train, y_train, test, y_test, class_order, categorical, source_files


def prepare_dataset(dataset: str, data_root: Path) -> PreparedData:
    train, y_train, test, y_test, class_order, categorical, source_files = _load_raw(dataset, data_root)
    tasks = make_tasks(len(class_order), 1 if dataset == "NSL-KDD" else 2)
    numeric = [column for column in train.columns if column not in categorical]
    train = train.copy()
    test = test.copy()
    train[numeric] = train[numeric].replace([np.inf, -np.inf], np.nan)
    test[numeric] = test[numeric].replace([np.inf, -np.inf], np.nan)
    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    task0_mask = np.isin(y_train, tasks[0])
    if not np.any(task0_mask):
        raise RuntimeError("Task 0 has no training rows")
    transformer.fit(train.loc[task0_mask])
    x_train = np.asarray(transformer.transform(train), dtype=np.float32)
    x_test = np.asarray(transformer.transform(test), dtype=np.float32)
    feature_names = [str(name) for name in transformer.get_feature_names_out()]
    return PreparedData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        class_order=class_order,
        tasks=tasks,
        feature_names=feature_names,
        source_files=source_files,
    )
