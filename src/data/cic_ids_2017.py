"""CIC-IDS-2017 loader.

Reads 8 daily CSV files (MachineLearningCSV format, 78 features),
merges 14 raw attack subtypes into 8 attack families, subsamples BENIGN
to 200k, and performs an 80/20 stratified train/test split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, SEED


ATTACK_FAMILY = {
    "BENIGN":            "Normal",
    "DoS Hulk":          "DoS",
    "DoS GoldenEye":     "DoS",
    "DoS slowloris":     "DoS",
    "DoS Slowhttptest":  "DoS",
    "Heartbleed":        "DoS",
    "DDoS":              "DDoS",
    "PortScan":          "PortScan",
    "FTP-Patator":       "Bruteforce",
    "SSH-Patator":       "Bruteforce",
    # Raw labels use Latin-1 \x96 dash; provide both spellings.
    "Web Attack \x96 Brute Force": "WebAttack",
    "Web Attack \x96 XSS":         "WebAttack",
    "Web Attack \x96 Sql Injection": "WebAttack",
    "Web Attack – Brute Force":   "WebAttack",
    "Web Attack – XSS":           "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Bot":               "Botnet",
    "Infiltration":      "Infiltration",
}

DROP_COLS = [
    "Label", "Flow ID", "Source IP", "Destination IP",
    "Source Port", "Timestamp", "External IP",
]

CLASS_ORDER = ["Normal", "DoS", "DDoS", "Bruteforce", "PortScan",
               "WebAttack", "Botnet", "Infiltration"]


def _normalise_labels(labels):
    """Standardise the dash variants found in CIC-IDS-2017 labels."""
    return (
        labels.astype(str)
        .str.strip()
        .str.replace("\u00ef\u00bf\u00bd", "\u2013", regex=False)
        .str.replace("\ufffd", "\u2013", regex=False)
        .str.replace("\x96", "\u2013", regex=False)
    )


def load_cicids_2017():
    data_dir = DATA_DIR / "cic-ids-2017"
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")
    print(f"  reading {len(csv_files)} CIC-IDS-2017 CSVs ...")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]   # CIC has leading spaces
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"  total raw rows: {len(df):,}")

    # Clean NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  after dropping NaN/Inf: {len(df):,}")

    # Family mapping
    df["Label"] = _normalise_labels(df["Label"])
    df["family"] = df["Label"].map(ATTACK_FAMILY)
    n_unmapped = df["family"].isna().sum()
    if n_unmapped > 0:
        unmapped = [repr(x) for x in df.loc[df["family"].isna(), "Label"].unique()[:5]]
        print(f"  WARNING: {n_unmapped} rows had unmapped labels: {unmapped}")
        df = df.dropna(subset=["family"])

    # Subsample BENIGN to balance the heavily-skewed distribution.
    # FULL_DATA=1 disables the cap; CIC17_NORMAL_CAP controls it otherwise.
    import os as _os
    _cap = int(_os.environ.get("CIC17_NORMAL_CAP", "200000"))
    is_normal = df["family"] == "Normal"
    if _os.environ.get("FULL_DATA") == "1":
        print(f"  FULL DATA (no BENIGN cap): {len(df):,} rows")
    elif is_normal.sum() > _cap:
        keep = df[is_normal].sample(n=_cap, random_state=SEED)
        df = pd.concat([keep, df[~is_normal]], ignore_index=True)
        print(f"  BENIGN capped to {_cap:,}: {len(df):,} rows")
    else:
        print(f"  no BENIGN cap needed: {len(df):,} rows")

    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    print(f"  family counts:\n{df['family'].value_counts()}")

    y = df.pop("family").values
    X = df.values.astype(np.float32)

    cls2idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_i = np.array([cls2idx[v] for v in y], dtype=np.int64)

    X_tr, X_te, y_tr_i, y_te_i = train_test_split(
        X, y_i, test_size=0.2, random_state=SEED, stratify=y_i
    )
    return X_tr, y_tr_i, X_te, y_te_i, list(CLASS_ORDER)
