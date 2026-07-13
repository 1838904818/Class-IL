"""CIC-IDS-2018 (CSE-CIC-IDS2018) loader.

Reads 9 daily CICFlowMeter CSV files (80 features) from
`datasets/cic-ids-2018/`, merges 14 raw attack sub-types into
7 attack families, subsamples BENIGN to 200k, and performs an
80/20 stratified train/test split.

Files used (downloaded from official AWS S3 open data bucket):
    Thursday-01-03, Wednesday-28-02   → Infiltration
    Thursday-15-02                    → DoS (GoldenEye, Slowloris)
    Friday-16-02                      → DoS (Hulk, SlowHTTPTest)
    Wednesday-21-02                   → DDoS (HOIC, LOIC-UDP)
    Wednesday-14-02                   → Bruteforce (FTP, SSH)
    Friday-23-02, Thursday-22-02      → WebAttack (BruteForce-Web, XSS, SQL)
    Friday-02-03                      → Botnet (Bot)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, SEED


ATTACK_FAMILY = {
    # Normal
    "Benign":                    "Normal",
    # Brute-force
    "FTP-BruteForce":            "Bruteforce",
    "SSH-Bruteforce":            "Bruteforce",
    # DoS
    "DoS attacks-GoldenEye":     "DoS",
    "DoS attacks-Slowloris":     "DoS",
    "DoS attacks-SlowHTTPTest":  "DoS",
    "DoS attacks-Hulk":          "DoS",
    # DDoS
    "DDOS attack-HOIC":          "DDoS",
    "DDOS attack-LOIC-UDP":      "DDoS",
    "DDoS attacks-LOIC-HTTP":    "DDoS",  # alias seen in some releases
    # Web attacks
    "Brute Force -Web":          "WebAttack",
    "Brute Force -XSS":          "WebAttack",
    "SQL Injection":              "WebAttack",
    # Infiltration — note: original data has a typo ("Infilteration")
    "Infilteration":             "Infiltration",
    "Infiltration":              "Infiltration",
    # Botnet
    "Bot":                       "Botnet",
}

# Only 'Label' and 'Timestamp' need dropping.
# Unlike CIC-IDS-2017, the 2018 CSVs have IP/port columns already removed.
DROP_COLS = ["Label", "Timestamp"]

# Class order chosen for Class-IL ordering: 2 per task →
#   T0: [Normal, DoS]
#   T1: [DDoS, Bruteforce]
#   T2: [WebAttack, Infiltration]
#   T3: [Botnet]
CLASS_ORDER = ["Normal", "DoS", "DDoS", "Bruteforce",
               "WebAttack", "Infiltration", "Botnet"]


def load_cicids_2018():
    data_dir = DATA_DIR / "cic-ids-2018"
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs found in {data_dir}. "
            "Download the dataset from the AWS S3 open data registry:\n"
            "  aws s3 sync --no-sign-request "
            "\"s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/\" "
            "./datasets/cic-ids-2018/"
        )
    print(f"  reading {len(csv_files)} CIC-IDS-2018 CSVs ...")

    # Chunked streaming with per-family reservoir cap.
    # Once a family accumulates MAX_PER_CLASS rows we stop collecting it,
    # so peak RAM ≈ n_classes × MAX_PER_CLASS × row_size ≈ 350 MB.
    # Override the default cap when running larger-scale experiments.
    import os as _os
    MAX_PER_CLASS = int(_os.environ.get("MAX_PER_CLASS", "50000"))
    CHUNK_SIZE = 50_000
    family_frames: dict[str, pd.DataFrame] = {}  # {family: collected_df}
    family_counts: dict[str, int] = {}            # total rows seen per family
    total_raw = 0

    for f in csv_files:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if "Label" not in chunk.columns:
                continue
            chunk = chunk[chunk["Label"] != "Label"]
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
            if len(chunk) == 0:
                continue
            chunk["family"] = chunk["Label"].map(ATTACK_FAMILY)
            chunk = chunk.dropna(subset=["family"])
            if len(chunk) == 0:
                continue
            total_raw += len(chunk)
            for fam, grp in chunk.groupby("family"):
                seen = family_counts.get(fam, 0)
                if seen >= MAX_PER_CLASS:
                    continue  # already have enough — skip
                remaining_quota = MAX_PER_CLASS - seen
                if len(grp) > remaining_quota:
                    grp = grp.sample(n=remaining_quota, random_state=SEED)
                if fam in family_frames:
                    family_frames[fam] = pd.concat(
                        [family_frames[fam], grp], ignore_index=True
                    )
                else:
                    family_frames[fam] = grp.copy()
                family_counts[fam] = seen + len(grp)

    print(f"  total mapped rows: {total_raw:,}")
    df = pd.concat(list(family_frames.values()), ignore_index=True)
    print(f"  after per-class cap ({MAX_PER_CLASS:,}): {len(df):,}")

    # Drop metadata columns
    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    print(f"  family counts:\n{df['family'].value_counts()}")

    y = df.pop("family").values
    X = df.values.astype(np.float32)

    # Only keep classes that actually appear in the data
    present = set(y)
    order = [c for c in CLASS_ORDER if c in present]
    cls2idx = {c: i for i, c in enumerate(order)}
    y_i = np.array([cls2idx[v] for v in y], dtype=np.int64)

    X_tr, X_te, y_tr_i, y_te_i = train_test_split(
        X, y_i, test_size=0.2, random_state=SEED, stratify=y_i
    )
    return X_tr, y_tr_i, X_te, y_te_i, order
