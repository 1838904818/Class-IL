"""NF-ToN-IoT-v2 loader — a 5th, modern NetFlow-v2 IoT benchmark.

NF-ToN-IoT-v2 (Sarhan et al., "Towards a Standard Feature Set for NIDS
Datasets", MONET 2022) is one of the four sources that make up the merged
NF-UQ-NIDS-v2 dataset; it shares the same standardized 43-feature NetFlow v2
schema. We use this single, downloadable component as our fifth benchmark:
it is modern (IoT testbed), in the standard NetFlow feature set, and has 10
classes (Benign + 9 IoT attack types) — the longest Class-IL sequence in our
suite (5 tasks at 2 classes/task).

Source: https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2
        (train + test CSVs, ~2.6 GB; mirror of the UQ NF-ToN-IoT-v2 release)

Columns: 43 NetFlow features + Label (binary) + Attack (multi-class). We drop
the four flow identifiers (src/dst IPv4 + src/dst L4 port) to avoid the model
memorising the testbed's fixed addressing — the same identifier-dropping done
for the CIC loaders — leaving 39 numeric features. The multi-class label is the
`Attack` column; `Label` (0/1) is discarded.

Like the CIC loaders, the two provided CSVs are pooled, capped per class with a
streaming reservoir, then split 80/20 stratified — so the Class-IL protocol is
identical across all five datasets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, SEED


# Attack column -> family. ToN-IoT uses 'Benign' (capitalised) + lowercase
# attack names; we normalise via a lowercase lookup so case never bites us.
ATTACK_FAMILY = {
    "benign":     "Normal",
    "dos":        "DoS",
    "ddos":       "DDoS",
    "scanning":   "Scanning",
    "backdoor":   "Backdoor",
    "injection":  "Injection",
    "xss":        "XSS",
    "password":   "Password",
    "mitm":       "MITM",
    "ransomware": "Ransomware",
}

# Flow identifiers + binary label dropped; 'Attack' is popped as the label.
DROP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR",
             "L4_SRC_PORT", "L4_DST_PORT", "Label"]

# Class-IL order (2 per task -> 5 tasks): Normal + two high-volume attacks
# first, rarer families later so late tasks are the harder few-shot ones.
CLASS_ORDER = ["Normal", "Scanning", "DoS", "DDoS", "Backdoor",
               "Injection", "Password", "XSS", "Ransomware", "MITM"]


def load_nf_ton_iot_v2():
    data_dir = DATA_DIR / "nf-ton-iot-v2"
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs found in {data_dir}. Download from Hugging Face:\n"
            "  curl -sL -o datasets/nf-ton-iot-v2/NF-ToN-IoT-v2-train.csv "
            "https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2/resolve/"
            "main/NF-ToN-IoT-v2-train.csv\n  (and the -test.csv likewise)"
        )
    print(f"  reading {len(csv_files)} NF-ToN-IoT-v2 CSVs ...")

    # Override the default cap when running larger-scale experiments.
    import os as _os
    MAX_PER_CLASS = int(_os.environ.get("MAX_PER_CLASS", "50000"))
    CHUNK_SIZE = 100_000
    family_frames: dict[str, pd.DataFrame] = {}
    family_counts: dict[str, int] = {}
    total_raw = 0

    for f in csv_files:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if "Attack" not in chunk.columns:
                continue
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
            if len(chunk) == 0:
                continue
            chunk["family"] = (chunk["Attack"].astype(str).str.strip()
                               .str.lower().map(ATTACK_FAMILY))
            chunk = chunk.dropna(subset=["family"])
            if len(chunk) == 0:
                continue
            total_raw += len(chunk)
            for fam, grp in chunk.groupby("family"):
                seen = family_counts.get(fam, 0)
                if seen >= MAX_PER_CLASS:
                    continue
                remaining = MAX_PER_CLASS - seen
                if len(grp) > remaining:
                    grp = grp.sample(n=remaining, random_state=SEED)
                family_frames[fam] = (
                    pd.concat([family_frames[fam], grp], ignore_index=True)
                    if fam in family_frames else grp.copy())
                family_counts[fam] = seen + len(grp)

    print(f"  total mapped rows: {total_raw:,}")
    df = pd.concat(list(family_frames.values()), ignore_index=True)
    print(f"  after per-class cap ({MAX_PER_CLASS:,}): {len(df):,}")

    # drop identifiers + binary label + the raw Attack string
    for c in DROP_COLS + ["Attack"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    y_str = df.pop("family").values
    # everything left must be numeric; coerce defensively, then keep rows that
    # are finite AND representable in float32. A few NetFlow byte/throughput
    # fields carry garbage values above the float32 max (~3.4e38); casting them
    # would yield +inf and poison StandardScaler, so we drop those flows.
    arr = df.apply(pd.to_numeric, errors="coerce").values  # float64
    good = np.isfinite(arr).all(axis=1) & (np.abs(arr) < 3.0e38).all(axis=1)
    X = arr[good].astype(np.float32)
    y_str = y_str[good]
    print(f"  feature dim: {X.shape[1]}  rows after numeric/finite filter: "
          f"{len(X):,} (dropped {int((~good).sum())})")

    present = set(y_str)
    order = [c for c in CLASS_ORDER if c in present]
    cls2idx = {c: i for i, c in enumerate(order)}
    y_i = np.array([cls2idx[v] for v in y_str], dtype=np.int64)
    counts = {c: int((y_i == i).sum()) for i, c in enumerate(order)}
    print(f"  classes ({len(order)}): {counts}")

    X_tr, X_te, y_tr_i, y_te_i = train_test_split(
        X, y_i, test_size=0.2, random_state=SEED, stratify=y_i
    )
    return X_tr, y_tr_i, X_te, y_te_i, order
