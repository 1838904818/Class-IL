"""CIC-IoT-2023 (CICIoT2023) dataset loader — D4-5.

Reads 169 part-files from `datasets/cic-iot-2023/` and groups the 33 raw
attack sub-labels into 8 attack families for Class-IL evaluation.

Official source:
  http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/
  (registration required — see download_cic_iot_2023.py)
Kaggle mirror (no auth):
  https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset

CSV format (47 columns):
  flow_duration, Header_Length, Protocol_Type, ..., label
  169 part-files named: part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv
  (and similar UUID-stamped names from different versions)

Families (8) — Class-IL ordering (2 per task):
  T0: [Normal, DDoS]
  T1: [DoS, Recon]
  T2: [WebAttack, Bruteforce]
  T3: [Spoofing, Mirai]
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, SEED

# ---------------------------------------------------------------------------
# Label → family mapping (all 33 known attack labels + BenignTraffic)
# ---------------------------------------------------------------------------
ATTACK_FAMILY = {
    # Normal
    "BenignTraffic":                "Normal",
    # DDoS — 13 sub-types
    "DDoS-ACK_Fragmentation":       "DDoS",
    "DDoS-UDP_Flood":               "DDoS",
    "DDoS-SlowLoris":               "DDoS",
    "DDoS-ICMP_Flood":              "DDoS",
    "DDoS-RSTFINFlood":             "DDoS",
    "DDoS-PSHACK_Flood":            "DDoS",
    "DDoS-SYN_Flood":               "DDoS",
    "DDoS-SynonymousIP_Flood":      "DDoS",
    "DDoS-ICMP_Fragmentation":      "DDoS",
    "DDoS-TCP_Flood":               "DDoS",
    "DDoS-HTTP_Flood":              "DDoS",
    "DDoS-DNS_Amplification":       "DDoS",   # alias seen in some releases
    "DDoS-UDP_Fragmentation":       "DDoS",   # alias
    # DoS — 4 sub-types
    "DoS-UDP_Flood":                "DoS",
    "DoS-TCP_Flood":                "DoS",
    "DoS-SYN_Flood":                "DoS",
    "DoS-HTTP_Flood":               "DoS",
    # Recon / Scanning
    "Recon-HostDiscovery":          "Recon",
    "Recon-OSScan":                 "Recon",
    "Recon-PingSweep":              "Recon",
    "Recon-PortScan":               "Recon",
    "VulnerabilityScan":            "Recon",
    # Web-based attacks
    "SqlInjection":                 "WebAttack",
    "XSS":                          "WebAttack",
    "CommandInjection":             "WebAttack",
    "Uploading_Attack":             "WebAttack",
    "BrowserHijacking":             "WebAttack",
    "Backdoor_Malware":             "WebAttack",
    # Brute force
    "DictionaryBruteForce":         "Bruteforce",
    # Spoofing
    "MITM-ArpSpoofing":             "Spoofing",
    "DNS_Spoofing":                 "Spoofing",
    # Mirai IoT botnet
    "Mirai-greip_flood":            "Mirai",
    "Mirai-greeth_flood":           "Mirai",
    "Mirai-udpplain":               "Mirai",
}

# Class-IL ordering: 2 per task
CLASS_ORDER = ["Normal", "DDoS", "DoS", "Recon", "WebAttack", "Bruteforce", "Spoofing", "Mirai"]

# Per-family cap to limit RAM (DDoS and DoS can be enormous)
MAX_PER_CLASS = 50_000
CHUNK_SIZE    = 50_000
BENIGN_CAP    = 100_000   # keep up to 100k benign samples

# Known non-feature columns to drop
DROP_COLS = ["label"]


def load_cic_iot_2023():
    """Load CIC-IoT-2023 dataset from `datasets/cic-iot-2023/` (CSV part-files).

    Returns:
        X_tr, y_tr, X_te, y_te: float32 arrays
        class_order:            list[str], 8 families in CIL order
    """
    data_dir = DATA_DIR / "cic-iot-2023"
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs found in {data_dir}.\n"
            "Download CIC-IoT-2023 using one of:\n"
            "  python download_cic_iot_2023.py          (auto from Kaggle)\n"
            "  python download_cic_iot_2023.py --manual (prints wget instructions)\n"
            "Or manually place the 169 part-files under datasets/cic-iot-2023/"
        )

    print(f"  reading {len(csv_files)} CIC-IoT-2023 CSV files ...")

    family_frames: dict[str, pd.DataFrame] = {}
    family_counts: dict[str, int]          = {}
    total_raw = 0

    for f in csv_files:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
            # Normalize column names (some releases use slightly different casing)
            chunk.columns = [c.strip() for c in chunk.columns]
            # Identify label column
            label_col = None
            for cand in ["label", "Label", "Attack"]:
                if cand in chunk.columns:
                    label_col = cand
                    break
            if label_col is None:
                continue

            chunk = chunk.replace([float("inf"), float("-inf")], float("nan")).dropna()
            if len(chunk) == 0:
                continue

            chunk["family"] = chunk[label_col].map(ATTACK_FAMILY)
            chunk = chunk.dropna(subset=["family"])
            if len(chunk) == 0:
                continue

            total_raw += len(chunk)
            for fam, grp in chunk.groupby("family"):
                cap = BENIGN_CAP if fam == "Normal" else MAX_PER_CLASS
                seen = family_counts.get(fam, 0)
                if seen >= cap:
                    continue
                remaining = cap - seen
                if len(grp) > remaining:
                    grp = grp.sample(n=remaining, random_state=SEED)
                if fam in family_frames:
                    family_frames[fam] = pd.concat([family_frames[fam], grp], ignore_index=True)
                else:
                    family_frames[fam] = grp.copy()
                family_counts[fam] = seen + len(grp)

    if not family_frames:
        raise RuntimeError("No valid data loaded from CIC-IoT-2023 files.")

    print(f"  total raw rows mapped: {total_raw:,}")
    df = pd.concat(list(family_frames.values()), ignore_index=True)
    print(f"  after per-class cap ({MAX_PER_CLASS:,}): {len(df):,}")

    # Drop label column (and any other metadata cols)
    for c in DROP_COLS + ["Attack", "Label"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    # Drop 'family' helper column
    family_labels = df.pop("family").values
    # Drop any remaining non-numeric columns
    df = df.select_dtypes(include=[np.number])

    print(f"  family counts:\n{pd.Series(family_labels).value_counts()}")

    X = df.values.astype(np.float32)
    present = set(family_labels)
    order = [c for c in CLASS_ORDER if c in present]
    cls2idx = {c: i for i, c in enumerate(order)}
    y = np.array([cls2idx[v] for v in family_labels], dtype=np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_tr, y_tr, X_te, y_te, order
