"""NSL-KDD loader.

Original data: KDDTrain+/KDDTest+ in `datasets/nsl-kdd/`.
Maps 23+ raw attack subtypes into 5 super-classes:
    Normal, DoS, Probe, R2L, U2R
"""
import numpy as np
import pandas as pd

from src.config import DATA_DIR


COLS = [
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

ATTACK_FAMILY = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}


def load_nslkdd():
    train_path = DATA_DIR / "nsl-kdd" / "KDDTrain+.txt"
    test_path = DATA_DIR / "nsl-kdd" / "KDDTest+.txt"
    df_tr = pd.read_csv(train_path, header=None, names=COLS)
    df_te = pd.read_csv(test_path, header=None, names=COLS)

    for df in (df_tr, df_te):
        df["family"] = df["label"].map(ATTACK_FAMILY).fillna("R2L")
        df.drop(columns=["label", "difficulty"], inplace=True)

    # One-hot encode categorical columns; concat then split to keep schema aligned
    cat_cols = ["protocol_type", "service", "flag"]
    df_all = pd.concat([df_tr.assign(_split="tr"), df_te.assign(_split="te")], ignore_index=True)
    df_all = pd.get_dummies(df_all, columns=cat_cols)
    df_tr = df_all[df_all._split == "tr"].drop(columns=["_split"])
    df_te = df_all[df_all._split == "te"].drop(columns=["_split"])

    y_tr = df_tr.pop("family").values
    y_te = df_te.pop("family").values
    X_tr = df_tr.values.astype(np.float32)
    X_te = df_te.values.astype(np.float32)

    # Manual class -> int mapping (preserves desired Class-IL ordering)
    class_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    cls2idx = {c: i for i, c in enumerate(class_order)}
    y_tr_i = np.array([cls2idx[v] for v in y_tr], dtype=np.int64)
    y_te_i = np.array([cls2idx[v] for v in y_te], dtype=np.int64)

    return X_tr, y_tr_i, X_te, y_te_i, class_order
