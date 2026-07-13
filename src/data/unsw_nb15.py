"""UNSW-NB15 loader.

Files: UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv.
Labels come from the `attack_cat` column (10 classes).
"""
import numpy as np
import pandas as pd

from src.config import DATA_DIR


def load_unsw():
    train_path = DATA_DIR / "unsw-nb15" / "UNSW_NB15_training-set.csv"
    test_path = DATA_DIR / "unsw-nb15" / "UNSW_NB15_testing-set.csv"
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    # Drop non-features
    for df in (df_tr, df_te):
        for col in ("id", "label"):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    cat_cols = ["proto", "service", "state"]
    df_all = pd.concat([df_tr.assign(_split="tr"), df_te.assign(_split="te")], ignore_index=True)
    df_all = pd.get_dummies(df_all, columns=cat_cols)
    df_tr = df_all[df_all._split == "tr"].drop(columns=["_split"])
    df_te = df_all[df_all._split == "te"].drop(columns=["_split"])

    y_tr = df_tr.pop("attack_cat").values
    y_te = df_te.pop("attack_cat").values
    X_tr = df_tr.values.astype(np.float32)
    X_te = df_te.values.astype(np.float32)

    # Normal first; attacks roughly in descending training-count order,
    # so rare classes (Worms, Shellcode) arrive last — realistic for Class-IL.
    class_order = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
                   "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
    cls2idx = {c: i for i, c in enumerate(class_order)}
    y_tr_i = np.array([cls2idx[v] for v in y_tr], dtype=np.int64)
    y_te_i = np.array([cls2idx[v] for v in y_te], dtype=np.int64)

    return X_tr, y_tr_i, X_te, y_te_i, class_order
