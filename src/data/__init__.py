"""Dataset loaders. Each module exposes a `load_<name>()` returning:

    X_tr, y_tr_int, X_te, y_te_int, class_order

where:
- X_tr/X_te: float32 arrays of shape (N, F)
- y_*: int64 arrays of class indices (0..len(class_order)-1)
- class_order: list[str] preserving the desired Class-IL ordering
              (LabelEncoder is NOT used — we use manual cls->idx mapping
              so that class_order is the source of truth for task partitioning).
"""
from src.data.nslkdd import load_nslkdd
from src.data.unsw_nb15 import load_unsw
from src.data.cic_ids_2017 import load_cicids_2017
from src.data.cic_ids_2018 import load_cicids_2018
from src.data.cic_iot_2023 import load_cic_iot_2023
from src.data.nf_ton_iot_v2 import load_nf_ton_iot_v2

# Registry — toggled in main.py
DATASET_LOADERS = {
    "NSL-KDD":      (load_nslkdd,        1),  # 5 classes, 1 new per task (task0 takes 2)
    "UNSW-NB15":    (load_unsw,          2),  # 10 classes, 2 per task
    "CIC-IDS-2017": (load_cicids_2017,   2),  # 8 classes, 2 per task
    "CIC-IDS-2018": (load_cicids_2018,   2),  # ~8 classes, 2 per task
    "CIC-IoT-2023": (load_cic_iot_2023,  2),  # ~10 classes, 2 per task (TBD)
    "NF-ToN-IoT-v2": (load_nf_ton_iot_v2, 2),  # 10 classes (NetFlow-v2 IoT), 2 per task
}

__all__ = [
    "load_nslkdd", "load_unsw", "load_cicids_2017",
    "load_cicids_2018", "load_cic_iot_2023", "load_nf_ton_iot_v2",
    "DATASET_LOADERS",
]
