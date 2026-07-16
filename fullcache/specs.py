"""Dataset contracts copied from the frozen OFRA loaders.

Only label grouping, feature removal, and class order are reused here.  The
old loaders' class caps and in-memory concatenation are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd


LabelNormalizer = Callable[[pd.Series], pd.Series]


def _strip_labels(labels: pd.Series) -> pd.Series:
    return labels.astype("string").str.strip()


def _lower_labels(labels: pd.Series) -> pd.Series:
    return _strip_labels(labels).str.lower()


def _normalise_cic17_labels(labels: pd.Series) -> pd.Series:
    """Normalise every observed CIC-IDS-2017 Web Attack dash spelling."""

    normalised = (
        labels.astype("string")
        .str.strip()
        .str.replace("\u00ef\u00bf\u00bd", "\u2013", regex=False)
        .str.replace("\ufffd", "\u2013", regex=False)
        .str.replace("\x96", "\u2013", regex=False)
    )
    suffix = normalised.str.extract(
        r"^Web Attack.*?(Brute Force|XSS|Sql Injection)$", expand=False
    )
    mask = suffix.notna()
    normalised.loc[mask] = "Web Attack \u2013 " + suffix.loc[mask]
    return normalised


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    subdirectory: str
    label_candidates: tuple[str, ...]
    label_mapping: Mapping[str, str]
    label_normalizer: LabelNormalizer
    class_order: tuple[str, ...]
    drop_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    split_strategy: str
    expected_feature_count: int
    expected_file_count: int
    tasks: tuple[tuple[int, ...], ...]
    expected_filenames: tuple[str, ...] = ()
    encoding: str | None = None
    file_glob: str = "*.csv"
    read_names: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    unknown_label_family: str | None = None
    drop_any_raw_missing: bool = False
    label_source: str = "column"
    problem_type: str = "intrusion_detection"
    task_semantics: str = "class_incremental"
    metric_profile: str = "nids_multiclass_with_binary_detection"
    normal_class_id: int | None = 0
    source_contract_relative: str | None = None
    bundled_contract_relative: str | None = None
    source_contract_sha256: str | None = None
    source_revision: str | None = None


CIC17_FILES = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
)

CIC18_FILES = (
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    # The official bucket spells Tuesday as "Thuesday".
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
)


CIC17_LABELS = {
    "BENIGN": "Normal",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Bruteforce",
    "SSH-Patator": "Bruteforce",
    "Web Attack \u2013 Brute Force": "WebAttack",
    "Web Attack \u2013 XSS": "WebAttack",
    "Web Attack \u2013 Sql Injection": "WebAttack",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
}

CIC18_LABELS = {
    "Benign": "Normal",
    "FTP-BruteForce": "Bruteforce",
    "SSH-Bruteforce": "Bruteforce",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "Brute Force -Web": "WebAttack",
    "Brute Force -XSS": "WebAttack",
    "SQL Injection": "WebAttack",
    "Infilteration": "Infiltration",
    "Infiltration": "Infiltration",
    "Bot": "Botnet",
}

CIC_IOT_LABELS = {
    "BenignTraffic": "Normal",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-DNS_Amplification": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DoS-UDP_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",
    "SqlInjection": "WebAttack",
    "XSS": "WebAttack",
    "CommandInjection": "WebAttack",
    "Uploading_Attack": "WebAttack",
    "BrowserHijacking": "WebAttack",
    "Backdoor_Malware": "WebAttack",
    "DictionaryBruteForce": "Bruteforce",
    "MITM-ArpSpoofing": "Spoofing",
    "DNS_Spoofing": "Spoofing",
    "Mirai-greip_flood": "Mirai",
    "Mirai-greeth_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
}

NF_TON_LABELS = {
    "benign": "Normal",
    "dos": "DoS",
    "ddos": "DDoS",
    "scanning": "Scanning",
    "backdoor": "Backdoor",
    "injection": "Injection",
    "xss": "XSS",
    "password": "Password",
    "mitm": "MITM",
    "ransomware": "Ransomware",
}

NSL_KDD_COLUMNS = (
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
)

NSL_KDD_LABELS = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "processtable": "DoS", "worm": "DoS",
    "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

UNSW_LABELS = {
    class_name: class_name
    for class_name in (
        "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
        "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
    )
}

MALAYA_NETWORK_GT_LABELS = {
    "Bittorent": "BitTorrent",
    "BitTorrent": "BitTorrent",
    "ChromeRDP": "ChromeRDP",
    "Discord": "Discord",
    "EA Origin": "EA Origin",
    "Microsoft Teams": "Microsoft Teams",
    "Slack": "Slack",
    "Steam": "Steam",
    "Teamviewer": "TeamViewer",
    "TeamViewer": "TeamViewer",
    "Webex": "Webex",
    "Zoom": "Zoom",
}


DATASET_SPECS: dict[str, DatasetSpec] = {
    "nsl-kdd": DatasetSpec(
        name="nsl-kdd",
        subdirectory="nsl-kdd",
        label_candidates=("label",),
        label_mapping=NSL_KDD_LABELS,
        label_normalizer=_strip_labels,
        class_order=("Normal", "DoS", "Probe", "R2L", "U2R"),
        drop_columns=("label", "difficulty"),
        identifier_columns=(),
        split_strategy="official_train_test_files",
        expected_feature_count=122,
        expected_file_count=2,
        tasks=((0, 1), (2,), (3,), (4,)),
        expected_filenames=("KDDTrain+.txt", "KDDTest+.txt"),
        file_glob="*.txt",
        read_names=NSL_KDD_COLUMNS,
        categorical_columns=("protocol_type", "service", "flag"),
        # This exactly preserves the frozen loader's fillna("R2L") fallback.
        unknown_label_family="R2L",
    ),
    "unsw-nb15": DatasetSpec(
        name="unsw-nb15",
        subdirectory="unsw-nb15",
        label_candidates=("attack_cat",),
        label_mapping=UNSW_LABELS,
        label_normalizer=_strip_labels,
        class_order=(
            "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
            "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
        ),
        drop_columns=("id", "label", "attack_cat"),
        identifier_columns=("id",),
        split_strategy="official_train_test_files",
        expected_feature_count=194,
        expected_file_count=2,
        tasks=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
        expected_filenames=(
            "UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv",
        ),
        categorical_columns=("proto", "service", "state"),
    ),
    "cic-ids-2017": DatasetSpec(
        name="cic-ids-2017",
        subdirectory="cic-ids-2017",
        label_candidates=("Label",),
        label_mapping=CIC17_LABELS,
        label_normalizer=_normalise_cic17_labels,
        class_order=(
            "Normal", "DoS", "DDoS", "Bruteforce", "PortScan",
            "WebAttack", "Botnet", "Infiltration",
        ),
        drop_columns=(
            "Label", "Flow ID", "Source IP", "Destination IP", "Source Port",
            "Timestamp", "External IP",
        ),
        identifier_columns=(
            "Flow ID", "Source IP", "Destination IP", "Source Port",
            "External IP",
        ),
        split_strategy="feature_hash_group_80_20",
        expected_feature_count=78,
        expected_file_count=8,
        tasks=((0, 1), (2, 3), (4, 5), (6, 7)),
        expected_filenames=CIC17_FILES,
        encoding="latin-1",
        drop_any_raw_missing=True,
    ),
    "cic-ids-2018": DatasetSpec(
        name="cic-ids-2018",
        subdirectory="cic-ids-2018",
        label_candidates=("Label",),
        label_mapping=CIC18_LABELS,
        label_normalizer=_strip_labels,
        class_order=(
            "Normal", "DoS", "DDoS", "Bruteforce", "WebAttack",
            "Infiltration", "Botnet",
        ),
        # Dst Port is a shared model feature and must not be dropped.  The
        # official Thuesday-20-02 file alone adds the four identifiers below.
        drop_columns=("Label", "Timestamp", "Flow ID", "Src IP", "Src Port", "Dst IP"),
        identifier_columns=("Flow ID", "Src IP", "Src Port", "Dst IP"),
        split_strategy="feature_hash_group_80_20",
        expected_feature_count=78,
        expected_file_count=10,
        tasks=((0, 1), (2, 3), (4, 5), (6,)),
        expected_filenames=CIC18_FILES,
        drop_any_raw_missing=True,
    ),
    "cic-iot-2023": DatasetSpec(
        name="cic-iot-2023",
        subdirectory="cic-iot-2023",
        label_candidates=("label", "Label", "Attack"),
        label_mapping=CIC_IOT_LABELS,
        label_normalizer=_strip_labels,
        class_order=(
            "Normal", "DDoS", "DoS", "Recon", "WebAttack", "Bruteforce",
            "Spoofing", "Mirai",
        ),
        drop_columns=("label", "Label", "Attack"),
        identifier_columns=(),
        split_strategy="feature_hash_group_80_20",
        expected_feature_count=46,
        expected_file_count=169,
        tasks=((0, 1), (2, 3), (4, 5), (6, 7)),
        drop_any_raw_missing=True,
    ),
    "nf-ton-iot-v2": DatasetSpec(
        name="nf-ton-iot-v2",
        subdirectory="nf-ton-iot-v2",
        label_candidates=("Attack",),
        label_mapping=NF_TON_LABELS,
        label_normalizer=_lower_labels,
        class_order=(
            "Normal", "Scanning", "DoS", "DDoS", "Backdoor", "Injection",
            "Password", "XSS", "Ransomware", "MITM",
        ),
        drop_columns=(
            "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
            "Label", "Attack",
        ),
        identifier_columns=(
            "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
        ),
        split_strategy="official_train_test_files",
        expected_feature_count=39,
        expected_file_count=2,
        tasks=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
        expected_filenames=("NF-ToN-IoT-v2-train.csv", "NF-ToN-IoT-v2-test.csv"),
        drop_any_raw_missing=True,
    ),
    "malaya-network-gt": DatasetSpec(
        name="malaya-network-gt",
        # The data root contains a pinned Hugging Face checkout named
        # ``malaya-network-gt``. Only its flow CSVs are read; PCAPs are unused.
        subdirectory="malaya-network-gt/csv_output",
        label_candidates=(),
        label_mapping=MALAYA_NETWORK_GT_LABELS,
        label_normalizer=_strip_labels,
        class_order=(
            "BitTorrent", "ChromeRDP", "Discord", "EA Origin",
            "Microsoft Teams", "Slack", "Steam", "TeamViewer", "Webex",
            "Zoom",
        ),
        drop_columns=("src_ip", "dst_ip", "src_port", "dst_port", "timestamp"),
        identifier_columns=("src_ip", "dst_ip", "src_port", "dst_port"),
        split_strategy="frozen_capture_manifest",
        expected_feature_count=77,
        expected_file_count=31,
        tasks=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
        file_glob="**/*.csv",
        label_source="parent_directory",
        problem_type="application_classification",
        task_semantics="class_incremental",
        metric_profile="generic_multiclass",
        normal_class_id=None,
        bundled_contract_relative="contracts/malaya_network_gt_capture_split.json",
        source_contract_sha256=(
            "a3b626ea5dfdd08ea33016a742d2dab19d354ef649773674855fb40fbaa3dee8"
        ),
        source_revision="384a59278f98490ee6e93aae017e748078d29b6a",
    ),
}
