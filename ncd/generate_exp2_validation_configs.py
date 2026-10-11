"""Generate the frozen fresh-seed Exp 2 boundary-validation configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (5, 6, 7, 8, 9)
RHO_VALUES = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
P2_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
CELLS = (
    ("nsl", "nsl-kdd", "replay", "exp2_v2_nsl_replay_seed42.json"),
    ("nsl", "nsl-kdd", "icarl", "exp2_v2_nsl_icarl_seed42.json"),
    ("unsw", "unsw-nb15", "replay", "exp2_v2_unsw_replay_seed42.json"),
    ("unsw", "unsw-nb15", "icarl", "exp2_v2_unsw_icarl_seed42.json"),
)


def build_configs(config_dir: Path) -> list[Path]:
    written: list[Path] = []
    for dataset_short, dataset_dir, method, template_name in CELLS:
        template = json.loads((config_dir / template_name).read_text(encoding="utf-8"))
        for seed in SEEDS:
            config = dict(template)
            config["seed"] = seed
            config["rho_values"] = list(RHO_VALUES)
            if method == "replay":
                config["p2_mask_fractions"] = list(P2_FRACTIONS)
                config["p2_marker_seed"] = seed * 1_000_000 + 715_001
                config["p2_train_seed"] = 3_000 + seed
                if dataset_short == "unsw":
                    config["p2_selection_seed"] = seed * 1_000_000 + 600_001
            output = f"ncd/results/exp2_validation_v1/{dataset_dir}/{method}/seed{seed}"
            config["output_dir"] = output
            config["log_path"] = f"ncd/logs/exp2_validation_v1_{dataset_short}_{method}_seed{seed}.log"
            name = f"exp2_validation_v1_{dataset_short}_{method}_seed{seed}.json"
            path = config_dir / name
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("ncd/configs"))
    args = parser.parse_args()
    written = build_configs(args.config_dir)
    print(f"wrote {len(written)} frozen validation configs")


if __name__ == "__main__":
    main()
