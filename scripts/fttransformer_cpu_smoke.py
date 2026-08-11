"""Generate the auditable CPU smoke/parameter report for the FT candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ofra_encoders import FTTransformerEncoder, verify_ft_transformer_dependency
from streaming_full.models import model_architecture_record


DATASETS = {
    "nsl-kdd": {"feature_dim": 122, "classes": 5},
    "unsw-nb15": {"feature_dim": 194, "classes": 10},
    "cic-ids-2017": {"feature_dim": 78, "classes": 8},
    "cic-ids-2018": {"feature_dim": 78, "classes": 7},
    "malaya-network-gt": {"feature_dim": 77, "classes": 10},
}


def state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_record() -> dict[str, object]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260716)
    values = torch.randn(4, 194)

    def run_once() -> tuple[str, str, list[int], int]:
        torch.manual_seed(314159)
        model = FTTransformerEncoder(
            n_features=194,
            d_model=64,
            depth=4,
            heads=4,
            dim_head=16,
            attn_dropout=0.1,
            ff_dropout=0.1,
            num_residual_streams=1,
        ).eval()
        with torch.no_grad():
            output = model(values)
        output_hash = hashlib.sha256(
            output.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        return (
            state_sha256(model),
            output_hash,
            list(output.shape),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    first = run_once()
    second = run_once()
    if first != second:
        raise RuntimeError("CPU FT-Transformer smoke is not bitwise deterministic")

    parameter_comparison = {}
    for dataset, spec in DATASETS.items():
        common = {
            "n_features": spec["feature_dim"],
            "n_classes": spec["classes"],
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "ft_heads": 4,
            "ft_dim_head": 16,
            "ft_attn_dropout": 0.1,
            "ft_ff_dropout": 0.1,
            "ft_num_residual_streams": 1,
        }
        mlp = model_architecture_record(
            encoder_type="mlp", d_model=128, n_layers=2, **common
        )
        ft = model_architecture_record(
            encoder_type="ft_transformer", d_model=64, n_layers=4, **common
        )
        mlp_count = mlp["parameter_counts"]["encoder"]
        ft_count = ft["parameter_counts"]["encoder"]
        mlp_total = mlp["parameter_counts"][
            "encoder_plus_final_family_heads"
        ]
        ft_total = ft["parameter_counts"][
            "encoder_plus_final_family_heads"
        ]
        parameter_comparison[dataset] = {
            "feature_dim": spec["feature_dim"],
            "classes": spec["classes"],
            "mlp_128x2_encoder_parameters": mlp_count,
            "ft_transformer_64x4_encoder_parameters": ft_count,
            "ft_to_mlp_encoder_parameter_ratio": ft_count / mlp_count,
            "mlp_full_parameter_record": mlp["parameter_counts"],
            "ft_full_parameter_record": ft["parameter_counts"],
            "ft_to_mlp_encoder_plus_final_heads_parameter_ratio": (
                ft_total / mlp_total
            ),
        }

    return {
        "report": "ofra_ft_transformer_cpu_smoke_v1",
        "device": "cpu",
        "torch": torch.__version__,
        "dependency": verify_ft_transformer_dependency(),
        "input_contract": {
            "categories": [],
            "continuous_features": "all standardised post-cache features",
        },
        "candidate": {
            "dim": 64,
            "depth": 4,
            "heads": 4,
            "dim_head": 16,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
            "num_residual_streams": 1,
        },
        "synthetic_worst_width_smoke": {
            "feature_dim": 194,
            "batch_rows": 4,
            "output_shape": first[2],
            "model_state_sha256": first[0],
            "output_sha256": first[1],
            "encoder_parameters": first[3],
            "repeat_bitwise_identical": True,
        },
        "parameter_comparison": parameter_comparison,
        "gpu_policy": {
            "gpu_training_not_run": True,
            "train_microbatch": 64,
            "gradient_accumulation_steps": 4,
            "nominal_effective_batch": 256,
            "evaluation_batch": 128,
            "first_probe_dataset": "unsw-nb15",
            "fallback_if_oom": {
                "train_microbatch": 32,
                "gradient_accumulation_steps": 8,
                "nominal_effective_batch": 256,
                "evaluation_batch": 64,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/fttransformer_cpu_smoke.json"),
    )
    args = parser.parse_args()
    record = build_record()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
