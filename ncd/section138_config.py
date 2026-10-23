"""Registered section138 factors; configuration enumeration is not a runner.

Confirmation cells are sealed until the selection commit. This module does
not launch cells, claim resource readiness, or authorize unsealing.
"""
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Configuration:
    encoding: int
    gate_schedule: int
    gate_capacity: int
    retention: int
    family_capacity: int

    def __post_init__(self):
        for name in ("encoding", "gate_schedule", "gate_capacity", "family_capacity"):
            if getattr(self, name) not in (1, 2):
                raise ValueError(f"Unregistered {name}")
        if self.retention not in (1, 2, 3):
            raise ValueError("Unregistered retention")

    @property
    def identifier(self):
        return (f"S138-a{self.encoding}b{self.gate_schedule}c{self.gate_capacity}"
                f"d{self.retention}e{self.family_capacity}")

    @property
    def is_reference(self):
        return self == Configuration(1, 1, 1, 2, 1)

    def switches(self):
        return {
            "encoding": "standardized" if self.encoding == 1 else "embedding8",
            "gate_schedule": "fixed" if self.gate_schedule == 1 else "section123_es",
            "gate_hidden": 128 if self.gate_capacity == 1 else 512,
            "gate_hidden_layers": 2 if self.gate_capacity == 1 else 3,
            "family_hidden": 128 if self.family_capacity == 1 else 512,
            "family_hidden_layers": 2 if self.family_capacity == 1 else 3,
            "family_schedule": "fixed",
            "derpp_alpha": 1.0,
            "derpp_beta": 1.0,
            "weight_alignment": self.retention in (2, 3),
            "last_layer_cbf_epochs": 1 if self.retention == 3 else 0,
            "gate_retention": False,
            "normal_training_rows": 50000,
            "attack_memory_per_class": 200,
        }


def configurations():
    return tuple(Configuration(*levels) for levels in
                 product((1, 2), (1, 2), (1, 2), (1, 2, 3), (1, 2)))


def cells(*, local=False):
    """Stable screening-first order; local reverses dataset order, not roles."""
    datasets = ("UNSW-NB15", "CIC-IDS-2017")
    if local:
        datasets = tuple(reversed(datasets))
    return tuple((cfg.identifier, dataset, seed, "screen" if seed in (1, 2) else "sealed")
                 for seeds in ((1, 2), (10, 11, 12, 13, 14))
                 for dataset in datasets for cfg in configurations() for seed in seeds)
