"""Owner-registered section156 design; enumeration alone is not execution."""
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Configuration:
    gate: int = 1
    family: int = 1
    normal: int = 1
    special: str = ""

    def __post_init__(self):
        if self.gate not in (1, 2, 3, 4) or self.family not in (1, 2, 3) or self.normal not in (1, 2):
            raise ValueError("Unregistered section156 factor")
        if self.special not in ("", "T3", "T4", "T5", "C1", "C2"):
            raise ValueError("Unregistered section156 special configuration")

    @property
    def identifier(self):
        return self.special or f"S156-g{self.gate}f{self.family}n{self.normal}"

    @property
    def selectable(self):
        return self.special not in ("C1", "C2")

    # Compatibility with the shared trajectory's registered role construction.
    encoding = 2
    gate_schedule = 2
    retention = 3

    def switches(self):
        gate = {1: "embedding", 2: "histgb", 3: "ensemble5", 4: "partial_auc"}[self.gate]
        family = "histgb" if self.family == 2 or (self.gate == 2 and self.family == 3) else "embedding"
        if self.special in ("T3", "T4"):
            gate = "tabtransformer"
            family = "embedding" if self.special == "T3" else "histgb"
        if self.special == "T5":
            gate, family = "histgb", "tabtransformer"
        control = self.special in ("C1", "C2")
        return dict(gate=gate, family=family, control=control,
                    benign_hard=self.family == 3 and not self.special,
                    normal_training_rows=500000 if self.normal == 2 or self.special == "C2" else 50000,
                    attack_memory_per_class=200, gate_retention=False,
                    gate_schedule="section123_es", family_hidden=128,
                    family_hidden_layers=2, derpp_alpha=1.0, derpp_beta=1.0,
                    weight_alignment=family != "histgb",
                    last_layer_cbf_epochs=int(family == "embedding"),
                    store_logits=family != "histgb" and not control)


def configurations():
    return tuple(Configuration(*x) for x in product(range(1, 5), range(1, 4), (1, 2))) + tuple(
        Configuration(special=x) for x in ("T3", "T4", "T5", "C1", "C2"))


def cells(*, lane="cpu", local=False):
    if lane not in ("cpu", "gpu"):
        raise ValueError("Unknown lane")
    configs = [c for c in configurations() if (c.special in ("T3", "T4", "T5")) == (lane == "gpu")]
    datasets = ("UNSW-NB15",) if local else ("UNSW-NB15", "CIC-IDS-2017")
    seed_blocks = ((1, 2),) if local else ((1, 2), (10, 11, 12, 13, 14))
    return tuple((c.identifier, d, s, "screen" if s in (1, 2) else "sealed")
                 for block in seed_blocks for d in datasets for c in configs for s in block)
