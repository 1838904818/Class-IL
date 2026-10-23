"""Section138 standard/embedding MLP building blocks, not an execution entrypoint."""
import copy

import torch
from torch import nn

from ncd.section81_tabtransformer import TabularClassifier


class EmbeddedFeatures(nn.Module):
    """Reuse the registered Task-0 vocabulary and unknown-token encoding."""
    encode = TabularClassifier.encode

    def __init__(self, spec, hidden, layers):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        if spec.get("test_used_for_fitting") is not False or spec.get("unknown_id") != 0:
            raise ValueError("Task-0 TRAIN vocabulary required")
        self.register_buffer("continuous_columns", torch.tensor(spec["continuous_columns"], dtype=torch.long))
        for i, group in enumerate(spec["groups"]):
            self.register_buffer(f"columns_{i}", torch.tensor(group["columns"], dtype=torch.long))
            dtype = torch.long if group["kind"] == "onehot" else torch.float32
            self.register_buffer(f"vocabulary_{i}", torch.tensor(group["values"], dtype=dtype))
        self.embeddings = nn.ModuleList(nn.Embedding(len(g["values"]) + 1, 8)
                                       for g in spec["groups"])
        width = len(spec["continuous_columns"]) + 8 * len(spec["groups"])
        self.network = dense_features(width, hidden, layers)

    def forward(self, x):
        tokens, continuous = self.encode(x)
        encoded = [embedding(tokens[:, i]) for i, embedding in enumerate(self.embeddings)]
        return self.network(torch.cat([continuous, *encoded], dim=1))


def dense_features(inputs, hidden, layers):
    if (hidden, layers) not in ((128, 2), (512, 3)):
        raise ValueError("Unregistered section138 capacity")
    modules = []
    for index in range(layers):
        modules.extend((nn.Linear(inputs if index == 0 else hidden, hidden), nn.ReLU()))
    return nn.Sequential(*modules)


class Classifier(nn.Module):
    """Expose features and head for the existing retention/WA interfaces."""
    def __init__(self, inputs, outputs, hidden=128, layers=2, *, spec=None):
        super().__init__()
        self.features = (dense_features(inputs, hidden, layers) if spec is None
                         else EmbeddedFeatures(spec, hidden, layers))
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        return self.head(self.features(x))
