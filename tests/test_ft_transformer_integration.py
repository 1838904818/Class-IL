"""CPU-only regression tests for the isolated FT-Transformer candidate."""
from __future__ import annotations

import hashlib
import unittest

import torch
import torch.nn.functional as F

from ofra_encoders import FTTransformerEncoder, verify_ft_transformer_dependency
from streaming_full.exposure_preflight import _optimizer_schedule
from streaming_full.models import (
    FamilyHead,
    MLPEncoder,
    build_encoder,
    model_architecture_record,
)
from streaming_full.validation import RunConfig


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FTTransformerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_dependency_is_exactly_pinned(self):
        record = verify_ft_transformer_dependency()
        self.assertEqual(record["installed_version"], "0.6.1")
        self.assertEqual(
            record["installed_implementation_sha256"],
            "db62c6e258467bb2d85b738fe1839f0b4279ec92f0bdbb83400ddd42fadd4d42",
        )

    def test_continuous_only_shape_and_family_head_contract(self):
        torch.manual_seed(11)
        encoder = FTTransformerEncoder(
            n_features=17,
            d_model=32,
            depth=2,
            heads=4,
            dim_head=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            num_residual_streams=1,
        ).eval()
        values = torch.randn(7, 17)
        embedding = encoder(values)
        self.assertEqual(tuple(embedding.shape), (7, 32))
        self.assertEqual(encoder.categories, ())
        head = FamilyHead(d_model=32, rank=8, alpha=16.0)
        self.assertEqual(tuple(head(embedding).shape), (7, 2))

    def test_embedding_only_path_matches_pinned_upstream_output_and_gradients(self):
        torch.manual_seed(31)
        encoder = FTTransformerEncoder(
            n_features=9,
            d_model=16,
            depth=2,
            heads=2,
            dim_head=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            num_residual_streams=1,
        ).train()
        values = torch.randn(5, 9)

        torch.manual_seed(77)
        optimized = encoder(values)
        optimized.square().mean().backward()
        optimized_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in encoder.named_parameters()
            if parameter.grad is not None
        }
        encoder.zero_grad(set_to_none=True)

        categories = torch.empty((len(values), 0), dtype=torch.long)
        torch.manual_seed(77)
        upstream = encoder.backbone(categories, values, return_attn=False)
        upstream.square().mean().backward()

        torch.testing.assert_close(optimized, upstream, rtol=0.0, atol=0.0)
        for name, parameter in encoder.named_parameters():
            if name in optimized_gradients:
                torch.testing.assert_close(
                    optimized_gradients[name], parameter.grad, rtol=0.0, atol=0.0
                )

    def test_initialisation_forward_and_one_step_are_deterministic(self):
        values = torch.linspace(-2.0, 2.0, steps=96).reshape(8, 12)
        target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

        def run_once() -> tuple[str, torch.Tensor]:
            torch.manual_seed(1729)
            encoder = FTTransformerEncoder(
                n_features=12,
                d_model=16,
                depth=2,
                heads=2,
                dim_head=8,
                attn_dropout=0.1,
                ff_dropout=0.1,
                num_residual_streams=1,
            )
            head = torch.nn.Linear(16, 2)
            optimizer = torch.optim.Adam(
                [*encoder.parameters(), *head.parameters()], lr=1e-3
            )
            encoder.train()
            torch.manual_seed(991)
            logits = head(encoder(values))
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            encoder.eval()
            with torch.no_grad():
                output = encoder(values).clone()
            combined = torch.nn.ModuleList([encoder, head])
            return _state_sha256(combined), output

        first_hash, first_output = run_once()
        second_hash, second_output = run_once()
        self.assertEqual(first_hash, second_hash)
        torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)

    def test_protocol_architecture_record_and_parameter_count(self):
        record = model_architecture_record(
            encoder_type="ft_transformer",
            n_features=194,
            n_classes=10,
            d_model=64,
            n_layers=4,
            lora_rank=8,
            lora_alpha=16.0,
            ft_heads=4,
            ft_dim_head=16,
            ft_attn_dropout=0.1,
            ft_ff_dropout=0.1,
            ft_num_residual_streams=1,
        )
        self.assertEqual(record["encoder"]["categories"], [])
        self.assertEqual(record["encoder"]["num_continuous"], 194)
        self.assertIn("not stacked", record["encoder"]["attention_output_policy"])
        self.assertGreater(record["parameter_counts"]["encoder"], 0)
        self.assertEqual(
            record["parameter_counts"]["encoder"],
            record["encoder"]["parameters"],
        )

    def test_gradient_accumulation_schedule_preserves_effective_batch(self):
        config = RunConfig(
            encoder_type="ft_transformer",
            d_model=64,
            n_layers=4,
            batch_size=64,
            gradient_accumulation_steps=4,
            eval_batch_size=128,
        )
        config.validate()
        self.assertEqual(
            config.batch_size * config.gradient_accumulation_steps, 256
        )
        schedule = _optimizer_schedule(101, 404, 12, 4)
        self.assertEqual(schedule["gradient_microbatches_per_epoch"], 9)
        self.assertEqual(schedule["optimizer_steps_per_epoch"], 3)

    def test_mlp_builder_keeps_the_copied_default_architecture(self):
        kwargs = dict(
            encoder_type="mlp",
            n_features=17,
            d_model=128,
            n_layers=2,
            ft_heads=4,
            ft_dim_head=16,
            ft_attn_dropout=0.1,
            ft_ff_dropout=0.1,
            ft_num_residual_streams=1,
        )
        torch.manual_seed(23)
        expected = MLPEncoder(17, d_model=128, n_layers=2)
        torch.manual_seed(23)
        actual = build_encoder(**kwargs)
        self.assertEqual(_state_sha256(expected), _state_sha256(actual))


if __name__ == "__main__":
    unittest.main()
