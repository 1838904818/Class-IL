"""Compatibility aliases for experiments recorded under the former name."""

from src_v2.methods.ofra import OFRAAgent, run_ofra

PILoRAAgent = OFRAAgent
run_pilora = run_ofra

__all__ = ["PILoRAAgent", "run_pilora"]
