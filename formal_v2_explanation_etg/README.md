# Formal-v2 explanation drift and ETG analyzer

This CPU-only offline analyzer consumes monitored FT-Transformer checkpoint
artifacts.  It validates the complete SHA-256 chain, reconstructs the official
`joint_cap3000` score, explains the class decision margin with SHAP expected
gradients, emits the full registered threshold-sensitivity grid, and simulates
the strict ETG-v2 governance ledger.

It does not write to W&B and does not modify training artifacts.  The output
manifest is fail-closed: any changed file, state, score, probe coordinate,
training result, method protocol, or analysis output causes validation to fail.
