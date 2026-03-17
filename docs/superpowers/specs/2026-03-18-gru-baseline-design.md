# GRU Baseline Design

## Goal

Add a plain GRU sentence-classification baseline that is parameter-matched to the existing tiny Transformer so DynaSent results can separate:

- from-scratch Transformer weakness
- recurrent encoder strength
- actual CFRM/cloud-state contribution

## Scope

This design adds one new model family only:

- a bidirectional GRU classifier for sentence classification

It must integrate with the existing single-run and multi-seed experiment CLIs, and it must be easy to run from Colab.

Out of scope:

- decoder work
- GRU attention hybrids
- new datasets
- visualization changes

## Success Criteria

1. A new `gru` model type is available in training scripts.
2. Its parameter count is close to the Transformer baseline, with a target of roughly `583k` and acceptable drift within about `5%`.
3. Existing tests still pass, and new model/CLI tests cover the new path.
4. Colab commands for `transformer`, `gru`, `cfrm`, and `cfrm_philosophy` stay uniform.

## Architecture

The GRU baseline should be a plain encoder classifier:

1. token embedding
2. one bidirectional GRU
3. pooled sequence representation
4. linear classification head

This keeps the comparison honest:

- `transformer`: attention encoder baseline
- `gru`: recurrent encoder baseline
- `cfrm`: recurrent plus cloud baseline
- `cfrm_philosophy`: heavier cloud-state hypothesis model

## Parameter Matching

The new GRU should be sized against the current Transformer baseline, not against CFRM-philosophy.

Current counts:

- Transformer: about `583k`
- CFRM: about `564k`
- CFRM-philosophy: about `1.23M`

The GRU baseline should therefore target the Transformer/CFRM band. The simplest route is:

- keep the same vocab interface
- use a hidden width chosen empirically to land near the Transformer count
- avoid extra projection stacks unless needed for count matching

## Testing

Minimal new tests are enough:

- forward shape test
- tiny loss-reduction smoke test
- CLI choice acceptance for `gru`
- train runner smoke through existing paths

## Colab Handoff

After implementation, the user should be able to run:

- `transformer`
- `gru`
- `cfrm`
- `cfrm_philosophy`

through both `scripts/train_experiment.py` and `scripts/run_experiment_suite.py` with the same dataset and challenge set flags.
