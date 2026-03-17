# Semantic Cloud Decoder V1 Design

## Goal

Design a first decoder-side experiment that tests whether a semantic-cloud model can support causal generation on a tightly scoped task.

This phase does not attempt full attention replacement. It is a small, controlled test of whether a cloud-state decoder can learn next-token prediction and show useful behavior on late-resolution completion tasks.

## Scope

This design covers one bounded decoder experiment cycle:

- a toy causal language modeling setup
- a prefix-to-completion generation task built from the current meaning-reinterpretation challenge patterns
- three decoder baselines:
  - tiny Transformer decoder
  - GRU decoder
  - CFRM causal decoder
- lightweight evaluation focused on learning viability and late-resolution behavior

Out of scope for this phase:

- long-context generation
- open-ended text generation quality benchmarks
- copy or retrieval stress tests
- split or merge cloud mechanics
- claims of attention replacement

## Why This Phase Exists

The encoder-side experiments suggest that the philosophy-heavy CFRM variant may have useful inductive bias for meaning integration and reinterpretation. That is encouraging, but it does not establish that the same structure can support generation.

The decoder phase starts with the smallest honest question:

Can a cloud-state model participate in causal next-token prediction on a task where the final clause resolves the meaning of the sentence?

## Success Criteria

Decoder v1 is successful if all of the following hold:

1. The CFRM decoder trains stably enough for validation loss to decrease.
2. The CFRM decoder produces coherent suffix completions on the toy task.
3. The CFRM decoder is not catastrophically worse than tiny Transformer or GRU baselines.
4. On late-resolution completions, the CFRM decoder shows either competitive or stronger ending accuracy.

This phase does not require the CFRM decoder to beat both baselines on overall language modeling loss.

## Data Strategy

### Source

The decoder corpus will be derived from the current meaning-reinterpretation challenge patterns rather than from a broad public language-modeling dataset.

This keeps the task aligned with the semantic-cloud hypothesis:

- late reversal
- qualified support
- qualified rejection
- apparent positive with hidden negative ending
- apparent negative with hidden positive ending
- uncertainty resolved in a late clause

### Task Format

The primary task is `prefix -> full sentence completion`.

For each example:

- an input prefix is taken from the early part of the sentence
- the target is the remaining suffix
- training remains causal next-token prediction over the target sequence

This is intentionally narrower than full-sentence language modeling because the experiment is meant to test late semantic resolution rather than generic text continuation.

### Dataset Splits

The dataset should provide:

- `train`
- `valid`
- `test`

Optional metadata per row:

- `challenge_type`
- `prefix_text`
- `target_text`
- `resolution_position`
- `final_signal`

These metadata fields are useful for subset evaluation and qualitative inspection but should not be required for the training loop itself.

## Models

### 1. Tiny Transformer Decoder

Purpose:

- attention-based causal baseline

Structure:

- token embedding
- positional embedding
- 2-layer causal Transformer decoder stack
- vocabulary projection head

This is a small from-scratch baseline, not a pretrained model.

### 2. GRU Decoder

Purpose:

- recurrent baseline that is likely to be data-efficient on small corpora

Structure:

- token embedding
- 1-layer GRU
- hidden-state to vocabulary projection head

This baseline helps separate cloud-state gains from ordinary recurrent modeling gains.

### 3. CFRM Causal Decoder

Purpose:

- first generation-oriented test of the semantic-cloud idea

Structure:

- token embedding
- local recurrent path: 1-layer GRU
- global path: cloud-state update per step
- readout built from:
  - local hidden state
  - cloud core
  - strongest cloud
  - uncertainty features
- vocabulary projection head

The decoder should begin with the dense philosophy variant, not the optimized sparse variants.

Reason:

- the first decoder question is whether the mechanism works at all
- efficiency optimization should follow only after the dense version is validated

## Training Objective

The objective is standard causal next-token prediction on the target suffix.

For a prefix-completion example:

- the prefix is fed as conditioning context
- the model predicts the suffix token by token
- teacher forcing is used during training

This keeps the objective conventional so that architectural differences remain interpretable.

## Evaluation

### Required Metrics

- validation loss
- test loss
- perplexity
- exact match on full suffix completion
- token accuracy on the target suffix

### Focused Metrics

For late-resolution subsets:

- ending-clause token accuracy
- exact match on the final clause

These metrics matter more than general fluency because the experiment is about whether the model can generate the meaning-resolving ending.

## Qualitative Checks

For each model, save a small sample of generated completions on held-out prefixes.

The sample should show:

- prefix
- gold suffix
- generated suffix
- challenge type

This is not for benchmark scoring. It is a sanity check that helps identify whether the CFRM decoder collapses, repeats, or drifts semantically before the metrics explain why.

## Colab Requirements

The workflow must remain simple enough to run in Colab:

1. clone or pull the repository
2. build the decoder toy dataset
3. train one model
4. evaluate on test and late-resolution subsets
5. repeat for the other decoders

All outputs should be written under `artifacts/`.

## Risks

### 1. Dense CFRM Decoder May Be Slow

This is expected. The first decoder phase optimizes for learning signal, not runtime.

Mitigation:

- keep sequence lengths short
- keep model width small
- keep the dataset tightly scoped

### 2. CFRM Decoder May Learn More Slowly Than GRU

This would not invalidate the hypothesis by itself.

The key question is whether it can learn at all and whether it shows relative strength on late-resolution endings.

### 3. The Task May Be Too Narrow

That is acceptable for v1.

The purpose of v1 is not broad language generation. It is to test whether cloud-state decoding is viable on a task chosen to reflect the semantic-cloud idea.

## Decision Gate After Decoder V1

Move to decoder v2 only if:

1. CFRM decoder trains stably
2. completion outputs are readable
3. late-resolution metrics are competitive with at least one baseline

If those conditions are not met, do not add more architectural complexity yet. First diagnose whether the failure is due to optimization, data design, or the decoder readout itself.
