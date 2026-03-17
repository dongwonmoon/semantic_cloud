# Semantic Cloud CFRM First Experiment Design

## Goal

Build a first CPU/Colab-feasible experiment that tests whether a semantic-cloud-style model can compete with a small attention baseline on a controlled meaning classification task.

The experiment is not a general language model benchmark. It is a compact hypothesis test for the claim in [CORE.md](/mnt/c/Users/anseh/Workspace/semantic_cloud/CORE.md):

> Language understanding is better modeled as deformation and condensation of a global semantic state than as pairwise token lookup alone.

## Scope

This design covers one experiment track only:

- English single-sentence classification
- Small-label supervised task
- A rewritten dataset built from real short English sentences
- One baseline model: tiny Transformer classifier
- One experimental model: GRU-assisted cloud-state classifier
- Metrics focused on accuracy, F1, learning speed, and subset behavior

Out of scope for this phase:

- Next-token prediction
- Large-scale pretraining
- Long-document hierarchy
- Internal-state visualization as a primary deliverable
- Particle-field or implicit-field variants

## Success Criteria

The first experiment is successful if all of the following are true:

1. The rewritten dataset is reproducible, balanced, and trainable on Colab or local CPU.
2. The tiny Transformer and CFRM prototype both learn above trivial baseline.
3. The evaluation includes at least one controlled subset where late meaning condensation and distractor pressure are measured separately.
4. The outputs are concrete enough to justify either iterating on CFRM or rejecting the current formulation.

## Task Definition

### Input

- One English sentence
- Target sentence length after rewriting: 20 to 40 tokens

### Output

- One class from a small fixed label set
- Recommended first label count: 8 classes

### Core Property of the Task

The final label should depend on sentence-level interpretation rather than one obvious local cue. The sentence should:

- build an early interpretation,
- contain distractor or misleading local signals,
- resolve its final class late in the sentence, and
- require integration of the entire sentence to classify well.

## Dataset Strategy

### Recommended Approach

Use real short English sentences as seeds, then rewrite them into controlled single-sentence examples that introduce:

- late reversal or qualification,
- misleading early cues,
- delayed class resolution, and
- metadata that records the hidden generation structure.

This avoids a purely artificial toy dataset while preserving control over the effect being tested.

### Seed Data

Preferred initial source:

- SST-2 from the GLUE family

Fallback sources, only if SST-2 download or licensing flow blocks progress:

- Rotten Tomatoes sentence-level sentiment data
- Other short single-sentence public datasets with permissive access and simple labels

The seed dataset is only a source of fluent short English clauses. Final labels for the experiment come from the rewrite generator, not the original dataset labels.

### Dataset Builder Components

The data pipeline should have these units:

- `seed_loader`
  - loads real short English seed sentences
  - filters out malformed, too short, too long, or duplicate examples
- `rewrite_templates`
  - applies controlled late-resolution sentence patterns
- `lexicon_pools`
  - provides transition cues, distractor phrases, qualifiers, and topical inserts
- `label_rule`
  - assigns the final class from the latent rewrite scenario
- `quality_filters`
  - removes broken outputs, obvious cue-only outputs, and duplicates
- `dataset_exporter`
  - saves train, validation, test, and challenge splits

### Rewrite Patterns

The first dataset should emphasize two effects:

1. Delayed interpretation
2. Local distraction followed by global correction

Examples of allowed structural patterns:

- early positive framing followed by late qualification
- early negative framing followed by late rescue
- attention drawn to a surface issue while the final clause states the real issue
- apparent endorsement that becomes limited or conditional late in the sentence

The generator should not rely on one fixed connector. It should vary markers such as:

- `but`
- `yet`
- `however`
- `although`
- `even though`
- `despite that`
- `in fact`
- `still`

### Labels

Use 8 classes for the first run. The class set should reflect final sentence-level meaning, not raw sentiment polarity alone.

The first implementation should use these exact class ids:

- `direct_positive`
- `direct_negative`
- `qualified_positive`
- `qualified_negative`
- `hidden_positive`
- `hidden_negative`
- `warning_dominant`
- `uncertainty_dominant`

### Per-Sample Metadata

Each example should store both text and latent generation metadata. Minimum fields:

- `text`
- `label`
- `seed_source`
- `template_id`
- `early_signal`
- `final_signal`
- `reversal_position`
- `distractor_strength`
- `length_tokens`

This metadata is required for subset evaluation and later analysis.

### Split Strategy

- Train: 10k to 30k rewritten samples
- Validation: 2k
- Test: 2k
- Challenge test: 200 to 500 hand-curated or manually reviewed samples

The challenge set should overrepresent hard cases:

- late resolution
- strong distractors
- weak surface cues
- rare rewrite templates

## Model Design

### Baseline

One tiny Transformer classifier modeled on the attention-centric design family from *Attention Is All You Need*, but scaled down to fit local CPU and Colab.

Recommended baseline properties:

- small token embedding
- 2 encoder layers
- small hidden size
- classification head over final pooled state

This is not a reproduction of the original paper's training regime. It is a scaled architectural baseline for the same task.

### Experimental Model

The first CFRM prototype is a classifier, not a generator.

Recommended structure:

1. Token embedding layer
2. One-layer GRU encoder for light local-context accumulation
3. Cloud-state updater that maintains `K` latent semantic components
4. Readout layer that converts the final cloud state into logits

### Cloud State

Represent the global semantic state as `K` latent components. Each component should have at least:

- center vector
- spread or scale parameter
- weight or salience

The implementation does not need to be a physically exact Gaussian density. A Gaussian-like latent mixture is sufficient for the first prototype if it preserves:

- multiple semantic centers,
- soft uncertainty around each center,
- updateable salience over time.

### Update Rule

At token step `t`, the GRU output `h_t` and previous cloud state `S_t` produce an updated state `S_{t+1}`.

Required update behaviors:

- center shift
- spread increase or decrease
- salience reweighting

Optional first-pass behavior:

- periodic lightweight reconfiguration of components

Full component split/merge can be deferred unless the implementation remains simple.

### Readout

The classifier should consume the final cloud state and produce logits over the class set.

Possible readout implementations:

- concatenate weighted component summaries
- pooled weighted mean plus uncertainty features

The readout must remain small and interpretable enough to compare with the baseline.

## Training Plan

### Environment

Supported environments:

- local `venv` with CPU-only PyTorch
- Google Colab with optional GPU

The implementation must avoid assuming persistent GPU access.

### Feasible Scale

The first run must remain small enough to finish within a practical Colab session and remain debuggable locally:

- one shared tokenizer for both models
- preferred tokenizer: whitespace-plus-basic-punctuation tokenization with a learned vocabulary capped for the experiment
- short sequence length
- small batch size as needed
- fixed random seeds

### Objectives

Primary objective:

- multiclass cross-entropy classification

Not included in the first pass:

- auxiliary condensation losses
- reconstruction losses
- contrastive paraphrase alignment losses

Those can be added only if the basic classifier experiment works.

## Evaluation

### Core Metrics

- accuracy
- macro F1
- per-class F1
- training time per epoch

### Subset Metrics

At minimum, report separate performance on:

- late-resolution subset
- high-distractor subset
- challenge subset

These subsets are defined from stored metadata rather than guessed from model outputs.

### Comparison Rules

The baseline and CFRM should be compared under:

- same train/validation/test split
- similar parameter scale where practical
- same tokenizer family if possible
- same number of training epochs or similar early-stopping logic

Perplexity is not a primary metric in this phase.

## Error Handling and Data Quality

The dataset builder should fail fast or skip examples when:

- the rewritten sentence is outside the token-length window
- the output is ungrammatical by simple heuristic checks
- the rewrite leaves the final label ambiguous relative to the latent scenario
- one connector or one lexical cue dominates the class too strongly

The pipeline should log:

- number of examples generated
- number filtered
- class distribution
- template distribution
- duplicate count

## Testing Strategy

Minimum required tests:

- seed filtering produces deterministic output
- rewrite generator produces labels consistent with latent scenario rules
- split generation is reproducible under fixed seed
- metadata fields are complete
- both models can overfit a tiny debug subset
- training loop runs end to end on a very small sample

## Risks

### Risk 1: Cue Leakage

One connector or phrase may become an almost direct label shortcut.

Mitigation:

- vary templates and connectors
- inspect top n-grams by class
- evaluate cue-heavy vs cue-light subsets

### Risk 2: Artificiality

The rewritten data may be too synthetic to support meaningful conclusions.

Mitigation:

- start from real seed sentences
- review a challenge set manually
- keep a path open for later evaluation on less-controlled data

### Risk 3: CFRM Collapse

The cloud components may learn redundant or unstable states.

Mitigation:

- keep `K` small in the first run
- start with only center, spread, and weight updates
- verify tiny-subset overfit before full training

### Risk 4: Over-Scope

Too many model mechanisms in the first pass will slow iteration.

Mitigation:

- no next-token objective
- no document hierarchy
- no visualization-first requirement
- no split/merge complexity unless needed

## Implementation Boundaries

The first implementation should produce:

- a reproducible rewritten dataset generator
- one tiny Transformer baseline
- one GRU-plus-cloud classifier
- one training and evaluation path
- one report of aggregate and subset metrics

The first implementation should not attempt:

- a full semantic-field theory
- large-scale language modeling
- broad benchmark coverage
- polished interpretability tooling

## Expected Outcome

This design should answer a narrow but useful question:

Can a small global-state semantic-cloud prototype remain competitive with a tiny attention baseline on controlled sentence-level meaning tasks where final meaning resolves late and local cues can mislead?

If yes, the next phase can add better cloud dynamics and stronger analysis.
If no, the failure mode should still reveal whether the issue is dataset design, update dynamics, or representational weakness.
