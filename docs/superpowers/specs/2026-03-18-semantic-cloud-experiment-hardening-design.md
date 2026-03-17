# Semantic Cloud Experiment Hardening Design

## Goal

Strengthen the current sentence-classification experiment so that CFRM, CFRM-philosophy, and the Transformer baseline can be compared with enough rigor to justify or reject a transition into decoder design.

This phase does not attempt to prove attention replacement. It is a tighter encoder-side validation pass focused on repeatability, held-out generalization, controlled challenge behavior, and inspectable internal-state summaries.

## Scope

This design covers one bounded experiment-hardening cycle:

- repeated runs across multiple random seeds
- validation and test evaluation for all supported classifiers
- a small hand-authored challenge set for meaning reinterpretation cases
- lightweight aggregation of state summaries from CFRM-philosophy
- a single consolidated result bundle that can inform the next decoder spec

Out of scope for this phase:

- causal language modeling
- long-context generation
- attention replacement claims
- statistical significance testing beyond simple mean and standard deviation
- rich visualization dashboards
- automatic challenge-set generation

## Why This Phase Exists

Current results on DynaSent are promising but not decisive. A single validation score is not enough to claim that the philosophy-heavy CFRM variant is meaningfully stronger than the Transformer baseline.

The current comparison must be tightened in four ways:

1. Measure stability across seeds
2. Evaluate on held-out test data, not only validation
3. Probe controlled reinterpretation cases that ordinary public datasets do not isolate
4. Preserve enough internal-state evidence to decide whether decoder work is justified

## Success Criteria

This phase is successful if all of the following are true:

1. A single command can run multi-seed experiments for `transformer`, `cfrm`, and `cfrm_philosophy`.
2. The output contains per-run metrics and aggregated mean and standard deviation for validation, test, and challenge splits.
3. A hand-authored challenge set is evaluated separately and produces per-type metrics.
4. CFRM-philosophy exports compact, comparable state summaries that can be inspected without manual tensor work.
5. The resulting report is concrete enough to support a decoder-spec decision with clear claims and clear limits.

## High-Level Approach

The existing single-run training script remains the base execution path. A new experiment harness will orchestrate repeated runs, collect reports, and optionally evaluate a shared challenge set.

The public dataset path remains DynaSent for the main benchmark. A new fixed challenge set will supplement it with cases that better reflect the semantic-cloud hypothesis:

- late reinterpretation
- qualified endorsement or rejection
- hidden final polarity beneath misleading surface cues
- uncertainty that resolves only after a late clause

The philosophy model's raw state dumps will be reduced into compact summary values and representative examples. This keeps Colab usage simple while still preserving evidence about novelty, cloud concentration, and final-state condensation.

## Architecture

### 1. Single-Run Training Path

The current `train_experiment.py` flow stays intact for one-off runs and Colab smoke tests.

It should be extended only where necessary:

- optional evaluation of the test split
- optional challenge-set evaluation
- optional compact state-summary export

This preserves the current interface for simple experiments while allowing a higher-level runner to reuse the same primitives.

### 2. Multi-Seed Experiment Harness

A new runner will orchestrate repeated experiments across a seed list.

Responsibilities:

- invoke the existing run path with one seed at a time
- store per-run reports in separate output directories
- collect validation, test, and challenge metrics
- aggregate mean and standard deviation across runs
- write a final summary artifact for direct comparison across models

The harness should not contain training logic. It is an orchestration and aggregation layer only.

### 3. Hand-Authored Challenge Set

The challenge set will be stored as a fixed JSONL dataset under version control.

It should be small enough to inspect manually and stable enough to compare across future model revisions. The initial target size is 40 to 80 examples.

Each row should include:

- `text`
- `label`
- `challenge_type`
- `target_cue_span`
- `notes`

Recommended `challenge_type` values:

- `late_reversal`
- `qualified_support`
- `qualified_rejection`
- `apparent_positive_hidden_negative`
- `apparent_negative_hidden_positive`
- `uncertainty_resolution`

The challenge set is a probe, not a training resource. It should never be mixed into train, validation, or test.

### 4. State Summary Export

The existing raw validation state dump is useful for debugging but too awkward for repeated experiments.

This phase should add a compact summary artifact with:

- `novelty_mean`
- `novelty_peak`
- `entropy_final`
- `alpha_max_final`
- `uncertainty_final`
- `diversity_final`
- a small number of representative sample rows

The summary should be generated for the philosophy model at minimum. The implementation may emit no-op or empty summaries for other model types if that keeps the interface consistent.

## Data Flow

### Main Benchmark Path

1. Build or reuse `dynasent_v1`
2. Train one model for one seed
3. Evaluate on validation split
4. Evaluate on test split
5. Optionally evaluate the challenge set
6. Optionally export state summaries
7. Write one report for that run

### Aggregation Path

1. Loop over seeds
2. Write one run directory per seed
3. Load per-run reports
4. Aggregate metrics into summary tables
5. Write:
   - `runs.json`
   - `summary.json`
   - optional markdown summary for quick reading

## Metrics

### Per Run

Required metrics for validation, test, and challenge:

- loss where applicable
- accuracy
- macro F1

For the challenge split, the report should also include per-`challenge_type` accuracy and macro F1.

### Aggregated

For each model and split:

- mean accuracy
- standard deviation of accuracy
- mean macro F1
- standard deviation of macro F1

This is enough for first-pass rigor without overbuilding the statistics layer.

## Challenge Set Authoring Rules

The challenge set should be intentionally human-written and narrow.

Authoring constraints:

- one sentence per example
- fluent English only
- no obvious label words repeated mechanically
- final interpretation should depend on the whole sentence or on a late clause
- labels must align with the existing classifier label space

Examples should be hard for shallow cue matching but still readable and defensible by a human reviewer.

## Colab Requirements

The hardened experiment flow must remain easy to run in Colab.

Requirements:

- simple CLI commands
- no hidden local-only paths
- output written to `artifacts/`
- repeated runs should be resumable or at least naturally partitioned by run directory

The primary supported workflow is:

1. clone or pull the repository
2. build or reuse the public dataset
3. run the multi-seed harness for one model
4. repeat for the other models
5. compare aggregate summaries

## Error Handling

The experiment runner should fail clearly in these cases:

- challenge set path missing
- unsupported split names
- empty aggregation input
- state-summary request for a model run that produced no state output

Failure messages should be brief and actionable because the main execution environment is Colab.

## Testing Strategy

This phase should be verified with focused unit tests and one small end-to-end smoke path.

Required coverage:

- multi-seed aggregation math
- challenge-set loading and evaluation
- per-type challenge summary generation
- state-summary reduction from dumped internal states
- CLI argument handling for the new runner

The smoke path can stay small and CPU-friendly, using the existing miniature dataset approach.

## Deliverables

This phase should produce:

- a reusable multi-seed experiment runner
- a checked-in challenge JSONL file
- aggregated report artifacts
- compact philosophy-state summary artifacts
- documentation or command examples for Colab usage

## Decoder Handoff Criteria

The next decoder design should start only after this phase produces one consolidated comparison table across:

- `transformer`
- `cfrm`
- `cfrm_philosophy`

and includes:

- validation mean and standard deviation
- test mean and standard deviation
- challenge-set metrics
- compact state-summary findings

The decoder design should assume only the following claim if supported:

> A cloud-state encoder can be competitive as a sentence-level meaning integrator and may expose interpretable global-state behavior worth testing in a causal setting.

The decoder design should not assume:

- attention is replaceable in general
- CFRM is better for token-level precision tasks
- sentence-classification gains imply generation gains

## Non-Goals for Decoder Work

When this phase hands off to decoder design, the first decoder should remain narrow:

- small causal language model
- cloud-state persistence across tokens
- stable next-token training on toy or compact text

It should not begin with:

- long-context retrieval benchmarks
- exact-copy superiority claims
- full attention removal claims at scale

That next phase will be a separate spec.
