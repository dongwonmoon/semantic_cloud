# Semantic Cloud Experiment Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible multi-seed evaluation, held-out test evaluation, a fixed reinterpretation challenge set, and compact philosophy-state summaries so the current encoder experiments can support the next decoder spec.

**Architecture:** Keep `scripts/train_experiment.py` and `semantic_cloud/training/train.py` as the single-run core, then add a separate orchestration layer for repeated runs and aggregation. Store the challenge set as checked-in JSONL data, reuse the existing dataset/tensor pipeline for evaluation, and reduce raw philosophy state dumps into compact summary artifacts instead of adding heavy visualization code.

**Tech Stack:** Python 3.12, PyTorch, pytest, tqdm, JSONL artifacts, existing shared `.venv`

---

## File Structure

Planned files and responsibilities:

- Modify: `semantic_cloud/training/train.py`
  - add split-aware evaluation, challenge-set evaluation, and state-summary reduction hooks
- Modify: `scripts/train_experiment.py`
  - expose new single-run options for test/challenge/state summary paths
- Create: `semantic_cloud/training/experiment_runner.py`
  - multi-seed orchestration and aggregation helpers
- Create: `scripts/run_experiment_suite.py`
  - CLI entry point for multi-seed runs
- Create: `semantic_cloud/data/challenge_sets.py`
  - load fixed challenge JSONL rows and validate required metadata
- Create: `artifacts/challenge_sets/meaning_reinterpretation_v1.jsonl`
  - small hand-authored reinterpretation probe set
- Modify: `semantic_cloud/training/metrics.py`
  - per-split and per-challenge-type summary helpers
- Create: `tests/data/test_challenge_sets.py`
  - challenge-set loading and schema tests
- Modify: `tests/training/test_train_loop.py`
  - test split evaluation, challenge evaluation, and state-summary output
- Create: `tests/training/test_experiment_runner.py`
  - aggregation math and suite output tests

## Chunk 1: Single-Run Evaluation Hardening

### Task 1: Add split-aware evaluation to the training core

**Files:**
- Modify: `semantic_cloud/training/train.py`
- Modify: `tests/training/test_train_loop.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:

```python
def test_run_experiment_reports_test_metrics(tmp_path):
    metrics = run_experiment(
        model_type="transformer",
        dataset_dir=str(debug_dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
        evaluate_test=True,
    )
    assert "test_accuracy" in metrics
    assert "test_macro_f1" in metrics
```

```python
def test_run_experiment_can_evaluate_a_challenge_dir(tmp_path):
    metrics = run_experiment(
        model_type="transformer",
        dataset_dir=str(debug_dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
        challenge_dir=str(challenge_dir),
    )
    assert "challenge_accuracy" in metrics
    assert "challenge_by_type" in metrics
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/training/test_train_loop.py -q -s
```

Expected: failures because `run_experiment()` does not yet report test or challenge metrics.

- [ ] **Step 3: Implement the minimal train-loop changes**

Update `run_experiment()` so it can:

- load the `test` split when `evaluate_test=True`
- evaluate a separate `challenge_dir` dataset when provided
- return `test_loss`, `test_accuracy`, `test_macro_f1`
- return `challenge_accuracy`, `challenge_macro_f1`, and `challenge_by_type`

Keep training behavior unchanged.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/training/test_train_loop.py -q -s
```

Expected: PASS

### Task 2: Add compact state-summary reduction

**Files:**
- Modify: `semantic_cloud/training/train.py`
- Modify: `tests/training/test_train_loop.py`

- [ ] **Step 1: Write the failing test**

Add a test that writes a state summary artifact and checks:

```python
def test_run_experiment_writes_state_summary(tmp_path):
    summary_path = tmp_path / "state_summary.json"
    run_experiment(
        model_type="cfrm_philosophy",
        dataset_dir=str(debug_dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
        state_summary_path=str(summary_path),
    )
    payload = json.loads(summary_path.read_text())
    assert "novelty_mean" in payload
    assert "representative_samples" in payload
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run the same `pytest ... tests/training/test_train_loop.py -q -s` command.  
Expected: failure because `state_summary_path` does not exist yet.

- [ ] **Step 3: Implement minimal summary reduction**

Add a compact reducer that:

- derives summary statistics from `alpha`, `entropy`, `novelty`, `uncertainty`, and `diversity`
- stores a small fixed number of representative samples
- writes an empty but valid summary for non-stateful models if needed

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/training/test_train_loop.py -q -s
```

Expected: PASS

## Chunk 2: Fixed Challenge Set

### Task 3: Add challenge-set loading and schema validation

**Files:**
- Create: `semantic_cloud/data/challenge_sets.py`
- Create: `tests/data/test_challenge_sets.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:

- challenge rows load from JSONL
- missing `challenge_type` raises a clear error
- only known labels pass through

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/data/test_challenge_sets.py -q -s
```

Expected: FAIL with missing module or missing validation behavior.

- [ ] **Step 3: Implement the loader**

Create a small loader that:

- reads JSONL rows
- validates `text`, `label`, `challenge_type`
- preserves `target_cue_span` and `notes`
- normalizes rows into the same shape expected by `ExperimentDataset`

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same test command.  
Expected: PASS

### Task 4: Check in the first reinterpretation probe set

**Files:**
- Create: `artifacts/challenge_sets/meaning_reinterpretation_v1.jsonl`
- Modify: `tests/data/test_challenge_sets.py`

- [ ] **Step 1: Write the failing fixture-backed test**

Add a test that loads the checked-in file and asserts:

- at least 40 rows exist
- at least 4 challenge types exist
- every row maps to the current classifier label space

- [ ] **Step 2: Run the focused tests and verify they fail**

Run the same `tests/data/test_challenge_sets.py` command.  
Expected: FAIL because the checked-in file does not exist yet.

- [ ] **Step 3: Add the fixed challenge set**

Author 40 to 80 one-sentence examples across:

- `late_reversal`
- `qualified_support`
- `qualified_rejection`
- `apparent_positive_hidden_negative`
- `apparent_negative_hidden_positive`
- `uncertainty_resolution`

Keep labels within the existing semantic-cloud classifier space.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/data/test_challenge_sets.py -q -s
```

Expected: PASS

## Chunk 3: Multi-Seed Suite Runner

### Task 5: Add aggregation helpers

**Files:**
- Create: `semantic_cloud/training/experiment_runner.py`
- Create: `tests/training/test_experiment_runner.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:

- multiple per-run reports aggregate into mean and standard deviation
- challenge-by-type metrics aggregate per type
- empty input raises a clear error

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/training/test_experiment_runner.py -q -s
```

Expected: FAIL with missing runner module.

- [ ] **Step 3: Implement the aggregation layer**

Add helpers that:

- iterate over run directories
- load `report.json`
- aggregate scalar metrics
- aggregate nested `challenge_by_type` metrics
- write `runs.json` and `summary.json`

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same test command.  
Expected: PASS

### Task 6: Add a suite CLI for repeated experiments

**Files:**
- Create: `scripts/run_experiment_suite.py`
- Modify: `tests/training/test_experiment_runner.py`

- [ ] **Step 1: Write the failing CLI test**

Add a test that invokes the suite entry point with a tiny debug dataset and verifies:

- one run directory per seed exists
- `summary.json` is written
- the CLI exits successfully

- [ ] **Step 2: Run the focused tests and verify they fail**

Run the same `tests/training/test_experiment_runner.py` command.  
Expected: FAIL because the CLI does not exist yet.

- [ ] **Step 3: Implement the CLI**

Expose options for:

- `--model-type`
- `--dataset-dir`
- `--seeds`
- `--epochs`
- `--batch-size`
- `--device`
- `--challenge-dir`
- `--output-dir`

The CLI should call the single-run API, write per-seed artifacts, then write an aggregate summary.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/training/test_experiment_runner.py -q -s
```

Expected: PASS

## Chunk 4: CLI Integration And Final Verification

### Task 7: Extend the single-run CLI

**Files:**
- Modify: `scripts/train_experiment.py`
- Modify: `tests/training/test_train_loop.py`

- [ ] **Step 1: Write the failing CLI-level tests**

Add tests that verify `parse_args()` accepts:

- `--evaluate-test`
- `--challenge-dir`
- `--state-summary-path`

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/training/test_train_loop.py -q -s
```

Expected: FAIL because the CLI options are not defined.

- [ ] **Step 3: Implement the CLI changes**

Thread the new options through to `run_experiment()` without changing existing default behavior.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same command.  
Expected: PASS

### Task 8: Run end-to-end verification

**Files:**
- Verify only

- [ ] **Step 1: Run the full test suite**

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests -q -s
```

Expected: all tests pass.

- [ ] **Step 2: Run one tiny suite smoke test**

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python scripts/build_dataset.py --dataset-source dynasent --output-dir artifacts/datasets/dynasent_smoke
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python scripts/run_experiment_suite.py --model-type cfrm_philosophy --dataset-dir artifacts/datasets/dynasent_smoke --seeds 7 11 --epochs 1 --batch-size 32 --device cpu --challenge-dir artifacts/challenge_sets/meaning_reinterpretation_v1.jsonl --output-dir artifacts/reports/smoke_suite
```

Expected:

- dataset build succeeds
- run directories are created
- `artifacts/reports/smoke_suite/summary.json` exists

- [ ] **Step 3: Commit**

```bash
git add scripts/train_experiment.py scripts/run_experiment_suite.py semantic_cloud/data/challenge_sets.py semantic_cloud/training/train.py semantic_cloud/training/experiment_runner.py artifacts/challenge_sets/meaning_reinterpretation_v1.jsonl tests/data/test_challenge_sets.py tests/training/test_train_loop.py tests/training/test_experiment_runner.py
git commit -m "feat: harden semantic cloud experiment evaluation"
```
