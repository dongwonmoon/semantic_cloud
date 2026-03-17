# GRU Baseline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parameter-matched bidirectional GRU baseline to the current experiment stack and expose it through the existing local and Colab training commands.

**Architecture:** Introduce one focused GRU classifier module, thread a new `gru` model type through the model builder and CLIs, and add only the smallest test set needed to lock the new path. The model should stay in the Transformer/CFRM parameter band so the comparison remains interpretable.

**Tech Stack:** Python 3.12, PyTorch, pytest

---

## File Structure

- Create: `semantic_cloud/models/gru_baseline.py`
- Modify: `semantic_cloud/training/train.py`
- Modify: `scripts/train_experiment.py`
- Modify: `scripts/run_experiment_suite.py`
- Create: `tests/models/test_gru_baseline.py`
- Modify: `tests/training/test_train_loop.py`

## Task 1: Add GRU model tests first

**Files:**
- Create: `tests/models/test_gru_baseline.py`

- [ ] Add a shape test for `GRUBaselineClassifier`
- [ ] Add a tiny loss-reduction smoke test
- [ ] Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/models/test_gru_baseline.py -q -s`
- [ ] Confirm it fails because the model file does not exist yet

## Task 2: Implement the GRU baseline

**Files:**
- Create: `semantic_cloud/models/gru_baseline.py`

- [ ] Implement a bidirectional GRU classifier with pooled readout
- [ ] Tune hidden width so parameter count lands close to Transformer
- [ ] Re-run: `PYTHONPATH=. .venv/bin/python -m pytest tests/models/test_gru_baseline.py -q -s`
- [ ] Confirm it passes

## Task 3: Thread the model through training and CLIs

**Files:**
- Modify: `semantic_cloud/training/train.py`
- Modify: `scripts/train_experiment.py`
- Modify: `scripts/run_experiment_suite.py`
- Modify: `tests/training/test_train_loop.py`

- [ ] Add a failing CLI/parser test that accepts `--model-type gru`
- [ ] Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/training/test_train_loop.py -q -s`
- [ ] Confirm the new parser assertion fails first
- [ ] Add `gru` to model builders and CLI choices
- [ ] Re-run the same test command and confirm it passes

## Task 4: Full verification and integration

**Files:**
- Verify only

- [ ] Run: `PYTHONPATH=. .venv/bin/python -m pytest tests -q -s`
- [ ] Run a quick parameter-count check for `transformer`, `gru`, `cfrm`, `cfrm_philosophy`
- [ ] Commit with one feature commit
