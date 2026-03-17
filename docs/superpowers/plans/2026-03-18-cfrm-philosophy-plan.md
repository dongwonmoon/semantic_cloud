# CFRM Philosophy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a philosophy-oriented CFRM classifier, expose it as a trainable model type, and support optional validation-state dumps for Colab analysis.

**Architecture:** The change adds one new model file, extends the training runner to recognize the new model and export selected internal states, and adds CLI flags so Colab can run the new path without notebook monkey patches. State dumping stays optional and JSON-based to avoid changing the default training flow.

**Tech Stack:** Python 3.12, PyTorch, pytest, JSON

---

## File Structure

- Create: `semantic_cloud/models/cfrm_philosophy.py`
  - philosophy-oriented classifier with `return_state=True`
- Modify: `semantic_cloud/training/train.py`
  - add model selection, optional state-dump extraction, and JSON export
- Modify: `scripts/train_experiment.py`
  - add `cfrm_philosophy` model type and state-dump CLI flag
- Create: `tests/models/test_cfrm_philosophy.py`
  - forward shape, state payload, and tiny loss-reduction tests
- Modify: `tests/training/test_train_loop.py`
  - verify optional state dump export through the debug runner or direct experiment call

## Chunk 1: Model And Runner

### Task 1: Add CFRM Philosophy Model

**Files:**
- Create: `semantic_cloud/models/cfrm_philosophy.py`
- Test: `tests/models/test_cfrm_philosophy.py`

- [ ] **Step 1: Write failing model tests**
- [ ] **Step 2: Run `PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/models/test_cfrm_philosophy.py -v` and verify failure**
- [ ] **Step 3: Implement `CFRMPhilosophyClassifier` with `return_state` support**
- [ ] **Step 4: Re-run the model tests and verify pass**

### Task 2: Wire Runner And State Dumps

**Files:**
- Modify: `semantic_cloud/training/train.py`
- Modify: `scripts/train_experiment.py`
- Modify: `tests/training/test_train_loop.py`

- [ ] **Step 1: Write failing state-dump test**
- [ ] **Step 2: Run `PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/training/test_train_loop.py -v` and verify failure**
- [ ] **Step 3: Add `cfrm_philosophy` model selection plus optional JSON state dump for validation samples**
- [ ] **Step 4: Re-run `tests/training/test_train_loop.py` and verify pass**

## Chunk 2: Verification

### Task 3: Run Full Verification

**Files:**
- Modify: none

- [ ] **Step 1: Run `PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests -q`**
- [ ] **Step 2: Run a local smoke command for `cfrm_philosophy` on the debug dataset**
- [ ] **Step 3: If Git is available, commit the branch changes**
