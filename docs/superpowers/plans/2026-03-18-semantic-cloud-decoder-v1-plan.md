# Semantic Cloud Decoder V1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a toy causal language modeling experiment around prefix-to-completion generation so `tiny Transformer`, `GRU`, and `dense CFRM` decoders can be compared on late-resolution suffix generation.

**Architecture:** Add a small decoder data pipeline derived from the existing reinterpretation challenge patterns, then implement three decoder models behind one shared training and evaluation path. Keep the first version narrow: teacher-forced suffix prediction, short sequences, compact metrics, and held-out completion outputs written under `artifacts/`.

**Tech Stack:** Python 3.12, PyTorch, pytest, tqdm, JSONL artifacts, existing shared `.venv`

---

## File Structure

Planned files and responsibilities:

- Create: `semantic_cloud/data/decoder_dataset.py`
  - build and load prefix-to-suffix causal LM rows and metadata
- Create: `scripts/build_decoder_dataset.py`
  - CLI for generating the toy decoder corpus under `artifacts/datasets/`
- Create: `semantic_cloud/models/transformer_decoder.py`
  - tiny causal Transformer decoder baseline
- Create: `semantic_cloud/models/gru_decoder.py`
  - GRU decoder baseline
- Create: `semantic_cloud/models/cfrm_decoder.py`
  - dense causal CFRM decoder with local GRU path plus cloud-state readout
- Create: `semantic_cloud/training/decoder_train.py`
  - shared decoder train/eval loop, generation helper, and metrics
- Create: `scripts/train_decoder_experiment.py`
  - single-run CLI for one decoder experiment
- Create: `scripts/run_decoder_suite.py`
  - optional multi-seed CLI once single runs are stable
- Create: `tests/data/test_decoder_dataset.py`
  - dataset build/load and masking tests
- Create: `tests/models/test_decoder_models.py`
  - forward-shape and generation smoke tests for all decoders
- Create: `tests/training/test_decoder_train.py`
  - train-loop, metrics, and artifact-output tests

## Chunk 1: Decoder Dataset

### Task 1: Add a prefix-to-suffix dataset format

**Files:**
- Create: `semantic_cloud/data/decoder_dataset.py`
- Create: `tests/data/test_decoder_dataset.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:

```python
def test_build_decoder_row_splits_prefix_and_target():
    row = build_decoder_row(
        text="It sounded kind at first, but the final offer was plainly exploitative.",
        label="warning_dominant",
        challenge_type="late_reversal",
    )
    assert row["prefix_text"]
    assert row["target_text"]
    assert row["resolution_position"] > 0
```

```python
def test_decoder_dataset_returns_input_ids_labels_and_loss_mask():
    item = dataset[0]
    assert item["tokens"].shape[0] == item["labels"].shape[0]
    assert item["loss_mask"].shape[0] == item["tokens"].shape[0]
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/data/test_decoder_dataset.py -q -s
```

Expected: FAIL because the decoder dataset module and helpers do not exist yet.

- [ ] **Step 3: Implement the minimal dataset layer**

Create a dataset module that:

- converts reinterpretation rows into `prefix_text` and `target_text`
- tokenizes them with the existing simple vocabulary approach
- returns `tokens`, `labels`, `loss_mask`, and metadata
- keeps training loss on suffix tokens only

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same `pytest` command.  
Expected: PASS

### Task 2: Add a decoder dataset builder CLI

**Files:**
- Create: `scripts/build_decoder_dataset.py`
- Modify: `tests/data/test_decoder_dataset.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds a tiny dataset into a temp directory and asserts:

```python
def test_build_decoder_dataset_writes_train_valid_test(tmp_path):
    build_decoder_dataset(output_dir=tmp_path, train_size=64, valid_size=16, test_size=16)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "valid.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run the same decoder-dataset test command.  
Expected: FAIL because the builder entry point does not exist yet.

- [ ] **Step 3: Implement the minimal builder**

Build a small corpus from the current reinterpretation templates and write:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `metadata.json`

Keep sequence lengths short and make the output Colab-friendly under `artifacts/datasets/`.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same `pytest` command.  
Expected: PASS

## Chunk 2: Baseline Decoders

### Task 3: Add a tiny Transformer decoder baseline

**Files:**
- Create: `semantic_cloud/models/transformer_decoder.py`
- Create: `tests/models/test_decoder_models.py`

- [ ] **Step 1: Write the failing test**

Add a test that asserts:

```python
def test_transformer_decoder_returns_vocab_logits():
    model = TinyTransformerDecoder(vocab_size=128)
    tokens = torch.randint(0, 128, (4, 20))
    logits = model(tokens)
    assert logits.shape == (4, 20, 128)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/models/test_decoder_models.py -q -s
```

Expected: FAIL because the decoder model does not exist yet.

- [ ] **Step 3: Implement the minimal causal Transformer decoder**

Add:

- token embedding
- positional embedding
- causal mask
- 2-layer Transformer encoder stack used causally
- vocabulary projection head

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same test command.  
Expected: PASS

### Task 4: Add a GRU decoder baseline

**Files:**
- Create: `semantic_cloud/models/gru_decoder.py`
- Modify: `tests/models/test_decoder_models.py`

- [ ] **Step 1: Write the failing test**

Add:

```python
def test_gru_decoder_returns_vocab_logits():
    model = GRUDecoder(vocab_size=128)
    tokens = torch.randint(0, 128, (4, 20))
    logits = model(tokens)
    assert logits.shape == (4, 20, 128)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run the same decoder-model test command.  
Expected: FAIL because the GRU decoder does not exist yet.

- [ ] **Step 3: Implement the minimal GRU decoder**

Add:

- token embedding
- 1-layer GRU
- hidden-to-vocab projection

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same `pytest` command.  
Expected: PASS

## Chunk 3: Dense CFRM Decoder

### Task 5: Add a causal CFRM decoder

**Files:**
- Create: `semantic_cloud/models/cfrm_decoder.py`
- Modify: `tests/models/test_decoder_models.py`

- [ ] **Step 1: Write the failing test**

Add:

```python
def test_cfrm_decoder_returns_vocab_logits():
    model = CFRMDecoder(vocab_size=128, num_clouds=4, hidden_dim=64)
    tokens = torch.randint(0, 128, (4, 20))
    logits = model(tokens)
    assert logits.shape == (4, 20, 128)
```

```python
def test_cfrm_decoder_can_return_state():
    model = CFRMDecoder(vocab_size=128, num_clouds=4, hidden_dim=64)
    tokens = torch.randint(0, 128, (2, 12))
    output = model(tokens, return_state=True)
    assert output["logits"].shape == (2, 12, 128)
    assert "core" in output
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run the same decoder-model test command.  
Expected: FAIL because the CFRM decoder does not exist yet.

- [ ] **Step 3: Implement the minimal dense decoder**

Use the encoder-side philosophy model as a guide, but keep the first decoder narrow:

- token embedding
- local 1-layer GRU path
- per-step cloud update
- `core`, `strongest_cloud`, and uncertainty-style readout features
- vocabulary head over `[local_hidden, core, strongest_cloud, uncertainty]`

Do not add sparse scheduling, split, or merge in v1.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same `pytest` command.  
Expected: PASS

## Chunk 4: Shared Training and Evaluation

### Task 6: Add the decoder training loop

**Files:**
- Create: `semantic_cloud/training/decoder_train.py`
- Create: `tests/training/test_decoder_train.py`

- [ ] **Step 1: Write the failing tests**

Add tests that assert:

```python
def test_run_decoder_experiment_reports_loss_and_perplexity(tmp_path):
    metrics = run_decoder_experiment(
        model_type="gru_decoder",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
    )
    assert "valid_loss" in metrics
    assert "valid_perplexity" in metrics
```

```python
def test_run_decoder_experiment_writes_sample_generations(tmp_path):
    run_decoder_experiment(..., sample_output_path=str(path))
    assert path.exists()
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests/training/test_decoder_train.py -q -s
```

Expected: FAIL because the decoder training module does not exist yet.

- [ ] **Step 3: Implement the minimal training loop**

Add:

- model builder for `transformer_decoder`, `gru_decoder`, `cfrm_decoder`
- teacher-forced suffix prediction
- masked loss over suffix tokens only
- validation/test loss and perplexity
- token accuracy over suffix
- ending-clause exact match
- sample completion output writer

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same `pytest` command.  
Expected: PASS

### Task 7: Add single-run decoder CLI

**Files:**
- Create: `scripts/train_decoder_experiment.py`
- Modify: `tests/training/test_decoder_train.py`

- [ ] **Step 1: Write the failing test**

Add a parser test:

```python
def test_train_decoder_experiment_parse_args_accepts_decoder_models():
    args = parse_args([
        "--model-type", "cfrm_decoder",
        "--dataset-dir", "artifacts/datasets/decoder_v1",
    ])
    assert args.model_type == "cfrm_decoder"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run the same decoder-train test command.  
Expected: FAIL because the CLI does not exist yet.

- [ ] **Step 3: Implement the CLI**

Expose:

- `--model-type`
- `--dataset-dir`
- `--epochs`
- `--batch-size`
- `--device`
- `--evaluate-test`
- `--sample-output-path`
- `--state-summary-path`

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same `pytest` command.  
Expected: PASS

## Chunk 5: Colab Execution Path

### Task 8: Add a simple decoder suite runner

**Files:**
- Create: `scripts/run_decoder_suite.py`
- Modify: `tests/training/test_decoder_train.py`

- [ ] **Step 1: Write the failing test**

Add a parser or orchestration test that asserts:

```python
def test_run_decoder_suite_accepts_multiple_seeds(tmp_path):
    args = parse_suite_args([
        "--model-type", "gru_decoder",
        "--dataset-dir", "artifacts/datasets/decoder_v1",
        "--seeds", "7", "11",
        "--output-dir", str(tmp_path),
    ])
    assert args.seeds == [7, 11]
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run the same decoder-train test command.  
Expected: FAIL because the suite CLI does not exist yet.

- [ ] **Step 3: Implement the minimal suite runner**

Reuse the encoder experiment-runner pattern:

- one subdirectory per seed
- one `report.json` per run
- one `summary.json` for aggregated metrics

Keep this lightweight. If single-run execution reveals the decoder is unstable or too slow, this step can be deferred behind a feature flag.

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run the same `pytest` command.  
Expected: PASS

## Chunk 6: Final Verification

### Task 9: Run the full decoder test suite

**Files:**
- Verify: `tests/data/test_decoder_dataset.py`
- Verify: `tests/models/test_decoder_models.py`
- Verify: `tests/training/test_decoder_train.py`
- Verify: existing regression tests as needed

- [ ] **Step 1: Run the focused decoder tests**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest \
  tests/data/test_decoder_dataset.py \
  tests/models/test_decoder_models.py \
  tests/training/test_decoder_train.py \
  -q -s
```

Expected: PASS

- [ ] **Step 2: Run the full test suite**

Run:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python -m pytest tests -q -s
```

Expected: PASS

- [ ] **Step 3: Run one Colab-equivalent smoke command**

Build data:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python scripts/build_decoder_dataset.py \
  --train-size 512 --valid-size 128 --test-size 128 \
  --output-dir artifacts/datasets/decoder_v1_debug
```

Train one model:

```bash
PYTHONPATH=. /mnt/c/Users/anseh/Workspace/semantic_cloud/.venv/bin/python scripts/train_decoder_experiment.py \
  --model-type gru_decoder \
  --dataset-dir artifacts/datasets/decoder_v1_debug \
  --epochs 1 \
  --batch-size 16 \
  --device cpu \
  --evaluate-test \
  --sample-output-path artifacts/reports/decoder_v1_debug_samples.json
```

Expected:

- dataset files created
- one decoder run completes
- report contains validation and test metrics
- sample outputs file exists

- [ ] **Step 4: Commit**

```bash
git add semantic_cloud/data/decoder_dataset.py \
  scripts/build_decoder_dataset.py \
  semantic_cloud/models/transformer_decoder.py \
  semantic_cloud/models/gru_decoder.py \
  semantic_cloud/models/cfrm_decoder.py \
  semantic_cloud/training/decoder_train.py \
  scripts/train_decoder_experiment.py \
  scripts/run_decoder_suite.py \
  tests/data/test_decoder_dataset.py \
  tests/models/test_decoder_models.py \
  tests/training/test_decoder_train.py
git commit -m "feat(decoder): add toy causal LM experiment"
```
