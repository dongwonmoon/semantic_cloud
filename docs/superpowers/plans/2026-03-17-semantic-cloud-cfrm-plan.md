# Semantic Cloud CFRM Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible first experiment that creates a rewritten English single-sentence dataset and compares a tiny Transformer classifier against a GRU-plus-cloud CFRM classifier on 8-way meaning classification.

**Architecture:** The implementation is split into four layers: project scaffolding, dataset generation, model definitions, and training/evaluation. Both models share the same tokenizer, splits, and metrics so the experiment isolates the effect of the semantic-cloud state update rather than accidental tooling differences.

**Tech Stack:** Python 3.11, `venv`, PyTorch, `datasets`, `pytest`, `numpy`, `pandas`

---

## File Structure

Planned files and responsibilities:

- Create: `requirements.txt`
  - runtime dependencies for local `venv` and Colab parity
- Create: `semantic_cloud/__init__.py`
  - package marker
- Create: `semantic_cloud/config.py`
  - dataclasses for dataset, model, and training configuration
- Create: `semantic_cloud/tokenization.py`
  - shared whitespace-plus-punctuation tokenizer, vocabulary builder, encode/decode helpers
- Create: `semantic_cloud/data/__init__.py`
  - package marker
- Create: `semantic_cloud/data/seed_loader.py`
  - load and filter SST-2 seed sentences
- Create: `semantic_cloud/data/rewrite_templates.py`
  - latent rewrite scenarios, templates, connector sampling, metadata assembly
- Create: `semantic_cloud/data/quality.py`
  - duplicate checks, token-length filters, cue leakage heuristics
- Create: `semantic_cloud/data/build_dataset.py`
  - end-to-end dataset builder and split exporter
- Create: `semantic_cloud/models/__init__.py`
  - package marker
- Create: `semantic_cloud/models/transformer_baseline.py`
  - tiny Transformer classifier
- Create: `semantic_cloud/models/cfrm_classifier.py`
  - GRU-plus-cloud classifier with Gaussian-like latent components
- Create: `semantic_cloud/training/__init__.py`
  - package marker
- Create: `semantic_cloud/training/datasets.py`
  - tensor dataset wrappers and collation
- Create: `semantic_cloud/training/metrics.py`
  - accuracy, macro F1, per-class F1, subset aggregation
- Create: `semantic_cloud/training/train.py`
  - train loop, eval loop, checkpoint-free experiment runner
- Create: `scripts/build_dataset.py`
  - CLI entry point for dataset generation
- Create: `scripts/train_experiment.py`
  - CLI entry point for baseline or CFRM training
- Create: `tests/test_tokenization.py`
  - tokenizer behavior and reproducibility tests
- Create: `tests/test_config.py`
  - config defaults and dataclass wiring tests
- Create: `tests/data/test_seed_loader.py`
  - seed filtering tests
- Create: `tests/data/test_rewrite_templates.py`
  - rewrite and label consistency tests
- Create: `tests/data/test_build_dataset.py`
  - split reproducibility and metadata coverage tests
- Create: `tests/models/test_transformer_baseline.py`
  - baseline forward-pass and tiny overfit smoke test
- Create: `tests/models/test_cfrm_classifier.py`
  - CFRM forward-pass and tiny overfit smoke test
- Create: `tests/training/test_train_loop.py`
  - end-to-end one-batch training smoke test
- Create: `tests/training/test_metrics.py`
  - accuracy, macro F1, and subset metric tests

Notes for the implementer:

- This workspace is not currently a Git repository, so commit steps cannot be executed unless the user initializes Git first.
- This workspace also has no existing code layout, so keep files small and avoid premature abstractions.

## Chunk 1: Scaffold And Shared Foundations

### Task 1: Set Up The Local Python Project

**Files:**
- Create: `requirements.txt`
- Create: `semantic_cloud/__init__.py`
- Create: `semantic_cloud/config.py`

- [ ] **Step 1: Write the failing dependency/config test**

```python
from semantic_cloud.config import DataConfig, ModelConfig, TrainConfig


def test_default_configs_are_instantiable():
    data = DataConfig()
    model = ModelConfig(model_type="transformer")
    train = TrainConfig()
    assert data.max_length == 40
    assert model.model_type == "transformer"
    assert train.batch_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL with `ModuleNotFoundError` or missing `semantic_cloud.config`

- [ ] **Step 3: Write minimal implementation**

Implement three small dataclasses:

```python
@dataclass
class DataConfig:
    min_length: int = 20
    max_length: int = 40
    vocab_size: int = 8000
    train_size: int = 12000
    valid_size: int = 2000
    test_size: int = 2000


@dataclass
class ModelConfig:
    model_type: str
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_classes: int = 8
    num_clouds: int = 6


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 8
    seed: int = 7
```

- [ ] **Step 4: Add runtime dependencies**

Add a minimal `requirements.txt` with:

```text
torch
datasets
numpy
pandas
pytest
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

If Git exists:

```bash
git add requirements.txt semantic_cloud/__init__.py semantic_cloud/config.py tests/test_config.py
git commit -m "chore: scaffold semantic cloud experiment config"
```

If Git does not exist: record that commit was skipped because the workspace is not a repository.

### Task 2: Implement The Shared Tokenizer

**Files:**
- Create: `semantic_cloud/tokenization.py`
- Test: `tests/test_tokenization.py`

- [ ] **Step 1: Write the failing tokenizer tests**

```python
from semantic_cloud.tokenization import BasicTokenizer, build_vocab


def test_basic_tokenizer_splits_punctuation():
    tokenizer = BasicTokenizer()
    assert tokenizer.tokenize("Wait, however, this changed.") == [
        "wait", ",", "however", ",", "this", "changed", "."
    ]


def test_build_vocab_is_deterministic():
    vocab = build_vocab(["alpha beta", "beta gamma"], vocab_size=10)
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert "beta" in vocab
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokenization.py -v`  
Expected: FAIL with missing tokenizer module or symbols

- [ ] **Step 3: Write minimal implementation**

Implement:

- `BasicTokenizer.tokenize(text) -> list[str]`
- `build_vocab(texts, vocab_size)`
- `encode(tokens, vocab, max_length)`

Rules:

- lowercase text
- split punctuation into separate tokens
- reserve `<pad>` and `<unk>`
- truncate or pad to fixed length

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokenization.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add semantic_cloud/tokenization.py tests/test_tokenization.py
git commit -m "feat: add shared experiment tokenizer"
```

Skip if Git is unavailable.

## Chunk 2: Dataset Generation

### Task 3: Load And Filter SST-2 Seed Sentences

**Files:**
- Create: `semantic_cloud/data/__init__.py`
- Create: `semantic_cloud/data/seed_loader.py`
- Test: `tests/data/test_seed_loader.py`

- [ ] **Step 1: Write the failing seed-loader tests**

```python
from semantic_cloud.data.seed_loader import filter_seed_sentences


def test_filter_seed_sentences_drops_duplicates_and_bad_lengths():
    rows = [
        {"sentence": "This works well."},
        {"sentence": "This works well."},
        {"sentence": "Bad"},
    ]
    result = filter_seed_sentences(rows, min_tokens=3, max_tokens=8)
    assert result == ["This works well."]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_seed_loader.py -v`  
Expected: FAIL with missing loader implementation

- [ ] **Step 3: Write minimal implementation**

Implement:

- `load_sst2_sentences(split: str) -> list[str]`
- `filter_seed_sentences(rows, min_tokens, max_tokens) -> list[str]`

Rules:

- use the `datasets` package for SST-2
- drop duplicates
- keep only English-like sentences with token count inside bounds
- isolate network access to `load_sst2_sentences`

- [ ] **Step 4: Run the unit test**

Run: `pytest tests/data/test_seed_loader.py -v`  
Expected: PASS

- [ ] **Step 5: Add a manual smoke command**

Run: `python scripts/build_dataset.py --check-seeds-only --limit 20`  
Expected: prints a small sample of filtered seed sentences

- [ ] **Step 6: Commit**

```bash
git add semantic_cloud/data/__init__.py semantic_cloud/data/seed_loader.py tests/data/test_seed_loader.py
git commit -m "feat: load and filter SST-2 seed sentences"
```

Skip if Git is unavailable.

### Task 4: Implement Rewrite Templates And Label Rules

**Files:**
- Create: `semantic_cloud/data/rewrite_templates.py`
- Test: `tests/data/test_rewrite_templates.py`

- [ ] **Step 1: Write the failing rewrite tests**

```python
from semantic_cloud.data.rewrite_templates import rewrite_sentence


def test_rewrite_sentence_returns_text_and_metadata():
    sample = rewrite_sentence("The acting is charming and direct.", seed=7)
    assert "text" in sample
    assert "label" in sample
    assert "template_id" in sample
    assert sample["label"] in {
        "direct_positive",
        "direct_negative",
        "qualified_positive",
        "qualified_negative",
        "hidden_positive",
        "hidden_negative",
        "warning_dominant",
        "uncertainty_dominant",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_rewrite_templates.py -v`  
Expected: FAIL with missing rewrite implementation

- [ ] **Step 3: Write minimal implementation**

Implement:

- latent scenario definitions for the 8 classes
- connector pools and distractor phrase pools
- `rewrite_sentence(seed_text, seed)` that returns:
  - `text`
  - `label`
  - `template_id`
  - `early_signal`
  - `final_signal`
  - `reversal_position`
  - `distractor_strength`

Initial constraints:

- final text should target 20 to 40 tokens
- use multiple connectors, not one fixed connector
- label should come from latent scenario choice, not from string matching on output

- [ ] **Step 4: Add a label consistency test**

Add a test that a forced `qualified_positive` scenario always produces `label == "qualified_positive"` even when distractor phrases are varied.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/data/test_rewrite_templates.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add semantic_cloud/data/rewrite_templates.py tests/data/test_rewrite_templates.py
git commit -m "feat: add rewritten sentence generator"
```

Skip if Git is unavailable.

### Task 5: Add Quality Filters And Dataset Export

**Files:**
- Create: `semantic_cloud/data/quality.py`
- Create: `semantic_cloud/data/build_dataset.py`
- Create: `scripts/build_dataset.py`
- Test: `tests/data/test_build_dataset.py`

- [ ] **Step 1: Write the failing dataset-builder tests**

```python
from semantic_cloud.data.build_dataset import build_splits


def test_build_splits_is_reproducible():
    first = build_splits(seed=11, train_size=20, valid_size=4, test_size=4)
    second = build_splits(seed=11, train_size=20, valid_size=4, test_size=4)
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_build_dataset.py -v`  
Expected: FAIL with missing builder implementation

- [ ] **Step 3: Write minimal implementation**

Implement:

- duplicate detection helper
- length filter helper
- connector-frequency heuristic check
- `build_splits(...)` that:
  - loads filtered seeds
  - rewrites them
  - filters low-quality examples
  - returns train, validation, and test lists with complete metadata
- CLI exporter that writes JSONL files to `artifacts/datasets/semantic_cloud_v1/`

- [ ] **Step 4: Add metadata coverage test**

Add assertions that each exported row includes:

```python
{
    "text",
    "label",
    "seed_source",
    "template_id",
    "early_signal",
    "final_signal",
    "reversal_position",
    "distractor_strength",
    "length_tokens",
}
```

- [ ] **Step 5: Run the unit tests**

Run: `pytest tests/data/test_build_dataset.py -v`  
Expected: PASS

- [ ] **Step 6: Run the dataset build smoke test**

Run: `python scripts/build_dataset.py --train-size 64 --valid-size 16 --test-size 16 --output-dir artifacts/datasets/debug_v1`  
Expected: writes JSONL split files and prints class counts

- [ ] **Step 7: Commit**

```bash
git add semantic_cloud/data/quality.py semantic_cloud/data/build_dataset.py scripts/build_dataset.py tests/data/test_build_dataset.py
git commit -m "feat: build reproducible rewritten dataset"
```

Skip if Git is unavailable.

## Chunk 3: Model Definitions

### Task 6: Implement The Tiny Transformer Baseline

**Files:**
- Create: `semantic_cloud/models/__init__.py`
- Create: `semantic_cloud/models/transformer_baseline.py`
- Test: `tests/models/test_transformer_baseline.py`

- [ ] **Step 1: Write the failing baseline model tests**

```python
import torch

from semantic_cloud.models.transformer_baseline import TinyTransformerClassifier


def test_tiny_transformer_classifier_shape():
    model = TinyTransformerClassifier(vocab_size=100, num_classes=8)
    tokens = torch.randint(0, 100, (4, 32))
    logits = model(tokens)
    assert logits.shape == (4, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_transformer_baseline.py -v`  
Expected: FAIL with missing model implementation

- [ ] **Step 3: Write minimal implementation**

Implement:

- token embedding
- positional embedding
- 2-layer Transformer encoder
- masked mean pooling or final-state pooling
- linear classifier head

Keep dimensions small enough for local CPU smoke tests.

- [ ] **Step 4: Add tiny overfit smoke test**

Add a second test that trains on 8 examples for a few optimizer steps and verifies the loss decreases.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/models/test_transformer_baseline.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add semantic_cloud/models/__init__.py semantic_cloud/models/transformer_baseline.py tests/models/test_transformer_baseline.py
git commit -m "feat: add tiny transformer baseline"
```

Skip if Git is unavailable.

### Task 7: Implement The GRU-Plus-Cloud CFRM Classifier

**Files:**
- Create: `semantic_cloud/models/cfrm_classifier.py`
- Test: `tests/models/test_cfrm_classifier.py`

- [ ] **Step 1: Write the failing CFRM tests**

```python
import torch

from semantic_cloud.models.cfrm_classifier import CFRMClassifier


def test_cfrm_classifier_shape():
    model = CFRMClassifier(vocab_size=100, num_classes=8, num_clouds=6)
    tokens = torch.randint(0, 100, (4, 32))
    logits = model(tokens)
    assert logits.shape == (4, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_cfrm_classifier.py -v`  
Expected: FAIL with missing model implementation

- [ ] **Step 3: Write minimal implementation**

Implement:

- token embedding
- 1-layer GRU encoder
- cloud state tensors for:
  - centers
  - spreads
  - weights
- token-step update that uses the GRU output to adjust center, spread, and salience
- final cloud-state readout into class logits

Keep the first implementation simple:

- no split/merge
- no extra auxiliary loss
- deterministic tensor shapes throughout

- [ ] **Step 4: Add tiny overfit smoke test**

Train on 8 examples for a few optimizer steps and verify the loss decreases.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/models/test_cfrm_classifier.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add semantic_cloud/models/cfrm_classifier.py tests/models/test_cfrm_classifier.py
git commit -m "feat: add GRU-plus-cloud classifier"
```

Skip if Git is unavailable.

## Chunk 4: Training And Experiment Execution

### Task 8: Implement Dataset Wrappers And Metrics

**Files:**
- Create: `semantic_cloud/training/__init__.py`
- Create: `semantic_cloud/training/datasets.py`
- Create: `semantic_cloud/training/metrics.py`
- Test: `tests/training/test_metrics.py`

- [ ] **Step 1: Write the failing metrics tests**

```python
from semantic_cloud.training.metrics import compute_accuracy


def test_compute_accuracy():
    preds = [0, 1, 1, 2]
    labels = [0, 0, 1, 2]
    assert compute_accuracy(preds, labels) == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_metrics.py -v`  
Expected: FAIL with missing metrics implementation

- [ ] **Step 3: Write minimal implementation**

Implement:

- tensor dataset wrapper for encoded text and labels
- batch collation with attention mask
- `compute_accuracy`
- macro F1 without adding extra heavyweight libraries
- subset aggregation by metadata predicates

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_metrics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add semantic_cloud/training/__init__.py semantic_cloud/training/datasets.py semantic_cloud/training/metrics.py tests/training/test_metrics.py
git commit -m "feat: add experiment dataset wrappers and metrics"
```

Skip if Git is unavailable.

### Task 9: Implement The Shared Train Loop

**Files:**
- Create: `semantic_cloud/training/train.py`
- Create: `scripts/train_experiment.py`
- Test: `tests/training/test_train_loop.py`

- [ ] **Step 1: Write the failing train-loop test**

```python
def test_train_one_epoch_returns_metrics(tmp_path):
    metrics = run_debug_experiment(output_dir=tmp_path)
    assert "train_loss" in metrics
    assert "valid_accuracy" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_train_loop.py -v`  
Expected: FAIL with missing train runner

- [ ] **Step 3: Write minimal implementation**

Implement:

- model selection by `model_type`
- one training epoch
- one evaluation epoch
- logging of:
  - train loss
  - validation accuracy
  - validation macro F1
  - subset metrics
- CLI flags for:
  - model type
  - dataset path
  - batch size
  - epochs
  - device

- [ ] **Step 4: Run the train-loop test**

Run: `pytest tests/training/test_train_loop.py -v`  
Expected: PASS

- [ ] **Step 5: Run a local debug training pass**

Run: `python scripts/train_experiment.py --model-type transformer --dataset-dir artifacts/datasets/debug_v1 --epochs 1 --batch-size 8 --device cpu`  
Expected: completes one epoch and prints validation metrics

- [ ] **Step 6: Run a second local debug training pass for CFRM**

Run: `python scripts/train_experiment.py --model-type cfrm --dataset-dir artifacts/datasets/debug_v1 --epochs 1 --batch-size 8 --device cpu`  
Expected: completes one epoch and prints validation metrics

- [ ] **Step 7: Commit**

```bash
git add semantic_cloud/training/train.py scripts/train_experiment.py tests/training/test_train_loop.py
git commit -m "feat: add experiment training runner"
```

Skip if Git is unavailable.

### Task 10: Run The First Comparable Experiment

**Files:**
- Modify: `artifacts/datasets/semantic_cloud_v1/*.jsonl`
- Modify: `artifacts/reports/semantic_cloud_v1_metrics.json`
- Modify: `artifacts/reports/semantic_cloud_v1_summary.md`

- [ ] **Step 1: Build the first trainable dataset**

Run: `python scripts/build_dataset.py --train-size 12000 --valid-size 2000 --test-size 2000 --output-dir artifacts/datasets/semantic_cloud_v1`  
Expected: JSONL files are created and class counts are printed for each split

- [ ] **Step 2: Train the tiny Transformer baseline**

Run: `python scripts/train_experiment.py --model-type transformer --dataset-dir artifacts/datasets/semantic_cloud_v1 --epochs 5 --batch-size 32 --device cpu --report-path artifacts/reports/transformer_metrics.json`  
Expected: training finishes and writes aggregate plus subset metrics

- [ ] **Step 3: Train the CFRM model**

Run: `python scripts/train_experiment.py --model-type cfrm --dataset-dir artifacts/datasets/semantic_cloud_v1 --epochs 5 --batch-size 32 --device cpu --report-path artifacts/reports/cfrm_metrics.json`  
Expected: training finishes and writes aggregate plus subset metrics

- [ ] **Step 4: Write the first comparison summary**

Create `artifacts/reports/semantic_cloud_v1_summary.md` with:

- dataset sizes
- parameter counts for each model
- final accuracy and macro F1
- late-resolution subset results
- high-distractor subset results
- one short note on whether CFRM remains competitive

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add artifacts/reports artifacts/datasets
git commit -m "feat: run first semantic cloud experiment"
```

Skip if Git is unavailable.
