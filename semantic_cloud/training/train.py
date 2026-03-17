from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from semantic_cloud.data.build_dataset import build_splits, export_splits
from semantic_cloud.models.cfrm_classifier import CFRMClassifier
from semantic_cloud.models.transformer_baseline import TinyTransformerClassifier
from semantic_cloud.training.datasets import (
    ExperimentDataset,
    build_vocab_from_rows,
    collate_batch,
    load_jsonl_rows,
)
from semantic_cloud.training.metrics import compute_accuracy, compute_macro_f1, summarize_subset


def build_model(model_type: str, vocab_size: int, num_classes: int) -> nn.Module:
    if model_type == "transformer":
        return TinyTransformerClassifier(vocab_size=vocab_size, num_classes=num_classes)
    if model_type == "cfrm":
        return CFRMClassifier(vocab_size=vocab_size, num_classes=num_classes, num_clouds=6)
    raise ValueError(f"Unsupported model_type: {model_type}")


def run_epoch(model: nn.Module, dataloader: DataLoader, optimizer=None) -> tuple[float, list[int], list[int], list[dict[str, object]]]:
    loss_fn = nn.CrossEntropyLoss()
    is_training = optimizer is not None
    model.train(is_training)

    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []

    for batch in dataloader:
        tokens = batch["tokens"]
        batch_labels = batch["labels"]
        logits = model(tokens)
        loss = loss_fn(logits, batch_labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds.extend(logits.argmax(dim=-1).tolist())
        labels.extend(batch_labels.tolist())
        metadata.extend(batch["metadata"])

    average_loss = sum(losses) / len(losses) if losses else 0.0
    return average_loss, preds, labels, metadata


def run_experiment(
    model_type: str,
    dataset_dir: str,
    batch_size: int = 8,
    epochs: int = 1,
    report_path: str | None = None,
) -> dict[str, object]:
    train_rows = load_jsonl_rows(dataset_dir, "train")
    valid_rows = load_jsonl_rows(dataset_dir, "valid")
    vocab = build_vocab_from_rows(train_rows)

    train_dataset = ExperimentDataset(train_rows, vocab=vocab, max_length=40)
    valid_dataset = ExperimentDataset(valid_rows, vocab=vocab, max_length=40)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = build_model(model_type=model_type, vocab_size=len(vocab), num_classes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = 0.0
    for _ in range(epochs):
        train_loss, _, _, _ = run_epoch(model, train_loader, optimizer=optimizer)

    valid_loss, preds, labels, metadata = run_epoch(model, valid_loader, optimizer=None)
    metrics = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_accuracy": compute_accuracy(preds, labels),
        "valid_macro_f1": compute_macro_f1(preds, labels, num_classes=8),
        "late_resolution": summarize_subset(
            preds,
            labels,
            metadata,
            predicate=lambda row: int(row["reversal_position"]) >= 12,
            num_classes=8,
        ),
        "high_distractor": summarize_subset(
            preds,
            labels,
            metadata,
            predicate=lambda row: float(row["distractor_strength"]) >= 0.7,
            num_classes=8,
        ),
    }

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def run_debug_experiment(output_dir: str | Path) -> dict[str, object]:
    output_path = Path(output_dir)
    dataset_dir = output_path / "debug_dataset"
    seed_sentences = [f"Seed sentence {idx} works well." for idx in range(32)]
    splits = build_splits(seed=7, train_size=16, valid_size=8, test_size=8, seed_sentences=seed_sentences)
    export_splits(splits, str(dataset_dir))
    return run_experiment(
        model_type="transformer",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
    )
