from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from semantic_cloud.data.build_dataset import build_dataset_source, build_splits, export_splits
from semantic_cloud.models.cfrm_classifier import CFRMClassifier
from semantic_cloud.models.cfrm_philosophy import CFRMPhilosophyClassifier
from semantic_cloud.models.transformer_baseline import TinyTransformerClassifier
from semantic_cloud.training.datasets import (
    ExperimentDataset,
    build_label_mapping,
    build_vocab_from_rows,
    collate_batch,
    load_dataset_metadata,
    load_jsonl_rows,
)
from semantic_cloud.training.metrics import compute_accuracy, compute_macro_f1, summarize_subset


def build_model(model_type: str, vocab_size: int, num_classes: int) -> nn.Module:
    if model_type == "transformer":
        return TinyTransformerClassifier(vocab_size=vocab_size, num_classes=num_classes)
    if model_type == "cfrm":
        return CFRMClassifier(vocab_size=vocab_size, num_classes=num_classes, num_clouds=6)
    if model_type == "cfrm_philosophy":
        return CFRMPhilosophyClassifier(vocab_size=vocab_size, num_classes=num_classes, num_clouds=6)
    raise ValueError(f"Unsupported model_type: {model_type}")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    optimizer=None,
    description: str = "epoch",
) -> tuple[float, list[int], list[int], list[dict[str, object]]]:
    loss_fn = nn.CrossEntropyLoss()
    is_training = optimizer is not None
    model.train(is_training)

    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []

    iterator = tqdm(dataloader, desc=description, leave=False)
    for batch in iterator:
        tokens = batch["tokens"].to(device)
        batch_labels = batch["labels"].to(device)
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
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = sum(losses) / len(losses) if losses else 0.0
    return average_loss, preds, labels, metadata


def dump_validation_state(
    model: nn.Module,
    dataset: ExperimentDataset,
    device: str,
    output_path: str,
    sample_count: int = 8,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    limit = min(sample_count, len(dataset))
    for index in range(limit):
        item = dataset[index]
        tokens = item["tokens"].unsqueeze(0).to(device)
        metadata = item["metadata"]
        if hasattr(model, "forward"):
            try:
                output = model(tokens, return_state=True)
            except TypeError:
                output = {"logits": model(tokens)}
        else:
            output = {"logits": model(tokens)}

        row = {
            "text": metadata["text"],
            "label": metadata["label"],
            "predicted_label_id": int(output["logits"].argmax(dim=-1).item()),
        }
        for key in ("alpha", "core", "entropy", "novelty", "uncertainty", "diversity", "energy"):
            if key in output:
                row[key] = output[key].detach().cpu().tolist()
        rows.append(row)

    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run_experiment(
    model_type: str,
    dataset_dir: str,
    batch_size: int = 8,
    epochs: int = 1,
    device: str = "cpu",
    report_path: str | None = None,
    state_dump_path: str | None = None,
) -> dict[str, object]:
    train_rows = load_jsonl_rows(dataset_dir, "train")
    valid_rows = load_jsonl_rows(dataset_dir, "valid")
    metadata = load_dataset_metadata(dataset_dir)
    vocab = build_vocab_from_rows(train_rows)
    label_to_id = metadata.get("label_to_id") or build_label_mapping(train_rows + valid_rows)
    num_classes = len(label_to_id)
    resolved_device = resolve_device(device)

    train_dataset = ExperimentDataset(train_rows, vocab=vocab, max_length=40, label_to_id=label_to_id)
    valid_dataset = ExperimentDataset(valid_rows, vocab=vocab, max_length=40, label_to_id=label_to_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = build_model(model_type=model_type, vocab_size=len(vocab), num_classes=num_classes)
    model.to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = 0.0
    for epoch_index in range(epochs):
        train_loss, _, _, _ = run_epoch(
            model,
            train_loader,
            device=resolved_device,
            optimizer=optimizer,
            description=f"train[{epoch_index + 1}/{epochs}]",
        )

    valid_loss, preds, labels, metadata_rows = run_epoch(
        model,
        valid_loader,
        device=resolved_device,
        optimizer=None,
        description="valid",
    )
    metrics = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_accuracy": compute_accuracy(preds, labels),
        "valid_macro_f1": compute_macro_f1(preds, labels, num_classes=num_classes),
        "device": resolved_device,
        "num_classes": num_classes,
        "late_resolution": summarize_subset(
            preds,
            labels,
            metadata_rows,
            predicate=lambda row: int(row["reversal_position"]) >= 12,
            num_classes=num_classes,
        ),
        "high_distractor": summarize_subset(
            preds,
            labels,
            metadata_rows,
            predicate=lambda row: float(row["distractor_strength"]) >= 0.7,
            num_classes=num_classes,
        ),
    }

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if state_dump_path is not None:
        dump_validation_state(
            model=model,
            dataset=valid_dataset,
            device=resolved_device,
            output_path=state_dump_path,
        )

    return metrics


def run_debug_experiment(output_dir: str | Path) -> dict[str, object]:
    output_path = Path(output_dir)
    dataset_dir = output_path / "debug_dataset"
    seed_sentences = [f"Seed sentence {idx} works well." for idx in range(32)]
    splits = build_splits(
        seed=7,
        train_size=16,
        valid_size=8,
        test_size=8,
        seed_sentences=seed_sentences,
    )
    label_to_id = build_label_mapping(splits["train"] + splits["valid"])
    export_splits(
        splits,
        str(dataset_dir),
        metadata={"dataset_source": "semantic_cloud", "label_to_id": label_to_id},
    )
    return run_experiment(
        model_type="transformer",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
    )
