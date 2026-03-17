from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from semantic_cloud.data.build_dataset import build_dataset_source, build_splits, export_splits
from semantic_cloud.data.challenge_sets import load_challenge_rows
from semantic_cloud.models.cfrm_classifier import CFRMClassifier
from semantic_cloud.models.cfrm_philosophy import CFRMPhilosophyClassifier
from semantic_cloud.models.gru_baseline import GRUBaselineClassifier
from semantic_cloud.models.transformer_baseline import TinyTransformerClassifier
from semantic_cloud.training.datasets import (
    ExperimentDataset,
    build_label_mapping,
    build_vocab_from_rows,
    collate_batch,
    load_dataset_metadata,
    load_jsonl_rows,
)
from semantic_cloud.training.metrics import (
    compute_accuracy,
    compute_macro_f1,
    summarize_by_metadata_field,
    summarize_subset,
)


def build_model(model_type: str, vocab_size: int, num_classes: int) -> nn.Module:
    if model_type == "transformer":
        return TinyTransformerClassifier(vocab_size=vocab_size, num_classes=num_classes)
    if model_type == "gru":
        return GRUBaselineClassifier(vocab_size=vocab_size, num_classes=num_classes)
    if model_type == "cfrm":
        return CFRMClassifier(vocab_size=vocab_size, num_classes=num_classes, num_clouds=6)
    if model_type == "cfrm_philosophy":
        return CFRMPhilosophyClassifier(vocab_size=vocab_size, num_classes=num_classes, num_clouds=6)
    if model_type == "cfrm_philosophy_fast":
        return CFRMPhilosophyClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            num_clouds=6,
            sparse_reconfiguration=True,
            reconfiguration_interval=4,
            novelty_threshold=0.6,
        )
    if model_type == "cfrm_philosophy_balanced":
        return CFRMPhilosophyClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            num_clouds=6,
            sparse_reconfiguration=True,
            reconfiguration_interval=2,
            novelty_threshold=0.5,
            always_apply_attractor=True,
        )
    if model_type == "cfrm_philosophy_topk":
        return CFRMPhilosophyClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            num_clouds=6,
            sparse_reconfiguration=True,
            reconfiguration_interval=2,
            novelty_threshold=0.5,
            always_apply_attractor=True,
            interaction_topk=2,
        )
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


def write_state_summary(
    model: nn.Module,
    dataset: ExperimentDataset,
    device: str,
    output_path: str,
    sample_count: int = 8,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    novelty_values: list[float] = []
    entropy_values: list[float] = []
    alpha_max_values: list[float] = []
    uncertainty_values: list[float] = []
    diversity_values: list[float] = []

    limit = min(sample_count, len(dataset))
    for index in range(limit):
        item = dataset[index]
        tokens = item["tokens"].unsqueeze(0).to(device)
        metadata = item["metadata"]
        try:
            output = model(tokens, return_state=True)
        except TypeError:
            output = {"logits": model(tokens)}

        summary_row = {
            "text": metadata["text"],
            "label": metadata["label"],
            "predicted_label_id": int(output["logits"].argmax(dim=-1).item()),
        }
        if "novelty" in output:
            novelty_tensor = output["novelty"].detach().cpu()
            novelty_values.extend(float(value) for value in novelty_tensor.reshape(-1).tolist())
            summary_row["novelty_peak"] = float(novelty_tensor.max().item())
        if "entropy" in output:
            entropy_value = float(output["entropy"].detach().cpu().reshape(-1)[-1].item())
            entropy_values.append(entropy_value)
            summary_row["entropy_final"] = entropy_value
        if "alpha" in output:
            alpha_value = float(output["alpha"].detach().cpu().max().item())
            alpha_max_values.append(alpha_value)
            summary_row["alpha_max_final"] = alpha_value
        if "uncertainty" in output:
            uncertainty_value = float(output["uncertainty"].detach().cpu().reshape(-1)[-1].item())
            uncertainty_values.append(uncertainty_value)
            summary_row["uncertainty_final"] = uncertainty_value
        if "diversity" in output:
            diversity_value = float(output["diversity"].detach().cpu().reshape(-1)[-1].item())
            diversity_values.append(diversity_value)
            summary_row["diversity_final"] = diversity_value
        rows.append(summary_row)

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    payload = {
        "novelty_mean": average(novelty_values),
        "novelty_peak": max(novelty_values) if novelty_values else 0.0,
        "entropy_final": average(entropy_values),
        "alpha_max_final": average(alpha_max_values),
        "uncertainty_final": average(uncertainty_values),
        "diversity_final": average(diversity_values),
        "representative_samples": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_rows(
    model: nn.Module,
    rows: list[dict[str, object]],
    vocab: dict[str, int],
    label_to_id: dict[str, int],
    batch_size: int,
    device: str,
    description: str,
) -> tuple[float, list[int], list[int], list[dict[str, object]]]:
    dataset = ExperimentDataset(rows, vocab=vocab, max_length=40, label_to_id=label_to_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return run_epoch(
        model,
        loader,
        device=device,
        optimizer=None,
        description=description,
    )


def run_experiment(
    model_type: str,
    dataset_dir: str,
    batch_size: int = 8,
    epochs: int = 1,
    device: str = "cpu",
    report_path: str | None = None,
    state_dump_path: str | None = None,
    evaluate_test: bool = False,
    challenge_dir: str | None = None,
    state_summary_path: str | None = None,
    seed: int = 7,
) -> dict[str, object]:
    torch.manual_seed(seed)
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

    if state_summary_path is not None:
        write_state_summary(
            model=model,
            dataset=valid_dataset,
            device=resolved_device,
            output_path=state_summary_path,
        )

    if evaluate_test:
        test_rows = load_jsonl_rows(dataset_dir, "test")
        test_loss, test_preds, test_labels, _ = evaluate_rows(
            model=model,
            rows=test_rows,
            vocab=vocab,
            label_to_id=label_to_id,
            batch_size=batch_size,
            device=resolved_device,
            description="test",
        )
        metrics["test_loss"] = test_loss
        metrics["test_accuracy"] = compute_accuracy(test_preds, test_labels)
        metrics["test_macro_f1"] = compute_macro_f1(test_preds, test_labels, num_classes=num_classes)

    if challenge_dir is not None:
        challenge_rows = load_challenge_rows(challenge_dir, allowed_labels=set(label_to_id.keys()))
        challenge_loss, challenge_preds, challenge_labels, challenge_meta = evaluate_rows(
            model=model,
            rows=challenge_rows,
            vocab=vocab,
            label_to_id=label_to_id,
            batch_size=batch_size,
            device=resolved_device,
            description="challenge",
        )
        metrics["challenge_loss"] = challenge_loss
        metrics["challenge_accuracy"] = compute_accuracy(challenge_preds, challenge_labels)
        metrics["challenge_macro_f1"] = compute_macro_f1(
            challenge_preds,
            challenge_labels,
            num_classes=num_classes,
        )
        metrics["challenge_by_type"] = summarize_by_metadata_field(
            challenge_preds,
            challenge_labels,
            challenge_meta,
            field="challenge_type",
            num_classes=num_classes,
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
