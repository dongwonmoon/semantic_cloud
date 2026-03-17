from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from semantic_cloud.data.decoder_dataset import (
    DecoderDataset,
    build_decoder_vocab_from_rows,
    collate_decoder_batch,
    load_decoder_jsonl_rows,
)
from semantic_cloud.models.cfrm_decoder import CFRMDecoder
from semantic_cloud.models.gru_decoder import GRUDecoder
from semantic_cloud.models.transformer_decoder import TinyTransformerDecoder
from semantic_cloud.tokenization import BasicTokenizer


def build_decoder_model(model_type: str, vocab_size: int) -> nn.Module:
    if model_type == "transformer_decoder":
        return TinyTransformerDecoder(vocab_size=vocab_size)
    if model_type == "gru_decoder":
        return GRUDecoder(vocab_size=vocab_size)
    if model_type == "cfrm_decoder":
        return CFRMDecoder(vocab_size=vocab_size, num_clouds=6, hidden_dim=64)
    raise ValueError(f"Unsupported decoder model_type: {model_type}")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def compute_masked_loss(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)
    masked = losses * loss_mask
    denom = loss_mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def decode_tokens(token_ids: list[int], inverse_vocab: dict[int, str]) -> str:
    tokens = [
        inverse_vocab.get(token_id, "<unk>")
        for token_id in token_ids
        if inverse_vocab.get(token_id, "<unk>") not in {"<pad>", "<bos>", "<eos>"}
    ]
    return " ".join(tokens).strip()


def run_decoder_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    optimizer=None,
    description: str = "decoder",
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    losses: list[float] = []
    token_hits = 0.0
    token_total = 0.0

    iterator = tqdm(dataloader, desc=description, leave=False)
    for batch in iterator:
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        logits = model(tokens)
        loss = compute_masked_loss(logits, labels, loss_mask)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        preds = logits.argmax(dim=-1)
        token_hits += float(((preds == labels).float() * loss_mask).sum().item())
        token_total += float(loss_mask.sum().item())
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = sum(losses) / len(losses) if losses else 0.0
    token_accuracy = token_hits / token_total if token_total else 0.0
    return average_loss, token_accuracy


def greedy_generate_suffix(
    model: nn.Module,
    prefix_text: str,
    vocab: dict[str, int],
    inverse_vocab: dict[int, str],
    device: str,
    max_new_tokens: int,
) -> str:
    tokenizer = BasicTokenizer()
    prefix_tokens = ["<bos>", *tokenizer.tokenize(prefix_text)]
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in prefix_tokens]

    for _ in range(max_new_tokens):
        current = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits = model(current)
        next_id = int(logits[0, -1].argmax().item())
        if inverse_vocab.get(next_id, "<unk>") == "<eos>":
            break
        token_ids.append(next_id)

    generated_ids = token_ids[len(prefix_tokens) :]
    return decode_tokens(generated_ids, inverse_vocab)


def write_sample_generations(
    model: nn.Module,
    rows: list[dict[str, object]],
    vocab: dict[str, int],
    device: str,
    output_path: str,
    sample_count: int = 8,
) -> None:
    inverse_vocab = {idx: token for token, idx in vocab.items()}
    payload = []
    for row in rows[:sample_count]:
        max_new_tokens = max(4, len(BasicTokenizer().tokenize(str(row["target_text"]))) + 2)
        generated = greedy_generate_suffix(
            model,
            prefix_text=str(row["prefix_text"]),
            vocab=vocab,
            inverse_vocab=inverse_vocab,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        payload.append(
            {
                "prefix_text": row["prefix_text"],
                "target_text": row["target_text"],
                "generated_suffix": generated,
                "challenge_type": row.get("challenge_type", ""),
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_decoder_experiment(
    model_type: str,
    dataset_dir: str,
    batch_size: int = 8,
    epochs: int = 1,
    device: str = "cpu",
    evaluate_test: bool = False,
    sample_output_path: str | None = None,
    report_path: str | None = None,
    seed: int = 7,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = resolve_device(device)

    train_rows = load_decoder_jsonl_rows(dataset_dir, "train")
    valid_rows = load_decoder_jsonl_rows(dataset_dir, "valid")
    vocab = build_decoder_vocab_from_rows(train_rows)

    train_dataset = DecoderDataset(train_rows, vocab=vocab, max_length=48)
    valid_dataset = DecoderDataset(valid_rows, vocab=vocab, max_length=48)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_decoder_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_decoder_batch)

    model = build_decoder_model(model_type, vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = 0.0
    for epoch in range(epochs):
        train_loss, _ = run_decoder_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            description=f"decoder_train[{epoch + 1}/{epochs}]",
        )

    valid_loss, valid_token_accuracy = run_decoder_epoch(
        model,
        valid_loader,
        device=device,
        optimizer=None,
        description="decoder_valid",
    )
    metrics: dict[str, object] = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_perplexity": math.exp(min(valid_loss, 20.0)),
        "valid_token_accuracy": valid_token_accuracy,
        "device": device,
    }

    if evaluate_test:
        test_rows = load_decoder_jsonl_rows(dataset_dir, "test")
        test_dataset = DecoderDataset(test_rows, vocab=vocab, max_length=48)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_decoder_batch)
        test_loss, test_token_accuracy = run_decoder_epoch(
            model,
            test_loader,
            device=device,
            optimizer=None,
            description="decoder_test",
        )
        metrics["test_loss"] = test_loss
        metrics["test_perplexity"] = math.exp(min(test_loss, 20.0))
        metrics["test_token_accuracy"] = test_token_accuracy

    if sample_output_path is not None:
        write_sample_generations(
            model,
            valid_rows,
            vocab=vocab,
            device=device,
            output_path=sample_output_path,
        )

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
