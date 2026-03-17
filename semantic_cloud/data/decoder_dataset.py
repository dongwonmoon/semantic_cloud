from __future__ import annotations

import json
from pathlib import Path
from random import Random

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from semantic_cloud.data.rewrite_templates import CONNECTORS, rewrite_sentence
from semantic_cloud.data.seed_loader import load_sst2_sentences
from semantic_cloud.tokenization import BasicTokenizer


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def build_decoder_row(
    text: str,
    label: str,
    challenge_type: str,
    template_id: str | None = None,
    early_signal: str | None = None,
    final_signal: str | None = None,
    distractor_strength: float = 0.0,
) -> dict[str, object]:
    stripped = text.strip()
    lowered = stripped.lower()
    prefix_text = stripped
    target_text = stripped
    resolution_position = 1

    for connector in CONNECTORS:
        needle = f" {connector} "
        found = lowered.find(needle)
        if found >= 0:
            split_at = found + len(needle)
            prefix_text = stripped[:split_at].strip()
            target_text = stripped[split_at:].strip()
            resolution_position = len(BasicTokenizer().tokenize(prefix_text))
            break
    else:
        tokens = BasicTokenizer().tokenize(stripped)
        split_index = max(1, int(len(tokens) * 0.6))
        prefix_tokens = tokens[:split_index]
        target_tokens = tokens[split_index:]
        prefix_text = " ".join(prefix_tokens).strip()
        target_text = " ".join(target_tokens).strip()
        resolution_position = len(prefix_tokens)

    return {
        "text": stripped,
        "label": label,
        "challenge_type": challenge_type,
        "prefix_text": prefix_text,
        "target_text": target_text,
        "resolution_position": resolution_position,
        "template_id": template_id or f"decoder_{challenge_type}",
        "early_signal": early_signal or "mixed",
        "final_signal": final_signal or label,
        "distractor_strength": float(distractor_strength),
    }


def build_decoder_splits(
    seed: int,
    train_size: int,
    valid_size: int,
    test_size: int,
    seed_sentences: list[str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    seeds = list(seed_sentences) if seed_sentences is not None else load_sst2_sentences("train")
    if not seeds:
        raise ValueError("No seed sentences available for decoder dataset")

    total_needed = train_size + valid_size + test_size
    rows: list[dict[str, object]] = []
    rng = Random(seed)
    index = 0
    with tqdm(total=total_needed, desc="decoder_rewrite", leave=False) as progress:
        while len(rows) < total_needed:
            sentence = seeds[index % len(seeds)]
            rewritten = rewrite_sentence(sentence, seed=seed + index + rng.randint(0, 999))
            rows.append(
                build_decoder_row(
                    text=str(rewritten["text"]),
                    label=str(rewritten["final_signal"]),
                    challenge_type=str(rewritten.get("template_id", "decoder")),
                    template_id=str(rewritten.get("template_id", "decoder")),
                    early_signal=str(rewritten.get("early_signal", "mixed")),
                    final_signal=str(rewritten.get("final_signal", "mixed")),
                    distractor_strength=float(rewritten.get("distractor_strength", 0.0)),
                )
            )
            index += 1
            progress.update(1)

    return {
        "train": rows[:train_size],
        "valid": rows[train_size : train_size + valid_size],
        "test": rows[train_size + valid_size : total_needed],
    }


def build_decoder_vocab_from_rows(
    rows: list[dict[str, object]],
    vocab_size: int = 8000,
) -> dict[str, int]:
    counter: dict[str, int] = {}
    tokenizer = BasicTokenizer()
    for row in rows:
        for field in ("prefix_text", "target_text"):
            for token in tokenizer.tokenize(str(row[field])):
                counter[token] = counter.get(token, 0) + 1

    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if len(vocab) >= vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_decoder_example(
    row: dict[str, object],
    vocab: dict[str, int],
    max_length: int,
) -> tuple[list[int], list[int], list[int]]:
    tokenizer = BasicTokenizer()
    prefix_tokens = tokenizer.tokenize(str(row["prefix_text"]))
    target_tokens = tokenizer.tokenize(str(row["target_text"]))
    full_tokens = ["<bos>", *prefix_tokens, *target_tokens, "<eos>"]
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in full_tokens]
    input_ids = token_ids[:-1][:max_length]
    label_ids = token_ids[1:][:max_length]

    prefix_len = 1 + len(prefix_tokens)
    mask = []
    for label_index in range(len(label_ids)):
        mask.append(1 if label_index >= max(prefix_len - 1, 0) else 0)

    if len(input_ids) < max_length:
        pad_count = max_length - len(input_ids)
        input_ids.extend([vocab["<pad>"]] * pad_count)
        label_ids.extend([vocab["<pad>"]] * pad_count)
        mask.extend([0] * pad_count)

    return input_ids, label_ids, mask


class DecoderDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], vocab: dict[str, int], max_length: int = 48):
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        tokens, labels, loss_mask = encode_decoder_example(row, self.vocab, self.max_length)
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
            "metadata": row,
        }


def collate_decoder_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "loss_mask": torch.stack([item["loss_mask"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }


def export_decoder_splits(
    splits: dict[str, list[dict[str, object]]],
    output_dir: str,
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        split_path = path / f"{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    if metadata is not None:
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


def load_decoder_jsonl_rows(dataset_dir: str, split: str) -> list[dict[str, object]]:
    path = Path(dataset_dir) / f"{split}.jsonl"
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows
