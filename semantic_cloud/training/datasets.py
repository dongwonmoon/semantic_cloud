from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from semantic_cloud.tokenization import BasicTokenizer, build_vocab, encode


def build_label_mapping(rows: list[dict[str, object]]) -> dict[str, int]:
    labels = sorted({str(row["label"]) for row in rows})
    return {label: index for index, label in enumerate(labels)}


class ExperimentDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, object]],
        vocab: dict[str, int],
        max_length: int,
        label_to_id: dict[str, int],
    ):
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length
        self.label_to_id = label_to_id
        self.tokenizer = BasicTokenizer()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        tokens = self.tokenizer.tokenize(str(row["text"]))
        encoded = encode(tokens, self.vocab, self.max_length)
        return {
            "tokens": torch.tensor(encoded, dtype=torch.long),
            "label": torch.tensor(self.label_to_id[str(row["label"])], dtype=torch.long),
            "metadata": row,
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    tokens = torch.stack([item["tokens"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    attention_mask = (tokens != 0).long()
    metadata = [item["metadata"] for item in batch]
    return {"tokens": tokens, "labels": labels, "attention_mask": attention_mask, "metadata": metadata}


def load_jsonl_rows(dataset_dir: str, split: str) -> list[dict[str, object]]:
    path = Path(dataset_dir) / f"{split}.jsonl"
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_dataset_metadata(dataset_dir: str) -> dict[str, object]:
    path = Path(dataset_dir) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_vocab_from_rows(rows: list[dict[str, object]], vocab_size: int = 8000) -> dict[str, int]:
    texts = [str(row["text"]) for row in rows]
    return build_vocab(texts, vocab_size=vocab_size)
