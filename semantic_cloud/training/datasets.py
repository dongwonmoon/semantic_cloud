from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from semantic_cloud.tokenization import BasicTokenizer, build_vocab, encode


LABEL_TO_ID = {
    "direct_positive": 0,
    "direct_negative": 1,
    "qualified_positive": 2,
    "qualified_negative": 3,
    "hidden_positive": 4,
    "hidden_negative": 5,
    "warning_dominant": 6,
    "uncertainty_dominant": 7,
}


class ExperimentDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], vocab: dict[str, int], max_length: int):
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length
        self.tokenizer = BasicTokenizer()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        tokens = self.tokenizer.tokenize(str(row["text"]))
        encoded = encode(tokens, self.vocab, self.max_length)
        return {
            "tokens": torch.tensor(encoded, dtype=torch.long),
            "label": torch.tensor(LABEL_TO_ID[str(row["label"])], dtype=torch.long),
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


def build_vocab_from_rows(rows: list[dict[str, object]], vocab_size: int = 8000) -> dict[str, int]:
    texts = [str(row["text"]) for row in rows]
    return build_vocab(texts, vocab_size=vocab_size)
