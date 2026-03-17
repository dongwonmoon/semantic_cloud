from __future__ import annotations

import json
from pathlib import Path
from random import Random

from semantic_cloud.data.quality import dedupe_rows, keep_length_window
from semantic_cloud.data.rewrite_templates import rewrite_sentence
from semantic_cloud.data.seed_loader import load_sst2_sentences


def build_splits(
    seed: int,
    train_size: int,
    valid_size: int,
    test_size: int,
    seed_sentences: list[str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    rng = Random(seed)
    seeds = list(seed_sentences) if seed_sentences is not None else load_sst2_sentences("train")
    if not seeds:
        raise ValueError("No seed sentences available")

    total_needed = train_size + valid_size + test_size
    rows: list[dict[str, object]] = []
    index = 0
    while len(rows) < total_needed:
        sentence = seeds[index % len(seeds)]
        local_seed = seed + index
        rows.append(rewrite_sentence(sentence, seed=local_seed))
        index += 1

    rows = dedupe_rows(rows)
    rows = keep_length_window(rows, min_tokens=20, max_tokens=40)

    if len(rows) < total_needed:
        while len(rows) < total_needed:
            sentence = seeds[index % len(seeds)]
            local_seed = seed + index + rng.randint(0, 999)
            candidate = rewrite_sentence(sentence, seed=local_seed)
            if 20 <= int(candidate["length_tokens"]) <= 40:
                rows.append(candidate)
            index += 1

    rows = rows[:total_needed]
    return {
        "train": rows[:train_size],
        "valid": rows[train_size : train_size + valid_size],
        "test": rows[train_size + valid_size : total_needed],
    }


def export_splits(splits: dict[str, list[dict[str, object]]], output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        split_path = path / f"{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
