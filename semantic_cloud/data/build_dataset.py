from __future__ import annotations

import json
from pathlib import Path
from random import Random

from tqdm.auto import tqdm

from semantic_cloud.data.quality import dedupe_rows, keep_length_window
from semantic_cloud.data.public_datasets import load_dynasent_splits
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
    with tqdm(total=total_needed, desc="rewrite", leave=False) as progress:
        while len(rows) < total_needed:
            sentence = seeds[index % len(seeds)]
            local_seed = seed + index
            rows.append(rewrite_sentence(sentence, seed=local_seed))
            index += 1
            progress.update(1)

    rows = dedupe_rows(rows)
    rows = keep_length_window(rows, min_tokens=20, max_tokens=40)

    if len(rows) < total_needed:
        with tqdm(total=total_needed - len(rows), desc="refill", leave=False) as progress:
            while len(rows) < total_needed:
                sentence = seeds[index % len(seeds)]
                local_seed = seed + index + rng.randint(0, 999)
                candidate = rewrite_sentence(sentence, seed=local_seed)
                if 20 <= int(candidate["length_tokens"]) <= 40:
                    rows.append(candidate)
                    progress.update(1)
                index += 1

    rows = rows[:total_needed]
    return {
        "train": rows[:train_size],
        "valid": rows[train_size : train_size + valid_size],
        "test": rows[train_size + valid_size : total_needed],
    }


def build_dataset_source(
    dataset_source: str,
    seed: int,
    train_size: int,
    valid_size: int,
    test_size: int,
) -> dict[str, list[dict[str, object]]]:
    if dataset_source == "semantic_cloud":
        return build_splits(
            seed=seed,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
        )
    if dataset_source == "dynasent":
        return load_dynasent_splits()
    raise ValueError(f"Unsupported dataset_source: {dataset_source}")


def export_splits(
    splits: dict[str, list[dict[str, object]]],
    output_dir: str,
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for split_name, rows in tqdm(splits.items(), desc="export", leave=False):
        split_path = path / f"{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    if metadata is not None:
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
