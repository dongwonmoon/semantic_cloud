from __future__ import annotations

import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path


SUPPORTED_DYNASENT_LABELS = {"positive", "negative", "neutral"}
AG_NEWS_LABELS = {
    0: "world",
    1: "sports",
    2: "business",
    3: "sci_tech",
}


def normalize_dynasent_rows(
    rows: list[dict[str, object]],
    split_name: str,
    source_name: str,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        label = row.get("gold_label")
        sentence = str(row.get("sentence", "")).strip()
        if label not in SUPPORTED_DYNASENT_LABELS or not sentence:
            continue
        normalized.append(
            {
                "text": sentence,
                "label": str(label),
                "seed_source": source_name,
                "template_id": f"{source_name}_{split_name}",
                "early_signal": "public",
                "final_signal": "public",
                "reversal_position": 0,
                "distractor_strength": 0.0,
                "length_tokens": len(sentence.split()),
                "source_id": row.get("text_id"),
            }
        )
    return normalized


DYNASENT_URL = "https://github.com/cgpotts/dynasent/raw/main/dynasent-v1.1.zip"
DYNASENT_ZIP = "dynasent-v1.1.zip"
DYNASENT_FILES = {
    "train": "dynasent-v1.1/dynasent-v1.1-round02-dynabench-train.jsonl",
    "valid": "dynasent-v1.1/dynasent-v1.1-round02-dynabench-dev.jsonl",
    "test": "dynasent-v1.1/dynasent-v1.1-round02-dynabench-test.jsonl",
}


def _read_dynasent_jsonl(zip_path: str | Path, member_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as handle:
            for raw_line in handle:
                rows.append(json.loads(raw_line))
    return rows


def _download_dynasent_zip() -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "semantic_cloud_public_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / DYNASENT_ZIP
    if not zip_path.exists():
        urllib.request.urlretrieve(DYNASENT_URL, zip_path)
    return zip_path


def load_dynasent_splits() -> dict[str, list[dict[str, object]]]:
    zip_path = _download_dynasent_zip()
    return {
        split_name: normalize_dynasent_rows(
            _read_dynasent_jsonl(zip_path, member_name),
            split_name=split_name,
            source_name="dynasent",
        )
        for split_name, member_name in DYNASENT_FILES.items()
    }


def normalize_ag_news_rows(
    rows: list[dict[str, object]],
    split_name: str,
    source_name: str,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        raw_label = row.get("label")
        label = AG_NEWS_LABELS.get(int(raw_label)) if raw_label is not None else None
        if not text or label is None:
            continue
        normalized.append(
            {
                "text": text,
                "label": label,
                "seed_source": source_name,
                "template_id": f"{source_name}_{split_name}",
                "early_signal": "public",
                "final_signal": "public",
                "reversal_position": 0,
                "distractor_strength": 0.0,
                "length_tokens": len(text.split()),
                "source_id": row.get("text_id", f"{split_name}_{index}"),
            }
        )
    return normalized


def load_ag_news_splits() -> dict[str, list[dict[str, object]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "AG News loading requires the `datasets` package. Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset("ag_news")
    train_valid = dataset["train"].train_test_split(test_size=0.1, seed=7)
    return {
        "train": normalize_ag_news_rows(
            list(train_valid["train"]),
            split_name="train",
            source_name="ag_news",
        ),
        "valid": normalize_ag_news_rows(
            list(train_valid["test"]),
            split_name="valid",
            source_name="ag_news",
        ),
        "test": normalize_ag_news_rows(
            list(dataset["test"]),
            split_name="test",
            source_name="ag_news",
        ),
    }
