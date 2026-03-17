from __future__ import annotations

import json
from pathlib import Path


SUPPORTED_CHALLENGE_LABELS = {"positive", "negative", "neutral"}


def load_challenge_rows(
    path_or_dir: str | Path,
    allowed_labels: set[str] | None = None,
) -> list[dict[str, object]]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Challenge set not found: {path}")

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            row = json.loads(raw_line)
            text = str(row.get("text", "")).strip()
            label = str(row.get("label", "")).strip()
            challenge_type = str(row.get("challenge_type", "")).strip()
            if not text:
                raise ValueError(f"Missing text in challenge row {line_number}")
            valid_labels = allowed_labels or SUPPORTED_CHALLENGE_LABELS
            if label not in valid_labels:
                raise ValueError(f"Unsupported challenge label: {label}")
            if not challenge_type:
                raise ValueError(f"Missing challenge_type in challenge row {line_number}")
            normalized = {
                "text": text,
                "label": label,
                "challenge_type": challenge_type,
                "target_cue_span": str(row.get("target_cue_span", "")).strip(),
                "notes": str(row.get("notes", "")).strip(),
                "seed_source": "challenge",
                "template_id": f"challenge_{challenge_type}",
                "early_signal": str(row.get("early_signal", "challenge")).strip() or "challenge",
                "final_signal": str(row.get("final_signal", label)).strip() or label,
                "reversal_position": int(row.get("reversal_position", 0)),
                "distractor_strength": float(row.get("distractor_strength", 0.0)),
                "length_tokens": int(row.get("length_tokens", len(text.split()))),
            }
            rows.append(normalized)
    return rows
