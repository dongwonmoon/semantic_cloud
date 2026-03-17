import json
from pathlib import Path

import pytest

from semantic_cloud.data.challenge_sets import load_challenge_rows


def test_load_challenge_rows_reads_jsonl(tmp_path):
    challenge_path = tmp_path / "challenge.jsonl"
    challenge_path.write_text(
        json.dumps(
            {
                "text": "It sounded reassuring at first, yet the final warning mattered more.",
                "label": "negative",
                "challenge_type": "late_reversal",
                "target_cue_span": "the final warning mattered more",
                "notes": "negative resolution",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_challenge_rows(challenge_path)

    assert len(rows) == 1
    assert rows[0]["challenge_type"] == "late_reversal"


def test_load_challenge_rows_requires_challenge_type(tmp_path):
    challenge_path = tmp_path / "challenge.jsonl"
    challenge_path.write_text(
        json.dumps(
            {
                "text": "The praise sounded strong, but the hidden drawback still dominated.",
                "label": "negative",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="challenge_type"):
        load_challenge_rows(challenge_path)


def test_load_challenge_rows_requires_supported_labels(tmp_path):
    challenge_path = tmp_path / "challenge.jsonl"
    challenge_path.write_text(
        json.dumps(
            {
                "text": "It felt supportive until the final condition weakened everything.",
                "label": "qualified_positive",
                "challenge_type": "qualified_support",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported challenge label"):
        load_challenge_rows(challenge_path)


def test_checked_in_challenge_set_has_expected_shape():
    challenge_path = Path("artifacts/challenge_sets/meaning_reinterpretation_v1.jsonl")

    rows = load_challenge_rows(challenge_path)

    assert len(rows) >= 40
    assert len({row["challenge_type"] for row in rows}) >= 4
    assert {row["label"] for row in rows}.issubset({"positive", "negative", "neutral"})
