from semantic_cloud.data.build_dataset import build_splits


EXPECTED_FIELDS = {
    "text",
    "label",
    "seed_source",
    "template_id",
    "early_signal",
    "final_signal",
    "reversal_position",
    "distractor_strength",
    "length_tokens",
}


def test_build_splits_is_reproducible():
    seed_sentences = [f"Seed sentence {idx} works well." for idx in range(16)]

    first = build_splits(
        seed=11,
        train_size=8,
        valid_size=4,
        test_size=4,
        seed_sentences=seed_sentences,
    )
    second = build_splits(
        seed=11,
        train_size=8,
        valid_size=4,
        test_size=4,
        seed_sentences=seed_sentences,
    )

    assert first == second


def test_build_splits_rows_include_required_metadata():
    seed_sentences = [f"Seed sentence {idx} works well." for idx in range(16)]

    splits = build_splits(
        seed=5,
        train_size=8,
        valid_size=4,
        test_size=4,
        seed_sentences=seed_sentences,
    )

    row = splits["train"][0]
    assert EXPECTED_FIELDS.issubset(row.keys())
