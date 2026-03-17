from semantic_cloud.data.public_datasets import normalize_dynasent_rows


def test_normalize_dynasent_rows_keeps_only_supported_labels():
    rows = [
        {"sentence": "A warm and generous movie.", "gold_label": "positive", "text_id": "r1"},
        {"sentence": "A confusing middle effort.", "gold_label": "neutral", "text_id": "r2"},
        {"sentence": "Workers disagreed on this one.", "gold_label": "mixed", "text_id": "r3"},
        {"sentence": "No majority label here.", "gold_label": None, "text_id": "r4"},
    ]

    result = normalize_dynasent_rows(rows, split_name="train", source_name="dynasent")

    assert [row["label"] for row in result] == ["positive", "neutral"]
    assert all(row["seed_source"] == "dynasent" for row in result)
    assert all("text" in row for row in result)
