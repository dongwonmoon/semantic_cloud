from semantic_cloud.training.datasets import build_label_mapping


def test_build_label_mapping_supports_public_three_class_labels():
    rows = [
        {"label": "positive"},
        {"label": "negative"},
        {"label": "neutral"},
        {"label": "positive"},
    ]

    mapping = build_label_mapping(rows)

    assert mapping == {"negative": 0, "neutral": 1, "positive": 2}
