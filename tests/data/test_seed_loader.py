from semantic_cloud.data.seed_loader import filter_seed_sentences


def test_filter_seed_sentences_drops_duplicates_and_bad_lengths():
    rows = [
        {"sentence": "This works well."},
        {"sentence": "This works well."},
        {"sentence": "Bad"},
    ]

    result = filter_seed_sentences(rows, min_tokens=3, max_tokens=8)

    assert result == ["This works well."]
