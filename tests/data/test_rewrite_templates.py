from semantic_cloud.data.rewrite_templates import rewrite_sentence


LABELS = {
    "direct_positive",
    "direct_negative",
    "qualified_positive",
    "qualified_negative",
    "hidden_positive",
    "hidden_negative",
    "warning_dominant",
    "uncertainty_dominant",
}


def test_rewrite_sentence_returns_text_and_metadata():
    sample = rewrite_sentence("The acting is charming and direct.", seed=7)

    assert "text" in sample
    assert "label" in sample
    assert "template_id" in sample
    assert sample["label"] in LABELS


def test_rewrite_sentence_honors_forced_label():
    sample = rewrite_sentence(
        "The acting is charming and direct.",
        seed=7,
        forced_label="qualified_positive",
    )

    assert sample["label"] == "qualified_positive"
