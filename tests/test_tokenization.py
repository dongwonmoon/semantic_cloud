from semantic_cloud.tokenization import BasicTokenizer, build_vocab


def test_basic_tokenizer_splits_punctuation():
    tokenizer = BasicTokenizer()

    assert tokenizer.tokenize("Wait, however, this changed.") == [
        "wait",
        ",",
        "however",
        ",",
        "this",
        "changed",
        ".",
    ]


def test_build_vocab_is_deterministic():
    vocab = build_vocab(["alpha beta", "beta gamma"], vocab_size=10)

    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert "beta" in vocab
