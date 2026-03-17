from semantic_cloud.tokenization import BasicTokenizer


def load_sst2_sentences(split: str) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset("glue", "sst2", split=split)
    return filter_seed_sentences(dataset, min_tokens=4, max_tokens=20)


def filter_seed_sentences(
    rows: list[dict[str, str]], min_tokens: int, max_tokens: int
) -> list[str]:
    tokenizer = BasicTokenizer()
    seen: set[str] = set()
    filtered: list[str] = []

    for row in rows:
        sentence = row["sentence"].strip()
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) < min_tokens or len(tokens) > max_tokens:
            continue
        normalized = " ".join(tokens)
        if normalized in seen:
            continue
        seen.add(normalized)
        filtered.append(sentence)

    return filtered
