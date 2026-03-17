import re
from collections import Counter


TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class BasicTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    tokenizer = BasicTokenizer()
    for text in texts:
        counter.update(tokenizer.tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if len(vocab) >= vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(tokens: list[str], vocab: dict[str, int], max_length: int) -> list[int]:
    encoded = [vocab.get(token, vocab["<unk>"]) for token in tokens[:max_length]]
    if len(encoded) < max_length:
        encoded.extend([vocab["<pad>"]] * (max_length - len(encoded)))
    return encoded
