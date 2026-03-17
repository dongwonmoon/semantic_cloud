from collections import Counter


def dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    deduped: list[dict[str, object]] = []
    for row in rows:
        text = str(row["text"])
        if text in seen:
            continue
        seen.add(text)
        deduped.append(row)
    return deduped


def keep_length_window(
    rows: list[dict[str, object]], min_tokens: int, max_tokens: int
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if min_tokens <= int(row["length_tokens"]) <= max_tokens
    ]


def connector_ratio(rows: list[dict[str, object]]) -> dict[str, float]:
    counts = Counter()
    total = max(len(rows), 1)
    for row in rows:
        text = str(row["text"])
        for connector in ("but", "yet", "however", "although", "still", "in fact"):
            if connector in text:
                counts[connector] += 1
                break
    return {connector: count / total for connector, count in counts.items()}
