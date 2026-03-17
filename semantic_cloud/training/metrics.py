from __future__ import annotations


def compute_accuracy(preds: list[int], labels: list[int]) -> float:
    correct = sum(int(pred == label) for pred, label in zip(preds, labels))
    return correct / len(labels) if labels else 0.0


def compute_macro_f1(preds: list[int], labels: list[int], num_classes: int) -> float:
    f1_scores: list[float] = []
    for class_id in range(num_classes):
        tp = sum(1 for pred, label in zip(preds, labels) if pred == class_id and label == class_id)
        fp = sum(1 for pred, label in zip(preds, labels) if pred == class_id and label != class_id)
        fn = sum(1 for pred, label in zip(preds, labels) if pred != class_id and label == class_id)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return sum(f1_scores) / num_classes if num_classes else 0.0


def summarize_subset(
    preds: list[int],
    labels: list[int],
    metadata: list[dict[str, object]],
    predicate,
    num_classes: int,
) -> dict[str, float]:
    filtered = [
        (pred, label)
        for pred, label, row in zip(preds, labels, metadata)
        if predicate(row)
    ]
    if not filtered:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    subset_preds = [pred for pred, _ in filtered]
    subset_labels = [label for _, label in filtered]
    return {
        "accuracy": compute_accuracy(subset_preds, subset_labels),
        "macro_f1": compute_macro_f1(subset_preds, subset_labels, num_classes=num_classes),
    }
