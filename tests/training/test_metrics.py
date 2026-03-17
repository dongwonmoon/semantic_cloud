from semantic_cloud.training.metrics import compute_accuracy, compute_macro_f1


def test_compute_accuracy():
    preds = [0, 1, 1, 2]
    labels = [0, 0, 1, 2]

    assert compute_accuracy(preds, labels) == 0.75


def test_compute_macro_f1():
    preds = [0, 1, 1, 2]
    labels = [0, 0, 1, 2]

    assert round(compute_macro_f1(preds, labels, num_classes=3), 4) == 0.7778
