import torch

from semantic_cloud.models.sparse_field_classifier import HierarchicalSparseFieldClassifier


def test_sparse_field_classifier_returns_logits():
    model = HierarchicalSparseFieldClassifier(vocab_size=128, num_classes=4)
    tokens = torch.randint(0, 128, (4, 24))

    logits = model(tokens)

    assert logits.shape == (4, 4)


def test_sparse_field_classifier_can_return_state():
    model = HierarchicalSparseFieldClassifier(vocab_size=128, num_classes=4)
    tokens = torch.randint(0, 128, (2, 24))

    output = model(tokens, return_state=True)

    assert output["logits"].shape == (2, 4)
    assert output["final_z"].shape[0] == 2
