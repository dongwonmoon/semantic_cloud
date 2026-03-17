import torch

from semantic_cloud.models.gru_baseline import GRUBaselineClassifier


def test_gru_baseline_classifier_shape():
    model = GRUBaselineClassifier(vocab_size=100, num_classes=8)
    tokens = torch.randint(0, 100, (4, 32))

    logits = model(tokens)

    assert logits.shape == (4, 8)


def test_gru_baseline_classifier_can_reduce_loss():
    model = GRUBaselineClassifier(vocab_size=32, num_classes=8, hidden_dim=48)
    tokens = torch.randint(0, 32, (8, 24))
    labels = torch.randint(0, 8, (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    first_loss = None
    final_loss = None
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(tokens)
        loss = loss_fn(logits, labels)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss is not None
    assert first_loss is not None
    assert final_loss < first_loss
