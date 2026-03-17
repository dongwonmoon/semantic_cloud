import torch

from semantic_cloud.models.cfrm_philosophy import CFRMPhilosophyClassifier


def test_cfrm_philosophy_classifier_shape():
    model = CFRMPhilosophyClassifier(vocab_size=100, num_classes=3, num_clouds=4)
    tokens = torch.randint(0, 100, (4, 24))

    logits = model(tokens)

    assert logits.shape == (4, 3)


def test_cfrm_philosophy_classifier_returns_state():
    model = CFRMPhilosophyClassifier(vocab_size=100, num_classes=3, num_clouds=4)
    tokens = torch.randint(0, 100, (2, 16))

    output = model(tokens, return_state=True)

    assert output["logits"].shape == (2, 3)
    assert output["alpha"].shape == (2, 4)
    assert output["core"].shape[0] == 2
    assert output["novelty"].shape == (2, 16, 1)


def test_cfrm_philosophy_classifier_can_reduce_loss():
    model = CFRMPhilosophyClassifier(vocab_size=32, num_classes=3, num_clouds=4, hidden_dim=32)
    tokens = torch.randint(0, 32, (8, 20))
    labels = torch.randint(0, 3, (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    first_loss = None
    final_loss = None
    for _ in range(8):
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


def test_cfrm_philosophy_sparse_schedule_reports_fewer_reconfigurations():
    dense_model = CFRMPhilosophyClassifier(
        vocab_size=100,
        num_classes=3,
        num_clouds=4,
        sparse_reconfiguration=False,
    )
    sparse_model = CFRMPhilosophyClassifier(
        vocab_size=100,
        num_classes=3,
        num_clouds=4,
        sparse_reconfiguration=True,
        reconfiguration_interval=4,
        novelty_threshold=0.6,
    )
    tokens = torch.randint(0, 100, (2, 16))

    dense_output = dense_model(tokens, return_state=True)
    sparse_output = sparse_model(tokens, return_state=True)

    assert sparse_output["logits"].shape == (2, 3)
    assert sparse_output["reconfiguration_count"] <= dense_output["reconfiguration_count"]


def test_cfrm_philosophy_topk_schedule_tracks_attractor_and_masks():
    model = CFRMPhilosophyClassifier(
        vocab_size=100,
        num_classes=3,
        num_clouds=4,
        sparse_reconfiguration=True,
        reconfiguration_interval=3,
        novelty_threshold=0.5,
        always_apply_attractor=True,
        interaction_topk=2,
    )
    tokens = torch.randint(0, 100, (2, 12))

    output = model(tokens, return_state=True)

    assert output["logits"].shape == (2, 3)
    assert len(output["reconfiguration_mask"]) == 12
    assert len(output["attractor_mask"]) == 12
    assert output["attractor_count"] >= output["reconfiguration_count"]
