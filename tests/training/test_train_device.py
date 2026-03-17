from semantic_cloud.training.train import resolve_device


def test_resolve_device_returns_cpu_for_auto_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert resolve_device("auto") == "cpu"
