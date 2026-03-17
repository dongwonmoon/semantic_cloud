from semantic_cloud.training.train import run_debug_experiment


def test_train_one_epoch_returns_metrics(tmp_path):
    metrics = run_debug_experiment(output_dir=tmp_path)

    assert "train_loss" in metrics
    assert "valid_accuracy" in metrics
