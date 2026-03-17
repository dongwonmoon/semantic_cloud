from pathlib import Path

from semantic_cloud.training.train import run_debug_experiment, run_experiment


def test_train_one_epoch_returns_metrics(tmp_path):
    metrics = run_debug_experiment(output_dir=tmp_path)

    assert "train_loss" in metrics
    assert "valid_accuracy" in metrics


def test_run_experiment_can_write_state_dump(tmp_path):
    metrics = run_debug_experiment(output_dir=tmp_path)
    dataset_dir = tmp_path / "debug_dataset"
    state_dump_path = tmp_path / "state_dump.json"

    rerun_metrics = run_experiment(
        model_type="transformer",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
        state_dump_path=str(state_dump_path),
    )

    assert "valid_accuracy" in metrics
    assert rerun_metrics["valid_accuracy"] >= 0.0
    assert Path(state_dump_path).exists()
