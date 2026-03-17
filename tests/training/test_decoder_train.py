import json
from pathlib import Path

from scripts.run_decoder_suite import parse_args as parse_suite_args
from scripts.train_decoder_experiment import parse_args
from semantic_cloud.data.decoder_dataset import build_decoder_splits, export_decoder_splits
from semantic_cloud.training.decoder_train import run_decoder_experiment


def _write_decoder_debug_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "decoder_debug_dataset"
    splits = build_decoder_splits(seed=7, train_size=32, valid_size=8, test_size=8)
    export_decoder_splits(
        splits,
        str(dataset_dir),
        metadata={"dataset_source": "decoder_debug", "task_type": "prefix_completion"},
    )
    return dataset_dir


def test_run_decoder_experiment_reports_loss_and_perplexity(tmp_path):
    dataset_dir = _write_decoder_debug_dataset(tmp_path)

    metrics = run_decoder_experiment(
        model_type="gru_decoder",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
    )

    assert "valid_loss" in metrics
    assert "valid_perplexity" in metrics


def test_run_decoder_experiment_writes_sample_generations(tmp_path):
    dataset_dir = _write_decoder_debug_dataset(tmp_path)
    sample_output_path = tmp_path / "samples.json"

    run_decoder_experiment(
        model_type="gru_decoder",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
        evaluate_test=True,
        sample_output_path=str(sample_output_path),
    )

    payload = json.loads(sample_output_path.read_text(encoding="utf-8"))
    assert payload
    assert "generated_suffix" in payload[0]


def test_train_decoder_experiment_parse_args_accepts_decoder_models():
    args = parse_args(
        [
            "--model-type",
            "cfrm_decoder",
            "--dataset-dir",
            "artifacts/datasets/decoder_v1",
        ]
    )

    assert args.model_type == "cfrm_decoder"


def test_run_decoder_suite_accepts_multiple_seeds(tmp_path):
    args = parse_suite_args(
        [
            "--model-type",
            "gru_decoder",
            "--dataset-dir",
            "artifacts/datasets/decoder_v1",
            "--seeds",
            "7",
            "11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert args.seeds == [7, 11]
