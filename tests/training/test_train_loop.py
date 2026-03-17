import json
from pathlib import Path

from scripts.train_experiment import parse_args
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


def test_run_experiment_reports_test_metrics(tmp_path):
    run_debug_experiment(output_dir=tmp_path)
    dataset_dir = tmp_path / "debug_dataset"

    metrics = run_experiment(
        model_type="transformer",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
        evaluate_test=True,
    )

    assert "test_accuracy" in metrics
    assert "test_macro_f1" in metrics


def test_run_experiment_can_evaluate_a_challenge_dir(tmp_path):
    run_debug_experiment(output_dir=tmp_path)
    dataset_dir = tmp_path / "debug_dataset"
    challenge_dir = tmp_path / "challenge_dataset"
    challenge_dir.mkdir()
    challenge_rows = [
        {
            "text": "It looked generous at first, but the final terms were plainly harmful.",
            "label": "warning_dominant",
            "challenge_type": "late_reversal",
            "target_cue_span": "but the final terms were plainly harmful",
            "notes": "late negative resolution",
            "seed_source": "challenge",
            "template_id": "challenge_late_reversal",
            "early_signal": "positive",
            "final_signal": "negative",
            "reversal_position": 10,
            "distractor_strength": 0.8,
            "length_tokens": 12,
        }
    ]
    (challenge_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(row) for row in challenge_rows) + "\n",
        encoding="utf-8",
    )

    metrics = run_experiment(
        model_type="transformer",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
        challenge_dir=str(challenge_dir),
    )

    assert "challenge_accuracy" in metrics
    assert "challenge_by_type" in metrics
    assert "late_reversal" in metrics["challenge_by_type"]


def test_run_experiment_writes_state_summary(tmp_path):
    run_debug_experiment(output_dir=tmp_path)
    dataset_dir = tmp_path / "debug_dataset"
    state_summary_path = tmp_path / "state_summary.json"

    run_experiment(
        model_type="cfrm_philosophy",
        dataset_dir=str(dataset_dir),
        batch_size=8,
        epochs=1,
        device="cpu",
        state_summary_path=str(state_summary_path),
    )

    payload = json.loads(state_summary_path.read_text(encoding="utf-8"))
    assert "novelty_mean" in payload
    assert "representative_samples" in payload


def test_train_experiment_parse_args_accepts_hardening_options():
    args = parse_args(
        [
            "--model-type",
            "cfrm_philosophy",
            "--dataset-dir",
            "artifacts/datasets/dynasent_v1",
            "--evaluate-test",
            "--challenge-dir",
            "artifacts/challenge_sets/meaning_reinterpretation_v1.jsonl",
            "--state-summary-path",
            "artifacts/reports/state_summary.json",
        ]
    )

    assert args.evaluate_test is True
    assert args.challenge_dir.endswith("meaning_reinterpretation_v1.jsonl")
    assert args.state_summary_path.endswith("state_summary.json")


def test_train_experiment_parse_args_accepts_gru_model_type():
    args = parse_args(
        [
            "--model-type",
            "gru",
            "--dataset-dir",
            "artifacts/datasets/dynasent_v1",
        ]
    )

    assert args.model_type == "gru"


def test_train_experiment_parse_args_accepts_fast_philosophy_model_type():
    args = parse_args(
        [
            "--model-type",
            "cfrm_philosophy_fast",
            "--dataset-dir",
            "artifacts/datasets/dynasent_v1",
        ]
    )

    assert args.model_type == "cfrm_philosophy_fast"


def test_train_experiment_parse_args_accepts_balanced_philosophy_model_type():
    args = parse_args(
        [
            "--model-type",
            "cfrm_philosophy_balanced",
            "--dataset-dir",
            "artifacts/datasets/dynasent_v1",
        ]
    )

    assert args.model_type == "cfrm_philosophy_balanced"


def test_train_experiment_parse_args_accepts_topk_philosophy_model_type():
    args = parse_args(
        [
            "--model-type",
            "cfrm_philosophy_topk",
            "--dataset-dir",
            "artifacts/datasets/dynasent_v1",
        ]
    )

    assert args.model_type == "cfrm_philosophy_topk"
