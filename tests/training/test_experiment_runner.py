import json
from pathlib import Path
import subprocess
import sys

import pytest

from semantic_cloud.training.experiment_runner import aggregate_run_reports, run_experiment_suite
from semantic_cloud.training.train import run_debug_experiment


def test_aggregate_run_reports_computes_mean_std_and_challenge_types():
    reports = [
        {
            "valid_accuracy": 0.5,
            "valid_macro_f1": 0.4,
            "test_accuracy": 0.6,
            "test_macro_f1": 0.5,
            "challenge_accuracy": 0.7,
            "challenge_macro_f1": 0.6,
            "challenge_by_type": {
                "late_reversal": {"accuracy": 0.8, "macro_f1": 0.7},
                "qualified_support": {"accuracy": 0.6, "macro_f1": 0.5},
            },
        },
        {
            "valid_accuracy": 0.7,
            "valid_macro_f1": 0.6,
            "test_accuracy": 0.8,
            "test_macro_f1": 0.7,
            "challenge_accuracy": 0.5,
            "challenge_macro_f1": 0.4,
            "challenge_by_type": {
                "late_reversal": {"accuracy": 0.4, "macro_f1": 0.3},
                "qualified_support": {"accuracy": 0.8, "macro_f1": 0.7},
            },
        },
    ]

    summary = aggregate_run_reports(reports)

    assert summary["valid_accuracy"]["mean"] == pytest.approx(0.6)
    assert summary["test_accuracy"]["mean"] == pytest.approx(0.7)
    assert summary["challenge_by_type"]["late_reversal"]["accuracy"]["mean"] == pytest.approx(0.6)


def test_aggregate_run_reports_rejects_empty_input():
    with pytest.raises(ValueError, match="No run reports"):
        aggregate_run_reports([])


def test_run_experiment_suite_writes_run_reports_and_summary(tmp_path):
    run_debug_experiment(output_dir=tmp_path)
    dataset_dir = tmp_path / "debug_dataset"
    output_dir = tmp_path / "suite"

    result = run_experiment_suite(
        model_type="transformer",
        dataset_dir=str(dataset_dir),
        seeds=[7, 11],
        batch_size=8,
        epochs=1,
        device="cpu",
        output_dir=str(output_dir),
    )

    assert len(result["runs"]) == 2
    assert (output_dir / "seed_7" / "report.json").exists()
    assert (output_dir / "summary.json").exists()
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "valid_accuracy" in payload


def test_suite_cli_writes_summary(tmp_path):
    run_debug_experiment(output_dir=tmp_path)
    dataset_dir = tmp_path / "debug_dataset"
    output_dir = tmp_path / "cli_suite"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_suite.py",
            "--model-type",
            "transformer",
            "--dataset-dir",
            str(dataset_dir),
            "--seeds",
            "7",
            "11",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "summary.json").exists()
