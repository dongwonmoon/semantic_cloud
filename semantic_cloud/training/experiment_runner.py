from __future__ import annotations

import json
import math
from pathlib import Path

from semantic_cloud.training.train import run_experiment


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def aggregate_run_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    if not reports:
        raise ValueError("No run reports to aggregate")

    scalar_keys = [
        "train_loss",
        "valid_loss",
        "valid_accuracy",
        "valid_macro_f1",
        "test_loss",
        "test_accuracy",
        "test_macro_f1",
        "challenge_loss",
        "challenge_accuracy",
        "challenge_macro_f1",
    ]
    summary: dict[str, object] = {}
    for key in scalar_keys:
        values = [float(report[key]) for report in reports if key in report]
        if values:
            summary[key] = {"mean": _mean(values), "std": _std(values)}

    challenge_types = sorted(
        {
            challenge_type
            for report in reports
            for challenge_type in report.get("challenge_by_type", {}).keys()
        }
    )
    if challenge_types:
        summary["challenge_by_type"] = {}
        for challenge_type in challenge_types:
            summary["challenge_by_type"][challenge_type] = {}
            for metric_key in ("accuracy", "macro_f1"):
                values = [
                    float(report["challenge_by_type"][challenge_type][metric_key])
                    for report in reports
                    if challenge_type in report.get("challenge_by_type", {})
                ]
                summary["challenge_by_type"][challenge_type][metric_key] = {
                    "mean": _mean(values),
                    "std": _std(values),
                }
    return summary


def run_experiment_suite(
    model_type: str,
    dataset_dir: str,
    seeds: list[int],
    batch_size: int,
    epochs: int,
    device: str,
    output_dir: str,
    challenge_dir: str | None = None,
    write_state_summary: bool = False,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("At least one seed is required")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for seed in seeds:
        run_dir = output_path / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = run_dir / "report.json"
        state_summary_path = run_dir / "state_summary.json" if write_state_summary else None
        metrics = run_experiment(
            model_type=model_type,
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            report_path=str(report_path),
            evaluate_test=True,
            challenge_dir=challenge_dir,
            state_summary_path=str(state_summary_path) if state_summary_path is not None else None,
            seed=seed,
        )
        reports.append(metrics)
        run_rows.append({"seed": seed, "report_path": str(report_path), "metrics": metrics})

    summary = aggregate_run_reports(reports)
    (output_path / "runs.json").write_text(json.dumps(run_rows, indent=2), encoding="utf-8")
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"runs": run_rows, "summary": summary}
