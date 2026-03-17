from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from semantic_cloud.training.decoder_train import run_decoder_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=("transformer_decoder", "gru_decoder", "cfrm_decoder"),
        required=True,
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def aggregate_metrics(reports: list[dict[str, object]]) -> dict[str, object]:
    keys = [key for key, value in reports[0].items() if isinstance(value, (int, float))]
    summary: dict[str, object] = {}
    for key in keys:
        values = [float(report[key]) for report in reports]
        summary[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for seed in args.seeds:
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = run_dir / "report.json"
        sample_path = run_dir / "samples.json"
        report = run_decoder_experiment(
            model_type=args.model_type,
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
            evaluate_test=args.evaluate_test,
            sample_output_path=str(sample_path),
            report_path=str(report_path),
            seed=seed,
        )
        reports.append(report)

    summary = aggregate_metrics(reports)
    (output_dir / "runs.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
