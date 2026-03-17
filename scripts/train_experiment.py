from __future__ import annotations

import argparse
import json

from semantic_cloud.training.train import run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=("transformer", "gru", "cfrm", "cfrm_philosophy", "cfrm_philosophy_fast"),
        required=True,
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--challenge-dir")
    parser.add_argument("--report-path")
    parser.add_argument("--state-dump-path")
    parser.add_argument("--state-summary-path")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        model_type=args.model_type,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        report_path=args.report_path,
        state_dump_path=args.state_dump_path,
        evaluate_test=args.evaluate_test,
        challenge_dir=args.challenge_dir,
        state_summary_path=args.state_summary_path,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
