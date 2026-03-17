from __future__ import annotations

import argparse
import json

from semantic_cloud.training.train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("transformer", "cfrm"), required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--report-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(
        model_type=args.model_type,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        report_path=args.report_path,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
