from __future__ import annotations

import argparse
import json

from semantic_cloud.training.decoder_train import run_decoder_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=("transformer_decoder", "gru_decoder", "cfrm_decoder"),
        required=True,
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--sample-output-path")
    parser.add_argument("--report-path")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    metrics = run_decoder_experiment(
        model_type=args.model_type,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        evaluate_test=args.evaluate_test,
        sample_output_path=args.sample_output_path,
        report_path=args.report_path,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
