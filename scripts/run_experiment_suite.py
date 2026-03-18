from __future__ import annotations

import argparse
import json

from semantic_cloud.training.experiment_runner import run_experiment_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=(
            "transformer",
            "gru",
            "sparse_field",
            "cfrm",
            "cfrm_philosophy",
            "cfrm_philosophy_fast",
            "cfrm_philosophy_balanced",
            "cfrm_philosophy_topk",
        ),
        required=True,
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--challenge-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-state-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment_suite(
        model_type=args.model_type,
        dataset_dir=args.dataset_dir,
        seeds=args.seeds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir,
        challenge_dir=args.challenge_dir,
        write_state_summary=args.write_state_summary,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
