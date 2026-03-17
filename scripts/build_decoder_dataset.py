from __future__ import annotations

import argparse

from semantic_cloud.data.decoder_dataset import (
    build_decoder_splits,
    export_decoder_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--valid-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--output-dir", default="artifacts/datasets/decoder_v1")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = build_decoder_splits(
        seed=args.seed,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
    )
    export_decoder_splits(
        splits,
        args.output_dir,
        metadata={
            "dataset_source": "decoder_v1",
            "task_type": "prefix_completion",
        },
    )
    for split_name, rows in splits.items():
        print(split_name, len(rows))


if __name__ == "__main__":
    main()
