from __future__ import annotations

import argparse

from semantic_cloud.data.build_dataset import build_dataset_source, export_splits
from semantic_cloud.data.seed_loader import load_sst2_sentences
from semantic_cloud.training.datasets import build_label_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-source",
        choices=("semantic_cloud", "dynasent", "ag_news"),
        default="semantic_cloud",
    )
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--valid-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--output-dir", default="artifacts/datasets/semantic_cloud_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-seeds-only", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_seeds_only:
        if args.dataset_source != "semantic_cloud":
            raise ValueError("--check-seeds-only is only supported for semantic_cloud source")
        for sentence in load_sst2_sentences("train")[: args.limit]:
            print(sentence)
        return

    splits = build_dataset_source(
        dataset_source=args.dataset_source,
        seed=args.seed,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
    )
    label_to_id = build_label_mapping(splits["train"] + splits["valid"])
    export_splits(
        splits,
        args.output_dir,
        metadata={
            "dataset_source": args.dataset_source,
            "label_to_id": label_to_id,
        },
    )
    for split_name, rows in splits.items():
        print(split_name, len(rows))


if __name__ == "__main__":
    main()
