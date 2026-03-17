import json

from semantic_cloud.data.decoder_dataset import (
    DecoderDataset,
    build_decoder_row,
    build_decoder_splits,
    build_decoder_vocab_from_rows,
)


def test_build_decoder_row_splits_prefix_and_target():
    row = build_decoder_row(
        text="It sounded kind at first, but the final offer was plainly exploitative.",
        label="negative",
        challenge_type="late_reversal",
    )

    assert row["prefix_text"]
    assert row["target_text"]
    assert row["resolution_position"] > 0
    assert row["target_text"] in row["text"]


def test_decoder_dataset_returns_input_ids_labels_and_loss_mask():
    rows = [
        build_decoder_row(
            text="It felt promising at first, but the closing evidence made the outcome look harmful.",
            label="negative",
            challenge_type="late_reversal",
        )
    ]
    vocab = build_decoder_vocab_from_rows(rows, vocab_size=128)
    dataset = DecoderDataset(rows=rows, vocab=vocab, max_length=24)

    item = dataset[0]

    assert item["tokens"].shape[0] == item["labels"].shape[0]
    assert item["loss_mask"].shape[0] == item["tokens"].shape[0]
    assert item["loss_mask"].sum().item() > 0


def test_build_decoder_dataset_writes_train_valid_test(tmp_path):
    splits = build_decoder_splits(seed=7, train_size=8, valid_size=4, test_size=4)
    for split_name, rows in splits.items():
        split_path = tmp_path / f"{split_name}.jsonl"
        split_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "valid.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
