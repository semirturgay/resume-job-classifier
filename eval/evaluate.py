from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.dataset import encode_labels, load_label_map, load_split, tokenize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned classifier.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to a saved checkpoint (e.g. checkpoints/run-001).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with train.jsonl, val.jsonl, test.jsonl.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_map = load_label_map()
    label_names = label_map["labels"]

    # 1. Load saved model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    use_cpu = not torch.cuda.is_available()

    # 2. Load and tokenize the chosen split
    dataset = load_split(args.data_dir, args.split)
    dataset = encode_labels(dataset, label_map)
    dataset = tokenize_dataset(dataset, tokenizer, max_length=args.max_length)

    columns_to_keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in dataset.column_names:
        columns_to_keep.add("token_type_ids")
    drop_cols = [c for c in dataset.column_names if c not in columns_to_keep]
    dataset = dataset.remove_columns(drop_cols)

    # 3. Run predictions in batches
    all_preds: list[int] = []
    all_labels: list[int] = []
    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        inputs = {k: torch.tensor(v) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        if use_cpu:
            inputs = {k: v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        preds = np.argmax(outputs.logits.numpy(), axis=-1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels)

    # 4. Print metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Split: {args.split} ({len(all_labels)} examples)")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Macro-F1:  {macro_f1:.2f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=label_names, digits=2))

    # 5. Show each prediction vs truth
    print("Predictions:")
    raw = load_split(args.data_dir, args.split)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    for i, (pred_id, true_id) in enumerate(zip(all_preds, all_labels)):
        pred = id2label[pred_id]
        true = id2label[true_id]
        mark = "✓" if pred_id == true_id else "✗"
        preview = raw[i]["text"][:80].replace("\n", " ")
        print(f"  {mark} true={true:10s} pred={pred:10s}  {preview!r}...")


if __name__ == "__main__":
    main()
