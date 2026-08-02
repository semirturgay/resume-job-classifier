from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from training.dataset import encode_labels, load_dataset_dict, load_label_map, tokenize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the resume/job document classifier.")
    parser.add_argument(
        "--model-name",
        default="microsoft/MiniLM-L12-H384-uncased",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with train.jsonl, val.jsonl, test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/run-001"),
        help="Where to save the trained model.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


class ProgressPrinter(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\nTraining: {state.max_steps} steps over {args.num_train_epochs} epochs")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"  step {int(state.global_step):>3}/{state.max_steps}  loss {logs['loss']:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(
                f"  epoch {metrics.get('epoch', '?')} done — "
                f"val acc {metrics['eval_accuracy']:.0%}, macro-F1 {metrics['eval_macro_f1']:.2f}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        if state.best_model_checkpoint:
            print(f"\nBest checkpoint: {state.best_model_checkpoint}")


def main() -> None:
    args = parse_args()

    # 1. Load label map (resume→0, job_post→1, other→2)
    label_map = load_label_map()
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = label_map["label2id"]

    # 2. Load train/val/test splits from data/processed/
    datasets = load_dataset_dict(args.data_dir)

    # 3. Download/load tokenizer + base model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_map["labels"]),
        id2label=id2label,
        label2id=label2id,
    )

    # 4. Convert labels to numbers + tokenize text for each split
    tokenized = {}
    for split, dataset in datasets.items():
        encoded = encode_labels(dataset, label_map)
        tokenized[split] = tokenize_dataset(encoded, tokenizer, max_length=args.max_length)

    # 5. Keep only columns the Trainer needs
    columns_to_keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in tokenized["train"].column_names:
        columns_to_keep.add("token_type_ids")

    for split in tokenized:
        drop_cols = [c for c in tokenized[split].column_names if c not in columns_to_keep]
        tokenized[split] = tokenized[split].remove_columns(drop_cols)

    # 6. Configure training
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = max(1, (len(tokenized["train"]) + args.batch_size - 1) // args.batch_size)
    logging_steps = max(1, min(steps_per_epoch, 10))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch" if "val" in tokenized else "no",
        save_strategy="epoch",
        load_best_model_at_end="val" in tokenized,
        metric_for_best_model="macro_f1" if "val" in tokenized else None,
        greater_is_better=True,
        logging_steps=logging_steps,
        seed=args.seed,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("val"),
        processing_class=tokenizer,
        compute_metrics=compute_metrics if "val" in tokenized else None,
        callbacks=[ProgressPrinter()],
    )

    # 7. Train and save
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    with (args.output_dir / "training_args.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
