from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP_PATH = REPO_ROOT / "schemas" / "label_map.json"
VALID_LABELS = frozenset({"resume", "job_post", "other"})


def load_label_map(path: Path | None = None) -> dict[str, Any]:
    label_map_path = path or LABEL_MAP_PATH
    with label_map_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "").strip()
            label = record.get("label", "").strip()
            if not text:
                raise ValueError(f"{path}:{line_no}: missing or empty 'text'")
            if label not in VALID_LABELS:
                raise ValueError(f"{path}:{line_no}: invalid label {label!r}")
            records.append({"text": text, "label": label})
    return records


def load_split(data_dir: Path, split: str) -> Dataset:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return Dataset.from_list(load_jsonl(path))


def load_dataset_dict(data_dir: Path) -> DatasetDict:
    splits = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            splits[split] = load_split(data_dir, split)
    if "train" not in splits:
        raise FileNotFoundError(f"No train.jsonl found in {data_dir}")
    return DatasetDict(splits)


def encode_labels(dataset: Dataset, label_map: dict[str, Any]) -> Dataset:
    label2id = label_map["label2id"]

    def _encode(example: dict[str, str]) -> dict[str, int]:
        return {"labels": label2id[example["label"]]}

    return dataset.map(_encode)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(_tokenize, batched=True)
