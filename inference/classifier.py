from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DocumentClassifier:
    """Load a fine-tuned checkpoint and classify text."""

    def __init__(self, model_dir: str | Path, max_length: int = 512) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict(self, text: str) -> dict:
        """Return label, confidence, and per-class scores."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0].numpy()

        probs = _softmax(logits)
        pred_id = int(np.argmax(probs))
        label = self.id2label[pred_id]
        scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return {
            "label": label,
            "confidence": float(probs[pred_id]),
            "scores": scores,
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()
