from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.classifier import DocumentClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify text as resume, job_post, or other.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Fine-tuned PyTorch model directory.",
    )
    parser.add_argument("--text", type=str, help="Text to classify.")
    parser.add_argument("--file", type=Path, help="Read text from a file instead of --text.")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Provide --text or --file.")

    classifier = DocumentClassifier(args.model_dir, max_length=args.max_length)
    result = classifier.predict(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
