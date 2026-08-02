# resume-job-classifier

CPU-friendly text classifier that labels documents as **`resume`**, **`job_post`**, or **`other`**.

Use it to route text in hiring pipelines. Send only resume-like or job-like content to parsers, indexers, or LLM extraction steps.

**Pre-trained weights:** [Hugging Face](https://huggingface.co/YOUR_USERNAME/resume-job-classifier)  
**License:** Apache-2.0

## Quick start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Classify text with the published model:

```python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="YOUR_USERNAME/resume-job-classifier",
    top_k=None,
)

result = clf("We are hiring a Senior Software Engineer with Python experience.")
print(result)
```

Or use the CLI with a downloaded checkpoint:

```bash
python -m inference.predict \
  --model-dir path/to/model \
  --text "We are hiring a Senior Software Engineer..."
```

For fast CPU inference, use the ONNX INT8 artifact from the Hub (`onnx/model_int8.onnx`). See the model card on Hugging Face.

## Labels

| Label | Examples |
|-------|----------|
| `resume` | CVs, experience blocks, skills lists |
| `job_post` | Job descriptions, requirements, compensation |
| `other` | Bios, emails, blog posts, product copy |

## Model (v1.0)

| | |
|---|---|
| Base model | [microsoft/MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) |
| Test accuracy | 95% (macro-F1 0.95) |
| ONNX INT8 size | ~34 MB |

## About this repo

This repository contains the code used to train and evaluate the classifier. It is not a general template for publishing models to Hugging Face.

What is included:

- `inference/`: load a checkpoint and classify text
- `training/`, `eval/`, `export/`: how the v1.0 model was built
- `data/templates/`: a small set of hand-curated boundary examples

Training data, checkpoints, planning docs, and internal scripts stay local and are not committed to git.
