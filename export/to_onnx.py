from __future__ import annotations

import argparse
import platform
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fine-tuned checkpoint to float ONNX + INT8 ONNX."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("checkpoints/run-003"),
        help="Path to fine-tuned PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("export/artifacts"),
        help="Root output directory (writes onnx/ subfolder).",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Only export float ONNX, skip INT8 quantization.",
    )
    return parser.parse_args()


def _quantization_config() -> AutoQuantizationConfig:
    """Pick a dynamic INT8 config suited to the host CPU."""
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
    return AutoQuantizationConfig.avx2(is_static=False, per_channel=True)


def export_to_onnx(
    model_dir: Path,
    output_dir: Path,
    *,
    skip_quantize: bool = False,
) -> None:
    onnx_dir = output_dir / "onnx"
    float_dir = output_dir / "_float_export"

    for path in (onnx_dir, float_dir):
        if path.exists():
            shutil.rmtree(path)

    print(f"Exporting float ONNX from {model_dir} ...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir,
        export=True,
    )
    float_dir.mkdir(parents=True, exist_ok=True)
    ort_model.save_pretrained(float_dir)

    onnx_dir.mkdir(parents=True, exist_ok=True)
    float_onnx_src = float_dir / "model.onnx"
    float_onnx_dst = onnx_dir / "model.onnx"
    shutil.copy2(float_onnx_src, float_onnx_dst)
    print(f"Saved {float_onnx_dst} ({float_onnx_dst.stat().st_size / 1e6:.1f} MB)")

    if skip_quantize:
        shutil.rmtree(float_dir)
        return

    print("Quantizing to INT8 (dynamic) ...")
    quantizer = ORTQuantizer.from_pretrained(float_dir, file_name="model.onnx")
    quantizer.quantize(
        quantization_config=_quantization_config(),
        save_dir=onnx_dir,
        file_suffix="int8",
    )

    int8_path = onnx_dir / "model_int8.onnx"
    print(f"Saved {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")

    shutil.rmtree(float_dir)


def main() -> None:
    args = parse_args()
    export_to_onnx(
        args.model_dir,
        args.output_dir,
        skip_quantize=args.skip_quantize,
    )


if __name__ == "__main__":
    main()
