#!/usr/bin/env python3

"""

Road-Sense Production Model Exporter

Fully compliant with deployment requirements:

- ONNX (FP16 + Dynamic Batch)

- TorchScript (Tracing + Scripting)

- TFLite (INT8 Quantization)

- Full validation pipeline

"""

import argparse
import logging
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ModelExporter:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        logger.info(f"Loading YOLO model: {weights_path}")

        self.model = YOLO(str(self.weights_path))

        self.torch_model = self.model.model

        self.device = device

        self.model.to(device)

    # =========================

    # ONNX EXPORT (FP16 + DYNAMIC)

    # =========================

    def export_onnx(self, output_dir: Path, imgsz=640, half=True):
        logger.info("Exporting ONNX (FP16 + Dynamic Batch)...")

        if half:
            logger.info("FP16 enabled for ONNX export")

        path = self.model.export(
            format="onnx",
            imgsz=imgsz,
            half=half,
            dynamic=True,  # REQUIRED
            simplify=True,
        )

        return self._move(path, output_dir, "onnx")

    # =========================

    # TORCHSCRIPT (TRACE + SCRIPT)

    # =========================

    def export_torchscript(self, output_dir: Path):
        logger.info("Exporting TorchScript (trace + script)...")

        output_dir.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(1, 3, 640, 640).to(self.device)

        # TRACE

        traced = torch.jit.trace(self.torch_model, dummy)

        traced_path = output_dir / "model_traced.pt"

        traced.save(traced_path)

        # SCRIPT

        scripted = torch.jit.script(self.torch_model)

        scripted_path = output_dir / "model_scripted.pt"

        scripted.save(scripted_path)

        logger.info("TorchScript export completed")

        return {"traced": traced_path, "scripted": scripted_path}

    # =========================

    # TFLITE (INT8 QUANTIZATION)

    # =========================

    def export_tflite(self, output_dir: Path, imgsz=640):
        logger.info("Exporting TFLite (INT8 Quantization)...")

        path = self.model.export(format="tflite", imgsz=imgsz, int8=True)

        return self._move(path, output_dir, "tflite")

    # =========================

    # SAFE MOVE UTILITY

    # =========================

    def _move(self, exported_path, output_dir: Path, fmt: str):
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_path = Path(exported_path)

        final_path = output_dir / exported_path.name

        if exported_path.exists():
            if final_path.exists():
                if final_path.is_dir():
                    shutil.rmtree(final_path)

                else:
                    final_path.unlink()

            shutil.move(str(exported_path), str(final_path))

        logger.info(f"{fmt.upper()} saved at: {final_path}")

        return final_path

    # =========================

    # VALIDATION LAYER (REQUIRED)

    # =========================

    def validate_onnx(self, path):
        import onnxruntime as ort

        ort.InferenceSession(str(path))

        logger.info("ONNX validation passed")

    def validate_torchscript(self, paths: dict):
        # validate BOTH traced + scripted

        torch.jit.load(str(paths["traced"]))

        torch.jit.load(str(paths["scripted"]))

        logger.info("TorchScript validation passed")

    def validate_tflite(self, path):
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(path))

        interpreter.allocate_tensors()

        logger.info("TFLite validation passed")

    # =========================

    # FULL PIPELINE

    # =========================

    def export_all(self, formats, output_dir: Path, imgsz=640, half=True):
        results = {}

        if "onnx" in formats:
            onnx_path = self.export_onnx(output_dir, imgsz, half)

            self.validate_onnx(onnx_path)

            results["onnx"] = onnx_path

        if "torchscript" in formats:
            ts_paths = self.export_torchscript(output_dir)

            self.validate_torchscript(ts_paths)

            results["torchscript"] = ts_paths

        if "tflite" in formats:
            tflite_path = self.export_tflite(output_dir, imgsz)

            self.validate_tflite(tflite_path)

            results["tflite"] = tflite_path

        return results


# =========================

# CLI ENTRY POINT

# =========================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)

    parser.add_argument("--format", type=str, nargs="+", default=["onnx", "tflite", "torchscript"])

    parser.add_argument("--output", type=str, default="models/deploy")

    parser.add_argument("--half", action="store_true")

    args = parser.parse_args()

    exporter = ModelExporter(args.weights)

    results = exporter.export_all(args.format, Path(args.output), half=args.half)

    print("\nEXPORT COMPLETED:")

    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
