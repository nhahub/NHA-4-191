#!/usr/bin/env python3
"""ONNX Runtime inference wrapper for CPU-optimized deployment.

Wraps ONNX models with preprocessing, post-processing (NMS),
and multi-provider support (CPU, CUDA, TensorRT).

Usage:
    from src.deployment.onnx_runner import ONNXRunner

    runner = ONNXRunner("models/exports/best.onnx", providers=["CPUExecutionProvider"])
    boxes, scores, class_ids = runner.predict(image)
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ONNXRunner:
    def __init__(
        self,
        model_path: str,
        providers: list[str] | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        class_names: list[str] | None = None,
    ):
        self.model_path = str(Path(model_path).resolve())
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.class_names = class_names or ["Vehicle", "Pedestrian", "Cyclist"]

        if providers is None:
            providers = self._auto_select_providers()

        self.providers = providers
        self.session = self._load_session()

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        _, _, self.input_h, self.input_w = self.input_shape

        logger.info(f"ONNXRunner initialized: {self.model_path}")
        logger.info(f"  Providers: {self.providers}")
        logger.info(f"  Input shape: {self.input_shape}")

    def _auto_select_providers(self) -> list[str]:
        providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            priority = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in priority if p in available]
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Provider auto-detection failed: {e}")
        return providers

    def _load_session(self) -> Any:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2

        session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=self.providers,
        )
        logger.info(f"ONNX session created with {len(self.providers)} provider(s)")
        return session

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (self.imgsz, self.imgsz):
            image = cv2.resize(image, (self.imgsz, self.imgsz))

        img = image.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = outputs[0][0]
        return self.postprocess(predictions, image.shape[:2])

    def postprocess(
        self, predictions: np.ndarray, original_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_h, orig_w = original_shape
        scale_x = orig_w / self.imgsz
        scale_y = orig_h / self.imgsz

        predictions = predictions.T
        boxes, scores, class_ids = [], [], []

        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            class_score = float(class_scores[class_id])

            if class_score < self.conf_threshold:
                continue

            xc, yc, w, h = pred[0], pred[1], pred[2], pred[3]
            x1 = max(0, (xc - w / 2) * scale_x)
            y1 = max(0, (yc - h / 2) * scale_y)
            x2 = min(orig_w, (xc + w / 2) * scale_x)
            y2 = min(orig_h, (yc + h / 2) * scale_y)

            boxes.append([x1, y1, x2, y2])
            scores.append(class_score)
            class_ids.append(class_id)

        if not boxes:
            return np.array([]), np.array([]), np.array([])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int32)

        indices = self._nms(boxes, scores, self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[1:][iou <= iou_threshold]

        return keep
