#!/usr/bin/env python3
"""Model versioning and registry.

Tracks model metadata (version, date, mAP, params, export formats)
and provides an API to list available model versions.

Usage:
    from src.mlops.model_registry import ModelRegistry

    registry = ModelRegistry("models/registry.json")
    registry.register("HPO_run/weights/best.pt", mAP50=0.935, mAP5095=0.725, fps=36.4)
    registry.list_models()
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, registry_path: str = "models/registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.models: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.models = json.load(f)

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.models, f, indent=2)

    def register(
        self,
        model_path: str,
        mAP50: float | None = None,
        mAP5095: float | None = None,
        fps: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        params_m: float | None = None,
        size_mb: float | None = None,
        export_formats: list[str] | None = None,
    ) -> dict:
        entry = {
            "version": len(self.models) + 1,
            "model_path": model_path,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "mAP50": mAP50,
            "mAP5095": mAP5095,
            "fps": fps,
            "precision": precision,
            "recall": recall,
            "params_m": params_m,
            "size_mb": size_mb,
            "export_formats": export_formats or [],
        }
        self.models.append(entry)
        self._save()
        logger.info("Registered model v%d: %s", entry["version"], model_path)
        return entry

    def list_models(self) -> list[dict]:
        return list(self.models)

    def get_latest(self) -> dict | None:
        return self.models[-1] if self.models else None

    def get_version(self, version: int) -> dict | None:
        for m in self.models:
            if m["version"] == version:
                return m
        return None
