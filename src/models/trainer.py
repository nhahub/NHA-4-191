"""
YOLO Trainer - Road-Sense

High-level training pipeline for YOLO object detection models.
Wraps Ultralytics YOLO with configuration management, logging,
and experiment tracking.
"""

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from .callbacks import ModelCheckpoint, TrainingLogger
from .model_factory import load_model

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load training configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with configuration values.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config  # noqa: RET504


class YOLOTrainer:
    """
    High-level training manager for YOLO object detection models.

    Manages the full training lifecycle:
    - Model initialization with pretrained weights
    - Configuration loading and validation
    - Training execution with callbacks
    - Checkpoint management
    - Results tracking
    """

    def __init__(  # type: ignore[valid-type]
        self,
        config: dict[str, Any],
        config_path: str | None = None,
        project_root: str | None = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            config: Training configuration dictionary.
            config_path: Path to config file (for saving/reproducing).
            project_root: Project root directory for resolving relative paths.
        """
        self.config = config
        self.config_path = config_path
        self.project_root = Path(project_root) if project_root else Path.cwd()

        self.model: YOLO | None = None
        self.results: dict[str, Any] | None = None
        self.save_dir: Path | None = None
        self._resolved_data_yaml: Path | None = None
        self.logger = TrainingLogger()
        self.checkpointer: ModelCheckpoint | None = None

    def setup(self) -> None:
        """
        Set up the training environment:
        - Load or create the model
        - Create output directories
        - Initialize callbacks
        - Validate configuration
        """
        self._setup_logging()
        self._create_directories()
        self._load_model()
        self._init_callbacks()
        self._save_config_snapshot()

    def train(self, **override_kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """
        Run the training loop.

        Args:
            **override_kwargs: Additional arguments to override config values.
                These are passed directly to model.train().

        Returns:
            Dictionary with training results and metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup() before train().")

        self.logger.on_train_start(self.config)

        # Build training arguments from config
        train_args = self._build_train_args(**override_kwargs)

        # Log training configuration
        logger.info(f"Training arguments: {json.dumps(train_args, indent=2, default=str)}")

        # Start training
        try:
            results = self.model.train(**train_args)

            # Capture results
            self.results = {
                "metrics": self._extract_metrics(results),
                "save_dir": str(self.model.trainer.save_dir) if hasattr(self.model, "trainer") else None,
                "epochs_trained": self.config["training"]["epochs"],
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.on_train_end(self.results)
            self._save_results()

            return self.results

        except Exception as e:
            self.logger.on_train_error(e)
            raise

    def validate(self) -> dict[str, Any]:
        """
        Run validation on the trained model.

        Returns:
            Dictionary with validation metrics.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Train or load a model first.")

        data_cfg = self.config["data"]
        data_yaml_path = self._prepare_dataset_yaml()
        val_args = {
            "data": str(data_yaml_path),
            "imgsz": data_cfg["imgsz"],
            "batch": data_cfg["batch_size"],
            "workers": data_cfg["workers"],
        }

        logger.info("Running validation...")
        results = self.model.val(**val_args)

        metrics = self._extract_metrics(results)
        logger.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")

        return metrics

    def export(self, format: str = "onnx", output_dir: str | None = None, **export_kwargs: Any) -> str:  # noqa: ANN401
        """
        Export the trained model to a deployment format.

        Args:
            format: Export format (onnx, torchscript, openvino, engine, tflite, etc.).
            output_dir: Output directory for exported model.
            **export_kwargs: Additional export arguments.

        Returns:
            Path to the exported model file.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        if output_dir is None:
            output_dir = self.project_root / "models" / "exports"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to {format} format...")
        export_path = self.model.export(format=format, **export_kwargs)

        logger.info(f"Model exported to: {export_path}")
        return export_path

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _setup_logging(self) -> None:
        """Configure logging for the training run."""
        log_cfg = self.config.get("logging", {})
        log_level = logging.DEBUG if log_cfg.get("verbose", True) else logging.INFO
        logger.setLevel(log_level)

    def _create_directories(self) -> None:
        """Create required output directories."""
        log_cfg = self.config.get("logging", {})
        ckpt_cfg = self.config.get("checkpoint", {})

        # Training output directory
        project_dir = self.project_root / log_cfg.get("project", "runs/train")
        exp_name = log_cfg.get("name", "exp")

        # Auto-increment experiment name
        project_dir.mkdir(parents=True, exist_ok=True)
        existing = [d for d in project_dir.iterdir() if d.is_dir() and d.name.startswith(exp_name)]
        if existing and not log_cfg.get("exist_ok", False):
            max_num = max(
                (int(d.name.replace(exp_name, "")) for d in existing if d.name[len(exp_name) :].isdigit()),
                default=0,
            )
            exp_name = f"{exp_name}{max_num + 1}"

        self.save_dir = project_dir / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint directory
        ckpt_dir = self.project_root / ckpt_cfg.get("save_dir", "models/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        """Load or create the YOLO model."""
        model_cfg = self.config.get("model", {})
        model_name = model_cfg.get("name", "yolo11m")
        pretrained = model_cfg.get("pretrained", True)
        weights_path = model_cfg.get("pretrained_weights")

        # Resolve weights path relative to project root
        if weights_path:
            weights_path = str(self.project_root / weights_path)

        logger.info(f"Loading model: {model_name} (pretrained={pretrained})")
        self.model = load_model(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
        )

        # Log model info
        from .model_factory import get_model_info

        info = get_model_info(self.model)
        logger.info(f"Model loaded: {info['num_parameters'] / 1e6:.2f}M parameters, ~{info['size_mb']:.1f} MB")

    def _init_callbacks(self) -> None:
        """Initialize training callbacks."""
        ckpt_cfg = self.config.get("checkpoint", {})

        self.checkpointer = ModelCheckpoint(
            save_dir=self.project_root / ckpt_cfg.get("save_dir", "models/checkpoints"),
            save_best=ckpt_cfg.get("save_best", True),
            save_last=ckpt_cfg.get("save_last", True),
            save_period=ckpt_cfg.get("save_period", -1),
            metric=ckpt_cfg.get("metric", "mAP50-95"),
        )

    def _save_config_snapshot(self) -> None:
        """Save a snapshot of the config for reproducibility."""
        if self.save_dir is None:
            return

        config_snapshot_path = self.save_dir / "config.yaml"
        with open(config_snapshot_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        # Save original config path for reference
        metadata = {
            "config_path": self.config_path,
            "project_root": str(self.project_root),
            "timestamp": datetime.now().isoformat(),
            "model_name": self.config.get("model", {}).get("name"),
        }
        metadata_path = self.save_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Config snapshot saved to: {config_snapshot_path}")

    def _build_train_args(self, **override_kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        data_cfg = self.config.get("data", {})
        train_cfg = self.config.get("training", {})
        aug_cfg = self.config.get("augmentation", {})
        reg_cfg = self.config.get("regularization", {})
        val_cfg = self.config.get("validation", {})
        device_cfg = self.config.get("device", {})
        adv_cfg = self.config.get("advanced", {})
        log_cfg = self.config.get("logging", {})
        ckpt_cfg = self.config.get("checkpoint", {})
        data_yaml_path = self._prepare_dataset_yaml()

        args = {
            **self._build_data_args(data_cfg, data_yaml_path),
            **self._build_train_hyperparams(train_cfg),
            **self._build_augmentation_args(aug_cfg),
            **self._build_regularization_args(reg_cfg),
            **self._build_validation_args(val_cfg),
            **self._build_device_args(device_cfg),
            **self._build_advanced_args(adv_cfg),
            **self._build_logging_args(log_cfg),
            **self._build_checkpoint_args(ckpt_cfg),
        }

        freeze = adv_cfg.get("freeze", [])
        if freeze:
            args["freeze"] = freeze

        args.update(override_kwargs)
        return args

    @staticmethod
    def _build_data_args(cfg: dict, yaml_path: Path) -> dict:
        return {
            "data": str(yaml_path),
            "imgsz": cfg.get("imgsz", 640),
            "batch": cfg.get("batch_size", 16),
            "workers": cfg.get("workers", 4),
        }

    @staticmethod
    def _build_train_hyperparams(cfg: dict) -> dict:
        return {
            "epochs": cfg.get("epochs", 100),
            "patience": cfg.get("patience", 30),
            "optimizer": cfg.get("optimizer", "auto"),
            "lr0": cfg.get("lr0", 0.01),
            "lrf": cfg.get("lrf", 0.01),
            "momentum": cfg.get("momentum", 0.937),
            "weight_decay": cfg.get("weight_decay", 0.0005),
            "warmup_epochs": cfg.get("warmup_epochs", 3.0),
            "warmup_momentum": cfg.get("warmup_momentum", 0.8),
            "warmup_bias_lr": cfg.get("warmup_bias_lr", 0.1),
        }

    @staticmethod
    def _build_augmentation_args(cfg: dict) -> dict:
        return {
            "hsv_h": cfg.get("hsv_h", 0.015),
            "hsv_s": cfg.get("hsv_s", 0.7),
            "hsv_v": cfg.get("hsv_v", 0.4),
            "degrees": cfg.get("degrees", 0.0),
            "translate": cfg.get("translate", 0.1),
            "scale": cfg.get("scale", 0.5),
            "shear": cfg.get("shear", 0.0),
            "perspective": cfg.get("perspective", 0.0),
            "flipud": cfg.get("flipud", 0.0),
            "fliplr": cfg.get("fliplr", 0.5),
            "mosaic": cfg.get("mosaic", 1.0),
            "mixup": cfg.get("mixup", 0.0),
            "copy_paste": cfg.get("copy_paste", 0.0),
        }

    @staticmethod
    def _build_regularization_args(cfg: dict) -> dict:
        return {
            "dropout": cfg.get("dropout", 0.0),
            "box": cfg.get("box", 7.5),
            "cls": cfg.get("cls", 1.0),
            "dfl": cfg.get("dfl", 1.5),
        }

    @staticmethod
    def _build_validation_args(cfg: dict) -> dict:
        return {"val": cfg.get("val_interval", 1) > 0}

    @staticmethod
    def _build_device_args(cfg: dict) -> dict:
        return {
            "device": cfg.get("device", "0"),
            "deterministic": cfg.get("deterministic", True),
            "amp": cfg.get("half_precision", False),
            "compile": cfg.get("compile", False),
        }

    @staticmethod
    def _build_advanced_args(cfg: dict) -> dict:
        return {
            "close_mosaic": cfg.get("close_mosaic", 10),
            "rect": cfg.get("rect", False),
            "cache": cfg.get("cache", False),
            "seed": cfg.get("seed", 42),
        }

    @staticmethod
    def _build_logging_args(cfg: dict) -> dict:
        return {
            "verbose": cfg.get("verbose", True),
            "exist_ok": cfg.get("exist_ok", False),
        }

    def _build_checkpoint_args(self, cfg: dict) -> dict:
        return {
            "project": str(self.save_dir.parent if self.save_dir else "runs/train"),
            "name": self.save_dir.name if self.save_dir else "exp",
            "save": True,
            "save_period": cfg.get("save_period", -1),
        }

    def _extract_metrics(self, results: Any) -> dict[str, float]:  # noqa: ANN401
        """Extract metrics from Ultralytics training/validation results."""
        metrics = {}

        try:
            # Training results object
            if hasattr(results, "results"):
                train_results = results.results
                if isinstance(train_results, dict):
                    metrics = train_results
                elif hasattr(train_results, "keys"):
                    metrics = dict(train_results)

            # Fallback: try common metric attributes
            if not metrics:
                if hasattr(results, "box"):
                    metrics["mAP50"] = float(results.box.map50) if hasattr(results.box, "map50") else None
                    metrics["mAP50-95"] = float(results.box.map) if hasattr(results.box, "map") else None

            # Ensure all values are serializable
            clean_metrics = {}
            for k, v in metrics.items():
                if v is not None:
                    try:
                        clean_metrics[str(k)] = float(v)
                    except (ValueError, TypeError):
                        clean_metrics[str(k)] = str(v)

            return clean_metrics

        except Exception as e:
            logger.warning(f"Could not extract metrics: {e}")
            return {}

    def _prepare_dataset_yaml(self) -> Path:
        return self._prepare_dataset_yaml_internal(attempt_autofix=True)

    def _prepare_dataset_yaml_internal(self, attempt_autofix: bool) -> Path:
        """
        Normalize dataset YAML paths for portability across machines.

        Ultralytics treats absolute `path:` values as authoritative. If a config
        was created on another machine, training fails when that absolute path no
        longer exists. This method rewrites the dataset YAML into a run-local file
        with corrected absolute paths.
        """
        data_cfg = self.config.get("data", {})
        relative_yaml = data_cfg.get("yaml_path", "data/processed/kitti/data.yaml")
        dataset_yaml = (self.project_root / relative_yaml).resolve()

        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

        with open(dataset_yaml) as f:
            dataset = yaml.safe_load(f) or {}

        if not isinstance(dataset, dict):
            raise ValueError(f"Invalid dataset YAML format: {dataset_yaml}")

        normalized = copy.deepcopy(dataset)

        raw_root = dataset.get("path")
        if raw_root:
            dataset_root = Path(str(raw_root))
            if dataset_root.is_absolute() and not dataset_root.exists():
                logger.warning(
                    "Dataset YAML path '%s' does not exist. Falling back to project root '%s'.",
                    dataset_root,
                    self.project_root,
                )
                dataset_root = self.project_root
            elif not dataset_root.is_absolute():
                candidate_project = (self.project_root / dataset_root).resolve()
                candidate_yaml_parent = (dataset_yaml.parent / dataset_root).resolve()
                dataset_root = candidate_project if candidate_project.exists() else candidate_yaml_parent
        else:
            dataset_root = dataset_yaml.parent

        normalized["path"] = str(dataset_root)

        for key in ("train", "val", "test", "minival"):
            value = dataset.get(key)
            if isinstance(value, str):
                value_path = Path(value)
                if value_path.is_absolute():
                    normalized[key] = str(value_path)
                else:
                    normalized[key] = str((dataset_root / value_path).resolve())
            elif isinstance(value, list):
                normalized[key] = [
                    str(Path(p)) if Path(p).is_absolute() else str((dataset_root / p).resolve()) for p in value
                ]

        missing_required = []
        for key in ("train", "val"):
            value = normalized.get(key)
            if isinstance(value, str) and not Path(value).exists():
                missing_required.append(Path(value))
            elif isinstance(value, list):
                for path_str in value:
                    p = Path(path_str)
                    if not p.exists():
                        missing_required.append(p)

        if missing_required and attempt_autofix and self._should_auto_preprocess(dataset_yaml):
            logger.warning("Detected missing processed dataset paths. Running KITTI preprocessing automatically...")
            self._run_kitti_preprocessing()
            return self._prepare_dataset_yaml_internal(attempt_autofix=False)

        if missing_required:
            preview = ", ".join(str(p) for p in missing_required[:3])
            if len(missing_required) > 3:
                preview += ", ..."
            raise FileNotFoundError(
                "Dataset split paths are missing: "
                f"{preview}. Run 'python -m src.data.preprocess_dataset --config configs/preprocessing.yaml' "
                "or set --data to a ready YOLO dataset YAML."
            )

        output_dir = self.save_dir if self.save_dir is not None else dataset_yaml.parent
        resolved_yaml = output_dir / "dataset_resolved.yaml"
        with open(resolved_yaml, "w") as f:
            yaml.safe_dump(normalized, f, sort_keys=False)

        self._resolved_data_yaml = resolved_yaml
        logger.info(f"Resolved dataset YAML written to: {resolved_yaml}")
        return resolved_yaml

    def _should_auto_preprocess(self, dataset_yaml: Path) -> bool:
        """Return True when we can safely build missing processed KITTI splits."""
        default_yaml = (self.project_root / "data/processed/kitti/data.yaml").resolve()
        raw_image_dir = self.project_root / "data/raw/KITTI/training/image_2"
        raw_label_dir = self.project_root / "data/raw/KITTI/training/label_2"

        return dataset_yaml.resolve() == default_yaml and raw_image_dir.exists() and raw_label_dir.exists()

    def _run_kitti_preprocessing(self) -> None:
        """Generate processed KITTI train/val/test splits when absent."""
        try:
            from src.data.preprocess_dataset import preprocess_dataset
        except Exception:
            # Fallback for environments where absolute import resolution differs.
            from ..data.preprocess_dataset import preprocess_dataset

        stats = preprocess_dataset(
            config_path="configs/preprocessing.yaml",
            project_root=str(self.project_root),
        )

        if stats.get("successful", 0) <= 0:
            raise RuntimeError("Automatic KITTI preprocessing completed with zero successful images.")

    def _save_results(self) -> None:
        """Save training results to the experiment directory."""
        if self.save_dir is None or self.results is None:
            return

        results_path = self.save_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Training results saved to: {results_path}")


def train(
    config_path: str | Path = "configs/training.yaml",
    project_root: str | Path | None = None,
    **override_kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """
    Convenience function to run training from a config file.

    Args:
        config_path: Path to the training YAML config.
        project_root: Project root directory.
        **override_kwargs: Override specific training arguments.

    Returns:
        Dictionary with training results and metrics.

    Example:
        >>> results = train("configs/training.yaml", epochs=50)
        >>> print(results["metrics"])
    """
    config = load_config(config_path)

    trainer = YOLOTrainer(
        config=config,
        config_path=str(config_path),
        project_root=str(project_root) if project_root else None,
    )

    trainer.setup()
    return trainer.train(**override_kwargs)
