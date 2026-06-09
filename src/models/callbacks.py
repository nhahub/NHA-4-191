"""
Training Callbacks - Road-Sense

Callback classes for training lifecycle events:
- Logging and progress tracking
- Model checkpointing
- Metrics monitoring
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Callback for logging training progress and events.
    """

    def __init__(self) -> None:
        self.train_start_time: datetime | None = None

    def on_train_start(self, config: dict[str, Any]) -> None:
        """Called when training begins."""
        self.train_start_time = datetime.now()

        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})

        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info("=" * 60)
        logger.info(f"Model: {model_cfg.get('name', 'unknown')}")
        logger.info(f"Dataset: {data_cfg.get('yaml_path', 'unknown')}")
        logger.info(f"Epochs: {train_cfg.get('epochs', 'unknown')}")
        logger.info(f"Batch size: {data_cfg.get('batch_size', 'unknown')}")
        logger.info(f"Image size: {data_cfg.get('imgsz', 'unknown')}")
        logger.info(f"Device: {config.get('device', {}).get('device', 'auto')}")
        logger.info(f"Start time: {self.train_start_time.isoformat()}")
        logger.info("=" * 60)

    def on_train_end(self, results: dict[str, Any]) -> None:
        """Called when training completes."""
        end_time = datetime.now()
        duration = end_time - self.train_start_time if self.train_start_time else None

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration}")
        logger.info(f"End time: {end_time.isoformat()}")

        metrics = results.get("metrics", {})
        if metrics:
            logger.info("Final Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")

        logger.info("=" * 60)

    def on_train_error(self, error: Exception) -> None:
        """Called when training fails."""
        logger.error("=" * 60)
        logger.error("TRAINING FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {error}")
        logger.error(f"Time: {datetime.now().isoformat()}")
        logger.error("=" * 60)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v is not None)
        logger.info(f"Epoch {epoch}: {metric_str}")


class ModelCheckpoint:
    """
    Callback for saving model checkpoints during training.

    Handles:
    - Saving best model based on a metric
    - Saving last checkpoint
    - Periodic checkpointing
    """

    def __init__(
        self,
        save_dir: Path,
        save_best: bool = True,
        save_last: bool = True,
        save_period: int = -1,
        metric: str = "mAP50-95",
    ) -> None:
        """
        Initialize the checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints.
            save_best: Whether to save the best model.
            save_last: Whether to save the last model.
            save_period: Save every N epochs (-1 = disabled).
            metric: Metric to track for best model selection.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_best = save_best
        self.save_last = save_last
        self.save_period = save_period
        self.metric = metric

        self.best_metric_value = float("-inf")
        self.best_model_path: Path | None = None

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """
        Called at the end of each epoch. Decides whether to save.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of epoch metrics.
        """
        # Save last model
        if self.save_last:
            self._save_checkpoint("last.pt")

        # Save periodic checkpoint
        if self.save_period > 0 and epoch % self.save_period == 0:
            self._save_checkpoint(f"epoch_{epoch}.pt")

        # Save best model
        if self.save_best and self.metric in metrics:
            current_value = metrics[self.metric]
            if current_value > self.best_metric_value:
                self.best_metric_value = current_value
                self._save_checkpoint("best.pt")
                self.best_model_path = self.save_dir / "best.pt"
                logger.info(f"New best model: {self.metric}={current_value:.4f} (saved to best.pt)")

    def _save_checkpoint(self, filename: str) -> None:
        """
        Save a checkpoint file.

        Note: Ultralytics YOLO handles checkpoint saving internally.
        This callback tracks which checkpoints to save and manages
        the best model logic. Actual saving is done by the trainer.

        Args:
            filename: Name for the checkpoint file.
        """
        # Ultralytics saves checkpoints automatically during training.
        # This method is a placeholder for custom checkpoint logic.
        pass

    def get_best_model_path(self) -> Path | None:
        """Return the path to the best model checkpoint."""
        if self.best_model_path and self.best_model_path.exists():
            return self.best_model_path

        # Fallback: check for best.pt in save_dir
        fallback = self.save_dir / "best.pt"
        if fallback.exists():
            return fallback

        return None

    def get_checkpoint_summary(self) -> dict[str, Any]:
        """Return a summary of saved checkpoints."""
        checkpoints = []
        if self.save_dir.exists():
            for ckpt in self.save_dir.glob("*.pt"):
                stat = ckpt.stat()
                checkpoints.append(
                    {
                        "name": ckpt.name,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

        return {
            "save_dir": str(self.save_dir),
            "best_metric": self.best_metric_value if self.best_metric_value != float("-inf") else None,
            "best_model": str(self.best_model_path) if self.best_model_path else None,
            "checkpoints": checkpoints,
        }
