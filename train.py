#!/usr/bin/env python3
"""
Training Entry Point - Road-Sense

CLI interface for training YOLO object detection models.

Usage:
    # Train with default config
    python train.py

    # Train with custom config
    python train.py --config configs/training.yaml

    # Override specific parameters
    python train.py --epochs 50 --batch-size 8 --device 0

    # Resume training from checkpoint
    python train.py --resume models/checkpoints/last.pt

    # Dry run (print config without training)
    python train.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.model_factory import list_available_models
from src.models.trainer import YOLOTrainer, load_config

logger = logging.getLogger(__name__)


class TeeStream:
    """Duplicate writes to multiple streams (e.g., terminal + log file)."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLO object detection model for Road-Sense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python train.py

  # Train with custom config and overrides
  python train.py --config configs/training.yaml --epochs 50 --batch-size 8

  # Use a different model
  python train.py --model yolo11s --epochs 100

  # Resume training from checkpoint
  python train.py --resume models/checkpoints/last.pt

  # List available models
  python train.py --list-models
        """,
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training YAML config (default: configs/training.yaml)",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=("Path to combined console log file. Default: logs/train_<timestamp>.log"),
    )

    # Model overrides
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model variant (e.g., yolo11m, yolov8s). Overrides config.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained weights. Overrides config.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use pretrained weights",
    )

    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Overrides config.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. Overrides config.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size. Overrides config.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (e.g., 0, cpu, 0,1). Overrides config.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader workers. Overrides config.",
    )

    # Data override
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML. Overrides config data.yaml_path.",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Utilities
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model variants and exit",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress logging output",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = True, quiet: bool = False) -> None:
    """Configure logging based on CLI arguments."""
    from src.utils import setup_logging as _setup_logging

    _setup_logging(verbose=verbose)
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)


def setup_log_file(log_file: str | None, project_root: Path) -> tuple[Path, TextIO]:
    """Create a log file and tee stdout/stderr to terminal + file."""
    if log_file:
        target = Path(log_file)
        if not target.is_absolute():
            target = project_root / target
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = project_root / "logs" / f"train_{timestamp}.log"

    target.parent.mkdir(parents=True, exist_ok=True)
    file_stream = target.open("a", encoding="utf-8")

    sys.stdout = TeeStream(sys.__stdout__, file_stream)
    sys.stderr = TeeStream(sys.__stderr__, file_stream)

    return target, file_stream


def list_models() -> None:
    """Print available model variants."""
    models = list_available_models()
    print("\nAvailable YOLO Models:")
    print("-" * 50)
    print(f"{'Model':<15} {'Params (M)':<12} {'Size (MB)':<12}")
    print("-" * 50)
    for m in models:
        print(f"{m['name']:<15} {m['params_m']:<12.1f} {m['size_mb']:<12.1f}")
    print("-" * 50)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides to the configuration."""
    # Model overrides
    if args.model:
        config["model"]["name"] = args.model
    if args.weights:
        config["model"]["pretrained_weights"] = args.weights
    if args.no_pretrained:
        config["model"]["pretrained"] = False

    # Training overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.imgsz:
        config["data"]["imgsz"] = args.imgsz
    if args.device:
        config["device"]["device"] = args.device
    if args.workers:
        config["data"]["workers"] = args.workers

    # Data override
    if args.data:
        config["data"]["yaml_path"] = args.data

    return config


def _handle_resume(config: dict, resume: str) -> bool:
    resume_path = Path(resume)
    if not resume_path.exists():
        logger.error(f"Checkpoint not found: {resume_path}")
        return False
    config["model"]["pretrained_weights"] = str(resume_path)
    config["model"]["pretrained"] = False
    config["advanced"]["resume"] = True
    logger.info(f"Resuming training from: {resume_path}")
    return True


def _run_training(config: dict, config_path: str, project_root: Path) -> int:
    try:
        trainer = YOLOTrainer(
            config=config,
            config_path=config_path,
            project_root=str(project_root),
        )
        trainer.setup()
        results = trainer.train()

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: {config['model']['name']}")
        print(f"Epochs: {results['epochs_trained']}")
        print(f"Save directory: {results['save_dir']}")

        if results["metrics"]:
            print("\nMetrics:")
            for key, value in results["metrics"].items():
                print(f"  {key}: {value}")
        print("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    log_stream: TextIO | None = None

    try:
        log_file_path, log_stream = setup_log_file(args.log_file, project_root)
        setup_logging(verbose=args.verbose, quiet=args.quiet)
        logger.info(f"Console output is being logged to: {log_file_path}")

        if args.list_models:
            list_models()
            return 0

        logger.info(f"Loading config from: {args.config}")
        try:
            config = load_config(args.config)
        except FileNotFoundError as e:
            logger.error(f"Config error: {e}")
            return 1

        config = apply_overrides(config, args)

        if args.dry_run:
            print("\nTraining Configuration:")
            print("=" * 60)
            print(json.dumps(config, indent=2, default=str))
            print("=" * 60)
            return 0

        if args.resume and not _handle_resume(config, args.resume):
            return 1

        return _run_training(config, args.config, project_root)

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_stream is not None:
            log_stream.flush()
            log_stream.close()


if __name__ == "__main__":
    sys.exit(main())
