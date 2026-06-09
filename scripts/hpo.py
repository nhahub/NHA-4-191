#!/usr/bin/env python3
"""YOLO Hyperparameter Optimization with Optuna.

Usage:
    # Quick smoke test:
    python scripts/hpo.py --trials 3 --epochs 3

    # Full Stage 1 (coarse search):
    python scripts/hpo.py --trials 20 --epochs 10

    # Stage 2 (fine search around best params):
    python scripts/hpo.py --trials 10 --epochs 20 --stage 2

    # Final training with best config:
    python train.py --config experiments/hpo/best_config.yaml --epochs 100
"""

from __future__ import annotations

import os

# Fix Lambda Labs NVML issue: pop CUDA_VISIBLE_DEVICES before torch import
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import optuna
import torch
import yaml
from optuna.pruners import MedianPruner
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hpo")


STAGE1_SPACE = {
    "lr0": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "lrf": {"type": "float", "low": 0.01, "high": 0.2},
    "optimizer": {"type": "categorical", "choices": ["SGD", "Adam", "AdamW"]},
    "momentum": {"type": "float", "low": 0.8, "high": 0.98},
    "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "mosaic": {"type": "float", "low": 0.0, "high": 1.0},
    "mixup": {"type": "float", "low": 0.0, "high": 1.0},
    "copy_paste": {"type": "float", "low": 0.0, "high": 0.5},
    "degrees": {"type": "float", "low": 0.0, "high": 45.0},
    "hsv_h": {"type": "float", "low": 0.0, "high": 0.1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO Hyperparameter Optimization")
    parser.add_argument("--trials", type=int, default=20, help="Number of HPO trials")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="Search stage")
    parser.add_argument("--base-config", type=str, default="configs/training.yaml", help="Base training config")
    parser.add_argument("--output", type=str, default="experiments/hpo", help="Output directory")
    parser.add_argument("--device", type=str, default="", help="CUDA device (auto-detected if empty)")
    return parser.parse_args()


def load_base_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def suggest_params(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        method = spec["type"]
        if method == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif method == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def apply_params(config: dict, params: dict) -> dict:
    config = copy.deepcopy(config)
    cfg_keys = {
        "lr0": ("training",),
        "lrf": ("training",),
        "optimizer": ("training",),
        "momentum": ("training",),
        "weight_decay": ("training",),
        "mosaic": ("augmentation",),
        "mixup": ("augmentation",),
        "copy_paste": ("augmentation",),
        "degrees": ("augmentation",),
        "hsv_h": ("augmentation",),
    }
    for key, value in params.items():
        section = cfg_keys.get(key, ("training",))
        parent = config
        for s in section:
            if s not in parent:
                parent[s] = {}
            parent = parent[s]
        parent[key] = value
    return config


def build_train_args(config: dict, epochs: int, device: str) -> dict:
    t = config["training"]
    d = config["data"]
    a = config["augmentation"]
    r = config.get("regularization", {})
    v = config.get("validation", {})
    dev = config.get("device", {})

    args = {
        "data": d.get("yaml_path", "data/processed/kitti/data.yaml"),
        "imgsz": d.get("imgsz", 640),
        "batch": d.get("batch_size", 16),
        "epochs": epochs,
        "patience": t.get("patience", 30),
        "optimizer": t.get("optimizer", "auto"),
        "lr0": t["lr0"],
        "lrf": t["lrf"],
        "momentum": t["momentum"],
        "weight_decay": t["weight_decay"],
        "warmup_epochs": t.get("warmup_epochs", 3.0),
        "hsv_h": a["hsv_h"],
        "hsv_s": a.get("hsv_s", 0.7),
        "hsv_v": a.get("hsv_v", 0.4),
        "degrees": a["degrees"],
        "translate": a.get("translate", 0.1),
        "scale": a.get("scale", 0.5),
        "shear": a.get("shear", 0.0),
        "perspective": a.get("perspective", 0.0),
        "flipud": a.get("flipud", 0.0),
        "fliplr": a.get("fliplr", 0.5),
        "mosaic": a["mosaic"],
        "mixup": a["mixup"],
        "copy_paste": a["copy_paste"],
        "box": r.get("box", 7.5),
        "cls": r.get("cls", 1.0),
        "cls_pw": r.get("cls_pw", 1.0),
        "dfl": r.get("dfl", 1.5),
        "val": v.get("val_interval", 1) > 0,
        "device": dev.get("device", device),
        "workers": d.get("workers", 4),
        "exist_ok": True,
        "verbose": False,
        "project": "runs/hpo",
        "name": f"trial_{int(time.time())}",
    }
    return args  # noqa: RET504


def run_trial(config: dict, epochs: int, device: str) -> float:
    model = YOLO(config["model"]["name"] + ".pt")
    train_args = build_train_args(config, epochs, device)
    model.train(**train_args)
    metrics = model.val()
    return float(metrics.box.map)


def create_best_config(base_config: dict, best_params: dict, output_path: Path) -> None:
    config = apply_params(base_config, best_params)
    config["training"]["epochs"] = 100
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Best config saved to %s", output_path)


def resolve_device(args_device: str) -> str:
    if args_device and torch.cuda.is_available():
        return args_device
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_base_config(args.base_config)
    space = STAGE1_SPACE
    results: list[dict] = []

    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="road_sense_hpo")

    logger.info("Starting HPO — %d trials, %d epochs each", args.trials, args.epochs)
    logger.info("Search space: %s", list(space.keys()))

    for trial_idx in range(args.trials):
        trial = study.ask()
        params = suggest_params(trial, space)
        config = apply_params(base_config, params)

        try:
            map50_95 = run_trial(config, args.epochs, device)
            study.tell(trial, map50_95)
            logger.info("Trial %d/%d — mAP@50:95=%.4f — params=%s", trial_idx + 1, args.trials, map50_95, params)
        except Exception as e:
            logger.error("Trial %d failed: %s", trial_idx + 1, e)
            study.tell(trial, float("-inf"))
            map50_95 = None

        trial_result = {
            "trial": trial.number,
            "params": params,
            "value": map50_95,
            "state": "COMPLETE" if map50_95 is not None else "FAIL",
        }
        results.append(trial_result)

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    best = study.best_trial
    logger.info("=" * 60)
    logger.info("HPO COMPLETE")
    logger.info("=" * 60)
    logger.info("Best trial: %d", best.number)
    logger.info("Best mAP@50:95: %.4f", best.value)
    logger.info("Best params:")
    for k, v in best.params.items():
        logger.info("  %s: %s", k, v)

    best_config_path = output_dir / "best_config.yaml"
    create_best_config(base_config, best.params, best_config_path)

    summary = {
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": args.trials,
        "n_epochs": args.epochs,
        "stage": args.stage,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
