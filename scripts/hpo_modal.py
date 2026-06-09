#!/usr/bin/env python3
"""Run YOLO HPO on Modal cloud GPU.

Setup (one-time):
    pip install modal
    python3 -m modal setup
    modal deploy scripts/hpo_modal.py

Upload dataset to Modal Volume:
    modal volume create road-sense-data
    modal volume put road-sense-data data data
    modal volume put road-sense-data configs configs
    modal volume put road-sense-data models models

Run HPO:
    modal run scripts/hpo_modal.py --trials 20 --epochs 10

Download results:
    modal volume get road-sense-data experiments/hpo/results.json ./results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("road-sense-hpo")

hpo_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3.0",
        "optuna>=4.0.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    )
)

data_volume = modal.Volume.from_name("road-sense-data", create_if_missing=True)

HPO_SEARCH_SPACE = {
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


def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def fix_data_yaml(original_path: str, data_root: str) -> str:
    import yaml

    with open(original_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["path"] = data_root
    cfg["train"] = "images/train"
    cfg["val"] = "images/val"
    cfg.pop("test", None)
    with open(original_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Fixed data.yaml: path={data_root}, train=images/train, val=images/val")
    return original_path


def suggest_and_apply(trial, base_cfg: dict, space: dict) -> dict:
    import copy

    cfg = copy.deepcopy(base_cfg)
    section_map = {
        "lr0": "training",
        "lrf": "training",
        "optimizer": "training",
        "momentum": "training",
        "weight_decay": "training",
        "mosaic": "augmentation",
        "mixup": "augmentation",
        "copy_paste": "augmentation",
        "degrees": "augmentation",
        "hsv_h": "augmentation",
    }
    for name, spec in space.items():
        if spec["type"] == "float":
            value = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "categorical":
            value = trial.suggest_categorical(name, spec["choices"])
        else:
            continue
        section = section_map[name]
        cfg[section][name] = value
    return cfg


def build_train_args(cfg: dict, epochs: int) -> dict:
    t = cfg["training"]
    d = cfg["data"]
    a = cfg["augmentation"]
    r = cfg.get("regularization", {})
    return {
        "data": d.get("yaml_path", "/data/data/data/processed/kitti/data.yaml"),
        "imgsz": d.get("imgsz", 640),
        "batch": d.get("batch_size", 16),
        "epochs": epochs,
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
        "flipud": a.get("flipud", 0.0),
        "fliplr": a.get("fliplr", 0.5),
        "mosaic": a["mosaic"],
        "mixup": a["mixup"],
        "copy_paste": a["copy_paste"],
        "box": r.get("box", 7.5),
        "cls": r.get("cls", 1.0),
        "dfl": r.get("dfl", 1.5),
        "device": "0",
        "workers": 4,
        "exist_ok": True,
        "verbose": False,
        "project": "/tmp/hpo",  # noqa: S108
        "name": f"trial_{int(time.time())}",
    }


def run_trial(cfg: dict, epochs: int) -> float:
    from ultralytics import YOLO

    model = YOLO(cfg["model"]["name"] + ".pt")
    args = build_train_args(cfg, epochs)
    model.train(**args)
    data_yaml = args["data"]
    metrics = model.val(data=data_yaml)
    return float(metrics.box.map)


def run_stage(study, base_cfg: dict, space: dict, trials: int, epochs: int, output_dir: Path):
    results = load_results(output_dir)
    start = len(results)
    print(f"\n{'=' * 60}", flush=True)
    total = start + trials
    print(f"Stage: {trials} trials × {epochs} epochs (trials {start + 1}-{total})", flush=True)
    print(f"Search space: {list(space.keys())}", flush=True)
    print(f"{'=' * 60}", flush=True)

    for trial_idx in range(start, total):
        trial = study.ask()
        cfg = suggest_and_apply(trial, base_cfg, space)
        print(f"\n--- Starting trial {trial_idx + 1}/{total} ---", flush=True)

        map50_95 = None
        try:
            map50_95 = run_trial(cfg, epochs)
            try:
                study.tell(trial, map50_95)
            except Exception as tell_err:
                print(f"  study.tell failed: {tell_err}", flush=True)
            print(f"✓ Trial {trial_idx + 1}/{total} — mAP@50:95={map50_95:.4f}", flush=True)
        except Exception as e:
            print(f"✗ Trial {trial_idx + 1}/{total} failed: {e}", flush=True)
            try:
                study.tell(trial, float("-inf"))
            except Exception:  # noqa: S110
                pass

        trial_result = {
            "trial": trial.number,
            "params": trial.params,
            "value": map50_95,
            "state": "COMPLETE" if map50_95 is not None else "FAIL",
        }
        results.append(trial_result)
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        data_volume.commit()
    return results


def build_narrow_space(best_params: dict) -> dict:
    narrow = {}
    for k, spec in HPO_SEARCH_SPACE.items():
        if spec["type"] != "float":
            narrow[k] = spec
            continue
        best_val = best_params.get(k, spec.get("low", 0))
        span = (spec["high"] - spec["low"]) * 0.2
        low = max(spec["low"], best_val - span)
        high = min(spec["high"], best_val + span)
        narrow[k] = {**spec, "low": low, "high": high}
    return narrow


def load_results(output_dir: Path) -> list:
    results_file = output_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return []


def save_best_config(best_params: dict, output_dir: Path, epochs: int = 100):
    import yaml

    cfg = {}
    section_map = {
        "lr0": "training",
        "lrf": "training",
        "optimizer": "training",
        "momentum": "training",
        "weight_decay": "training",
        "mosaic": "augmentation",
        "mixup": "augmentation",
        "copy_paste": "augmentation",
        "degrees": "augmentation",
        "hsv_h": "augmentation",
    }
    for k, v in best_params.items():
        section = section_map[k]
        if section not in cfg:
            cfg[section] = {}
        cfg[section][k] = v
    cfg["training"]["epochs"] = epochs
    path = output_dir / "best_config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Best config saved to {path}")


@app.function(
    image=hpo_image,
    gpu="A100-80GB:1",
    timeout=3600 * 4,
    volumes={"/data": data_volume},
)
def run_hpo(trials: int = 20, epochs: int = 10, stage: int = 1, base_config: str = "/data/data/configs/training.yaml"):
    import optuna
    from optuna.pruners import MedianPruner

    output_dir = Path("/data/experiments/hpo")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(base_config)
    kitti_yaml = "/data/data/data/processed/kitti/data.yaml"
    fixed_yaml = fix_data_yaml(kitti_yaml, "/data/data/data/processed/kitti")
    base_cfg["data"]["yaml_path"] = fixed_yaml

    if stage == 1:
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
        study = optuna.create_study(direction="maximize", pruner=pruner, study_name="road_sense_hpo")
        _ = run_stage(study, base_cfg, HPO_SEARCH_SPACE, trials, epochs, output_dir)
    elif stage == 2:
        results = load_results(output_dir)
        if not results:
            print("No Stage 1 results found. Run Stage 1 first.")
            return
        best_params = max((r for r in results if r["value"] is not None), key=lambda r: r["value"], default=None)
        if not best_params:
            print("No successful trials from Stage 1.")
            return
        best_params = best_params["params"]
        print(f"Narrowing space around best Stage 1 params: {best_params}")
        narrow_space = build_narrow_space(best_params)
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=2)
        study = optuna.create_study(direction="maximize", pruner=pruner, study_name="road_sense_hpo")
        _ = run_stage(study, base_cfg, narrow_space, trials, epochs, output_dir)

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        print("\nNo trials completed successfully. Check the logs above for errors.")
        data_volume.commit()
        return

    best = study.best_trial
    print("\n" + "=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    print(f"Best trial: {best.number}")
    print(f"Best mAP@50:95: {best.value:.4f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    save_best_config(best.params, output_dir)
    summary = {
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": trials,
        "n_epochs": epochs,
        "stage": stage,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    data_volume.commit()


@app.local_entrypoint()
def main(trials: int = 20, epochs: int = 10, stage: int = 1):
    print(f"Running HPO Stage {stage} on Modal — {trials} trials, {epochs} epochs each")
    run_hpo.remote(trials=trials, epochs=epochs, stage=stage)
