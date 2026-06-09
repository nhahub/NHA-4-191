#!/usr/bin/env python3
"""Train YOLO with HPO-tuned params on Modal cloud GPU.

Usage:
    modal run scripts/train_modal.py --epochs 100

Download results:
    modal volume get road-sense-data /data/models/checkpoints/best_hpo.pt ./best_hpo.pt
    modal volume get road-sense-data /data/models/exports/best_hpo.onnx ./best_hpo.onnx
"""

from pathlib import Path

import modal

app = modal.App("road-sense-train")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "pillow>=10.0.0",
    )
)

data_volume = modal.Volume.from_name("road-sense-data", create_if_missing=True)


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


def build_train_args(hpo_cfg: dict, base_cfg: dict, epochs: int) -> dict:
    t = hpo_cfg.get("training", {})
    a = hpo_cfg.get("augmentation", {})
    b = base_cfg.get("data", {})
    r = base_cfg.get("regularization", {})

    return {
        "data": "/data/data/data/processed/kitti/data.yaml",
        "imgsz": b.get("imgsz", 640),
        "batch": b.get("batch_size", 16),
        "workers": b.get("workers", 4),
        "epochs": epochs,
        "patience": 50,
        "optimizer": t.get("optimizer", "auto"),
        "lr0": t.get("lr0", 0.01),
        "lrf": t.get("lrf", 0.01),
        "momentum": t.get("momentum", 0.937),
        "weight_decay": t.get("weight_decay", 0.0005),
        "warmup_epochs": 3.0,
        "hsv_h": a.get("hsv_h", 0.015),
        "hsv_s": a.get("hsv_s", 0.7),
        "hsv_v": a.get("hsv_v", 0.4),
        "degrees": a.get("degrees", 0.0),
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": a.get("mosaic", 1.0),
        "mixup": a.get("mixup", 0.0),
        "copy_paste": a.get("copy_paste", 0.0),
        "box": r.get("box", 7.5),
        "cls": r.get("cls", 1.0),
        "cls_pw": r.get("cls_pw", 1.0),
        "dfl": r.get("dfl", 1.5),
        "val": True,
        "device": "0",
        "exist_ok": True,
        "verbose": True,
        "project": "/data/experiments/train",
        "name": "hpo_best",
        "save": True,
        "save_period": 10,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 12,
    volumes={"/data": data_volume},
)
def train(epochs: int = 100):
    import yaml
    from ultralytics import YOLO

    # Fix dataset paths
    fix_data_yaml("/data/data/data/processed/kitti/data.yaml", "/data/data/data/processed/kitti")

    # Load HPO best config
    hpo_path = "/data/experiments/hpo/best_config.yaml"
    with open(hpo_path) as f:
        hpo_cfg = yaml.safe_load(f)
    print("HPO best config loaded:", hpo_cfg)

    # Load base config for defaults
    base_path = "/data/data/configs/training.yaml"
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    # Build training args
    args = build_train_args(hpo_cfg, base_cfg, epochs)
    print(f"\nStarting training: {epochs} epochs")
    for k, v in args.items():
        print(f"  {k}: {v}")

    # Train
    model = YOLO("yolo11m.pt")
    model.train(**args)

    # Export
    export_dir = Path("/data/models/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    pt_path = export_dir / "best_hpo.pt"
    onnx_path = export_dir / "best_hpo.onnx"

    model.save(str(pt_path))
    print(f"Model saved to {pt_path}")

    model.export(format="onnx", imgsz=args["imgsz"], half=True)
    print(f"ONNX exported to {onnx_path}")

    data_volume.commit()
    print("\nTraining complete. Results saved to volume.")


@app.local_entrypoint()
def main(epochs: int = 100):
    print(f"Starting training on Modal A10G — {epochs} epochs")
    train.remote(epochs=epochs)
