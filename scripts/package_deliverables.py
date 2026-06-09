#!/usr/bin/env python3
"""
Package deliverables for Road-Sense release.

Creates a structured deliverables directory with:
- Source code snapshot (git archive)
- Trained model weights (copied from checkpoints/)
- Documentation (PDF/HTML compilation)
- Docker image build script
- Export format summary

Usage:
    python scripts/package_deliverables.py [--version X.Y.Z]
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Road-Sense deliverables")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Release version (e.g., 1.0.0). Default: from git tag or date.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dist",
        help="Output directory for deliverables (default: dist/)",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker image build",
    )
    return parser.parse_args()


def get_version(args_version: str | None) -> str:
    if args_version:
        return args_version
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        if tag.stdout.strip():
            return tag.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return datetime.now().strftime("%Y%m%d")


def create_deliverables(version: str, output_dir: Path, skip_docker: bool) -> int:
    project_root = Path(__file__).resolve().parents[1]
    dist_dir = output_dir / f"road-sense-{version}"
    dist_dir.mkdir(parents=True, exist_ok=True)

    print(f"Packaging Road-Sense v{version}")
    print(f"Output: {dist_dir}")
    print("=" * 50)

    # 1. Source code archive
    print("\n[1/5] Creating source archive...")
    archive_name = f"road-sense-{version}-source"
    archive_path = output_dir / archive_name
    try:
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={archive_path}.zip", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
        )
        print(f"  ✓ {archive_path}.zip")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  ⚠ Git archive failed ({e}), copying files manually...")
        shutil.copytree(
            project_root,
            dist_dir / "source",
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                ".git",
                ".pytest_cache",
                ".ruff_cache",
                ".coverage",
                "*.pt",
                "*.onnx",
                "*.torchscript",
                "venv",
                ".venv",
            ),
        )
        print(f"  ✓ {dist_dir / 'source'}")

    # 2. Model weights
    print("\n[2/5] Copying model weights...")
    models_dir = dist_dir / "models"
    models_dir.mkdir(exist_ok=True)

    checkpoints_dir = project_root / "models" / "checkpoints"
    exports_dir = project_root / "models" / "exports"

    if checkpoints_dir.exists():
        for f in checkpoints_dir.iterdir():
            if f.suffix in (".pt", ".onnx"):
                dest = models_dir / f.name
                shutil.copy2(f, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  ✓ {f.name} ({size_mb:.1f} MB)")

    if exports_dir.exists():
        for f in exports_dir.iterdir():
            if f.suffix in (".pt", ".onnx", ".torchscript"):
                dest = models_dir / f.name
                shutil.copy2(f, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  ✓ {f.name} ({size_mb:.1f} MB)")

    # 3. Documentation
    print("\n[3/5] Collecting documentation...")
    docs_dir = dist_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Copy key markdown docs
    key_docs = [
        "README.md",
        "docs/QUICK_SETUP_GUIDE.md",
        "docs/DEPLOYMENT_GUIDE.md",
        "docs/DOCKER_USAGE.md",
        "docs/API_DOCUMENTATION.md",
        "docs/TRAINING_REPORT_EXP34332.md",
        "docs/PROJECT_DETAILS.md",
        "docs/MODEL_COMPARISON_REPORT.md",
        "reports/HPO_REPORT.md",
        "reports/HPO_TRAINING_REPORT.md",
        "reports/MILESTONE_2_EXECUTIVE_SUMMARY_EXP34332.md",
        "reports/MILESTONE_2_TECHNICAL_REPORT_EXP34332.md",
    ]
    for doc_rel in key_docs:
        src = project_root / doc_rel
        if src.exists():
            dest = docs_dir / src.name
            shutil.copy2(src, dest)
            print(f"  ✓ {src.name}")

    # 4. Docker image info
    print("\n[4/5] Preparing Docker artifacts...")
    docker_artifacts_dir = dist_dir / "docker"
    docker_artifacts_dir.mkdir(exist_ok=True)

    for f in ["Dockerfile", "docker-compose.yml", ".dockerignore"]:
        src = project_root / f
        if src.exists():
            shutil.copy2(src, docker_artifacts_dir / f)
            print(f"  ✓ {f}")

    # Build Docker image if not skipped
    if not skip_docker:
        print("  Building Docker image...")
        image_tag = f"road-sense:{version}"
        try:
            subprocess.run(
                ["docker", "build", "-t", image_tag, "."],
                cwd=project_root,
                check=True,
                capture_output=True,
            )
            print(f"  ✓ Docker image built: {image_tag}")

            # Save image to tar
            tar_path = dist_dir / f"road-sense-{version}.docker.tar"
            subprocess.run(
                ["docker", "save", "-o", str(tar_path), image_tag],
                check=True,
                capture_output=True,
            )
            print(f"  ✓ Image saved: {tar_path.name}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  ⚠ Docker build failed ({e}), skipping...")
    else:
        print("  ⚠ Docker build skipped (--skip-docker)")

    # 5. Summary
    print("\n[5/5] Creating summary...")
    readme = dist_dir / "README.md"
    readme.write_text(
        f"# Road-Sense v{version} Deliverables\n\n"
        f"Packaged: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "## Contents\n"
        f"- Source archive: `road-sense-{version}-source.zip`\n"
        f"- Model weights under `models/`\n"
        f"- Documentation under `docs/`\n"
        f"- Docker artifacts under `docker/`\n"
        "\n## Quick Start\n"
        "1. Extract source archive\n"
        "2. Copy model weights to `models/checkpoints/`\n"
        "3. `pip install -r requirements.txt`\n"
        "4. `python src/models/api_server.py`\n"
    )
    print(f"  ✓ Summary: {readme}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file())
    print(f"\n{'=' * 50}")
    print(f"Deliverables packaged: {dist_dir}")
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    print("Done.")
    return 0


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    version = get_version(args.version)
    return create_deliverables(version, output_dir, args.skip_docker)


if __name__ == "__main__":
    sys.exit(main())
