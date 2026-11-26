#!/usr/bin/env python3
"""
Sync this Git project to a private Kaggle dataset.

- Uploads only files tracked by Git (so .gitignore'd things like .env,
  node_modules, venv, data dumps are never sent).
- Creates the dataset the first time, then pushes a new version on later runs.

Usage:
    python kaggle_sync.py                 # uses defaults below
    python kaggle_sync.py -m "new model"  # custom version message

Requirements:
    pip install kaggle
    Kaggle API token in ~/.kaggle/kaggle.json
    (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
    or env vars KAGGLE_USERNAME and KAGGLE_KEY.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---- Edit these two lines -------------------------------------------------
DATASET_SLUG = "my-project-code"      # lowercase, dashes, 6-50 chars
DATASET_TITLE = "My Project Code"     # 6-50 chars
# ---------------------------------------------------------------------------

BUILD_DIR = Path(".kaggle_build")


def run(cmd, check=True, capture=False):
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True,
                          capture_output=capture)


def kaggle_username():
    user = os.environ.get("KAGGLE_USERNAME")
    if user:
        return user
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    sys.exit("No Kaggle credentials found. Put kaggle.json in ~/.kaggle/ "
             "or set KAGGLE_USERNAME and KAGGLE_KEY.")


def stage_tracked_files():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()
    files = run(["git", "ls-files"], capture=True).stdout.splitlines()
    for f in files:
        src = Path(f)
        if not src.is_file():          # deleted but not yet committed
            continue
        dst = BUILD_DIR / src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"Staged {len(files)} tracked files in {BUILD_DIR}/")


def write_metadata(dataset_id):
    meta = {
        "title": DATASET_TITLE,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (BUILD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def dataset_exists(dataset_id):
    r = run(["kaggle", "datasets", "files", dataset_id],
            check=False, capture=True)
    return r.returncode == 0 and "404" not in (r.stdout + r.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default=None,
                    help="version message (default: last commit subject)")
    args = ap.parse_args()

    if shutil.which("kaggle") is None:
        sys.exit("Kaggle CLI not found. Run: pip install kaggle")

    dataset_id = f"{kaggle_username()}/{DATASET_SLUG}"
    message = args.message or run(
        ["git", "log", "-1", "--pretty=%s"], capture=True).stdout.strip()

    stage_tracked_files()
    write_metadata(dataset_id)

    if dataset_exists(dataset_id):
        run(["kaggle", "datasets", "version", "-p", str(BUILD_DIR),
             "-m", message or "update", "--dir-mode", "zip"])
    else:
        # New datasets are private unless you pass --public
        run(["kaggle", "datasets", "create", "-p", str(BUILD_DIR),
             "--dir-mode", "zip"])

    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    print(f"Done: https://www.kaggle.com/datasets/{dataset_id}")


if __name__ == "__main__":
    main()