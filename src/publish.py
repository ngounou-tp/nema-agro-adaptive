"""Publish the model and dataset on Hugging Face AND Kaggle (both are required by the challenge rules).

Before: run src/make_cards.py and remove every REPLACE from release/<project>/.
Needs:  HF_TOKEN (write token) and Kaggle credentials (KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json).

Commands:
  python -m src.publish hf-dataset    --project agro
  python -m src.publish hf-model-card --project agro           # adds README, prompts and results to the adapter repo
  python -m src.publish hf-model      --project agro --adapter-dir path/to/adapter   # only if you did NOT use Adaption's one-click export
  python -m src.publish kaggle-model  --project agro
  python -m src.publish kaggle-dataset --project agro
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from src.common import ROOT


def cfg_for(project):
    return json.loads((ROOT / "configs" / f"project_{project}.json").read_text(encoding="utf-8"))


def check_no_replace(folder: Path):
    bad = [p.name for p in folder.glob("*.md") if any(w in p.read_text(encoding="utf-8") for w in ("REPLACE", "PASTE"))]
    if bad:
        raise SystemExit(f"Fill the REPLACE / PASTE placeholders first in {folder}: {bad}")


def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hf-dataset", "hf-model-card", "hf-model", "kaggle-model", "kaggle-dataset"])
    ap.add_argument("--project", choices=["agro", "waste"], required=True)
    ap.add_argument("--adapter-dir", default=None)
    a = ap.parse_args()
    cfg = cfg_for(a.project)
    rel = ROOT / "release" / a.project

    if a.cmd == "hf-dataset":
        check_no_replace(rel / "dataset")
        api = hf_api()
        api.create_repo(cfg["hf_dataset_repo"], repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(rel / "dataset"), repo_id=cfg["hf_dataset_repo"], repo_type="dataset",
                          commit_message="Release dataset + card")
        print(f"https://huggingface.co/datasets/{cfg['hf_dataset_repo']}")

    elif a.cmd in ("hf-model-card", "hf-model"):
        check_no_replace(rel / "model")
        api = hf_api()
        api.create_repo(cfg["hf_model_repo"], repo_type="model", exist_ok=True)
        if a.cmd == "hf-model":
            if not a.adapter_dir:
                raise SystemExit("--adapter-dir is required for hf-model")
            api.upload_folder(folder_path=a.adapter_dir, repo_id=cfg["hf_model_repo"], commit_message="Upload LoRA adapter")
        api.upload_folder(folder_path=str(rel / "model"), repo_id=cfg["hf_model_repo"], commit_message="Model card, prompts, results")
        print(f"https://huggingface.co/{cfg['hf_model_repo']}")

    elif a.cmd == "kaggle-model":
        check_no_replace(rel / "model")
        import kagglehub
        from huggingface_hub import snapshot_download
        tmp = Path(tempfile.mkdtemp())
        snapshot_download(cfg["hf_model_repo"], local_dir=tmp, token=os.environ.get("HF_TOKEN"))
        shutil.rmtree(tmp / ".cache", ignore_errors=True)
        for f in (rel / "model").iterdir():
            shutil.copy(f, tmp / f.name)
        kagglehub.model_upload(cfg["kaggle_model_handle"], str(tmp), license_name="Apache 2.0",
                               version_notes="Adaptive Intelligence Challenge release")
        print(f"https://www.kaggle.com/models/{'/'.join(cfg['kaggle_model_handle'].split('/')[:2])}")

    elif a.cmd == "kaggle-dataset":
        check_no_replace(rel / "dataset")
        import kagglehub
        kagglehub.dataset_upload(cfg["kaggle_dataset_handle"], str(rel / "dataset"),
                                 version_notes="Adaptive Intelligence Challenge release")
        print(f"https://www.kaggle.com/datasets/{cfg['kaggle_dataset_handle']}")


if __name__ == "__main__":
    main()
