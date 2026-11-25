"""Optional command-line client for the Adaption Adaptive Data REST API.

You can do everything in the Adaption web app instead (the playbook shows both). Use this script when you want
repeatable runs and a log of exactly what you sent.

IMPORTANT: field names below follow the public OpenAPI description of the Datasets API (v0.4.0). The platform moves
fast: before your first real run, compare them with the API reference linked in your Adaption dashboard, or with
`pip install adaption` (official SDK, see its api.md). Every command supports --dry-run to print the request only.

Setup:  export ADAPTION_API_KEY=pt_live_...      (or put it in .env)
Examples:
    python -m src.adaption_client list
    python -m src.adaption_client create-hf --name nema-agro-structured --url https://huggingface.co/datasets/<user>/nema-agro-seed --files adaption_seed.jsonl
    python -m src.adaption_client run --dataset <id> --config configs/run_agro_structured.json --estimate
    python -m src.adaption_client run --dataset <id> --config configs/run_agro_structured.json
    python -m src.adaption_client status --dataset <id>
    python -m src.adaption_client evaluation --dataset <id>
    python -m src.adaption_client download --dataset <id> --format jsonl --out data/build/agro/adapted_structured.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.environ.get("ADAPTION_BASE_URL", "https://api.prod.adaptionlabs.ai")


def load_env():
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def headers():
    key = os.environ.get("ADAPTION_API_KEY")
    if not key:
        sys.exit("ADAPTION_API_KEY is not set (see .env.example)")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def call(method, path, body=None, params=None, dry=False, stream_to=None):
    url = f"{BASE_URL}{path}"
    if dry:
        print(json.dumps({"method": method, "url": url, "params": params, "body": body}, indent=2, ensure_ascii=False))
        return None
    r = requests.request(method, url, headers=headers(), json=body, params=params, timeout=120, stream=bool(stream_to))
    if r.status_code >= 400:
        sys.exit(f"HTTP {r.status_code}: {r.text[:2000]}")
    if stream_to:
        Path(stream_to).parent.mkdir(parents=True, exist_ok=True)
        ctype = r.headers.get("content-type", "")
        if "application/json" in ctype and "url" in (data := r.json()):  # some APIs return a signed link instead
            r = requests.get(data["url"], timeout=600)
            Path(stream_to).write_bytes(r.content)
        else:
            Path(stream_to).write_bytes(r.content)
        print(f"saved {stream_to}")
        return None
    data = r.json() if r.content else {}
    print(json.dumps(data, indent=2, ensure_ascii=False)[:6000])
    return data


def main():
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("create-hf")
    c.add_argument("--name", required=True)
    c.add_argument("--url", required=True, help="https://huggingface.co/datasets/<user>/<repo>")
    c.add_argument("--files", nargs="+", required=True)
    k = sub.add_parser("create-kaggle")
    k.add_argument("--name", required=True)
    k.add_argument("--url", required=True)
    k.add_argument("--files", nargs="+", required=True)
    for name in ("status", "evaluation", "get"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--config", required=True, help="JSON file with the run body (see configs/)")
    r.add_argument("--estimate", action="store_true", help="only return the credit estimate, do not start")
    w = sub.add_parser("wait")
    w.add_argument("--dataset", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dataset", required=True)
    d.add_argument("--format", default="jsonl", choices=["csv", "json", "jsonl", "parquet"])
    d.add_argument("--out", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--dataset", required=True)
    p.add_argument("--target", choices=["huggingface", "kaggle"], required=True)
    for sp in sub.choices.values():
        sp.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run

    if a.cmd == "list":
        call("GET", "/api/v1/datasets", params={"limit": 50}, dry=dry)
    elif a.cmd in ("create-hf", "create-kaggle"):
        src_type = "huggingface" if a.cmd == "create-hf" else "kaggle"
        call("POST", "/api/v1/datasets", {"name": a.name, "source": {"type": src_type, "url": a.url, "files": a.files}}, dry=dry)
    elif a.cmd == "get":
        call("GET", f"/api/v1/datasets/{a.dataset}", dry=dry)
    elif a.cmd == "status":
        call("GET", f"/api/v1/datasets/{a.dataset}/status", dry=dry)
    elif a.cmd == "evaluation":
        call("GET", f"/api/v1/datasets/{a.dataset}/evaluation", dry=dry)
    elif a.cmd == "run":
        body = json.loads(Path(a.config).read_text(encoding="utf-8"))
        body.pop("_comment", None)
        bc = body.get("brand_controls", {})
        if str(bc.get("blueprint", "")).startswith("PASTE "):  # fill the Blueprint text from the file named in the config
            path = Path(__file__).resolve().parents[1] / bc["blueprint"].split()[1]
            bc["blueprint"] = path.read_text(encoding="utf-8").strip()
        body["estimate"] = bool(a.estimate)
        call("POST", f"/api/v1/datasets/{a.dataset}/run", body, dry=dry)
    elif a.cmd == "wait":
        while True:
            data = call("GET", f"/api/v1/datasets/{a.dataset}/status", dry=dry)
            if dry or not data or str(data.get("status", "")).lower() in ("completed", "succeeded", "failed", "error", "ready"):
                break
            time.sleep(60)
    elif a.cmd == "download":
        call("GET", f"/api/v1/datasets/{a.dataset}/download", params={"fileFormat": a.format}, dry=dry, stream_to=None if dry else a.out)
    elif a.cmd == "publish":
        call("POST", f"/api/v1/datasets/{a.dataset}/publish", {"target": a.target}, dry=dry)


if __name__ == "__main__":
    main()
