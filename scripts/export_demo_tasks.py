"""Write demo/tasks_<project>.json: for each task, the fixed instruction, a sample input and its context.
Run after build_dataset:  python scripts/export_demo_tasks.py --project agro"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import BUILD, ROOT, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--project", choices=["agro", "waste"], required=True)
a = ap.parse_args()
tasks = {}
for r in read_jsonl(BUILD / a.project / "test.jsonl"):
    if r["task"] in tasks or r["task"].endswith("_mcq"):
        continue
    instruction, sample = r["prompt"].split("\n", 1)
    tasks[r["task"]] = {"instruction": instruction, "sample_input": sample, "context": r["context"], "lang": r["lang"]}
out = ROOT / "demo" / f"tasks_{a.project}.json"
out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(tasks)} tasks -> {out}")
