"""Merge the datasets you downloaded from Adaptive Data into ONE final training file for AutoScientist,
after checking them. Rows are dropped when:
  - a structured task's completion is not valid JSON or has the wrong keys (adaptation can damage labels)
  - the prompt also appears in the frozen test set (leakage)
  - the prompt is empty or duplicated

Run:
    python -m src.merge_adapted --project agro \
        --inputs data/build/agro/adapted_structured.jsonl data/build/agro/adapted_advisory.jsonl \
        --out data/build/agro/final_train
Writes final_train.jsonl and final_train.csv (prompt, completion) + a report of what was dropped.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.common import BUILD, full_prompt, norm_text, read_jsonl, write_jsonl


def read_any(path: Path):
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("rows", [])
    if path.suffix == ".csv":
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    raise SystemExit(f"unsupported file type: {path}")


def pick(row, *names):
    for n in names:
        if row.get(n):
            return row[n]
    return ""


def main(project, inputs, out):
    train = read_jsonl(BUILD / project / "train.jsonl")
    test = read_jsonl(BUILD / project / "test.jsonl")
    expected = {}
    for r in train:
        if r["completion"].startswith("{"):
            expected.setdefault(r["task"], set(json.loads(r["completion"])))
    test_keys = {norm_text(full_prompt(r)) for r in test}
    kept, seen, dropped = [], set(), Counter()
    for path in inputs:
        for row in read_any(Path(path)):
            prompt = pick(row, "prompt", "instruction", "input", "question")
            completion = pick(row, "completion", "response", "output", "answer")
            if not prompt or not completion:
                dropped["empty"] += 1
                continue
            key = norm_text(prompt)
            if key in test_keys:
                dropped["in_test_set"] += 1
                continue
            if key in seen:
                dropped["duplicate"] += 1
                continue
            m = re.search(r"Task: ([a-z_]+)\.", prompt)
            task = m.group(1) if m else "advisory"
            if task in expected:
                try:
                    obj = json.loads(completion)
                    if set(obj) != expected[task]:
                        dropped[f"{task}:wrong_keys"] += 1
                        continue
                except Exception:
                    dropped[f"{task}:invalid_json"] += 1
                    continue
            seen.add(key)
            kept.append({"prompt": prompt, "completion": completion, "task": task, "lang": row.get("lang", "")})
    write_jsonl(kept, f"{out}.jsonl")
    with open(f"{out}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "completion"])
        w.writeheader()
        for r in kept:
            w.writerow({"prompt": r["prompt"], "completion": r["completion"]})
    report = {"kept": len(kept), "kept_by_task": Counter(r["task"] for r in kept), "dropped": dropped}
    Path(f"{out}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", choices=["agro", "waste"], required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.project, a.inputs, a.out)
