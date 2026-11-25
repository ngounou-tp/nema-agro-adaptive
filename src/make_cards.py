"""Fill the model card and dataset card from your config, dataset stats and benchmark results.

Inputs : configs/project_<p>.json, data/build/<p>/stats.json, results/<p>_results.md (from score.py),
         adapter_config.json (local folder or downloaded from your Hugging Face repo)
Outputs: release/<p>/model/README.md, release/<p>/model/prompts.md, release/<p>/dataset/README.md (+ data files)

Run:  python -m src.make_cards --project agro [--adapter-dir path/to/adapter]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from src.common import BUILD, ROOT, full_prompt, read_jsonl

TASK_DESCRIPTIONS = {
    "agro_listing_extract": "Marketplace message (sell offer or buy request) -> JSON listing: product, quantity, unit, kg, price, basis, place, date, quality, contact",
    "agro_unit_price": "Local unit + price -> kg, price per kg, total, gap to a reference price (from the context table)",
    "agro_coop_ledger": "Cooperative delivery notes -> ledger entries and totals per member",
    "agro_trace_record": "Harvest story -> traceability lot record, with pre-harvest-interval check",
    "agro_symptom_triage": "Symptom description -> problem from the context table (or unknown), technician flag, first actions",
    "agro_input_safety": "Pesticide question -> safe decision; dose only if on the label in context",
    "agro_advice_mcq": "Knowledge check from fiches techniques (multiple choice)",
    "waste_sort": "Waste item -> category, valorisable, hazardous, what to do",
    "waste_pickup_extract": "Pickup request -> city, neighbourhood, type, bags, urgency, contact",
    "waste_compost_calc": "Tonnage + parameters -> organic kg, compost kg, value",
    "waste_hygiene_mcq": "Knowledge check on sanitation (multiple choice)",
}


def fill(template: str, values: dict) -> str:
    for k, v in values.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


def yaml_list(items):
    return "\n".join(f"- {x}" for x in items)


def adapter_config(cfg, adapter_dir):
    path = Path(adapter_dir) / "adapter_config.json" if adapter_dir else None
    if not (path and path.exists()):
        try:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(cfg["hf_model_repo"], "adapter_config.json", token=os.environ.get("HF_TOKEN")))
        except Exception as e:
            print(f"(could not read adapter_config.json: {e}); LoRA fields left as REPLACE")
            return {}
    return json.loads(path.read_text())


def main(project, adapter_dir):
    cfg = json.loads((ROOT / "configs" / f"project_{project}.json").read_text(encoding="utf-8"))
    stats = json.loads((BUILD / project / "stats.json").read_text(encoding="utf-8"))
    res_path = ROOT / "results" / f"{project}_results.md"
    results = res_path.read_text(encoding="utf-8").split("\n\n")[0] if res_path.exists() else "REPLACE: run src/eval/score.py first"
    ac = adapter_config(cfg, adapter_dir)
    train = read_jsonl(BUILD / project / "train.jsonl")
    final = BUILD / project / "final_train.jsonl"
    test = read_jsonl(BUILD / project / "test.jsonl")
    tasks = sorted({r["task"] for r in train + test})
    example = next(r for r in test if r["task"] == tasks[0])

    common = {
        "BASE_MODEL": cfg["base_model"], "LICENSE": cfg["license"], "DATA_LICENSE": cfg["data_license"],
        "LANGUAGES_YAML": yaml_list(cfg["languages"]), "TAGS_YAML": yaml_list(cfg["tags"]),
        "HF_DATASET": cfg["hf_dataset_repo"], "HF_MODEL": cfg["hf_model_repo"], "MODEL_NAME": cfg["model_name"],
        "DATASET_NAME": cfg["model_name"].replace("LoRA", "dataset"), "ONE_LINE": cfg["one_line"],
        "BENCH_NAME": cfg["bench_name"], "TEST_N": len(test), "RESULTS_TABLE": results,
        "AUTOSCIENTIST_WINRATE": cfg["autoscientist_winrate"],
        "TASKS_TABLE": "| Task | What it does |\n| --- | --- |\n" + "\n".join(f"| `{t}` | {TASK_DESCRIPTIONS.get(t, '')} |" for t in tasks),
        "EXAMPLE_PROMPT": full_prompt(example).replace('"""', "'''"),
        "LORA_NAME": cfg["lora_name"], "LORA_R": ac.get("r", "REPLACE"), "LORA_ALPHA": ac.get("lora_alpha", "REPLACE"),
        "LORA_TARGETS": ", ".join(ac.get("target_modules", []) or ["REPLACE"]),
        "TRAINING_TABLE": "\n".join(f"| {a} | {b} | {c} |" for a, b, c in cfg["training_steps"]),
        "GITHUB_REPO": cfg["github_repo"], "GITHUB_DIR": cfg["github_repo"].rstrip("/").split("/")[-1],
        "PROJECT": project, "PER_BUCKET": cfg["per_bucket"], "EVAL_TIME": cfg["eval_time"],
        "KAGGLE_NOTEBOOK": cfg["kaggle_notebook_url"], "KAGGLE_DATASET": f"https://www.kaggle.com/datasets/{cfg['kaggle_dataset_handle']}",
        "LIMITATIONS": cfg["limitations"], "AUTHOR": cfg["author"], "PROJECT_URL": cfg["project_url"],
        "TRAIN_N": len(read_jsonl(final)) if final.exists() else stats["train"],
        "TRAIN_BY_TASK": ", ".join(f"{k} {v}" for k, v in stats["train_by_task"].items()),
        "TRAIN_BY_LANG": ", ".join(f"{k} {v}" for k, v in stats["train_by_lang"].items()),
        "HOW_MADE": cfg["how_made"],
        "SOURCES_TABLE": "| Source | Publisher / link | Licence |\n| --- | --- | --- |\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in cfg["sources"]),
    }
    out = ROOT / "release" / project
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "dataset").mkdir(parents=True, exist_ok=True)
    (out / "model" / "README.md").write_text(fill((ROOT / "cards" / "MODEL_CARD_TEMPLATE.md").read_text(encoding="utf-8"), common), encoding="utf-8")
    (out / "dataset" / "README.md").write_text(fill((ROOT / "cards" / "DATASET_CARD_TEMPLATE.md").read_text(encoding="utf-8"), common), encoding="utf-8")
    # one instruction per task, exactly as trained, for people integrating the model
    prompts = ["# Task instructions (use them verbatim)\n"]
    for t in tasks:
        ex = next(r for r in train + test if r["task"] == t)
        prompts.append(f"## {t}\n\n```text\n{full_prompt(ex)}\n```\n\nExpected answer:\n\n```json\n{ex['completion']}\n```\n")
    (out / "model" / "prompts.md").write_text("\n".join(prompts), encoding="utf-8")
    if res_path.exists():
        for f in (ROOT / "results").glob(f"{project}_*"):
            shutil.copy(f, out / "model" / f.name)
    # dataset files
    shutil.copy(final if final.exists() else BUILD / project / "train.jsonl", out / "dataset" / "train.jsonl")
    shutil.copy(BUILD / project / "test.jsonl", out / "dataset" / "test.jsonl")
    shutil.copy(BUILD / project / "stats.json", out / "dataset" / "stats.json")
    left = [p for p in (out / "model" / "README.md", out / "dataset" / "README.md") if any(w in p.read_text(encoding="utf-8") for w in ("REPLACE", "PASTE"))]
    print(f"cards written to {out}")
    if left:
        print("STILL TO FILL (search for REPLACE or PASTE): " + ", ".join(str(p) for p in left))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", choices=["agro", "waste"], required=True)
    ap.add_argument("--adapter-dir", default=None)
    a = ap.parse_args()
    main(a.project, a.adapter_dir)
