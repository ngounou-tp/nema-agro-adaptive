"""Merge generated + hand-written examples, remove duplicates, freeze a test set, and export files for Adaption.

Outputs in data/build/<project>/:
    train.jsonl            full records used for training (goes to Adaptive Data)
    test.jsonl             frozen benchmark (NEVER upload this to Adaption, never train on it)
    adaption_seed.jsonl    the training file for Adaption: prompt (context included), completion, task, lang
    adaption_seed.csv      same, as CSV (for the web upload form)
    test_view.csv          the same test set as a spreadsheet, to read (never upload it to Adaption)
    stats.json             counts per task / language / split, and the leakage check result

The test set has three parts:
    test_id   a random 15% of every task (same distribution as training)
    test_ood  whole templates / items / towns held out of training (shows the model generalises)
    test_real half of the real messages you collected by hand (data/seeds/manual_listings.csv, manual_*.jsonl)

Run:  python -m src.build_dataset --project agro
      python -m src.build_dataset --project waste
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict

from src.common import BUILD, ROOT, full_prompt, norm_text, read_jsonl, write_jsonl

PROJECTS = {
    "agro": {
        "inputs": ["gen_marketplace.jsonl", "gen_agro_calc.jsonl", "gen_agro_knowledge.jsonl"],
        "manual": ROOT / "data" / "seeds" / "manual_agro.jsonl",
        # held out of training entirely -> test_ood
        "ood_templates": {"fr_s4", "en_s4", "pcm_s3", "fr_b3", "pcm_b2"},
        "ood_towns": {"Kumbo", "Batouri", "Kousséri", "Akonolinga"},
        "cap_per_task": 1200,
    },
    "waste": {
        "inputs": ["gen_waste.jsonl"],
        "manual": ROOT / "data" / "seeds" / "manual_waste.jsonl",
        "ood_items": {"glass jar", "broken charger", "cocoa pod husks", "fluorescent bulb", "old newspapers"},
        "ood_neighbourhoods": {"Logbaba", "Ekounou", "Great Soppo"},
        "cap_per_task": 1000,
    },
}


def bucket(example_id: str, seed: int) -> float:
    """A number in [0, 1) fixed for each example id: the split never depends on the order or number of rows."""
    return int(hashlib.sha1(f"{seed}:{example_id}".encode()).hexdigest()[:8], 16) / 0x100000000


def is_ood(ex, cfg) -> bool:
    if ex["template_id"] in cfg.get("ood_templates", set()):
        return True
    gold = ex.get("meta", {}).get("gold") or {}
    if isinstance(gold, dict):
        if gold.get("location") in cfg.get("ood_towns", set()):
            return True
        if gold.get("neighbourhood") in cfg.get("ood_neighbourhoods", set()):
            return True
    if ex.get("meta", {}).get("town") in cfg.get("ood_towns", set()):
        return True
    if ex.get("meta", {}).get("item") in cfg.get("ood_items", set()):
        return True
    return False


def load_manual_listings(path):
    """Real marketplace messages you labelled in a spreadsheet (data/seeds/manual_listings.csv, one row per message).
    Easy to edit in the GitHub web editor or Google Sheets; converted here into agro_listing_extract examples."""
    if not path.exists():
        return []
    from src.common import dumps_gold, load_json, LEXICON, make_example
    from src.generators.gen_marketplace import INSTRUCTION, reference_list
    lex = load_json(LEXICON / "agro_lexicon.json")
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (r.get("message") or "").strip() or r["message"].startswith("EXAMPLE"):
                continue
            num = lambda v, cast=float: (cast(v) if str(v).strip() not in ("", "null") else None)  # noqa: E731
            unit = r["unit"].strip()
            qty = num(r["quantity"])
            qty = int(qty) if qty is not None and qty == int(qty) else qty
            kg = lex["units"].get(unit, {}).get("kg")
            price = num(r["price"], lambda v: int(float(v)))
            gold = {
                "intent": r["intent"].strip(), "product": r["product"].strip(), "quantity": qty, "unit": unit,
                "quantity_kg": qty * kg if (kg and qty is not None) else None,
                "price": price, "price_basis": (r["price_basis"].strip() or None) if price is not None else None,
                "currency": "XAF" if price is not None else None, "location": r["location"].strip() or None,
                "available_from": r["available_from"].strip() or None, "quality": r["quality"].strip() or None,
                "contact": (r["contact"].replace(" ", "").strip() or None),
            }
            prompt = f"{INSTRUCTION}\nMessage date: {r['message_date'].strip()}\nMessage: {r['message'].strip()}"
            ex = make_example("agro_listing_extract", r["lang"].strip(), "manual", prompt, dumps_gold(gold),
                              context=reference_list(lex), meta={"gold": gold, "manual": True})
            out.append(ex)
    return out


def to_adaption_row(ex):
    # "prompt" already contains the context (see common.full_prompt), so map only prompt -> completion in Adaption.
    return {"prompt": full_prompt(ex), "completion": ex["completion"], "task": ex["task"], "lang": ex["lang"]}


def main(project: str, test_share: float, seed: int):
    cfg = PROJECTS[project]
    rows = []
    for name in cfg["inputs"]:
        path = BUILD / name
        if not path.exists():
            raise SystemExit(f"missing {path}: run the generators first (see playbook section 5)")
        rows += read_jsonl(path)
    if cfg["manual"].exists():
        manual = read_jsonl(cfg["manual"])
        for m in manual:
            m.setdefault("template_id", "manual")
            m.setdefault("meta", {})["manual"] = True
            m.setdefault("context", "")
        rows += manual
        print(f"added {len(manual)} hand-written examples from {cfg['manual'].name}")
    if project == "agro":
        listings = load_manual_listings(ROOT / "data" / "seeds" / "manual_listings.csv")
        if listings:
            rows += listings
            print(f"added {len(listings)} real labelled messages from manual_listings.csv")

    # de-duplicate on normalised prompt + context
    seen, uniq = set(), []
    for r in rows:
        key = norm_text(r["context"] + " || " + r["prompt"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    # cap per task so no task drowns the others
    by_task = defaultdict(list)
    for r in uniq:
        by_task[r["task"]].append(r)
    capped = []
    for task, items in by_task.items():
        # deterministic: order by a hash of the id, so adding your own rows never changes which generated rows are kept
        gen = sorted([r for r in items if not r.get("meta", {}).get("manual")], key=lambda r: bucket(r["id"], seed))
        capped += gen[: cfg["cap_per_task"]] + [r for r in items if r.get("meta", {}).get("manual")]

    train, test = [], []
    for r in capped:
        if r["task"].endswith("_mcq"):
            r["split"] = "test_id"  # MCQs only measure knowledge learned from the documents: never train on them
            test.append(r)
        elif is_ood(r, cfg):
            r["split"] = "test_ood"
            test.append(r)
        elif r.get("meta", {}).get("manual") and bucket(r["id"], seed) < 0.5:
            r["split"] = "test_real"  # half of your real, hand-written examples go to the test set
            test.append(r)
        elif bucket(r["id"], seed) < test_share:
            r["split"] = "test_id"
            test.append(r)
        else:
            r["split"] = "train"
            train.append(r)

    # leakage check: no test prompt may also appear in train
    train_keys = {norm_text(r["context"] + " || " + r["prompt"]) for r in train}
    leaks = sum(1 for r in test if norm_text(r["context"] + " || " + r["prompt"]) in train_keys)

    out = BUILD / project
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(test, out / "test.jsonl")
    write_jsonl([to_adaption_row(r) for r in train], out / "adaption_seed.jsonl")
    with open(out / "adaption_seed.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "completion", "task", "lang"])
        w.writeheader()
        for r in train:
            w.writerow(to_adaption_row(r))

    stats = {
        "project": project,
        "train": len(train), "test": len(test), "leaks": leaks,
        "train_by_task": Counter(r["task"] for r in train),
        "test_by_task_split": Counter(f"{r['task']}|{r['split']}" for r in test),
        "train_by_lang": Counter(r["lang"] for r in train),
        "test_by_lang": Counter(r["lang"] for r in test),
    }
    with open(out / "test_view.csv", "w", encoding="utf-8", newline="") as f:  # read-only view of the benchmark
        w = csv.DictWriter(f, fieldnames=["task", "split", "lang", "prompt", "completion"])
        w.writeheader()
        for r in test:
            w.writerow({"task": r["task"], "split": r["split"], "lang": r["lang"], "prompt": full_prompt(r), "completion": r["completion"]})
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if leaks:
        raise SystemExit(f"LEAKAGE: {leaks} test prompts also in train. Fix before uploading.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", choices=list(PROJECTS), required=True)
    ap.add_argument("--test-share", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    main(a.project, a.test_share, a.seed)
