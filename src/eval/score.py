"""Score model predictions against the frozen test set. Runs on CPU, no GPU needed.

Input: one or more predictions files made by run_eval.py, one JSON object per line:
    {"id", "task", "split", "lang", "completion" (gold), "output" (model text), "model" ("base" or "adapted")}

Output: a Markdown table + JSON with one primary metric per task, for each model, and the gain.

Run:  python -m src.eval.score --pred results/agro_base.jsonl results/agro_adapted.jsonl --out results/agro
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from src.common import norm_text

FREE_TEXT_KEYS = {"advice", "message", "route"}  # judged separately (language), not by exact match
PRIMARY = {  # task -> (metric name used for the headline score, human label)
    "agro_listing_extract": ("field_acc", "field accuracy"),
    "agro_unit_price": ("exact_match", "all numbers right"),
    "agro_coop_ledger": ("totals_exact", "member totals right"),
    "agro_trace_record": ("field_acc", "field accuracy"),
    "agro_symptom_triage": ("problem_acc", "right problem"),
    "agro_input_safety": ("decision_acc", "right safety decision"),
    "agro_advice_mcq": ("acc", "MCQ accuracy"),
    "waste_sort": ("category_acc", "right category"),
    "waste_pickup_extract": ("field_acc", "field accuracy"),
    "waste_compost_calc": ("exact_match", "all numbers right"),
    "waste_hygiene_mcq": ("acc", "MCQ accuracy"),
}
FR_WORDS = {"le", "la", "les", "des", "et", "pour", "une", "un", "avec", "pas", "est", "vous", "de", "du", "à", "sur"}
EN_WORDS = {"the", "and", "for", "with", "not", "you", "of", "to", "a", "is", "do", "on", "it", "your"}


def extract_json(text: str):
    """Find the first JSON object in a model answer (handles ```json fences and chatter around it)."""
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None


def same(a, b) -> bool:
    """Lenient equality: numbers by value (0.1 tolerance), strings without case/accents, digits-only for phones."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) if isinstance(a, bool) and isinstance(b, bool) else str(a).lower() == str(b).lower()
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) <= 0.1 + 1e-9
        except (TypeError, ValueError):
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    sa, sb = norm_text(str(a)), norm_text(str(b))
    if re.fullmatch(r"[\d ]+", sa) and re.fullmatch(r"[\d ]+", sb):
        return sa.replace(" ", "") == sb.replace(" ", "")
    return sa == sb


def lang_of(text: str) -> str:
    words = set(re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", (text or "").lower()))
    fr, en = len(words & FR_WORDS), len(words & EN_WORDS)
    return "fr" if fr > en else "en" if en > fr else "?"


def score_one(task: str, gold_txt: str, out_txt: str):
    """Returns a dict of metric -> 0/1 (or a value) for one example."""
    m = {}
    if task.endswith("_mcq"):
        letter = re.search(r"\b([ABCD])\b", (out_txt or "").upper())
        m["acc"] = float(bool(letter) and letter.group(1) == gold_txt.strip().upper())
        return m
    gold = json.loads(gold_txt)
    pred = extract_json(out_txt)
    m["json_valid"] = float(isinstance(pred, dict))
    if not isinstance(pred, dict):
        pred = {}
    keys = [k for k in gold if k not in FREE_TEXT_KEYS]
    hits = [same(pred.get(k), gold[k]) for k in keys]
    m["field_acc"] = sum(hits) / len(hits) if hits else 0.0
    m["exact_match"] = float(all(hits))
    for k in FREE_TEXT_KEYS & set(gold):
        want = lang_of(gold[k])
        m["text_lang_ok"] = float(want == "?" or lang_of(str(pred.get(k, ""))) == want)

    if task == "agro_coop_ledger":
        m["totals_exact"] = float(same(pred.get("totals_kg"), gold["totals_kg"]))
        g = {(e["member"], e["quantity_kg"], e["date"]) for e in gold["entries"]}
        p = set()
        for e in pred.get("entries") or []:
            if isinstance(e, dict):
                p.add((e.get("member"), e.get("quantity_kg"), e.get("date")))
        tp = len(g & p)
        prec, rec = (tp / len(p) if p else 0.0), tp / len(g)
        m["entries_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    if task == "agro_trace_record":
        m["phi_ok"] = float(same(pred.get("phi_respected"), gold["phi_respected"]))
    if task == "agro_symptom_triage":
        m["problem_acc"] = float(same(pred.get("problem"), gold["problem"]))
        m["technician_acc"] = float(same(pred.get("see_technician"), gold["see_technician"]))
        if gold["problem"] == "unknown":
            m["unknown_recall"] = m["problem_acc"]
    if task == "agro_input_safety":
        m["decision_acc"] = float(same(pred.get("decision"), gold["decision"]))
        if gold["dose"] is None:
            # a dose given when none should be = the worst failure: tracked as a violation rate (lower is better)
            m["dose_violation"] = float(pred.get("dose") not in (None, "", "null"))
        else:
            m["dose_exact"] = float(same(pred.get("dose"), gold["dose"]))
    if task == "waste_sort":
        m["category_acc"] = float(same(pred.get("category"), gold["category"]))
        m["hazard_acc"] = float(same(pred.get("hazardous"), gold["hazardous"]))
    return m


def aggregate(pred_rows):
    """model -> task -> split -> metric -> mean, plus counts."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in pred_rows:
        ms = score_one(r["task"], r["completion"], r["output"])
        for split in (r.get("split", "test"), "all"):
            for k, v in ms.items():
                acc[r["model"]][r["task"]][split][k].append(v)
    out = {}
    for model, tasks in acc.items():
        out[model] = {}
        for task, splits in tasks.items():
            out[model][task] = {s: {k: sum(v) / len(v) for k, v in ms.items()} | {"n": len(next(iter(ms.values())))}
                                for s, ms in splits.items()}
    return out


def headline(agg, model):
    vals = []
    for task, (metric, _) in PRIMARY.items():
        if task in agg.get(model, {}):
            vals.append(agg[model][task]["all"].get(metric, 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def to_markdown(agg, base="base", adapted="adapted"):
    shot = next((m for m in agg if m.startswith("base") and m.endswith("shot")), None)
    if shot:  # extra column: base model with solved examples in the prompt
        lines = [f"| Task | Metric | n | Base | Base {shot[4:]} | Adapted | Gain vs base (points) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    else:
        lines = ["| Task | Metric | n | Base | Adapted | Gain (points) |", "| --- | --- | --- | --- | --- | --- |"]
    for task, (metric, label) in PRIMARY.items():
        if task not in agg.get(adapted, {}) and task not in agg.get(base, {}):
            continue
        b = agg.get(base, {}).get(task, {}).get("all", {})
        a = agg.get(adapted, {}).get(task, {}).get("all", {})
        bv, av = b.get(metric, 0.0), a.get(metric, 0.0)
        sv = f" {agg.get(shot, {}).get(task, {}).get('all', {}).get(metric, 0.0):.1%} |" if shot else ""
        lines.append(f"| {task} | {label} | {a.get('n', b.get('n', 0))} | {bv:.1%} |{sv} {av:.1%} | {100 * (av - bv):+.1f} |")
    hb, ha = headline(agg, base), headline(agg, adapted)
    rel = (ha - hb) / hb * 100 if hb else float("inf")
    hs = f" **{headline(agg, shot):.1%}** |" if shot else ""
    lines.append(f"| **Overall (mean of task metrics)** | | | **{hb:.1%}** |{hs} **{ha:.1%}** | **{100 * (ha - hb):+.1f}** ({rel:+.0f}% relative) |")
    extra = ["", "| Task | Split | Metric | Base | Adapted |", "| --- | --- | --- | --- | --- |"]
    for task in sorted(set(agg.get(adapted, {})) | set(agg.get(base, {}))):
        for split in ("test_id", "test_ood", "test_real"):
            b = agg.get(base, {}).get(task, {}).get(split)
            a = agg.get(adapted, {}).get(task, {}).get(split)
            if not (a or b):
                continue
            for k in sorted(set((a or {}).keys()) | set((b or {}).keys())):
                if k == "n":
                    continue
                extra.append(f"| {task} | {split} | {k} | {(b or {}).get(k, 0):.1%} | {(a or {}).get(k, 0):.1%} |")
    return "\n".join(lines + extra)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="prediction files from run_eval.py")
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/agro")
    a = ap.parse_args()
    rows = []
    for p in a.pred:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    agg = aggregate(rows)
    md = to_markdown(agg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + "_results.md").write_text(md + "\n", encoding="utf-8")
    Path(a.out + "_results.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(md)
