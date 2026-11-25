"""Shared helpers for the NEMA-Agro / REVALO-Waste starter kit.

Every generated example is one JSON object with these keys:
    id           unique id (task + hash)
    task         task name, e.g. "agro_listing_extract"
    lang         "fr", "en" or "pcm" (Cameroonian Pidgin)
    template_id  which template produced it (used to hold whole templates out of training)
    prompt       the user turn: instruction + input (what the model sees)
    context      optional reference text the model must use (tables, label data)
    completion   the gold answer (a JSON string for structured tasks)
    meta         anything useful for scoring or analysis (never shown to the model)
"""
from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEXICON = ROOT / "data" / "lexicon"
BUILD = ROOT / "data" / "build"


def load_json(path: Path | str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_text(s: str) -> str:
    """Lowercase, strip accents and extra spaces: used for de-duplication and lenient matching."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


def stable_id(task: str, text: str) -> str:
    return f"{task}-{hashlib.sha1(norm_text(text).encode()).hexdigest()[:12]}"


def dumps_gold(obj) -> str:
    """Canonical JSON for gold answers: stable key order, no ASCII escaping."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def full_prompt(ex: dict) -> str:
    """The exact text the model sees, in training AND evaluation AND production: context first, then the task.
    Keeping one function for all three is what makes the benchmark fair and the deployment match training."""
    ctx = (ex.get("context") or "").strip()
    return f"Context:\n{ctx}\n\n{ex['prompt']}" if ctx else ex["prompt"]


def make_example(task, lang, template_id, prompt, completion, context=None, meta=None):
    return {
        "id": stable_id(task, (context or "") + prompt),
        "task": task,
        "lang": lang,
        "template_id": template_id,
        "prompt": prompt,
        "context": context or "",
        "completion": completion,
        "meta": meta or {},
    }


def fmt_xaf(amount: int, rng: random.Random, lang: str) -> str:
    """Write a CFA franc amount the way people type it in messages."""
    styles_fr = ["{n} F", "{n}F", "{n} FCFA", "{n} frs", "{n} francs", "{n_sp} FCFA"]
    styles_en = ["{n} FCFA", "{n} frs", "{n}F", "{n_sp} francs", "XAF {n}"]
    styles_pcm = ["{n} frs", "{n}F", "{n} francs", "{n_sp} FCFA"]
    style = rng.choice({"fr": styles_fr, "en": styles_en, "pcm": styles_pcm}[lang])
    n_sp = f"{amount:,}".replace(",", " ")
    if amount >= 1000 and amount % 1000 == 0 and rng.random() < 0.25:
        k = amount // 1000
        return {"fr": f"{k} mille", "en": f"{k}k", "pcm": f"{k} thousand"}[lang]
    return style.format(n=amount, n_sp=n_sp)


def fake_phone(rng: random.Random) -> str:
    """Synthetic Cameroon-style mobile number (6XX XX XX XX). Not a real person."""
    digits = "6" + rng.choice("579") + "".join(rng.choice("0123456789") for _ in range(7))
    if rng.random() < 0.5:
        return f"{digits[:3]} {digits[3:5]} {digits[5:7]} {digits[7:]}"
    return digits


def end_of_month(d: date) -> date:
    nxt = (d.replace(day=28) + timedelta(days=4))
    return nxt - timedelta(days=nxt.day)


def next_monday(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) or 7)


def random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))
