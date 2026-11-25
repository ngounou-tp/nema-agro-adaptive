"""Task A4/A5: turn informal marketplace messages (sell offers and buy requests) into a clean JSON listing.

Messages are generated in French, English and Cameroonian Pidgin with typing habits seen on WhatsApp
(F / frs / mille, no accents, mixed order). The gold answer is computed from the same random choices,
so every label is exactly right by construction.

Run:  python -m src.generators.gen_marketplace --n 1500 --out data/build/gen_marketplace.jsonl
"""
from __future__ import annotations

import argparse
import random
import re
from datetime import date, timedelta

from src.common import (LEXICON, dumps_gold, end_of_month, fake_phone, fmt_xaf, load_json,
                        make_example, next_monday, random_date, write_jsonl)

TASK = "agro_listing_extract"

INSTRUCTION = (
    "Task: agro_listing_extract. Read the marketplace message and return ONLY a JSON object with keys: "
    "intent (sell|buy), product, quantity, unit, quantity_kg, price, price_basis (per_unit|per_kg|total|null), "
    "currency, location, available_from (YYYY-MM-DD or null), quality, contact. "
    "Use the product and unit codes from the reference list. Resolve relative dates from the message date. "
    "quantity_kg = quantity x kg per unit only when the unit has a fixed weight, otherwise null. "
    "Use null for anything the message does not say."
)

SELL_TEMPLATES = {
    "fr": [
        ("fr_s1", "Bonjour, j'ai {qty} {unit} de {product} à vendre à {place}{price_part}.{quality_part}.{when_part}{phone_part}"),
        ("fr_s2", "{product} dispo à {place} : {qty} {unit}{price_part}{quality_part}.{when_part}{phone_part}"),
        ("fr_s3", "Vente {product} {qty} {unit}, {place}{price_part}{when_part}{phone_part}"),
        ("fr_s4", "Qui veut du {product}? {qty} {unit} {quality_part2}à {place}{price_part}.{when_part}{phone_part}"),
        ("fr_s5", "GIC de {place}: nous avons {qty} {unit} de {product}{quality_part}{price_part}.{when_part}{phone_part}"),
    ],
    "en": [
        ("en_s1", "Hello, I have {qty} {unit} of {product} for sale in {place}{price_part}.{quality_part}.{when_part}{phone_part}"),
        ("en_s2", "{product} available in {place}: {qty} {unit}{price_part}{quality_part}.{when_part}{phone_part}"),
        ("en_s3", "Selling {qty} {unit} {product}, {place}{price_part}{when_part}{phone_part}"),
        ("en_s4", "Our cooperative in {place} has {qty} {unit} of {product}{quality_part}{price_part}.{when_part}{phone_part}"),
    ],
    "pcm": [
        ("pcm_s1", "I get {qty} {unit} {product} for sell for {place}{price_part}.{quality_part}.{when_part}{phone_part}"),
        ("pcm_s2", "{product} dey for {place}, {qty} {unit}{price_part}{quality_part}.{when_part}{phone_part}"),
        ("pcm_s3", "Who wan buy {product}? {qty} {unit} dey for {place}{price_part}{when_part}{phone_part}"),
    ],
}

BUY_TEMPLATES = {
    "fr": [
        ("fr_b1", "Je cherche {qty} {unit} de {product} livrés à {place}{price_part}.{quality_part}.{when_part}{phone_part}"),
        ("fr_b2", "Besoin urgent: {product}, {qty} {unit}, à {place}{price_part}{when_part}{phone_part}"),
        ("fr_b3", "Acheteur à {place} cherche {product} {qty} {unit}{quality_part}{price_part}.{when_part}{phone_part}"),
    ],
    "en": [
        ("en_b1", "Looking for {qty} {unit} of {product} delivered to {place}{price_part}.{quality_part}.{when_part}{phone_part}"),
        ("en_b2", "Buyer in {place} needs {product}, {qty} {unit}{price_part}{when_part}{phone_part}"),
    ],
    "pcm": [
        ("pcm_b1", "I want buy {qty} {unit} {product} for {place}{price_part}.{when_part}{phone_part}"),
        ("pcm_b2", "Who get {product}? I need {qty} {unit} for {place}{price_part}{quality_part}.{when_part}{phone_part}"),
    ],
}

BASIS_WORDS = {
    "per_unit": {"fr": ["le {u}", "par {u}", "l'unité"], "en": ["per {u}", "a {u}", "each"], "pcm": ["one {u}", "for one {u}"]},
    "per_kg": {"fr": ["le kilo", "le kg", "par kg"], "en": ["per kg", "a kilo"], "pcm": ["one kilo", "for one kilo"]},
    "total": {"fr": ["le tout", "pour l'ensemble", "au total"], "en": ["for everything", "for the lot", "total"], "pcm": ["for everything", "for all"]},
}
PRICE_INTRO = {"fr": [" à {p} {b}", ", prix {p} {b}", " ({p} {b})"], "en": [" at {p} {b}", ", price {p} {b}", " ({p} {b})"],
               "pcm": [" na {p} {b}", ", {p} {b}", " ({p} {b})"]}
BUDGET_INTRO = {"fr": [", budget {p} {b}", ", je paie {p} {b}"], "en": [", budget {p} {b}", ", paying {p} {b}"],
                "pcm": [", I go pay {p} {b}", ", my budget na {p} {b}"]}
PHONE_INTRO = {"fr": [" Appelez le {t}.", " Contact: {t}", " WhatsApp {t}"], "en": [" Call {t}.", " Contact: {t}", " WhatsApp {t}"],
               "pcm": [" Call me for {t}.", " Ma number na {t}", " WhatsApp {t}"]}


def when_phrase(rng, lang, msg_date):
    """Returns (text, resolved_date_or_None)."""
    kind = rng.choice(["none", "none", "tomorrow", "in_days", "eom", "next_monday", "explicit"])
    if kind == "none":
        return "", None
    if kind == "tomorrow":
        return rng.choice({"fr": [" Dispo demain."], "en": [" Ready tomorrow."], "pcm": [" E go ready tumoro.", " E go dey tomorrow."]}[lang]), msg_date + timedelta(days=1)
    if kind == "in_days":
        n = rng.randint(2, 10)
        return {"fr": f" Disponible dans {n} jours.", "en": f" Available in {n} days.", "pcm": f" E go ready for {n} days time."}[lang], msg_date + timedelta(days=n)
    if kind == "eom":
        return rng.choice({"fr": [" Livraison fin du mois."], "en": [" Delivery end of the month."], "pcm": [" Delivery for end of dis month."]}[lang]), end_of_month(msg_date)
    if kind == "next_monday":
        return {"fr": " A partir de lundi prochain.", "en": " From next Monday.", "pcm": " From next Monday."}[lang], next_monday(msg_date)
    d = msg_date + timedelta(days=rng.randint(3, 40))
    months_en = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    txt = {"fr": f" Disponible le {d.day:02d}/{d.month:02d}.", "en": f" Available on {d.day} {months_en[d.month - 1]}.",
           "pcm": f" E go ready for {d.day} {months_en[d.month - 1]}."}[lang]
    return txt, d


def build_one(rng: random.Random, lex: dict, lang: str, intent: str, msg_date: date):
    product = rng.choice(lex["products"])
    unit_id = rng.choice(product["units"])
    unit = lex["units"][unit_id]
    qty = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100]) if unit_id != "tonne" else rng.choice([1, 2, 3, 5, 10])
    unit_words = unit[lang]
    unit_txt = unit_words[1] if (qty > 1 and len(unit_words) > 1) else unit_words[0]
    prod_txt = rng.choice(product[lang])
    region = rng.choice(list(lex["places"]))
    place = rng.choice(lex["places"][region])

    # price
    price, basis = None, None
    if rng.random() < 0.8:
        basis = rng.choice(["per_unit", "per_kg", "total"]) if unit["kg"] else rng.choice(["per_unit", "total"])
        if unit_id == "kg" and basis == "per_unit":
            basis = "per_kg"
        kg_unit = unit["kg"] or product.get("kg_per_unit", {}).get(unit_id, 20)
        lo, hi = product.get("price_per_kg", [500, 1500])
        if "price_per_unit" in product and unit_id in product["price_per_unit"]:
            lo_u, hi_u = product["price_per_unit"][unit_id]
            per_unit = rng.randrange(lo_u, hi_u, 500)
        elif "price_per_litre" in product:
            per_unit = rng.randrange(*product["price_per_litre"], 50) * (20 if unit_id == "jerrycan_20l" else 1)
        else:
            per_unit = int(round(rng.uniform(lo, hi) * kg_unit / 500.0)) * 500 or 500
        if basis == "per_unit":
            price = per_unit
        elif basis == "per_kg":
            price = int(round(rng.uniform(lo, hi) / 25.0)) * 25
        else:
            price = per_unit * qty
    u_single = unit_words[0]
    price_part = ""
    if price is not None:
        b = rng.choice(BASIS_WORDS[basis][lang]).format(u=u_single)
        intro = rng.choice((PRICE_INTRO if intent == "sell" else BUDGET_INTRO)[lang])
        price_part = intro.format(p=fmt_xaf(price, rng, lang), b=b)

    when_txt, when_date = when_phrase(rng, lang, msg_date)

    quality = None
    quality_part = quality_part2 = ""
    if rng.random() < 0.35:
        qi = rng.choice(quality_choices(product["id"]))
        quality = lex["quality"]["en"][qi]
        q_txt = lex["quality"][lang][qi]
        quality_part = f", {q_txt}"
        quality_part2 = f"{q_txt} "

    phone = None
    phone_part = ""
    if rng.random() < 0.7:
        phone_txt = fake_phone(rng)
        phone = phone_txt.replace(" ", "")
        phone_part = rng.choice(PHONE_INTRO[lang]).format(t=phone_txt)

    templates = (SELL_TEMPLATES if intent == "sell" else BUY_TEMPLATES)[lang]
    template_id, tpl = rng.choice(templates)
    msg = tpl.format(qty=qty, unit=unit_txt, product=prod_txt, place=place, price_part=price_part,
                     when_part=when_txt, quality_part=quality_part, quality_part2=quality_part2, phone_part=phone_part)
    msg = " ".join(msg.split())
    msg = re.sub(r"\s+([.,:])", r"\1", msg)
    msg = re.sub(r"\.{2,}", ".", msg)
    msg = re.sub(r"\.,", ",", msg)
    msg = re.sub(r",\.", ".", msg)
    if rng.random() < 0.3:  # people often type without accents
        msg = msg.replace("é", "e").replace("è", "e").replace("à", "a")

    gold = {
        "intent": intent,
        "product": product["id"],
        "quantity": qty,
        "unit": unit_id,
        "quantity_kg": qty * unit["kg"] if unit["kg"] else None,
        "price": price,
        "price_basis": basis,
        "currency": "XAF" if price is not None else None,
        "location": place,
        "available_from": when_date.isoformat() if when_date else None,
        "quality": quality,
        "contact": phone,
    }
    prompt = f"{INSTRUCTION}\nMessage date: {msg_date.isoformat()}\nMessage: {msg}"
    return make_example(TASK, lang, template_id, prompt, dumps_gold(gold), context=reference_list(lex),
                        meta={"region": region, "gold": gold})


DRY = {"maize", "cassava_flour", "garri", "cocoa", "coffee", "groundnut", "beans", "rice", "egusi", "soybean"}
FRESH = {"cassava", "plantain", "banana", "tomato", "irish_potato", "yam", "cocoyam", "pepper", "onion"}


def quality_choices(pid: str):
    """Indexes into lexicon quality lists that make sense for this product."""
    idx = [1, 2, 6]  # first grade, grade 1, sorted: any product
    if pid in DRY:
        idx += [0, 4]  # well dried, no mould
    if pid in FRESH:
        idx += [5]  # fresh
    if pid == "cocoa":
        idx += [3]  # well fermented
    return idx


_REF_CACHE = {}


def reference_list(lex: dict) -> str:
    """Short code list given as context so the model learns to map local words to codes."""
    if "ref" not in _REF_CACHE:
        prods = "; ".join(f"{p['id']} = {', '.join(p['fr'][:1] + p['en'][:1] + p['pcm'][:1])}" for p in lex["products"])
        units = "; ".join(f"{k} ({v['kg']} kg)" if v["kg"] else f"{k} (no fixed weight)" for k, v in lex["units"].items())
        _REF_CACHE["ref"] = f"Product codes: {prods}.\nUnit codes: {units}."
    return _REF_CACHE["ref"]


def generate(n: int, seed: int = 7):
    rng = random.Random(seed)
    lex = load_json(LEXICON / "agro_lexicon.json")
    rows, seen = [], set()
    while len(rows) < n:
        lang = rng.choices(["fr", "en", "pcm"], weights=[0.5, 0.3, 0.2])[0]
        intent = rng.choices(["sell", "buy"], weights=[0.65, 0.35])[0]
        msg_date = random_date(rng, date(2026, 1, 5), date(2026, 12, 20))
        ex = build_one(rng, lex, lang, intent, msg_date)
        if ex["id"] in seen:
            continue
        seen.add(ex["id"])
        rows.append(ex)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/build/gen_marketplace.jsonl")
    a = ap.parse_args()
    print(f"wrote {write_jsonl(generate(a.n, a.seed), a.out)} rows -> {a.out}")
