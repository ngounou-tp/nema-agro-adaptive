"""Tasks with exact numeric answers (the easiest place to prove a measurable gain):

  agro_unit_price   convert a local unit to kg using the reference table in context, compute price per kg,
                    total price, and the gap to a reference market price.
  agro_coop_ledger  turn a cooperative's free-text delivery notes into ledger rows + totals per member.
  agro_trace_record turn a farmer's harvest story into a traceability lot record, including whether the
                    pre-harvest interval (PHI) of the last pesticide was respected, using the label table in context.

All reference numbers in contexts are INPUTS the model must use, not facts it must remember.
If data/seeds/wfp_food_prices_cmr.csv exists (HDX "Cameroon - Food Prices"), real reference prices are used.

Run:  python -m src.generators.gen_agro_calc --n 600 --out data/build/gen_agro_calc.jsonl
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

from src.common import (LEXICON, ROOT, dumps_gold, fmt_xaf, load_json, make_example, random_date,
                        write_jsonl)

# ---------------------------------------------------------------- unit + price
UNIT_INSTR = (
    "Task: agro_unit_price. Using ONLY the conversion table and reference prices in the context, return a JSON object "
    "with keys: quantity_kg, price_per_kg, total_price, reference_price_per_kg, difference_pct. "
    "Round price_per_kg to the nearest whole franc and difference_pct to one decimal "
    "((price_per_kg - reference) / reference x 100). Use null if the table lacks what you need."
)

WFP_NAME = {  # HDX commodity name -> lexicon id (extend after you open the CSV)
    "Maize": "maize", "Maize (white)": "maize", "Cassava": "cassava", "Cassava flour": "cassava_flour",
    "Rice (imported)": "rice", "Rice (local)": "rice", "Beans": "beans", "Beans (dry)": "beans",
    "Groundnuts": "groundnut", "Groundnuts (shelled)": "groundnut", "Tomatoes": "tomato", "Onions": "onion",
    "Potatoes (Irish)": "irish_potato", "Plantains": "plantain", "Yam": "yam", "Cocoyam": "cocoyam",
    "Oil (palm)": "palm_oil", "Gari": "garri", "Soybeans": "soybean",
}


def load_wfp_refs(path: Path):
    """Latest price per (product, market) in XAF per KG from the HDX CSV; empty dict if file missing."""
    if not path.exists():
        return {}
    latest = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("date", "").startswith("#"):
                continue  # HDX files have a second header row of tags
            pid = WFP_NAME.get(row.get("commodity", ""))
            unit = (row.get("unit") or "").upper().replace(" ", "")
            if not pid or row.get("currency") != "XAF" or row.get("pricetype") not in ("Retail", "Wholesale"):
                continue
            try:
                price = float(row["price"])
            except (KeyError, ValueError):
                continue
            per_kg = {"KG": 1, "100KG": 100, "50KG": 50, "25KG": 25}.get(unit)
            if not per_kg:
                continue
            key = (pid, row.get("market", ""))
            if key not in latest or row["date"] > latest[key][0]:
                latest[key] = (row["date"], round(price / per_kg), row.get("pricetype"))
    return latest


def unit_price_example(rng, lex, wfp, lang):
    conversions = {}
    for product in lex["products"]:
        values = {}
        for unit_id in product["units"]:
            unit = lex["units"][unit_id]
            kg_per = unit["kg"]
            if kg_per is None:
                kg_per = product.get("kg_per_unit", {}).get(unit_id)
            if kg_per is not None:
                values[unit_id] = kg_per
        if values:
            conversions[product["id"]] = values

    eligible = list(conversions)
    pid = rng.choice(eligible)
    product = next(p for p in lex["products"] if p["id"] == pid)
    unit_id = rng.choice(list(conversions[pid]))
    kg_per = conversions[pid][unit_id]
    qty = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    lo, hi = product.get("price_per_kg", [150, 600])
    per_unit = int(round(rng.uniform(lo, hi) * kg_per / 250.0)) * 250
    wfp_rows = [(k, v) for k, v in wfp.items() if k[0] == pid]
    if wfp_rows:
        (_, market), (when, ref, ptype) = rng.choice(wfp_rows)
        ref_txt = f"{pid}: {ref} XAF/kg ({ptype.lower()}, {market}, {when}, source WFP/HDX)"
    else:
        ref = int(round(rng.uniform(lo, hi) / 5.0)) * 5
        market = rng.choice(sum(lex["places"].values(), []))
        ref_txt = f"{pid}: {ref} XAF/kg (example reference, {market})"
    # sometimes the table lacks the unit -> answer must be nulls (teaches the model not to guess)
    missing = rng.random() < 0.08
    table_units = {u: k for u, k in conversions[pid].items() if not (missing and u == unit_id)}
    conv = "; ".join(f"1 {u} of {pid} = {k} kg" for u, k in table_units.items()) or "(no conversion listed)"
    context = f"Conversion table (cooperative reference): {conv}.\nReference prices: {ref_txt}."
    unit_word = lex["units"][unit_id][lang][1 if qty > 1 and len(lex["units"][unit_id][lang]) > 1 else 0]
    prod_word = rng.choice(product[lang])
    q = {
        "fr": f"Un vendeur propose {qty} {unit_word} de {prod_word} à {fmt_xaf(per_unit, rng, 'fr')} l'unité. Calcule pour moi.",
        "en": f"A seller offers {qty} {unit_word} of {prod_word} at {fmt_xaf(per_unit, rng, 'en')} each. Work it out for me.",
        "pcm": f"Some man di sell {qty} {unit_word} {prod_word}, {fmt_xaf(per_unit, rng, 'pcm')} for one. Calculate am for me.",
    }[lang]
    if missing:
        gold = {"quantity_kg": None, "price_per_kg": None, "total_price": per_unit * qty,
                "reference_price_per_kg": ref, "difference_pct": None}
    else:
        ppk = round(per_unit / kg_per)
        gold = {"quantity_kg": qty * kg_per, "price_per_kg": ppk, "total_price": per_unit * qty,
                "reference_price_per_kg": ref, "difference_pct": round((ppk - ref) / ref * 100, 1)}
    return make_example("agro_unit_price", lang, f"unit_{lang}", f"{UNIT_INSTR}\nQuestion: {q}", dumps_gold(gold),
                        context=context, meta={"gold": gold, "missing_unit": missing})


# ---------------------------------------------------------------- cooperative ledger
LEDGER_INSTR = (
    "Task: agro_coop_ledger. Turn the delivery notes into JSON with keys: entries (list of objects with member, "
    "product, quantity_kg, date YYYY-MM-DD) and totals_kg (object: member -> total kg across all products). "
    "Convert bags using the bag weights given in the notes header. Keep member names exactly as written."
)
NAMES = ["Mbarga Jean", "Ngo Bilong Marie", "Tchoumi Paul", "Fon Peter", "Ayuk Grace", "Nana Brigitte", "Ewane Eric",
         "Abena Sylvie", "Ndi Clovis", "Mballa Rose", "Kamga Alain", "Nkeng Judith", "Ateba Joseph", "Mefire Aicha",
         "Bello Ousmane", "Tabi Esther", "Fouda Martin", "Ekane Divine", "Djoumessi Lucie", "Ngum Florence"]


def ledger_example(rng, lex, lang):
    pid = rng.choice(["cocoa", "coffee", "maize", "beans", "groundnut", "soybean", "rice"])
    bag_kg = {"cocoa": 65, "coffee": 60, "maize": 100, "beans": 100, "groundnut": 50, "soybean": 100, "rice": 50}[pid]
    members = rng.sample(NAMES, rng.randint(3, 6))
    day0 = random_date(rng, date(2026, 1, 10), date(2026, 12, 10))
    entries, lines = [], []
    for _ in range(rng.randint(4, 9)):
        m = rng.choice(members)
        d = day0 + timedelta(days=rng.randint(0, 6))
        if rng.random() < 0.5:
            bags = rng.randint(1, 6)
            kg = bags * bag_kg
            amount_txt = {"fr": f"{bags} sac(s)", "en": f"{bags} bag(s)", "pcm": f"{bags} bag"}[lang]
        else:
            kg = rng.randint(15, 400)
            amount_txt = f"{kg} kg"
        entries.append({"member": m, "product": pid, "quantity_kg": kg, "date": d.isoformat()})
        lines.append({
            "fr": f"- {d.strftime('%d/%m')} : {m} a livré {amount_txt}",
            "en": f"- {d.strftime('%d/%m')}: {m} delivered {amount_txt}",
            "pcm": f"- {d.strftime('%d/%m')}: {m} bring {amount_txt}",
        }[lang])
    header = {
        "fr": f"Notes de collecte {pid} (année {day0.year}). 1 sac = {bag_kg} kg.",
        "en": f"Collection notes, {pid} ({day0.year}). 1 bag = {bag_kg} kg.",
        "pcm": f"Collection notes for {pid} ({day0.year}). One bag = {bag_kg} kg.",
    }[lang]
    totals = {}
    for e in entries:
        totals[e["member"]] = totals.get(e["member"], 0) + e["quantity_kg"]
    gold = {"entries": entries, "totals_kg": totals}
    prompt = f"{LEDGER_INSTR}\nNotes:\n{header}\n" + "\n".join(lines)
    return make_example("agro_coop_ledger", lang, f"ledger_{lang}", prompt, dumps_gold(gold), meta={"gold": gold})


# ---------------------------------------------------------------- traceability record
TRACE_INSTR = (
    "Task: agro_trace_record. Build a traceability lot record as JSON with keys: lot_id, producer, group, product, "
    "harvest_date, quantity_kg, last_treatment (product name or null), last_treatment_date, phi_days, "
    "phi_respected (true/false/null). lot_id format: <PRODUCT>-<TOWN first 3 letters, upper case, no accents>-<YYYYMMDD of harvest>. "
    "Take phi_days from the label table in the context; phi_respected is true when harvest_date minus "
    "last_treatment_date is at least phi_days; null when there was no treatment."
)
# Fictional product names on purpose: the task teaches the date arithmetic and record format,
# not real pesticide facts. Replace with real label data ONLY from the official registered list.
LABELS = {"Product K-12 (fungicide)": 14, "Product T-7 (insecticide)": 7, "Product C-21 (fungicide)": 21,
          "Product B-3 (biological insecticide)": 3, "Product F-10 (fungicide)": 10}


def trace_example(rng, lex, lang):
    pid = rng.choice(["cocoa", "coffee", "maize", "tomato", "pepper", "beans", "groundnut"])
    product = next(p for p in lex["products"] if p["id"] == pid)
    town = rng.choice(sum(lex["places"].values(), []))
    town_code = (town.upper().replace("É", "E").replace("È", "E").replace("-", "")[:3])
    producer = rng.choice(NAMES)
    group = f"GIC {rng.choice(['Espoir', 'Unité', 'Femmes Actives', 'Jeunes Agriculteurs', 'Avenir', 'Solidarité'])} {town}"
    harvest = random_date(rng, date(2026, 1, 10), date(2026, 12, 15))
    qty = rng.randint(50, 3000)
    treated = rng.random() < 0.8
    label = rng.choice(list(LABELS)) if treated else None
    phi = LABELS[label] if treated else None
    gap = rng.randint(1, 30) if treated else None
    tdate = harvest - timedelta(days=gap) if treated else None
    ctx = "Label table (pre-harvest interval, days): " + "; ".join(f"{k} = {v}" for k, v in LABELS.items()) + "."
    if lang == "fr":
        story = (f"Je suis {producer}, membre du {group}. J'ai récolté {qty} kg de {product['fr'][0]} le "
                 f"{harvest.strftime('%d/%m/%Y')}. ")
        story += (f"Dernier traitement avec {label} le {tdate.strftime('%d/%m/%Y')}." if treated else "Aucun traitement cette saison.")
    elif lang == "en":
        story = (f"My name is {producer}, member of {group}. I harvested {qty} kg of {product['en'][0]} on "
                 f"{harvest.strftime('%d/%m/%Y')}. ")
        story += (f"Last spray was {label} on {tdate.strftime('%d/%m/%Y')}." if treated else "No treatment this season.")
    else:
        story = (f"Na me {producer}, I dey for {group}. I harvest {qty} kg {product['pcm'][0]} for "
                 f"{harvest.strftime('%d/%m/%Y')}. ")
        story += (f"Last time wey I spray na {label} for {tdate.strftime('%d/%m/%Y')}." if treated else "I no spray anything dis season.")
    gold = {"lot_id": f"{pid.upper()}-{town_code}-{harvest.strftime('%Y%m%d')}", "producer": producer, "group": group,
            "product": pid, "harvest_date": harvest.isoformat(), "quantity_kg": qty, "last_treatment": label,
            "last_treatment_date": tdate.isoformat() if treated else None, "phi_days": phi,
            "phi_respected": (gap >= phi) if treated else None}
    return make_example("agro_trace_record", lang, f"trace_{lang}", f"{TRACE_INSTR}\nStory: {story}", dumps_gold(gold),
                        context=ctx, meta={"gold": gold, "town": town})


def generate(n, seed=11, wfp_path=None):
    rng = random.Random(seed)
    lex = load_json(LEXICON / "agro_lexicon.json")
    default = ROOT / "data" / "seeds" / "wfp_food_prices_cmr.csv"  # upload it there with GitHub "Add file"
    wfp = load_wfp_refs(Path(wfp_path) if wfp_path else default)
    rows, seen = [], set()
    makers = [lambda l: unit_price_example(rng, lex, wfp, l), lambda l: ledger_example(rng, lex, l),
              lambda l: trace_example(rng, lex, l)]
    while len(rows) < n:
        lang = rng.choices(["fr", "en", "pcm"], weights=[0.5, 0.3, 0.2])[0]
        ex = makers[len(rows) % 3](lang)
        if ex["id"] not in seen:
            seen.add(ex["id"])
            rows.append(ex)
    return rows, bool(wfp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--wfp", default=None, help="path to the HDX Cameroon food prices CSV (optional)")
    ap.add_argument("--out", default="data/build/gen_agro_calc.jsonl")
    a = ap.parse_args()
    rows, used_wfp = generate(a.n, a.seed, a.wfp)
    print(f"wrote {write_jsonl(rows, a.out)} rows -> {a.out} (real WFP reference prices: {used_wfp})")
