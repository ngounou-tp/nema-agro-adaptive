"""Project B (REVALO-Waste) tasks:

  waste_sort            item description -> category, valorisable, hazardous, route (from data/seeds/waste_items.csv)
  waste_pickup_extract  WhatsApp-style pickup request -> structured request (city, neighbourhood, type, bags, urgency, contact)
  waste_compost_calc    tonnage + parameters in context -> organic kg, compost kg, value in XAF
  waste_hygiene_mcq     multiple-choice questions YOU write from WHO/UNICEF sanitation material (data/seeds/hygiene_mcq.csv)

Run:  python -m src.generators.gen_waste --n 900 --out data/build/gen_waste.jsonl
"""
from __future__ import annotations

import argparse
import csv
import random

from src.common import ROOT, dumps_gold, fake_phone, make_example, stable_id, write_jsonl

SEEDS = ROOT / "data" / "seeds"

SORT_INSTR = ("Task: waste_sort. Classify the waste item. Return ONLY a JSON object with keys: category "
              "(organic | plastic_rigid | plastic_film | paper_cardboard | glass | metal | ewaste | hazardous | textile | residual), "
              "valorisable (true/false), hazardous (true/false), route (what to do with it, in the user's language: "
              "French for French, English for English or Pidgin).")
PICKUP_INSTR = ("Task: waste_pickup_extract. Read the pickup request and return ONLY a JSON object with keys: city, neighbourhood, "
                "waste_type (household | green | construction | ewaste | bulky | mixed), bags (integer or null), "
                "urgency (low | normal | high), contact (digits only or null).")
COMPOST_INSTR = ("Task: waste_compost_calc. Use ONLY the parameters in the context. Return ONLY a JSON object with keys: "
                 "input_kg, organic_kg, compost_kg, value_xaf. organic_kg = input_kg x organic share; compost_kg = organic_kg x "
                 "compost yield; value_xaf = compost_kg x price per kg. Round every number to the nearest whole number.")
MCQ_INSTR = "Task: waste_hygiene_mcq. Answer the multiple-choice question about waste and sanitation. Return ONLY the letter (A, B, C or D)."

ASK = {"fr": ["Je fais quoi avec {x} ?", "Comment jeter {x} ?", "{X}: poubelle, recyclage ou autre ?", "J'ai des {x}, on les met où ?"],
       "en": ["What do I do with {x}?", "How should I throw away {x}?", "{X}: bin, recycling or something else?"],
       "pcm": ["Wetin I go do with {x}?", "How I go throw {x}?", "{X}, I go put am for where?"]}

NEIGHBOURHOODS = {
    "Douala": ["Bonamoussadi", "Akwa", "Bonabéri", "Makepe", "Ndogbong", "Logbaba", "Bépanda", "Deido", "New Bell", "PK14"],
    "Yaoundé": ["Bastos", "Mvog-Ada", "Essos", "Biyem-Assi", "Nkolbisson", "Emana", "Mokolo", "Ekounou"],
    "Bafoussam": ["Tamdja", "Djeleng", "Kamkop"],
    "Buea": ["Molyko", "Bonduma", "Great Soppo"],
    "Bamenda": ["Nkwen", "Mile 4", "Up Station"],
}
WASTE_TYPES = {
    "household": {"fr": ["ordures ménagères", "déchets de la maison"], "en": ["household rubbish", "house waste"], "pcm": ["house dirty", "dirty for house"]},
    "green": {"fr": ["herbes et branches coupées", "déchets verts"], "en": ["cut grass and branches", "garden waste"], "pcm": ["grass and stick wey we cut"]},
    "construction": {"fr": ["gravats de chantier", "débris de construction"], "en": ["building rubble", "construction debris"], "pcm": ["block and sand dirty from building"]},
    "ewaste": {"fr": ["vieux appareils électroniques", "vieux téléphones et chargeurs"], "en": ["old electronics", "old phones and chargers"], "pcm": ["old phone and charger dem"]},
    "bulky": {"fr": ["un vieux matelas et un canapé", "des meubles cassés"], "en": ["an old mattress and a sofa", "broken furniture"], "pcm": ["old mattress and chair wey don spoil"]},
    "mixed": {"fr": ["déchets mélangés", "un peu de tout"], "en": ["mixed waste", "a bit of everything"], "pcm": ["all kind dirty mix"]},
}
URGENCY = {
    "high": {"fr": [" C'est urgent, ça commence à sentir.", " Urgent, il y a des rats.", " Le bac déborde, svp vite."],
             "en": [" Urgent, it has started to smell.", " Urgent, rats are coming.", " The bin is overflowing, please hurry."],
             "pcm": [" E urgent, e don di smell.", " Rat dem don start come, abeg quick."]},
    "low": {"fr": [" Pas pressé, quand vous pouvez.", " Cette semaine ou la prochaine, ça va."],
            "en": [" No hurry, whenever you can.", " This week or next is fine."],
            "pcm": [" No hurry, any time wey you fit."]},
    "normal": {"fr": [""], "en": [""], "pcm": [""]},
}


def load_items():
    with open(SEEDS / "waste_items.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sort_example(rng, items, lang):
    it = rng.choice(items)
    x = it[f"item_{lang}"]
    q = rng.choice(ASK[lang]).format(x=x, X=x[0].upper() + x[1:])
    out = "fr" if lang == "fr" else "en"
    gold = {"category": it["category"], "valorisable": it["valorisable"] == "true", "hazardous": it["hazardous"] == "true",
            "route": it[f"route_{out}"]}
    return make_example("waste_sort", lang, f"sort_{lang}", f"{SORT_INSTR}\nQuestion: {q}", dumps_gold(gold), meta={"gold": gold, "item": it["item_en"]})


def pickup_example(rng, lang):
    city = rng.choice(list(NEIGHBOURHOODS))
    hood = rng.choice(NEIGHBOURHOODS[city])
    wtype = rng.choice(list(WASTE_TYPES))
    wtxt = rng.choice(WASTE_TYPES[wtype][lang])
    bags = rng.choice([None, 1, 2, 3, 4, 5, 6, 8, 10]) if wtype not in ("bulky",) else None
    urg = rng.choices(["normal", "high", "low"], weights=[5, 3, 2])[0]
    phone = fake_phone(rng) if rng.random() < 0.75 else None
    bag_txt = ""
    if bags:
        bag_txt = {"fr": f"{bags} sac{'s' if bags > 1 else ''} de ", "en": f"{bags} bag{'s' if bags > 1 else ''} of ", "pcm": f"{bags} bag "}[lang]
    tpl = {"fr": ["Bonjour, je suis à {h} ({c}). J'ai {b}{w} à enlever.{u}{p}", "Ramassage svp à {c}, quartier {h}: {b}{w}.{u}{p}"],
           "en": ["Hello, I'm in {h}, {c}. I have {b}{w} to collect.{u}{p}", "Pickup please in {c}, {h} area: {b}{w}.{u}{p}"],
           "pcm": ["Hello, I dey for {h} ({c}). I get {b}{w} for carry.{u}{p}", "Abeg come carry {b}{w} for {h}, {c}.{u}{p}"]}[lang]
    p_txt = ""
    if phone:
        p_txt = {"fr": f" Tel: {phone}", "en": f" Phone: {phone}", "pcm": f" Call me {phone}"}[lang]
    msg = rng.choice(tpl).format(h=hood, c=city, b=bag_txt, w=wtxt, u=rng.choice(URGENCY[urg][lang]), p=p_txt)
    gold = {"city": city, "neighbourhood": hood, "waste_type": wtype, "bags": bags, "urgency": urg,
            "contact": phone.replace(" ", "") if phone else None}
    return make_example("waste_pickup_extract", lang, f"pickup_{lang}", f"{PICKUP_INSTR}\nMessage: {msg}", dumps_gold(gold), meta={"gold": gold})


def compost_example(rng, lang):
    share = rng.choice([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    yld = rng.choice([0.25, 0.3, 0.35, 0.4])
    price = rng.choice([50, 75, 100, 125, 150])
    if rng.random() < 0.5:
        bags, kg_bag = rng.randint(5, 200), rng.choice([10, 15, 20, 25])
        input_kg = bags * kg_bag
        q = {"fr": f"Notre quartier collecte {bags} sacs de {kg_bag} kg de déchets par semaine. Combien de compost et quelle valeur ?",
             "en": f"Our area collects {bags} bags of {kg_bag} kg of waste a week. How much compost and what value?",
             "pcm": f"Our quarter di gather {bags} bag of {kg_bag} kg dirty every week. How much compost and how much money?"}[lang]
    else:
        t = rng.choice([0.5, 1, 1.5, 2, 3, 5, 8, 10])
        input_kg = int(t * 1000)
        q = {"fr": f"Le marché produit {t} tonne(s) de déchets par jour. Combien de compost et quelle valeur ?",
             "en": f"The market produces {t} tonne(s) of waste a day. How much compost and what value?",
             "pcm": f"Di market di produce {t} ton dirty every day. How much compost and how much money?"}[lang]
    ctx = (f"Parameters: organic share of this waste = {int(share * 100)}%; compost yield = {int(yld * 100)}% of organic mass; "
           f"selling price of compost = {price} XAF per kg.")
    organic = round(input_kg * share)
    compost = round(organic * yld)
    gold = {"input_kg": input_kg, "organic_kg": organic, "compost_kg": compost, "value_xaf": round(compost * price)}
    return make_example("waste_compost_calc", lang, f"compost_{lang}", f"{COMPOST_INSTR}\nQuestion: {q}", dumps_gold(gold),
                        context=ctx, meta={"gold": gold})


def mcq_examples():
    path = SEEDS / "hygiene_mcq.csv"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("question") or r["question"].startswith("EXAMPLE"):
                continue
            opts = "\n".join(f"{k}. {r['option_' + k.lower()]}" for k in "ABCD")
            out.append(make_example("waste_hygiene_mcq", r.get("lang", "fr"), "mcq", f"{MCQ_INSTR}\n{r['question']}\n{opts}",
                                    r["answer"].strip().upper(), meta={"source": r.get("source", ""), "gold": r["answer"].strip().upper()}))
    return out


def all_sort_examples(items):
    """Every distinct (item, language, phrasing) question: the sort task has a finite set, so we enumerate it."""
    out = []
    for it in items:
        for lang in ("fr", "en", "pcm"):
            for i, tpl in enumerate(ASK[lang]):
                rng = random.Random(f"{it['item_en']}-{lang}-{i}")
                ex = sort_example(rng, [it], lang)
                x = it[f"item_{lang}"]
                q = tpl.format(x=x, X=x[0].upper() + x[1:])
                ex["prompt"] = f"{SORT_INSTR}\nQuestion: {q}"
                ex["id"] = stable_id("waste_sort", ex["prompt"])
                ex["template_id"] = f"sort_{lang}_{i}"
                out.append(ex)
    return out


def generate(n, seed=17):
    rng = random.Random(seed)
    items = load_items()
    sorts = all_sort_examples(items)
    rng.shuffle(sorts)
    rows = sorts[: min(len(sorts), n // 3)]
    seen = {r["id"] for r in rows}
    while len(rows) < n:
        lang = rng.choices(["fr", "en", "pcm"], weights=[0.5, 0.3, 0.2])[0]
        ex = pickup_example(rng, lang) if len(rows) % 2 else compost_example(rng, lang)
        if ex["id"] not in seen:
            seen.add(ex["id"])
            rows.append(ex)
    return rows + mcq_examples()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="data/build/gen_waste.jsonl")
    a = ap.parse_args()
    print(f"wrote {write_jsonl(generate(a.n, a.seed), a.out)} rows -> {a.out}")
