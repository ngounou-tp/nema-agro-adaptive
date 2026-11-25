"""Grounded knowledge tasks (the model reads a table in the context, it does not invent facts):

  agro_symptom_triage  farmer describes symptoms -> crop, problem code from the table, whether to call a technician,
                       first actions. Includes 'unknown' cases where the right answer is not in the table.
  agro_input_safety    questions about pesticides/inputs -> decision (label_dose | not_registered | refuse_unsafe |
                       seek_medical_help), the dose ONLY when it is on the label in context, and a short message.
  agro_advice_mcq      multiple-choice questions YOU write from your fiches techniques (data/seeds/advice_mcq.csv).

Product names in the safety task are fictional on purpose. In NEMA, the backend puts the real registered list
(from the official MINADER list) into the context, so the model learns the behaviour, never memorised doses.

Run:  python -m src.generators.gen_agro_knowledge --n 900 --out data/build/gen_agro_knowledge.jsonl
"""
from __future__ import annotations

import argparse
import csv
import random

from src.common import ROOT, dumps_gold, make_example, write_jsonl

SEEDS = ROOT / "data" / "seeds"

TRIAGE_INSTR = (
    "Task: agro_symptom_triage. Compare the farmer's description with the problem table in the context. "
    "Return ONLY a JSON object with keys: crop, problem (a code from the table, or \"unknown\" if nothing matches), "
    "see_technician (true for high-severity problems or unknown), advice (the table's first actions, in the farmer's "
    "language: French for French, English for English or Pidgin; for unknown, advise a photo and a visit by an extension officer)."
)
UNKNOWN_ADVICE = {
    "fr": "Je ne peux pas identifier le problème avec ces informations. Prenez des photos nettes et faites venir un technicien agricole.",
    "en": "I cannot identify the problem from this description. Take clear photos and ask an extension officer to visit.",
}
OPENERS = {
    "fr": ["Bonjour, dans mon champ de {crop}, ", "Mon {crop} a un problème: ", "Aidez-moi, sur mes plants de {crop} ", ""],
    "en": ["Hello, in my {crop} farm ", "My {crop} has a problem: ", "Please help, on my {crop} plants ", ""],
    "pcm": ["Abeg help me, for my {crop} farm ", "My {crop} get problem: ", "Massa, for my {crop} ", ""],
}
CROP_WORD = {"maize": {"fr": "maïs", "en": "maize", "pcm": "corn"}, "cassava": {"fr": "manioc", "en": "cassava", "pcm": "cassava"},
             "cocoa": {"fr": "cacao", "en": "cocoa", "pcm": "cocoa"}, "coffee": {"fr": "café", "en": "coffee", "pcm": "coffee"},
             "tomato": {"fr": "tomate", "en": "tomato", "pcm": "tomato"}, "plantain": {"fr": "plantain", "en": "plantain", "pcm": "plantain"},
             "groundnut": {"fr": "arachide", "en": "groundnut", "pcm": "groundnut"},
             "pepper": {"fr": "piment", "en": "pepper", "pcm": "pepe"}, "beans": {"fr": "haricot", "en": "beans", "pcm": "beans"},
             "cabbage": {"fr": "chou", "en": "cabbage", "pcm": "cabbage"}, "banana": {"fr": "banane", "en": "banana", "pcm": "banana"}}


def load_symptoms():
    with open(SEEDS / "symptom_table.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table_text(rows):
    return "Problem table:\n" + "\n".join(
        f"- crop={r['crop']}; code={r['code']}; name={r['name_en']}; severity={r['severity']}; "
        f"signs: {r['symptoms_en'].replace('|', '; ')}; first actions (fr): {r['actions_fr']} | (en): {r['actions_en']}"
        for r in rows)


def triage_example(rng, table, lang):
    crop = rng.choice(sorted({r["crop"] for r in table}))
    crop_rows = [r for r in table if r["crop"] == crop]
    others = [r for r in table if r["crop"] != crop]
    unknown = rng.random() < 0.12
    if unknown:
        # describe a problem from ANOTHER crop's row that we leave out of the context
        source = rng.choice(others)
        ctx_rows = crop_rows + rng.sample([r for r in others if r["code"] != source["code"]], 1)
    else:
        source = rng.choice(crop_rows)
        ctx_rows = crop_rows + rng.sample(others, min(2, len(others)))
    rng.shuffle(ctx_rows)
    signs = source[f"symptoms_{lang}"].split("|")
    picked = rng.sample(signs, rng.randint(1, len(signs)))
    joiner = {"fr": " et ", "en": " and ", "pcm": " and "}[lang]
    opener = rng.choice(OPENERS[lang]).format(crop=CROP_WORD[crop][lang])
    desc = opener + joiner.join(picked) + "."
    out_lang = "fr" if lang == "fr" else "en"
    if unknown:
        gold = {"crop": crop, "problem": "unknown", "see_technician": True, "advice": UNKNOWN_ADVICE[out_lang]}
    else:
        gold = {"crop": crop, "problem": source["code"], "see_technician": source["severity"] == "high",
                "advice": source[f"actions_{out_lang}"]}
    return make_example("agro_symptom_triage", lang, f"triage_{lang}", f"{TRIAGE_INSTR}\nFarmer: {desc[0].upper() + desc[1:]}",
                        dumps_gold(gold), context=table_text(ctx_rows), meta={"gold": gold, "unknown": unknown})


# ---------------------------------------------------------------- input safety
SAFETY_INSTR = (
    "Task: agro_input_safety. Answer the question using ONLY the registered-products table in the context. "
    "Return ONLY a JSON object with keys: decision (label_dose | not_registered | refuse_unsafe | seek_medical_help), "
    "dose (copy the label dose exactly when decision is label_dose, else null), message (short answer in the user's "
    "language; always remind protective equipment when spraying is involved). Never invent a dose."
)
FAKE_LABELS = [
    {"name": "Product K-12", "type": "fungicide", "crops": ["tomato", "cocoa", "pepper"], "dose": "40 g per 15 L sprayer", "phi": 14},
    {"name": "Product T-7", "type": "insecticide", "crops": ["maize", "beans"], "dose": "25 mL per 15 L sprayer", "phi": 7},
    {"name": "Product C-21", "type": "fungicide", "crops": ["cocoa", "coffee"], "dose": "50 g per 15 L sprayer", "phi": 21},
    {"name": "Product B-3", "type": "biological insecticide", "crops": ["tomato", "cabbage", "maize"], "dose": "30 mL per 15 L sprayer", "phi": 3},
    {"name": "Product F-10", "type": "fungicide", "crops": ["plantain", "banana"], "dose": "35 mL per 15 L sprayer", "phi": 10},
    {"name": "Product H-5", "type": "herbicide", "crops": ["maize", "cassava"], "dose": "100 mL per 15 L sprayer", "phi": 30},
]
UNREGISTERED = ["Product Z-99", "Product X-40", "le produit rouge du marché", "the red bottle from the market", "dat red bottle for market"]
PPE = {"fr": "Portez gants, masque, lunettes et bottes, et ne mangez pas pendant le traitement.",
       "en": "Wear gloves, mask, goggles and boots, and do not eat while spraying."}


def safety_example(rng, lang):
    labels = rng.sample(FAKE_LABELS, 4)
    ctx = "Registered products (label data):\n" + "\n".join(
        f"- {l['name']} ({l['type']}): crops = {', '.join(l['crops'])}; dose = {l['dose']}; pre-harvest interval = {l['phi']} days"
        for l in labels)
    kind = rng.choices(["label_dose", "not_registered", "refuse_unsafe", "seek_medical_help"], weights=[4, 2, 3, 1])[0]
    out = "fr" if lang == "fr" else "en"
    lab = rng.choice(labels)
    crop_code = rng.choice(lab["crops"])
    crop = CROP_WORD[crop_code][lang]
    dose = None
    if kind == "label_dose":
        q = {"fr": f"Quelle dose de {lab['name']} pour traiter mon champ de {crop} ?", "en": f"What dose of {lab['name']} should I use on my {crop}?",
             "pcm": f"How much {lab['name']} I go put for my {crop}?"}[lang]
        dose = lab["dose"]
        msg = {"fr": f"Selon l'étiquette: {dose}. Respectez un délai de {lab['phi']} jours avant la récolte. {PPE['fr']}",
               "en": f"According to the label: {dose}. Wait {lab['phi']} days before harvest. {PPE['en']}"}[out]
    elif kind == "not_registered":
        bad = rng.choice(UNREGISTERED)
        q = {"fr": f"Je veux utiliser {bad} sur mon champ de {crop}, je mets combien ?", "en": f"I want to use {bad} on my {crop}, how much do I put?",
             "pcm": f"I wan use {bad} for my {crop}, how much I go put?"}[lang]
        msg = {"fr": "Ce produit n'est pas dans la liste des produits homologués fournie; je ne peux pas donner de dose. Demandez à un technicien un produit homologué pour cette culture.",
               "en": "This product is not in the registered list provided, so I cannot give a dose. Ask an extension officer for a product registered for this crop."}[out]
    elif kind == "refuse_unsafe":
        variant = rng.choice(["double", "mix", "wrong_crop", "phi"])
        other_crop = CROP_WORD[rng.choice([c for c in ["cocoa", "tomato", "maize", "coffee", "plantain", "cassava"] if c not in lab["crops"]])][lang]
        q = {
            "double": {"fr": f"Si je double la dose de {lab['name']}, ça marche mieux ?", "en": f"If I double the dose of {lab['name']}, will it work better?", "pcm": f"If I put {lab['name']} two times di dose, e go work fine pass?"},
            "mix": {"fr": f"Je peux mélanger {lab['name']} avec d'autres produits dans le même pulvérisateur ?", "en": f"Can I mix {lab['name']} with other products in the same sprayer?", "pcm": f"I fit mix {lab['name']} with other medicine for one sprayer?"},
            "wrong_crop": {"fr": f"Je peux utiliser {lab['name']} sur mon champ de {other_crop} ?", "en": f"Can I use {lab['name']} on my {other_crop}?", "pcm": f"I fit use {lab['name']} for my {other_crop}?"},
            "phi": {"fr": f"J'ai traité avec {lab['name']} hier, je peux récolter et vendre aujourd'hui ?", "en": f"I sprayed {lab['name']} yesterday, can I harvest and sell today?", "pcm": f"I spray {lab['name']} yesterday, I fit harvest sell today?"},
        }[variant][lang]
        msg = {"fr": f"Non. Suivez strictement l'étiquette (cultures autorisées, dose, délai de {lab['phi']} jours avant récolte). Ne dépassez pas la dose et ne faites pas de mélange non prévu. {PPE['fr']}",
               "en": f"No. Follow the label strictly (approved crops, dose, {lab['phi']}-day pre-harvest interval). Do not exceed the dose or make mixtures the label does not allow. {PPE['en']}"}[out]
    else:
        q = {"fr": "Après avoir pulvérisé, j'ai mal à la tête, je vomis et je vois flou. Que faire ?",
             "en": "After spraying I have a headache, I am vomiting and my vision is blurred. What should I do?",
             "pcm": "After I spray, my head dey pain me, I dey vomit and my eye no see well. Wetin I go do?"}[lang]
        msg = {"fr": "Arrêtez tout de suite, enlevez les vêtements contaminés, lavez la peau à l'eau et au savon, et allez immédiatement au centre de santé le plus proche avec l'emballage du produit.",
               "en": "Stop now, remove contaminated clothes, wash skin with soap and water, and go immediately to the nearest health centre with the product packaging."}[out]
    gold = {"decision": kind, "dose": dose, "message": msg}
    return make_example("agro_input_safety", lang, f"safety_{kind}_{lang}", f"{SAFETY_INSTR}\nQuestion: {q}", dumps_gold(gold),
                        context=ctx, meta={"gold": gold})


# ---------------------------------------------------------------- MCQ from your fiches
MCQ_INSTR = "Task: agro_advice_mcq. Answer the multiple-choice question about farming in Cameroon. Return ONLY the letter (A, B, C or D)."


def mcq_examples():
    path = SEEDS / "advice_mcq.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("question") or r["question"].startswith("EXAMPLE"):
                continue
            opts = "\n".join(f"{k}. {r['option_' + k.lower()]}" for k in "ABCD")
            rows.append(make_example("agro_advice_mcq", r.get("lang", "fr"), "mcq", f"{MCQ_INSTR}\n{r['question']}\n{opts}",
                                     r["answer"].strip().upper(), meta={"source": r.get("source", ""), "gold": r["answer"].strip().upper()}))
    return rows


def generate(n, seed=13):
    rng = random.Random(seed)
    table = load_symptoms()
    rows, seen = [], set()
    while len(rows) < n:
        lang = rng.choices(["fr", "en", "pcm"], weights=[0.5, 0.3, 0.2])[0]
        ex = triage_example(rng, table, lang) if len(rows) % 2 == 0 else safety_example(rng, lang)
        if ex["id"] not in seen:
            seen.add(ex["id"])
            rows.append(ex)
    return rows + mcq_examples()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/build/gen_agro_knowledge.jsonl")
    a = ap.parse_args()
    print(f"wrote {write_jsonl(generate(a.n, a.seed), a.out)} rows -> {a.out}")
