"""Self-check for the kit, CPU only, about 10 seconds.

It fakes two models on the test set (a weak 'base' and a better 'adapted'), scores them, and checks that the
scorer ranks them correctly. Run it after any change to the generators or the scorer:

    python scripts/smoke_test.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import BUILD, read_jsonl  # noqa: E402
from src.eval.score import aggregate, headline, score_one, to_markdown  # noqa: E402


def corrupt(gold_txt, rng):
    if not gold_txt.startswith("{"):
        return rng.choice("ABCD")
    g = json.loads(gold_txt)
    k = rng.choice(list(g))
    g[k] = None if g[k] is not None else "wrong"
    return "Here is the result:\n```json\n" + json.dumps(g, ensure_ascii=False) + "\n```"


def main():
    rng = random.Random(0)
    preds = []
    for project in ("agro", "waste"):
        path = BUILD / project / "test.jsonl"
        if not path.exists():
            print(f"skip {project}: build it first")
            continue
        for ex in read_jsonl(path):
            for model, p_ok in (("base", 0.3), ("adapted", 0.85)):
                out = ex["completion"] if rng.random() < p_ok else corrupt(ex["completion"], rng)
                preds.append({"id": ex["id"], "task": ex["task"], "split": ex["split"], "lang": ex["lang"],
                              "completion": ex["completion"], "output": out, "model": model})
    # unit checks
    assert score_one("agro_unit_price", '{"a": 1}', '```json\n{"a": 1.0}\n```')["exact_match"] == 1.0
    assert score_one("agro_advice_mcq", "B", "The answer is B.")["acc"] == 1.0
    assert score_one("agro_input_safety", '{"decision": "not_registered", "dose": null, "message": "x"}',
                     '{"decision": "label_dose", "dose": "10 mL", "message": "x"}')["dose_violation"] == 1.0
    agg = aggregate(preds)
    print(to_markdown(agg).split("\n\n")[0])
    hb, ha = headline(agg, "base"), headline(agg, "adapted")
    assert ha > hb, "scorer should rank the better fake model higher"
    print(f"\nOK: base {hb:.1%} < adapted {ha:.1%}")


if __name__ == "__main__":
    main()
