"""Generate answers on the frozen test set with the BASE model and with the ADAPTED model (base + your LoRA adapter).

Designed for a free Kaggle notebook GPU (T4 16 GB): the base is loaded once in 4-bit, and the adapter is switched
off for the baseline run, so both runs use exactly the same weights, prompts and decoding.

Run (in the Kaggle notebook, see notebooks/kaggle_eval.ipynb):
    python -m src.eval.run_eval --test data/build/agro/test.jsonl --adapter <your-hf-user>/nema-agro-lora \
        --out results/agro --per-bucket 40 --batch-size 8
Then:
    python -m src.eval.score --pred results/agro_base.jsonl results/agro_adapted.jsonl --out results/agro
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from src.common import full_prompt, read_jsonl

MAX_NEW = {"agro_coop_ledger": 900, "agro_listing_extract": 260, "agro_trace_record": 260, "agro_unit_price": 160,
           "agro_symptom_triage": 260, "agro_input_safety": 260, "waste_sort": 200, "waste_pickup_extract": 160,
           "waste_compost_calc": 120}


def sample(rows, per_bucket, seed=0):
    """Same subset every time: up to N examples per (task, split) bucket."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["task"], r["split"])].append(r)
    out = []
    for key in sorted(buckets):
        items = sorted(buckets[key], key=lambda r: r["id"])
        rng.shuffle(items)
        out += items[:per_bucket] if per_bucket else items
    return out


def resolve_base(adapter: str, override: str | None) -> str:
    if override:
        return override
    from huggingface_hub import hf_hub_download
    cfg_path = Path(adapter) / "adapter_config.json"
    if not cfg_path.exists():
        cfg_path = Path(hf_hub_download(adapter, "adapter_config.json", token=os.environ.get("HF_TOKEN")))
    base = json.loads(cfg_path.read_text())["base_model_name_or_path"]
    if base.startswith("togethercomputer/"):
        # Adaption exports sometimes record Together's internal id; it may not exist on the Hub. Pass --base instead.
        print(f"WARNING: adapter says base = {base}. If loading fails, re-run with --base <official repo id>.")
    return base


def load(base_id: str, adapter: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    kw = dict(quantization_config=bnb, device_map="auto", token=os.environ.get("HF_TOKEN"))
    try:
        model = AutoModelForCausalLM.from_pretrained(base_id, **kw)
    except (ValueError, KeyError):
        # some new multimodal bases (e.g. Qwen3.5) only load with the image-text class
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(base_id, **kw)
    model = PeftModel.from_pretrained(model, adapter, token=os.environ.get("HF_TOKEN"))
    model.eval()
    try:
        tok = AutoTokenizer.from_pretrained(adapter, token=os.environ.get("HF_TOKEN"))
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_id, token=os.environ.get("HF_TOKEN"))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def render(tok, text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    try:  # Qwen3-style templates: turn off the "thinking" block so answers are comparable
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def fewshot_prefix(train_rows, task, k, seed=0):
    """k solved training examples of the same task, shown before the question (a stronger baseline)."""
    pool = sorted([r for r in train_rows if r["task"] == task], key=lambda r: r["id"])
    random.Random(seed).shuffle(pool)
    parts = [f"Example {i + 1}:\n{full_prompt(r)}\nAnswer:\n{r['completion']}" for i, r in enumerate(pool[:k])]
    return "\n\n".join(parts) + "\n\nNow your turn.\n" if parts else ""


def generate(model, tok, rows, batch_size, adapter_on: bool, shots=None):
    import torch
    outputs = []
    by_len = defaultdict(list)
    for r in rows:
        by_len[MAX_NEW.get(r["task"], 16 if r["task"].endswith("_mcq") else 300)].append(r)
    for max_new, group in by_len.items():
        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            texts = [render(tok, (shots.get(r["task"], "") if shots else "") + full_prompt(r)) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            gen_kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            with torch.no_grad():
                if adapter_on:
                    out = model.generate(**enc, **gen_kw)
                else:
                    with model.disable_adapter():
                        out = model.generate(**enc, **gen_kw)
            for r, seq in zip(batch, out):
                text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                outputs.append((r, clean(text)))
            print(f"  {'adapted' if adapter_on else 'base'}{' few-shot' if shots else ''}: {len(outputs)}/{len(rows)}", flush=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--adapter", required=True, help="Hugging Face repo id or local folder of the LoRA adapter")
    ap.add_argument("--base", default=None, help="override the base model id from adapter_config.json")
    ap.add_argument("--out", required=True, help="prefix, e.g. results/agro")
    ap.add_argument("--per-bucket", type=int, default=40, help="examples per (task, split); 0 = all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fewshot", type=int, default=0, help="also run the base model with N solved examples in the prompt")
    ap.add_argument("--train", default=None, help="train.jsonl to draw the few-shot examples from")
    a = ap.parse_args()

    rows = sample(read_jsonl(a.test), a.per_bucket)
    print(f"{len(rows)} test examples")
    base_id = resolve_base(a.adapter, a.base)
    print(f"base model: {base_id}")
    t0 = time.time()
    model, tok = load(base_id, a.adapter)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    runs = [("base", False, None), ("adapted", True, None)]
    if a.fewshot:
        train_rows = read_jsonl(a.train or a.test.replace("test.jsonl", "train.jsonl"))
        shots = {t: fewshot_prefix(train_rows, t, a.fewshot) for t in {r["task"] for r in rows}}
        runs.insert(1, (f"base{a.fewshot}shot", False, shots))
    for name, on, shots in runs:
        preds = generate(model, tok, rows, a.batch_size, on, shots)
        with open(f"{a.out}_{name}.jsonl", "w", encoding="utf-8") as f:
            for r, text in preds:
                f.write(json.dumps({"id": r["id"], "task": r["task"], "split": r["split"], "lang": r["lang"],
                                    "completion": r["completion"], "output": text, "model": name}, ensure_ascii=False) + "\n")
        print(f"wrote {a.out}_{name}.jsonl ({time.time() - t0:.0f}s so far)")


if __name__ == "__main__":
    main()
