---
base_model: {{BASE_MODEL}}
library_name: peft
license: {{LICENSE}}
language:
{{LANGUAGES_YAML}}
tags:
{{TAGS_YAML}}
datasets:
- {{HF_DATASET}}
pipeline_tag: text-generation
---

# {{MODEL_NAME}}

{{ONE_LINE}}

**Built for the Adaptive Intelligence Challenge by Adaption (HackCulture, Nov–Dec 2026)** with Adaptive Data (Forge, Blueprint, Expand, Invent, Visualize) and AutoScientist. Weights: LoRA adapter for `{{BASE_MODEL}}`.

## Results on the held-out benchmark

Benchmark: {{BENCH_NAME}}, {{TEST_N}} test examples frozen **before** any training (never uploaded to Adaption). `test_id` = same distribution as training; `test_ood` = message templates, towns or items never seen in training; `test_real` = real messages collected by hand. Scores use up to {{PER_BUCKET}} examples per task and split (fixed seed, same subset for both models). Same prompts and greedy decoding for both models; the base model is the same weights with the adapter switched off.

{{RESULTS_TABLE}}

AutoScientist's own head-to-head evaluation: adapted model won {{AUTOSCIENTIST_WINRATE}} against the base.

## What it does

{{TASKS_TABLE}}

Every request starts with `Task: <task_name>.` and the exact instruction used in training (see `prompts.md` in this repo). Reference data (product codes, conversion tables, registered-product labels, problem tables) is passed in a `Context:` block. The model is trained to use only that context, return JSON when asked, and return `null` / `unknown` instead of guessing.

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "{{BASE_MODEL}}"
ADAPTER = "{{HF_MODEL}}"
tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER).eval()

prompt = """{{EXAMPLE_PROMPT}}"""
msgs = [{"role": "user", "content": prompt}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=300, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

Serving with vLLM (OpenAI-compatible API):
```bash
vllm serve {{BASE_MODEL}} --enable-lora --lora-modules {{LORA_NAME}}={{HF_MODEL}} --max-lora-rank {{LORA_R}}
```

## Training

| Step | Tool | What was done |
| --- | --- | --- |
{{TRAINING_TABLE}}

LoRA configuration (from `adapter_config.json`): r = {{LORA_R}}, alpha = {{LORA_ALPHA}}, target modules = {{LORA_TARGETS}}.

## Reproduce the evaluation

```bash
git clone {{GITHUB_REPO}} && cd {{GITHUB_DIR}}
pip install -r requirements.txt
python -m src.eval.run_eval --test data/build/{{PROJECT}}/test.jsonl --adapter {{HF_MODEL}} --out results/{{PROJECT}} --per-bucket {{PER_BUCKET}} --fewshot 3
python -m src.eval.score --pred results/{{PROJECT}}_base.jsonl results/{{PROJECT}}_base3shot.jsonl results/{{PROJECT}}_adapted.jsonl --out results/{{PROJECT}}
```
A free Kaggle T4 GPU runs it in about {{EVAL_TIME}}. Notebook: {{KAGGLE_NOTEBOOK}}.

## Limitations and safety

{{LIMITATIONS}}

## Data and licences

Dataset: [{{HF_DATASET}}](https://huggingface.co/datasets/{{HF_DATASET}}) (also on Kaggle: {{KAGGLE_DATASET}}). Sources and their licences are listed in the dataset card. Phone numbers in the data are synthetic.

## Author

{{AUTHOR}} — Douala, Cameroon. Project home: {{PROJECT_URL}}.
