"""Gradio demo for the NEMA-Agro and REVALO-Waste adapters (Hugging Face Space, ZeroGPU or any GPU machine).

Space setup (see playbook section 11):
  - SDK: Gradio, hardware: ZeroGPU (or a paid GPU), files: app.py, requirements.txt, tasks_agro.json, tasks_waste.json
  - Space secrets / variables: BASE_MODEL, AGRO_ADAPTER, WASTE_ADAPTER (optional), HF_TOKEN (if repos are private)
"""
import json
import os
import re
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import spaces  # available on Hugging Face Spaces (ZeroGPU)
    gpu = spaces.GPU(duration=90)
except ImportError:  # local machine
    def gpu(f):
        return f

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-9B")
ADAPTERS = {"NEMA-Agro": os.environ.get("AGRO_ADAPTER", ""), "REVALO-Waste": os.environ.get("WASTE_ADAPTER", "")}
ADAPTERS = {k: v for k, v in ADAPTERS.items() if v}
HERE = Path(__file__).parent
TASKS = {"NEMA-Agro": json.loads((HERE / "tasks_agro.json").read_text(encoding="utf-8")),
         "REVALO-Waste": json.loads((HERE / "tasks_waste.json").read_text(encoding="utf-8"))}
TOKEN = os.environ.get("HF_TOKEN")

USE_4BIT = os.environ.get("LOAD_4BIT", "0") == "1"  # set to 1 on a 16 GB GPU (Kaggle T4)
kw = dict(token=TOKEN)
if USE_4BIT:
    kw.update(device_map="auto", quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
else:
    kw.update(torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(next(iter(ADAPTERS.values())), token=TOKEN)
try:
    model = AutoModelForCausalLM.from_pretrained(BASE, **kw)
except (ValueError, KeyError):  # some new multimodal bases only load with the image-text class
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(BASE, **kw)
names = list(ADAPTERS)
model = PeftModel.from_pretrained(model, ADAPTERS[names[0]], adapter_name=names[0], token=TOKEN)
for n in names[1:]:
    model.load_adapter(ADAPTERS[n], adapter_name=n, token=TOKEN)
if not USE_4BIT:
    model.to("cuda" if torch.cuda.is_available() or "spaces" in globals() else "cpu")
model.eval()


@gpu
def run(project, task, user_input, context, compare):
    t = TASKS[project][task]
    prompt = f"{t['instruction']}\n{user_input}"
    if context.strip():
        prompt = f"Context:\n{context.strip()}\n\n{prompt}"
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)

    def gen():
        out = model.generate(**ids, max_new_tokens=700, do_sample=False, pad_token_id=tok.eos_token_id)
        return re.sub(r"<think>.*?</think>", "", tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S).strip()

    model.set_adapter(project)
    adapted = gen()
    base = ""
    if compare:
        with model.disable_adapter():
            base = gen()
    return adapted, base


def on_task(project, task):
    t = TASKS[project][task]
    return t["sample_input"], t["context"]


def on_project(project):
    first = next(iter(TASKS[project]))
    return gr.update(choices=list(TASKS[project]), value=first), *on_task(project, first)


with gr.Blocks(title="NEMA-Agro / REVALO-Waste") as demo:
    gr.Markdown("# NEMA-Agro & REVALO-Waste\nSmall open models for Cameroonian agriculture and waste, trained with Adaption. "
                "Pick a task, edit the message (French, English or Pidgin), and compare with the base model.")
    with gr.Row():
        project = gr.Dropdown(names, value=names[0], label="Model")
        task = gr.Dropdown(list(TASKS[names[0]]), value=next(iter(TASKS[names[0]])), label="Task")
    user_input = gr.Textbox(lines=4, label="Input (edit me)")
    context = gr.Textbox(lines=6, label="Context passed to the model (reference tables, labels)")
    compare = gr.Checkbox(value=True, label="Also show the base model's answer")
    btn = gr.Button("Run", variant="primary")
    with gr.Row():
        out_a = gr.Code(label="Adapted model", language="json")
        out_b = gr.Code(label="Base model", language="json")
    project.change(on_project, project, [task, user_input, context])
    task.change(on_task, [project, task], [user_input, context])
    demo.load(on_task, [project, task], [user_input, context])
    btn.click(run, [project, task, user_input, context, compare], [out_a, out_b])

if __name__ == "__main__":
    demo.launch(share=os.environ.get("GRADIO_SHARE", "0") == "1")
