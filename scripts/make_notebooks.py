"""Writes the two Kaggle notebooks in notebooks/. Run: python scripts/make_notebooks.py"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def save(name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


BUILD = [
    md("# 1. Build the datasets (CPU, no GPU needed)\n\nRuns the kit's generators in the cloud and writes the files you upload to Adaption.\n\n"
       "**Settings:** Accelerator = None, Internet = On. Edit `REPO` below, then *Run All* (about 1 minute).\n\n"
       "Your own data goes in the GitHub repo first (edit in the GitHub web editor): `data/seeds/manual_listings.csv` "
       "(real messages), `data/seeds/advice_mcq.csv` and `data/seeds/hygiene_mcq.csv` (MCQs), `data/seeds/symptom_table.csv`."),
    code("REPO = 'https://github.com/YOUR_GITHUB_USER/nema-agro-adaptive.git'  # your public kit repo\n"
         "!git clone -q $REPO kit\n%cd kit\n!bash scripts/build_all.sh"),
    code("import json, shutil, os\n"
         "os.makedirs('/kaggle/working/for_adaption', exist_ok=True)\n"
         "for p in ('agro', 'waste'):\n"
         "    shutil.copy(f'data/build/{p}/adaption_seed.csv', f'/kaggle/working/for_adaption/{p}_adaption_seed.csv')\n"
         "    shutil.copy(f'data/build/{p}/test.jsonl', f'/kaggle/working/for_adaption/{p}_test_DO_NOT_UPLOAD.jsonl')\n"
         "    s = json.load(open(f'data/build/{p}/stats.json'))\n"
         "    print(p, 'train', s['train'], 'test', s['test'], 'leaks', s['leaks'])\n"
         "print(os.listdir('/kaggle/working/for_adaption'))"),
    md("**Next:** open the **Output** panel (right side), download `agro_adaption_seed.csv` and `waste_adaption_seed.csv`, "
       "and upload them in the Adaption app (playbook section 7). Never upload the `*_test_DO_NOT_UPLOAD.jsonl` files.\n\n"
       "If you changed seeds, commit the rebuilt `data/build/agro/test.jsonl` to GitHub too (download it from `kit/data/build/agro/`)."),
]

EVAL = [
    md("# 2. Evaluate and release (GPU)\n\nRuns the benchmark (base vs base+3 examples vs adapted), writes the model and dataset cards, "
       "and publishes on Hugging Face and Kaggle. Run the parts in order; stop after part C to read the results first.\n\n"
       "**Settings:** Accelerator = **GPU T4 x2**, Internet = **On** (phone-verified Kaggle account).\n\n"
       "**Add-ons \u2192 Secrets** (tick each for this notebook): `HF_TOKEN` (Hugging Face **write** token), "
       "`ADAPTION_API_KEY` (only for part B), `KAGGLE_USERNAME` and `KAGGLE_KEY` (from your kaggle.json, for part E)."),
    md("## A. Settings"),
    code("REPO = 'https://github.com/YOUR_GITHUB_USER/nema-agro-adaptive.git'\n"
         "PROJECT = 'agro'                         # 'agro' or 'waste'\n"
         "HF_USER = 'YOUR_HF_USER'\n"
         "KAGGLE_USER = 'YOUR_KAGGLE_USER'\n"
         "ADAPTER = f'{HF_USER}/nema-agro-lora' if PROJECT == 'agro' else f'{HF_USER}/revalo-waste-lora'\n"
         "ADAPTION_RUN_ID = ''   # AutoScientist run id: fill it to let part B copy the weights to Hugging Face for you\n"
         "BASE = 'Qwen/Qwen3.5-9B'  # the exact base model you chose in AutoScientist (official Hugging Face id)\n"
         "PER_BUCKET = 40          # test examples per task and split; 25 if your GPU quota is short\n\n"
         "import os\nfrom kaggle_secrets import UserSecretsClient\nsecrets = UserSecretsClient()\n"
         "for name in ('HF_TOKEN', 'ADAPTION_API_KEY', 'KAGGLE_USERNAME', 'KAGGLE_KEY'):\n"
         "    try:\n        os.environ[name] = secrets.get_secret(name)\n    except Exception:\n        print('secret not set (fine if you skip the part that needs it):', name)"),
    code("!git clone -q $REPO kit\n%cd kit\n"
         "!pip install -q -U transformers peft bitsandbytes accelerate huggingface_hub kagglehub adaption\n"
         "!bash scripts/build_all.sh > /dev/null && echo 'datasets rebuilt (same frozen test set)'\n"
         "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    md("## B. Copy the trained weights from Adaption to Hugging Face (skip if already there)\n\n"
       "No-code alternative: in the Adaption app, open the trained model, click **Weights export** to download the archive, "
       "create an empty model repo on huggingface.co and upload the extracted files with **Add file \u2192 Upload files**."),
    code("import glob, tarfile\nfrom pathlib import Path\nfrom adaption import Adaption\nfrom huggingface_hub import HfApi\n\n"
         "assert ADAPTION_RUN_ID, 'set ADAPTION_RUN_ID in part A'\n"
         "client = Adaption()  # reads ADAPTION_API_KEY\n"
         "run = client.autoscientist.get(ADAPTION_RUN_ID)\n"
         "print('status =', run.status, '| download_available =', getattr(run, 'download_available', None))\n"
         "assert getattr(run, 'download_available', False), 'run not finished or no download yet: use the no-code path above'\n"
         "# streamed download of the best checkpoint (see docs: AutoScientist > Download the model)\n"
         "with client.autoscientist.with_streaming_response.download(ADAPTION_RUN_ID) as resp:\n"
         "    resp.stream_to_file('best-checkpoint.tgz')\n"
         "Path('best-checkpoint').mkdir(exist_ok=True)\n"
         "with tarfile.open('best-checkpoint.tgz') as t:\n    t.extractall('best-checkpoint')\n"
         "adapter_dir = str(Path(glob.glob('best-checkpoint/**/adapter_config.json', recursive=True)[0]).parent)\n"
         "print('adapter files:', os.listdir(adapter_dir))\n"
         "api = HfApi(token=os.environ['HF_TOKEN'])\n"
         "api.create_repo(ADAPTER, repo_type='model', exist_ok=True, private=True)\n"
         "api.upload_folder(folder_path=adapter_dir, repo_id=ADAPTER, commit_message='LoRA adapter from Adaption AutoScientist')\n"
         "print('uploaded to https://huggingface.co/' + ADAPTER, '(private until part E)')"),
    md("## C. Benchmark: base, base + 3 examples, adapted"),
    code("!python -m src.eval.run_eval --test data/build/$PROJECT/test.jsonl --adapter $ADAPTER --base $BASE "
         "--out results/$PROJECT --per-bucket $PER_BUCKET --batch-size 8 --fewshot 3\n"
         "!python -m src.eval.score --pred results/{PROJECT}_base.jsonl results/{PROJECT}_base3shot.jsonl results/{PROJECT}_adapted.jsonl --out results/{PROJECT}"),
    code("from IPython.display import Markdown\nMarkdown(open(f'results/{PROJECT}_results.md').read())"),
    md("Read 10 outputs per task in `results/<project>_adapted.jsonl` before trusting the numbers. Stop here if something looks wrong."),
    md("## D. Model and dataset cards\n\nFill the values below (from AutoScientist's results page and your Adaptive Data runs), then run."),
    code("import json\ncfg_path = f'configs/project_{PROJECT}.json'\ncfg = json.load(open(cfg_path))\n"
         "cfg.update({\n"
         "    'hf_user': HF_USER, 'kaggle_user': KAGGLE_USER, 'base_model': BASE,\n"
         "    'hf_model_repo': ADAPTER,\n"
         "    'hf_dataset_repo': f\"{HF_USER}/{'nema-agro' if PROJECT == 'agro' else 'revalo-waste'}-dataset\",\n"
         "    'kaggle_model_handle': f\"{KAGGLE_USER}/{'nema-agro' if PROJECT == 'agro' else 'revalo-waste'}/transformers/lora-v1\",\n"
         "    'kaggle_dataset_handle': f\"{KAGGLE_USER}/{'nema-agro' if PROJECT == 'agro' else 'revalo-waste'}-dataset\",\n"
         "    'github_repo': REPO.removesuffix('.git'),\n"
         "    'kaggle_notebook_url': 'PASTE the public URL of this notebook',\n"
         "    'eval_time': 'PASTE how long part C took',\n"
         "    'autoscientist_winrate': 'PASTE from AutoScientist > Training Winrates (e.g. 68%)',\n"
         "})\n"
         "# training_steps and sources: edit configs/project_<project>.json in GitHub, or change cfg['training_steps'] here\n"
         "json.dump(cfg, open(cfg_path, 'w'), ensure_ascii=False, indent=2)\n"
         "!python -m src.make_cards --project $PROJECT\n"
         "!grep -n 'REPLACE\\|PASTE' release/$PROJECT/model/README.md release/$PROJECT/dataset/README.md || echo 'cards complete'"),
    md("## E. Publish on Hugging Face and Kaggle\n\nRuns only when no `REPLACE` is left. Then make the Hugging Face repos public "
       "(repo \u2192 Settings \u2192 Change visibility) and finish the Kaggle pages in the browser (playbook section 10)."),
    code("!python -m src.publish hf-dataset    --project $PROJECT\n"
         "!python -m src.publish hf-model-card --project $PROJECT\n"
         "!python -m src.publish kaggle-model  --project $PROJECT\n"
         "!python -m src.publish kaggle-dataset --project $PROJECT"),
    code("# keep results and cards in the notebook Output (Save Version -> Save & Run All)\nimport shutil\n"
         "shutil.copytree('results', '/kaggle/working/results', dirs_exist_ok=True)\n"
         "shutil.copytree('release', '/kaggle/working/release', dirs_exist_ok=True)"),
    md("## F. Optional: live demo with a temporary public link\nRuns the Gradio demo in 4-bit on this GPU. The `*.gradio.live` link works while the cell runs."),
    code("!pip install -q gradio\n"
         "!cd demo && LOAD_4BIT=1 GRADIO_SHARE=1 BASE_MODEL={BASE} AGRO_ADAPTER={ADAPTER} python app.py"),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    save("kaggle_1_build_data.ipynb", BUILD)
    save("kaggle_2_eval_and_release.ipynb", EVAL)
    print("wrote notebooks/kaggle_1_build_data.ipynb and notebooks/kaggle_2_eval_and_release.ipynb")
