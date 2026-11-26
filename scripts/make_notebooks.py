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


BUILD_SOURCES = """\
## Data sources

| File | What it is | Used for |
| --- | --- | --- |
| `data/lexicon/agro_lexicon.json` | 42 products with French, English, Pidgin and local names; units with kg weights; towns in all 10 regions; quality words | every synthetic agro message. Price ranges there are illustrative, not market facts |
| `data/seeds/wfp_food_prices_cmr.csv` | WFP / HDX *Cameroon - Food Prices* (real market prices, XAF) | the reference prices in `agro_unit_price` |
| `data/seeds/wfp_markets_cmr.csv` | WFP / HDX *Cameroon - Markets* (market names, regions, coordinates) | reference only |
| `data/seeds/manual_listings.csv` | your real WhatsApp/Facebook messages, anonymised (target 20 per language) | half go to the `test_real` split |
| `data/seeds/advice_mcq.csv`, `data/seeds/hygiene_mcq.csv` | your MCQs from the fiches (target 40-80) | test set only, never training |
| `data/seeds/symptom_table.csv` | crop symptoms and actions | agro knowledge examples |
| `data/lexicon/pcm_review_sheet.csv` | every Pidgin string, for one native speaker to check | not used by the build; copy corrections back into the lexicon |

If the two WFP files are not in your repo, the next cell downloads them from HDX. Commit them to GitHub so every run uses the same prices: new HDX data changes the reference prices in the examples.
"""

BUILD_CHECK = """\
import csv, json, urllib.request
from pathlib import Path

SEEDS = Path('data/seeds')
HDX = 'https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-cameroon'

# 1. WFP / HDX Cameroon prices (real reference prices for agro_unit_price)
missing = [n for n in ('wfp_food_prices_cmr.csv', 'wfp_markets_cmr.csv') if not (SEEDS / n).exists()]
if missing:
    print('not in the repo, downloading from HDX:', missing)
    try:
        for r in json.load(urllib.request.urlopen(HDX, timeout=60))['result']['resources']:
            name = r['url'].rsplit('/', 1)[-1]
            if name in missing:
                urllib.request.urlretrieve(r['url'], SEEDS / name)
    except OSError as e:
        raise RuntimeError(f'{missing} are not in the kit and HDX is unreachable (Internet off?). '
                           'Commit them to data/seeds/ in GitHub and update the Kaggle dataset.') from e
with open(SEEDS / 'wfp_food_prices_cmr.csv', encoding='utf-8') as f:
    wfp = [r for r in csv.DictReader(f) if not r['date'].startswith('#')]
print(f"WFP prices: {len(wfp)} rows, {min(r['date'] for r in wfp)} to {max(r['date'] for r in wfp)}, "
      f"{len({r['market'] for r in wfp})} markets")
from src.generators.gen_agro_calc import load_wfp_refs
covered = sorted({pid for pid, _ in load_wfp_refs(SEEDS / 'wfp_food_prices_cmr.csv')})
print('lexicon products with real WFP reference prices:', ', '.join(covered))

# 2. Lexicon
lex = json.load(open('data/lexicon/agro_lexicon.json', encoding='utf-8'))
print(f"lexicon: {len(lex['products'])} products, {len(lex['units'])} units, "
      f"{sum(map(len, lex['places'].values()))} towns")
review = lex['_pcm_native_speaker_review']
print('Pidgin native-speaker review:', review['status'], '| sheet:', review['review_sheet'])
if not review['completed']:
    print('  WARNING: the Pidgin forms are not checked by a native speaker yet; do not say they are.')

# 3. Your own data
def filled(path, col):
    with open(path, encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if (r.get(col) or '').strip() and not r[col].startswith('EXAMPLE')]

msgs = filled(SEEDS / 'manual_listings.csv', 'message')
print('real messages per language:', {l: sum(r['lang'].strip() == l for r in msgs) for l in ('fr', 'en', 'pcm')},
      '(target 20 each; half go to test_real)')
bad = [r['message'][:50] for r in msgs
       if r['product'].strip() not in {p['id'] for p in lex['products']} or r['unit'].strip() not in lex['units']]
assert not bad, f'manual_listings.csv uses product or unit codes missing from the lexicon: {bad}'
for name in ('advice_mcq.csv', 'hygiene_mcq.csv'):
    print(f'{name}: {len(filled(SEEDS / name, "question"))} MCQs (target 40-80, all go to the test set)')
"""

BUILD_WFP_USED = """\
with open('data/build/gen_agro_calc.jsonl', encoding='utf-8') as f:
    n_wfp = sum('source WFP/HDX' in line for line in f)
print('agro_unit_price examples with real WFP reference prices:', n_wfp)
assert n_wfp, 'no WFP reference prices were used: check the WFP cell above'
"""

GET_KIT = """\
# Gets the kit code. Works with Internet OFF if the GitHub-linked Kaggle dataset is attached (Add Input).
import glob, os, shutil, subprocess
from pathlib import Path

REPO = 'https://github.com/ngounou-tp/nema-agro-adaptive.git'  # used only when Internet is on
KIT = Path('/kaggle/working/kit')
if KIT.exists():
    shutil.rmtree(KIT)

found = sorted(glob.glob('/kaggle/input/**/scripts/build_all.sh', recursive=True))
if found:
    src = Path(found[0]).parents[1]
    shutil.copytree(src, KIT)
    subprocess.run(['chmod', '-R', 'u+w', str(KIT)], check=True)  # /kaggle/input is read-only
    print('kit copied from the attached Kaggle dataset:', src)
else:
    r = subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO, str(KIT)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('Could not get the kit: no attached dataset contains scripts/build_all.sh and git clone failed '
                           '(Internet is off). Click Add Input and attach the dataset made from the GitHub repo.\\n' + r.stderr)
    print('kit cloned from', REPO)

os.chdir(KIT)
print('working in', os.getcwd())
"""

BUILD = [
    md("# 1. Build the datasets (CPU, no GPU needed)\n\nRuns the kit's generators in the cloud and writes the files you upload to Adaption.\n\n"
       "**Settings:** Accelerator = None. Internet can stay **Off**: in the right panel click **Add Input** and attach the Kaggle "
       "dataset created from the GitHub repo (Datasets → New Dataset → GitHub link). With Internet On, the next cell clones "
       "the repo instead. Then *Run All* (about 2 minutes).\n\n"
       "Your own data goes in the GitHub repo first (edit in the GitHub web editor): `data/seeds/manual_listings.csv` "
       "(real messages), `data/seeds/advice_mcq.csv` and `data/seeds/hygiene_mcq.csv` (MCQs), `data/seeds/symptom_table.csv`. "
       "After a push, update the Kaggle dataset (its page → Update / New version) so the notebook sees the change."),
    md(BUILD_SOURCES),
    code(GET_KIT),
    md("## Check the data sources"),
    code(BUILD_CHECK),
    md("## Build"),
    code("!bash scripts/build_all.sh"),
    code(BUILD_WFP_USED),
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
         "!bash scripts/build_all.sh > /dev/null && echo 'datasets rebuilt'\n"
         "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code("# The rebuilt test set must be the one frozen by notebook 1 (committed to GitHub), or results are not comparable.\n"
         "import subprocess\n"
         "test_file = f'data/build/{PROJECT}/test.jsonl'\n"
         "if subprocess.run(['git', 'ls-files', '--error-unmatch', test_file], capture_output=True).returncode:\n"
         "    print(f'WARNING: {test_file} is not committed, so this run cannot prove it uses the same test set as notebook 1.')\n"
         "    print('Commit the test.jsonl from notebook 1 (see its last cell) and run again.')\n"
         "else:\n"
         "    changed = subprocess.run(['git', 'diff', '--quiet', '--', test_file]).returncode\n"
         "    assert not changed, (f'the rebuilt {test_file} differs from the committed one: the lexicon, generators or seeds '\n"
         "                         'changed after notebook 1. Run notebook 1 again and commit its test.jsonl.')\n"
         "    print('test set matches the committed frozen copy')"),
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
