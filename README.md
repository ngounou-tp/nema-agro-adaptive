# NEMA-Agro + REVALO-Waste — Adaptive Intelligence Challenge kit

Everything runs in the cloud: data adaptation and training on the Adaption app (adaptionlabs.ai/app),
data building, benchmark and publishing in two free Kaggle notebooks. Nothing to install on your laptop.
Follow the playbook document step by step; this README is only the map.

## Where each step runs
| Step | Where | File |
| --- | --- | --- |
| Edit seeds (real messages, MCQs, problem table) | GitHub web editor | `data/seeds/*.csv` |
| Build the training files + frozen test set | Kaggle notebook 1 (CPU) | `notebooks/kaggle_1_build_data.ipynb` |
| Improve, Invent, Expand, Visualize, train | Adaption app | upload `agro_adaption_seed.csv` / `waste_adaption_seed.csv` |
| Benchmark, cards, publish to Hugging Face + Kaggle | Kaggle notebook 2 (GPU) | `notebooks/kaggle_2_eval_and_release.ipynb` |
| Demo / serving / NEMA integration | Hugging Face Space or GPU server; Laravel + Flutter | `demo/`, `serve/`, `integration/` |

## Folder map
| Path | What it is |
| --- | --- |
| `data/lexicon/agro_lexicon.json` | products, units, towns, quality words in fr / en / pcm (edit and extend) |
| `data/seeds/symptom_table.csv` | pest & disease table used as context; fill `source_url` and `verified_by` for each row |
| `data/seeds/waste_items.csv` | waste items with category and route |
| `data/seeds/manual_listings.csv` | real marketplace messages you label by hand, one row each (half go to the test set) |
| `data/seeds/advice_mcq.csv`, `hygiene_mcq.csv` | your multiple-choice questions written from documents (test only) |
| `src/generators/` | synthetic task generators with exact labels |
| `src/build_dataset.py` | dedupe, freeze test set (random + held-out templates/towns/items + real messages), export for Adaption |
| `src/adaption_client.py` | optional REST client for Adaptive Data (the app does the same) |
| `src/merge_adapted.py` | optional check of downloaded adapted files |
| `src/eval/run_eval.py`, `src/eval/score.py` | benchmark generation (GPU) and scoring |
| `src/make_cards.py`, `src/publish.py` | model + dataset cards, upload to Hugging Face and Kaggle |
| `notebooks/` | the two Kaggle notebooks (rebuild them with `scripts/make_notebooks.py`) |
| `demo/`, `serve/`, `integration/` | Gradio Space, vLLM server, Laravel + Flutter client |
| `configs/` | Blueprint rules, Adaptive Data run settings, release settings |

## What was tested
Generators, dataset builder (stable test split, leakage check), scorer, card generator, Adaption client (dry-run),
notebook syntax, PHP syntax. `run_eval.py`, `demo/app.py`, `publish.py`, notebook 2 and `serve/` need a GPU or your
accounts and were only syntax-checked: run them first on a few examples.

