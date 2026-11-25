---
license: {{DATA_LICENSE}}
language:
{{LANGUAGES_YAML}}
task_categories:
- text-generation
- question-answering
tags:
{{TAGS_YAML}}
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: test
    path: test.jsonl
---

# {{DATASET_NAME}}

{{ONE_LINE}}

Made for the Adaptive Intelligence Challenge by Adaption. Used to train [{{HF_MODEL}}](https://huggingface.co/{{HF_MODEL}}).

## Contents

| Split | Rows | Notes |
| --- | --- | --- |
| train | {{TRAIN_N}} | after Adaptive Data in the Adaption app (deduplication, Blueprint, Translate, Invent rows), checked in the app's View tab |
| test | {{TEST_N}} | frozen before training; `split` = test_id, test_ood or test_real |

Rows per task (train): {{TRAIN_BY_TASK}}
Rows per language (train): {{TRAIN_BY_LANG}}

## Fields

| Field | Meaning |
| --- | --- |
| prompt | train file: exact text given to the model, `Context:` block included; test file: the task text only |
| context | test file only: the reference block; the model sees `Context:\n<context>\n\n<prompt>` (empty context = prompt alone) |
| completion | gold answer: JSON string for structured tasks, letter for MCQ, text for advice |
| task | task name |
| lang | fr, en or pcm (Cameroonian Pidgin, written) |
| split | test file only: test_id, test_ood or test_real |

## How it was made

{{HOW_MADE}}

## Sources and licences

{{SOURCES_TABLE}}

## Personal data

No real personal data. Names are common Cameroonian names used at random; phone numbers are synthetic; real messages collected for the test set were anonymised with the senders' consent.

## Limitations

{{LIMITATIONS}}
