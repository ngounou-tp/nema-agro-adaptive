#!/usr/bin/env bash
# Regenerate every synthetic file, rebuild both datasets, run the self-check. About 15 seconds.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.generators.gen_marketplace --n 1500
python -m src.generators.gen_agro_calc --n 900
python -m src.generators.gen_agro_knowledge --n 900
python -m src.generators.gen_waste --n 1500
python -m src.build_dataset --project agro > /dev/null
python -m src.build_dataset --project waste > /dev/null
python -c "import json
for p in ('agro', 'waste'):
    s = json.load(open(f'data/build/{p}/stats.json'))
    print(f\"{p}: train {s['train']}, test {s['test']}, leaks {s['leaks']}\")"
python scripts/export_demo_tasks.py --project agro
python scripts/export_demo_tasks.py --project waste
python scripts/smoke_test.py | tail -1
