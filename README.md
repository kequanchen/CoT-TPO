# CoT-TP Public Code

This repository contains the paper-aligned implementation of the CoT-TP
pipeline. The released code covers the proposed method: LLM teacher reasoning,
strategy-vector parsing, MLP student distillation, and FiLM-conditioned
trajectory prediction.

## Repository Structure

```text
.
├── scripts/
│   ├── generate_llm_teacher.py
│   ├── parse_strategy_vectors.py
│   ├── train_strategy_student_mlp.py
│   └── train_cot_tp_film.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated data, model checkpoints, and outputs are intentionally excluded from
the repository. Place local datasets and outputs under ignored folders such as
`data/`, `doc/`, `outputs/`, or `checkpoints/`.

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Generate LLM Teacher Responses

Generate prompts only:

```bash
python scripts/generate_llm_teacher.py \
  --data-path data/test_dataset.mat \
  --data-key test_data \
  --sample-idx 10 \
  --vehicle-length VEHICLE_LENGTH_METERS \
  --vehicle-width VEHICLE_WIDTH_METERS \
  --ego-x-col EGO_X_COL \
  --ego-y-col EGO_Y_COL \
  --ego-lon-v-col EGO_LON_V_COL \
  --ego-lat-v-col EGO_LAT_V_COL \
  --ego-acc-col EGO_ACC_COL \
  --neighbor-x-col NEIGHBOR_X_COL \
  --neighbor-y-col NEIGHBOR_Y_COL \
  --neighbor-lon-v-col NEIGHBOR_LON_V_COL \
  --neighbor-lat-v-col NEIGHBOR_LAT_V_COL \
  --neighbor-acc-col NEIGHBOR_ACC_COL \
  --output-dir outputs/llm_cot \
  --skip-llm
```

Call an OpenAI-compatible LLM endpoint:

```bash
set LLM_API_KEY=your_api_key
set LLM_BASE_URL=your_openai_compatible_base_url

python scripts/generate_llm_teacher.py \
  --data-path data/test_dataset.mat \
  --data-key test_data \
  --sample-idx 10 \
  --vehicle-length VEHICLE_LENGTH_METERS \
  --vehicle-width VEHICLE_WIDTH_METERS \
  --ego-x-col EGO_X_COL \
  --ego-y-col EGO_Y_COL \
  --ego-lon-v-col EGO_LON_V_COL \
  --ego-lat-v-col EGO_LAT_V_COL \
  --ego-acc-col EGO_ACC_COL \
  --neighbor-x-col NEIGHBOR_X_COL \
  --neighbor-y-col NEIGHBOR_Y_COL \
  --neighbor-lon-v-col NEIGHBOR_LON_V_COL \
  --neighbor-lat-v-col NEIGHBOR_LAT_V_COL \
  --neighbor-acc-col NEIGHBOR_ACC_COL \
  --output-dir outputs/llm_cot
```

### 2. Parse Strategy Vectors

```bash
python scripts/parse_strategy_vectors.py \
  --responses-dir outputs/llm_cot \
  --prompts-dir outputs/llm_cot \
  --out-dir doc/traininput
```

The parser writes `ids.npy`, `c.npy`, `vocab.json`, `meta.csv`,
`records.jsonl`, `summary.json`, and optionally `errors.log`.

### 3. Distill The MLP Strategy Student

```bash
python scripts/train_strategy_student_mlp.py \
  --train-mat data/train_dataset.mat \
  --test-mat data/test_dataset.mat \
  --train-vector-dir doc/traininput \
  --test-vector-dir doc/testinput \
  --out-dir outputs/student_mlp \
  --ego-x-col EGO_X_COL \
  --ego-y-col EGO_Y_COL \
  --ego-lon-v-col EGO_LON_V_COL \
  --ego-lat-v-col EGO_LAT_V_COL \
  --neighbor-x-col NEIGHBOR_X_COL \
  --neighbor-y-col NEIGHBOR_Y_COL \
  --neighbor-lon-v-col NEIGHBOR_LON_V_COL \
  --neighbor-lat-v-col NEIGHBOR_LAT_V_COL
```

### 4. Train The CoT-TP FiLM Predictor

```bash
python scripts/train_cot_tp_film.py \
  --train-mat data/train_dataset.mat \
  --test-mat data/test_dataset.mat \
  --train-vector-dir doc/traininput \
  --test-vector-dir doc/testinput \
  --student-ckpt outputs/student_mlp/strategy_student_distill.pth \
  --out-dir outputs/cot_tp_film \
  --require-student-ckpt
```

Run any script with `--help` to inspect all options.

## GitHub Upload

From this folder:

```bash
git init
git add .
git commit -m "Release CoT-TP core code"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

Before pushing, run:

```bash
rg -n "sk-|E:\\|D:\\|[\p{Han}]|dashscope|compatible-mode" .
python -m py_compile scripts/generate_llm_teacher.py scripts/parse_strategy_vectors.py scripts/train_strategy_student_mlp.py scripts/train_cot_tp_film.py
```
