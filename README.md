# CoT-TP Public Code

This repository contains the paper-aligned implementation of CoT-TP. The core
pipeline covers LLM teacher reasoning, strategy-vector parsing, MLP student
distillation, and FiLM-conditioned trajectory prediction. It also contains
independently implemented, post-crash adaptations of the **LLC-PC** and
**LC-LLM** baselines used in the study.

## Repository Structure

```text
.
|-- scripts/
|   |-- generate_llm_teacher.py
|   |-- parse_strategy_vectors.py
|   |-- train_strategy_student_mlp.py
|   `-- train_cot_tp_film.py
|-- baselines/
|   |-- llc_pc/
|   |   |-- configs/
|   |   |-- docs/
|   |   |-- llc_pc/
|   |   |-- scripts/
|   |   |-- tests/
|   |   |-- README.md
|   |   |-- CITATION.md
|   |   `-- THIRD_PARTY.md
|   `-- lc_llm/
|       |-- configs/
|       |-- docs/
|       |-- lc_llm/
|       |-- scripts/
|       |-- tests/
|       |-- README.md
|       |-- CITATION.md
|       `-- THIRD_PARTY.md
|-- requirements.txt
|-- .gitignore
`-- README.md
```

Generated data, model checkpoints, LLM responses, rendered context maps, and
outputs are intentionally excluded. Place private datasets and generated
artifacts only under ignored directories such as `data/`, `doc/`, `outputs/`,
or `checkpoints/`.

## Installation

```bash
pip install -r requirements.txt
```

## CoT-TP Pipeline

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
export LLM_API_KEY="YOUR_API_KEY"
export LLM_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"

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

On PowerShell, set the variables with
`$env:LLM_API_KEY="YOUR_API_KEY"` and
`$env:LLM_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"`.

### 2. Parse Strategy Vectors

```bash
python scripts/parse_strategy_vectors.py \
  --responses-dir outputs/llm_cot \
  --prompts-dir outputs/llm_cot \
  --out-dir doc/traininput
```

The parser writes `ids.npy`, `c.npy`, `vocab.json`, `meta.csv`,
`records.jsonl`, `summary.json`, and optionally `errors.log`.

### 3. Distill the MLP Strategy Student

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

### 4. Train the CoT-TP FiLM Predictor

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

## LLC-PC Baseline

LLC-PC is an independent, domain-adapted implementation following the semantic
context conditioning design of Zheng et al., *Large Language Models Powered
Context-aware Motion Prediction*. It converts structured LLM output into a
17-dimensional context representation. The context and training-derived
intention anchors are projected separately and combined into the queries of a
compact MTR-style decoder. It is not an exact reproduction of the original WOMD
implementation and does not copy the original source code or prompt.

The public repository contains no crash trajectories, generated LLM responses,
or model weights. See [the LLC-PC guide](baselines/llc_pc/README.md) for the
expected local data schema, leakage controls, configuration, and commands.

## LC-LLM Baseline

LC-LLM (adapted) is an independent, paper-based reconstruction of the direct
joint reasoning, lane-change intention, and trajectory-token generation method
described by Peng et al. Its Figure 3 prompt organization and reported LoRA
settings are retained where specified, while the inputs and output sampling are
adapted to the post-crash dataset. It is not presented as author-released code
or an exact numerical reproduction. See [the LC-LLM guide](baselines/lc_llm/README.md)
for the adaptation table, private-data interface, training and evaluation
commands, and attribution notice.

## Release Checks

Before committing public changes, inspect tracked and untracked files and scan
for accidental credentials or machine-specific paths:

```bash
git status --short
rg -l --hidden -g '!.git/**' -g '!*.md' -g '!*.example.json' "sk-[A-Za-z0-9_-]{20,}|dashscope|compatible-mode" .
rg -l -F --hidden -g '!.git/**' -g '!*.md' 'C:\' .
rg -l -F --hidden -g '!.git/**' -g '!*.md' 'D:\' .
rg -l -F --hidden -g '!.git/**' -g '!*.md' 'E:\' .
python -m py_compile scripts/generate_llm_teacher.py scripts/parse_strategy_vectors.py scripts/train_strategy_student_mlp.py scripts/train_cot_tp_film.py
```

Never commit datasets, API credentials, generated responses, checkpoints, or
absolute local paths.
