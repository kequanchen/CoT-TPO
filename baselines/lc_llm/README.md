# LC-LLM (Adapted)

This directory provides an independently written, paper-based reconstruction of
LC-LLM for the post-crash lane-changing data used by CoT-TP. LC-LLM reformulates
lane-change prediction as supervised language modeling: one fine-tuned LLM
jointly generates reasoning, a lane-change intention, and future coordinates.

No author-released LC-LLM code, machine-readable prompt, preprocessing program,
or model adapter was available to this project when this baseline was prepared.
This release must therefore be reported as **LC-LLM (adapted)**, not as an exact
reproduction or an execution of the authors' code.

## Why This Baseline Is Included

LC-LLM represents a different way of connecting language reasoning to motion
prediction. It directly generates the future coordinates as language tokens,
whereas CoT-TP converts LLM reasoning into a predefined numerical behavioral
interface and conditions a separate trajectory generator. The comparison helps
separate the proposed structured-conditioning design from a recent direct,
joint reasoning-intention-trajectory formulation.

## Paper-Based Reconstruction

The implementation retains the main elements stated by Peng et al.:

- the Figure 3 system/user prompt organization;
- a target-vehicle coordinate system;
- map, target-vehicle, history, and surrounding-vehicle descriptions;
- a joint `Thought`, `Intention`, and `Trajectory` answer;
- Llama-2-13B-chat with answer-only causal language-model loss; and
- supervised LoRA with rank 64 and alpha 16.

The prompt structure and adapted task wording are attributed to Figure 3 of the
open-access article under CC BY 4.0. See [THIRD_PARTY.md](THIRD_PARTY.md) for the
source, license, and modification notice.

## Domain Adaptation

| Item | Published LC-LLM | This post-crash adaptation |
| --- | --- | --- |
| Dataset | highD | Private post-crash LC samples |
| Observed history | 2 s | 1 s, 10 points at 10 Hz |
| Surrounding vehicles | nearest vehicles in eight directions | TL, TF, TFF, OL, OF, and OFF roles |
| Requested future | 4 points over 4 s (Figure 3) | 50 points over 5 s at 10 Hz |
| Coordinate serialization | coordinate sequence shown in Figure 3 | strict JSON-style `[x, y]` pairs for auditable parsing |
| Road description | known lane count and left/middle/right position | included only when explicitly configured |
| Coordinate frame | current target position, heading-aligned | retained; local x is forward and local y is left |
| Output tasks | CoT, three-class intention, coordinates | retained, with the denser trajectory schema |
| Reported trajectory metrics | lateral/longitudinal RMSE | top-1 ADE/FDE at 1-5 s for manuscript consistency |

The public configuration leaves `road.num_lanes` null and
`road.lane_id_to_position` empty. Lane IDs in the private files are not assumed
to encode leftmost, middle, or rightmost position. Fill these fields only if the
corresponding road facts are verified.

The default `phase_left_event` label mode matches the current private
preprocessing, which retains left-change events. Statuses 0 and 1 supervise the
left-change class and status 2 supervises keep lane. This supervised phase is
not exposed in the inference prompt. A displacement-based three-class option is
provided for a future dataset containing left, right, and keep-lane outcomes.
Consequently, intention metrics on the current data cover only keep-lane and
left-lane-change samples; there is no right-lane-change ground-truth class.
This limitation does not alter trajectory ADE/FDE.

## What Is Released

The directory includes:

- a strict MATLAB adapter and target-centered coordinate transform;
- the reconstructed and adapted Figure 3 prompt builder;
- deterministic train/validation CoT and intention labels;
- supervised and prompt-only JSONL builders;
- strict joint-output parsing for exactly 50 coordinate points;
- LoRA training and restartable batch inference entry points; and
- exact-ID evaluation with intention accuracy, macro-F1, and 1-5 s ADE/FDE.

It does **not** include real data, generated prompts, model responses, Llama 2
weights, LoRA adapters, API credentials, private paths, or reported results.
The expected private input is documented with synthetic shapes in
[docs/SYNTHETIC_DATA_SCHEMA.md](docs/SYNTHETIC_DATA_SCHEMA.md).

## Requirements

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r baselines/lc_llm/requirements.txt
```

Llama-2-13B-chat is not downloaded automatically. Obtain lawful access under
the Meta Llama 2 Community License and set `model.base_model` to the approved
local path or model identifier. The default eight-bit configuration requires a
compatible CUDA environment and bitsandbytes installation. Actual memory needs
depend on sequence length and software versions; reduce the per-device batch
size or use the four-bit option only if that change is disclosed in the
experimental setup.

## Configuration

Run the following commands from `baselines/lc_llm`:

```bash
cd baselines/lc_llm
cp configs/post_crash_lc.example.json configs/post_crash_lc.local.json
```

For PowerShell:

```powershell
Set-Location baselines/lc_llm
Copy-Item configs/post_crash_lc.example.json configs/post_crash_lc.local.json
```

Edit only the ignored local copy. At minimum, replace:

```json
{
  "data": {
    "train_mat": "<PATH_TO_TRAIN_MAT>",
    "validation_mat": "<PATH_TO_VALIDATION_MAT>",
    "test_mat": "<PATH_TO_TEST_MAT>"
  },
  "model": {
    "base_model": "<PATH_OR_HF_ID_TO_LLAMA_2_13B_CHAT_HF>"
  }
}
```

Do not put real paths into the public example. The default generated artifacts
are written below the ignored `outputs/` directory.

## End-to-End Commands

### 1. Prepare Supervised Training Records

```bash
python scripts/prepare_dataset.py \
  --config configs/post_crash_lc.local.json \
  --split train \
  --mode supervised
```

The training JSONL contains the observed prompt and its supervised reasoning,
intention, and future-coordinate answer. It must be generated locally and must
not be committed.

### 2. Prepare Supervised Validation Records

```bash
python scripts/prepare_dataset.py \
  --config configs/post_crash_lc.local.json \
  --split validation \
  --mode supervised
```

The validation JSONL has the same supervised schema as training but must come
from disjoint crash episodes. It is evaluated during fine-tuning and is the
only split used to choose the best LoRA checkpoint.

### 3. Prepare Prompt-Only Test Records

```bash
python scripts/prepare_dataset.py \
  --config configs/post_crash_lc.local.json \
  --split test \
  --mode inference
```

The test builder calls the adapter with `include_future=False`. The output has
no answer, intention, trajectory label, or future coordinates. Inspect a small
local run first by adding `--max-samples 10` and, when rebuilding an existing
file, `--overwrite`.

### 4. Fine-Tune the LoRA Adapter

```bash
python scripts/train_lora.py \
  --config configs/post_crash_lc.local.json
```

Training requires both `paths.train_jsonl` and `paths.validation_jsonl`. Every
validation pass is paired with a saved checkpoint, `eval_loss` is minimized,
and the LoRA artifacts are copied directly from the winning checkpoint to the
stable `paths.adapter_dir`. Missing checkpoint weights cause training to fail
closed. The training manifest records the selected checkpoint and validation
loss. The training program never opens the test JSONL or test MATLAB file.

For an installation smoke test, use `--max-samples` and a private configuration
with smaller equal `training.eval_steps` and `training.save_steps`. At least one
validation/save step must occur. A smoke test is not a paper result.

### 5. Generate Test Predictions

```bash
python scripts/predict.py \
  --config configs/post_crash_lc.local.json
```

Prediction is restartable by default. Each attempted sample is written once by
`sample_id`. Successfully parsed lines contain top-level `intention` and
`trajectory` fields; generation and parsing failures remain explicit. To retry
failed IDs, move or remove the local prediction file and start a fresh run.

### 6. Evaluate Exact Test Coverage

```bash
python scripts/evaluate.py \
  --config configs/post_crash_lc.local.json
```

The evaluator joins predictions and labels by `sample_id`, requires one valid
prediction for every test sample, rejects extra or duplicate IDs, and evaluates
the single generated trajectory. It reports intention accuracy and macro-F1,
plus top-1 ADE and FDE at 1, 2, 3, 4, and 5 seconds. A parse failure is not
silently removed. `--allow-partial` exists only to diagnose an incomplete run;
partial metrics must not be reported as the baseline result.

## Leakage Controls

1. The prompt builder accepts `ObservationSample`, which has no future member.
2. Test JSONL construction uses `include_future=False` and emits no supervised
   answer fields.
3. `lane_status`, `time_since_crossing`, ground-truth intention, and future CoT
   labels are excluded from the inference prompt.
4. Training labels and CoT rules are called only when constructing supervised
   train or validation records.
5. Training checks that train and validation have disjoint `sample_id` and
   `event_id` sets, and selects the adapter only by validation loss.
6. The training program does not read any test path. Prediction reads only the
   prompt-only test JSONL after the adapter has been fixed.
7. Evaluation loads the test future only after a prediction JSONL already
   exists; it never invokes generation.
8. Predictions are matched by stable IDs, never by file order.

## Reproducibility and Scope

The paper reports learning rate `5e-4`, batch size 8, two epochs, LoRA rank 64,
LoRA alpha 16, gradient accumulation 8, warmup 600, and eight-bit loading. These
values initialize the example configuration. Details not recoverable from the
paper, including the authors' complete preprocessing code, exact prompt file,
random sample list, and adapter weights, cannot be reproduced exactly. The
included label rules and implementation choices are documented and tested so
the adaptation itself is auditable. The public workflow requires separate
train, validation, and test inputs. `training.eval_steps` and
`training.save_steps` must be equal so every measured validation loss belongs
to a selectable checkpoint. `scenario_id` must be a globally stable crash
episode identifier shared by all sliding windows from that episode; do not
renumber it independently inside each split. Form the 35/5/10 crash-episode
partition before constructing sliding windows.

## Tests

All tests use synthetic arrays, fake tokenizers/models, and temporary files:

```bash
python -m unittest discover -s baselines/lc_llm/tests -p "test_*.py" -v
```

They require neither the private data nor Llama 2 weights.

## Reporting

Use **LC-LLM (adapted)** in tables and figures. Suggested wording is provided in
[CITATION.md](CITATION.md). The comparison should be described as a domain
adaptation that follows the published formulation, not as a claim of numerical
reproduction on highD.
