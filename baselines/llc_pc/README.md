# LLC-PC

LLC-PC is an independently implemented baseline that adapts the semantic
context conditioning workflow of Zheng et al., *Large Language Models Powered
Context-aware Motion Prediction*, to the post-crash lane-changing data format
used by CoT-TP.

This is not an exact reproduction of the original WOMD implementation. The
released code uses observed ego and six-neighbor histories, simplified local
lane geometry, an independently written task prompt, and an independently
implemented MTR-style predictor. No upstream prompt or source code is copied.

## Method Summary

```text
observed history + local context map + task prompt
                         |
                         v
                 structured LLM output
                         |
                         v
          8 action + 4 affordance + 5 scenario values
                         |
                         v
                 17-dimensional context
                         |
                         v
 training-only KNN over standardized observation tensors (K = 4)
                         |
                         v
 separately projected semantic contexts + intention-point anchors
                         |
                         v
     combined MTR-style queries + multimodal trajectory decoder
```

The 17-dimensional interface contains:

- eight ranked action values: `STATIONARY`, `STRAIGHT`, `STRAIGHT_LEFT`,
  `STRAIGHT_RIGHT`, `LEFT_TURN`, `RIGHT_TURN`, `LEFT_U_TURN`, and
  `RIGHT_U_TURN`;
- four multi-label affordances: `SLOW_ALLOW`, `ACCELERATE_ALLOW`,
  `LEFT_ALLOW`, and `RIGHT_ALLOW`; and
- five scenario categories: `INTERSECTION`, `ON_STRAIGHT_ROAD`,
  `PARKING_LOT`, `ON_ROADSIDE`, and `UNSURE`.

Ranked actions receive descending integer weights, while affordances and
scenarios use multi-hot and one-hot encoding, respectively. The encoded context
and the training-set future endpoint clusters are projected independently and
then added to form the decoder query tokens. This is a compact MTR-style
adaptation, rather than the full query-position and query-content pathway of the
original WOMD implementation.

The KNN database uses standardized, flattened observation tensors constructed
from the ego and six-neighbor histories, validity masks, and simplified local
lane polylines. It does not use the learned MTR encoder embeddings employed in
the reference implementation. The database and its feature standardizer are
fitted on the training split only.

## What Is and Is Not Released

The repository includes the adapter, local context-map renderer, structured
context parser and encoder, training-only retrieval utilities, intention-point
estimator, MTR-style conditioning components, configuration, tests, and pipeline
entry points.

The repository does **not** include:

- real post-crash trajectories or crash videos;
- generated transportation context maps or LLM responses;
- API credentials or private endpoint URLs;
- context indices, intention-point files, checkpoints, or evaluation outputs;
- WOMD data, upstream MTR source code, or the LLM-Augmented-MTR prompt.

See [the synthetic schema](docs/SYNTHETIC_DATA_SCHEMA.md) for the expected local
MATLAB fields and [third-party references](THIRD_PARTY.md) for provenance.

## Configuration

LLC-PC requires Python 3.10 or newer and the packages listed in the repository
root `requirements.txt`.

All commands below are run from `baselines/llc_pc`:

```bash
cd baselines/llc_pc
```

Copy the public example to a local configuration that Git ignores:

```bash
cp configs/post_crash_lc.example.json configs/post_crash_lc.local.json
```

For PowerShell:

```powershell
Copy-Item configs/post_crash_lc.example.json configs/post_crash_lc.local.json
```

In the local file, replace only the private input placeholders:

```json
{
  "data": {
    "train_mat": "<PATH_TO_TRAIN_MAT>",
    "test_mat": "<PATH_TO_TEST_MAT>"
  }
}
```

Do not replace the public example with real paths. Generated artifacts are
written below the ignored `outputs/` directory by default.

## Pipeline Commands

### 1. Validate Context Generation Without an API Call

Always begin with `--skip-llm`. This exercises data loading, observation-only
prompt construction, and local context-map rendering without contacting an
external service:

```bash
python scripts/generate_contexts.py \
  --config configs/post_crash_lc.local.json \
  --split train \
  --max-samples 10 \
  --skip-llm
```

Inspect `python scripts/generate_contexts.py --help` and the generated dry-run
artifacts before enabling an endpoint.

### 2. Generate Structured Training Contexts

Set credentials only through environment variables. Never place credentials in
the JSON configuration.

```bash
export LLM_API_KEY="YOUR_API_KEY"
export LLM_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"

python scripts/generate_contexts.py \
  --config configs/post_crash_lc.local.json \
  --split train \
  --max-samples 10 \
  --overwrite
```

For PowerShell:

```powershell
$env:LLM_API_KEY="YOUR_API_KEY"
$env:LLM_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"
python scripts/generate_contexts.py --config configs/post_crash_lc.local.json --split train --max-samples 10 --overwrite
```

Only observed history is provided to the renderer and LLM. Review request cost,
model availability, endpoint data policy, and local sample-selection settings
before running this step at scale. The commands above intentionally process only
10 samples; remove `--max-samples 10` only after validating those outputs. Keep
`--overwrite` when replacing a dry run or the 10-sample smoke-test manifest.

### 3. Build Training-Only Artifacts

```bash
python scripts/build_intention_points.py \
  --config configs/post_crash_lc.local.json

python scripts/build_context_index.py \
  --config configs/post_crash_lc.local.json
```

The context database is fitted only from training contexts. The intention points
are fitted only from training-set future endpoints. Both artifacts are frozen
before validation or test evaluation.

### 4. Train the Predictor

```bash
python scripts/train.py \
  --config configs/post_crash_lc.local.json
```

### 5. Evaluate with the Highest-Probability Trajectory

```bash
python scripts/evaluate.py \
  --config configs/post_crash_lc.local.json \
  --checkpoint <PATH_TO_LLC_PC_CHECKPOINT>
```

The paper-aligned configuration uses `prediction_mode: "top1"`. This evaluates
the trajectory with the highest predicted probability rather than selecting an
oracle best trajectory from the six generated modes.

## Leakage Controls

The following constraints are part of the baseline definition:

1. `y_future` never enters the task prompt, context map, context retrieval query,
   or inference model input.
2. The context index contains training samples only. Validation and test samples
   are retrieval queries and are never inserted into the database.
3. Intention-point clustering uses training endpoints only.
4. A sample may be excluded from retrieving itself, and the default configuration
   also excludes contexts from the same event when identifiers are available.
5. Test future trajectories are read only after prediction to calculate metrics.

These rules are more important than matching a particular local directory
layout. Do not weaken them when adapting the loader to another private dataset.

## Domain Adaptation Notes

The post-crash LC data do not contain the complete lane graph, intersections,
crosswalks, or traffic-control elements available in WOMD. LLC-PC therefore
constructs local parallel lane geometry from observed lane information and uses
the ego plus six defined neighboring-vehicle roles. Missing neighbors are
masked.

This limitation changes the map encoder input but does not require changing the
17-dimensional semantic interface. Semantic contexts and locally fitted
intention points are projected independently and then combined in the compact
decoder queries. Some generic scenario dimensions may be nearly constant on
straight post-crash road segments; they are retained to preserve the reference
method's semantic vocabulary. The adapted JSON schema requires exactly one
scenario label per sample rather than allowing overlapping scenario categories;
this task-specific restriction is recorded here so the baseline is not mistaken
for an exact reproduction.

## Tests

From the repository root:

```bash
python -m unittest discover -s baselines/llc_pc/tests -p "test_*.py"
```

The tests use synthetic arrays and temporary files only. They do not require the
private crash dataset or an LLM API call.

## Reporting and Citation

Use the name **LLC-PC** in tables and figures. A precise methodological
description is:

> LLC-PC is an independently implemented, domain-adapted baseline following the
> semantic-context conditioning design of Zheng et al. Structured action,
> affordance, and scenario outputs are encoded into a 17-dimensional context
> representation. The context and training-derived intention points are
> projected independently and combined into the queries of a compact MTR-style
> decoder. The input adapter, retrieval features, local context map, prediction
> horizon, and evaluation protocol are adapted to the post-crash lane-changing
> data.

See [CITATION.md](CITATION.md) for the required method and architecture
references.
