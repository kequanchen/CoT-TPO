# Direct LLM (Adapted)

Direct LLM is an independently implemented post-crash trajectory baseline that
follows the zero-shot coordinate-sequence prompting paradigm of LMTraj-ZERO
(Bae et al., CVPR 2024). Observed coordinates are serialized as text, and an
LLM directly returns future coordinates. There is no intermediate behavioral
vector, separate trajectory decoder, or explicit chain-of-thought interface.

The original LMTrajectory authors provide official code. This directory does
not copy or modify that code or its prompt strings. It implements a deliberately
narrow post-crash adaptation and must be reported as **Direct LLM (adapted)**,
not as a complete reproduction of LMTraj-ZERO.

## Why This Baseline Is Included

This baseline tests the most direct use of an LLM for trajectory prediction:
whether the same language model can map observed coordinates straight to a
future path without CoT-TP's predefined behavioral interface and downstream
generative model. It is therefore a framework-level contrast rather than merely
another recent publication.

For the manuscript comparison, the public example uses the same LLM family and
temperature setting selected for the other LLM-based methods. This helps avoid
attributing differences solely to a stronger foundation model. It does not make
the adapted implementation numerically equivalent to the original pedestrian
benchmark.

## Relation to LMTraj-ZERO

| Item | Official LMTraj-ZERO zero-shot setup | This Direct LLM adaptation |
| --- | --- | --- |
| Task | pedestrian trajectory prediction on ETH/UCY | post-crash vehicle trajectory prediction |
| Observations | 8 target positions | 10 target positions at 10 Hz |
| Future | 12 positions | 50 positions at 10 Hz, covering 5 s |
| Output candidates | 5 candidates per round, 4 conversational rounds | one strict top-1 trajectory |
| Temperature | 0.7 in the released zero-shot script | 0.3 in the public comparison config |
| Context | target coordinate sequence | target sequence plus optional anonymized neighbors |
| Response format | coordinate lists | one JSON object with exactly 50 xy pairs |
| Evaluation | multimodal pedestrian benchmark protocol | top-1 ADE/FDE at 1-5 s |

A target-only context option, closer to the original input scope, remains
available through:

```json
{"prompt": {"context_mode": "target_only"}}
```

The default `target_and_neighbors` setting gives the direct baseline access to
the observed interaction context available to the other methods. Up to six
valid neighbor histories are dynamically anonymized by distance at their latest
available observation; their observed time offsets are retained, and private
role names are not revealed to the model. Report the selected mode in the
experimental setup.

## Independent Implementation

The code retains only the published method paradigm:

```text
observed target and optional neighbor coordinates
                         |
                         v
        independently written direct-prediction prompt
                         |
                         v
             OpenAI-compatible chat completion
                         |
                         v
       strict JSON parser: one finite trajectory [50,2]
                         |
                         v
         exact-ID top-1 ADE/FDE at 1, 2, 3, 4, 5 s
```

The prompt wording, MATLAB adapter, target-centered coordinate conversion,
neighbor representation, JSON schema, API retry behavior, and evaluator are
newly written for this repository. See [THIRD_PARTY.md](THIRD_PARTY.md) for the
official source and its CC BY-NC 4.0 license.

## What Is and Is Not Released

Included:

- an observation-only MATLAB adapter;
- a target-centered, heading-aligned coordinate transform;
- target-only and target-plus-neighbor prompts;
- prompt-only test JSONL preparation;
- environment-only API configuration and bounded retries;
- strict 50-point JSON parsing and restartable prediction; and
- exact-coverage, top-1 ADE/FDE evaluation.

Not included:

- private post-crash data or crash videos;
- ETH/UCY data from the official project;
- prompts populated with real samples or LLM response dumps;
- API keys, endpoint URLs, account information, or model weights; and
- predictions, metrics, or private filesystem paths.

The private input contract is described using synthetic shapes in
[docs/SYNTHETIC_DATA_SCHEMA.md](docs/SYNTHETIC_DATA_SCHEMA.md).

## Requirements

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r baselines/direct_llm/requirements.txt
```

The code calls an OpenAI-compatible endpoint. Before using a hosted model,
review its price, rate limits, data handling, retention, geographic constraints,
and model availability. A dry prompt-preparation run does not contact an API.

## Configuration

From `baselines/direct_llm`:

```bash
cd baselines/direct_llm
cp configs/post_crash_lc.example.json configs/post_crash_lc.local.json
```

For PowerShell:

```powershell
Set-Location baselines/direct_llm
Copy-Item configs/post_crash_lc.example.json configs/post_crash_lc.local.json
```

Set `data.test_mat` only in the ignored local copy. Do not put a real path,
credential, or endpoint into the checked-in example. Generated artifacts are
written under the ignored `outputs/` directory by default.

## End-to-End Commands

### 1. Prepare Observation-Only Test Prompts

Start with a small local sample:

```bash
python scripts/prepare_prompts.py \
  --config configs/post_crash_lc.local.json \
  --max-samples 10
```

Inspect the generated JSONL before any API call. It must contain only IDs,
system/user prompts, and an audit string. Rebuild an existing local file with
`--overwrite`. Remove `--max-samples 10` only after verifying the schema.

### 2. Set Credentials Through Environment Variables

```bash
export LLM_API_KEY="YOUR_API_KEY"
export LLM_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"
```

For PowerShell:

```powershell
$env:LLM_API_KEY="YOUR_API_KEY"
$env:LLM_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"
```

The JSON configuration stores only the environment-variable names. The client
rejects inline keys and endpoint URLs.

### 3. Generate Direct Predictions

Run the same small-sample check first:

```bash
python scripts/predict.py \
  --config configs/post_crash_lc.local.json \
  --max-samples 10
```

Prediction resumes by `sample_id` by default and checkpoints every attempted
sample. Each successful row contains one top-level `trajectory`. API and parse
failures remain explicit with safe errors and attempt metadata. To retry failed
IDs, move or remove the local prediction file and start a fresh run; resuming
intentionally skips all existing IDs to prevent duplicates.

After checking cost and outputs, run the complete prompt file:

```bash
python scripts/predict.py \
  --config configs/post_crash_lc.local.json
```

### 4. Evaluate Exact Test Coverage

```bash
python scripts/evaluate.py \
  --config configs/post_crash_lc.local.json
```

The evaluator matches by `sample_id`, rejects duplicate and extra IDs, and by
default requires exactly one valid prediction for every test sample. It reports
top-1 ADE and FDE at 1, 2, 3, 4, and 5 seconds. Failed API calls and malformed
responses cannot silently reduce the denominator. `--allow-partial` is for
diagnosis only; partial metrics must not be reported as a baseline result.

## Leakage Controls

1. `ObservationSample` contains no future, lane-change phase, or intention.
2. Prompt preparation calls the loader with `include_future=False`.
3. Prompt JSONL validation rejects trajectory targets and label fields.
4. The LLM client reads only the prepared prompt JSONL.
5. Format retries add a fixed schema reminder; they never use ground truth.
6. Evaluation first loads the completed predictions and only then reads the
   labelled test future.
7. The test future is never used to repair, resample, rank, or select a result.

## Fair-Comparison Notes

The example uses `qwen-plus`, temperature 0.3, and one top-1 trajectory to match
the manuscript's LLM setting and ADE/FDE protocol. These are intentional
comparison controls, not claims about the official LMTraj-ZERO configuration.
If the model, temperature, context mode, sampling policy, or retry policy is
changed, record the change and do not mix the resulting predictions with a
previous output file. For reported experiments, record an exact provider model
version or snapshot when one is available because a service alias such as
`qwen-plus` may change over time.

Because the original method generated 20 multimodal candidates through four
rounds, its published benchmark values are not directly comparable to the
single-trajectory values produced here.

## Tests

All tests are synthetic and make no external API call:

```bash
python -m unittest discover -s baselines/direct_llm/tests -p "test_*.py" -v
```

## Reporting

Use **Direct LLM (adapted)** in manuscript tables. Cite Bae et al. for the
coordinate-sequence prompting paradigm and describe the implementation as an
independent domain adaptation. See [CITATION.md](CITATION.md) for suggested
wording.
