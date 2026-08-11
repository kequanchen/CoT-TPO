# Synthetic Data Schema for Direct LLM (Adapted)

No trajectory sample is distributed here. This document uses synthetic names
and array shapes to describe the private MATLAB and generated JSONL interfaces.

## MATLAB Test Container

The public example expects a configurable `test_data` struct array. Each sample
contains:

| Field | Shape | Use |
| --- | --- | --- |
| `scenario_id` | scalar | Event identifier used for stable matching. |
| `traj_id` | scalar | Trajectory identifier within the event. |
| `x_hist` | `(T, 6)` | Observed target history. |
| `y_future` | `(H, 2)` | Future labels read only by offline evaluation. |
| `ctx` | struct | Observed target and neighbor histories. |

The comparison protocol fixes `T = 10`, `H = 50`, and 10 Hz. Extra private
fields such as lane-change phase or time-since-crossing are ignored by the
Direct LLM observation adapter and never become prompt fields.

`x_hist` uses:

```text
[X, Y, VX, VY, ACC, YAW]
```

`y_future` uses:

```text
[X, Y]
```

`ctx.ego` has shape `(T, 19)`. Relevant zero-based columns are vehicle ID at 0,
lane ID at 1, frame at 2, X at 3, and Y at 4. The source neighbor fields are:

```text
ctx.phys_tl
ctx.phys_tf
ctx.phys_tff
ctx.phys_ol
ctx.phys_of
ctx.phys_off
```

Each neighbor matrix has shape `(T, 8)`:

```text
[ID, LANE_ID, X, Y, VX, VY, ACC, YAW]
```

Missing neighbors remain invalid under an explicit mask. When neighbor context
is enabled, valid histories are dynamically anonymized as Vehicle A, B, and so
on by distance at their latest available observation. The prompt retains each
valid point's time offset relative to the target's final observation, so a
partially observed history is not treated as uniformly complete. Dataset-specific
role names are not exposed to the LLM.

## Coordinate Frame

Observed and future global positions are converted into a target-centered
frame. The last observed target position is `(0, 0)`, local x points forward
along the last observed heading, and local y points left. Coordinates are in
metres.

## Stable Sample Identifier

```text
sample_id = scenario_id:traj_id:current_frame
```

The frame suffix distinguishes sliding windows from the same trajectory.
Evaluation matches by this ID rather than JSONL order.

## Prompt-Only Test JSONL

`scripts/prepare_prompts.py` writes exactly these top-level fields:

```json
{
  "sample_id": "synthetic_event:synthetic_track:100",
  "event_id": "synthetic_event",
  "source_split": "test",
  "system_prompt": "<DIRECT-COORDINATE SYSTEM INSTRUCTION>",
  "user_prompt": "<OBSERVATION-ONLY COORDINATE SEQUENCES>",
  "prompt_text": "<MODEL-NEUTRAL AUDIT STRING>"
}
```

There is no `future`, `y_future`, answer, intention, lane-change status, or
other target-bearing field.

## Prediction JSONL

A successfully parsed prediction contains:

```json
{
  "sample_id": "synthetic_event:synthetic_track:100",
  "status": "ok",
  "attempts": "<NONSECRET ATTEMPT AUDIT>",
  "raw_output": "<MODEL RESPONSE>",
  "trajectory": "float[50][2]"
}
```

The shape strings above are documentation notation, not literal production
values. In an actual prediction, `trajectory` is a 50-row JSON array. Failed
records retain `sample_id`, `attempts`, a safe error message, and either
`parse_error` or `api_error` status. They count as invalid coverage.

## Leakage Boundary

1. `ObservationSample` has no future or lane-change label member.
2. Prompt preparation calls the loader with `include_future=False`.
3. The JSONL prompt schema rejects future and label fields.
4. Inference consumes only the prepared prompt JSONL.
5. `scripts/evaluate.py` reads the completed prediction file first and then
   separately loads `y_future` for ADE/FDE.

The test future must never be used to repair, resample, select, or rank an LLM
prediction.
