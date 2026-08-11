# Synthetic Data Schema for LC-LLM (Adapted)

No trajectory data are distributed with this repository. The identifiers and
shapes below document the private MATLAB interface and generated JSONL records
without revealing a real crash, vehicle, prompt, or trajectory.

## MATLAB Container

The example configuration expects `train_data` and `test_data` struct arrays.
The keys are configurable. Each element contains:

| Field | Shape | Use |
| --- | --- | --- |
| `scenario_id` | scalar | Event identifier used only for bookkeeping. |
| `traj_id` | scalar | Trajectory identifier within the event. |
| `lane_status` | scalar | Supervised phase label; never inserted into an inference prompt. |
| `time_since_crossing` | scalar | Preprocessing metadata; not inserted into the Figure 3 prompt. |
| `x_hist` | `(T, 6)` | Observed target-vehicle history. |
| `y_future` | `(H, 2)` | Future labels used only for training answers and post-prediction evaluation. |
| `ctx` | struct | Observed target and neighboring-vehicle histories. |

The released protocol uses `T = 10` observations at 10 Hz and `H = 50` future
points at 10 Hz. Thus, the adapted prompt describes one second of observation
and requests five seconds of future positions.

## Array Layout

`x_hist` columns are:

```text
[X, Y, VX, VY, ACC, YAW]
```

`y_future` columns are:

```text
[X, Y]
```

`ctx.ego` has shape `(T, 19)`. Its zero-based column indices are:

| Index | Value |
| ---: | --- |
| 0 | vehicle ID |
| 1 | lane ID |
| 2 | frame |
| 3 | X |
| 4 | Y |
| 5 | distance to the upper lane boundary |
| 6 | distance to the lower lane boundary |
| 15 | lateral speed |
| 16 | longitudinal speed |
| 17 | acceleration |
| 18 | yaw |

The six neighboring-vehicle fields are:

```text
ctx.phys_tl   # target-lane leader
ctx.phys_tf   # target-lane follower
ctx.phys_tff  # second target-lane follower
ctx.phys_ol   # original-lane leader
ctx.phys_of   # original-lane follower
ctx.phys_off  # second original-lane follower
```

Each has shape `(T, 8)` with columns:

```text
[ID, LANE_ID, X, Y, VX, VY, ACC, YAW]
```

Missing neighbors remain invalid under the adapter's mask and are described as
not observed. They are not treated as vehicles at `(0, 0)`. Coordinates use
metres, velocities metres per second, accelerations metres per second squared,
and yaw radians. A private dataset with a different layout must change its
local adapter configuration rather than silently reinterpreting columns.

## Coordinate Frame

The MATLAB positions are transformed into the vehicle-centered frame described
in the LC-LLM paper. The last observed target position is the origin. The local
`x` axis points forward along the last observed heading and local `y` points to
the left. Both training targets and evaluated predictions use this frame.

## Stable Sample Identifier

Every record is matched by:

```text
sample_id = scenario_id:traj_id:current_frame
```

The frame suffix prevents collisions among sliding windows from the same
trajectory. Evaluation rejects duplicate IDs and, by default, requires exactly
one valid prediction for every test ID.

## Generated Training JSONL

One supervised training line contains these top-level fields:

```json
{
  "sample_id": "synthetic_event:synthetic_track:100",
  "event_id": "synthetic_event",
  "source_split": "train",
  "system_prompt": "<PAPER-BASED SYSTEM INSTRUCTION>",
  "user_prompt": "<OBSERVATION-ONLY SCENE DESCRIPTION>",
  "prompt_text": "<FORMATTED LLAMA-2 INSTRUCTION>",
  "answer": "<REASONING, INTENTION, AND 50 FUTURE POINTS>",
  "intention": 1,
  "future_local": "float[50][2]"
}
```

`intention` uses the paper convention `0 = keep lane`, `1 = left lane change`,
and `2 = right lane change`. With the default `phase_left_event` label mode,
the current private preprocessing contains only left-change events: statuses 0
and 1 supervise left lane change, while status 2 supervises keep lane. The
phase value itself is not exposed to the model prompt. For other datasets,
`future_lateral_displacement` can derive all three labels from training targets
using the configured threshold.

## Generated Inference JSONL

Inference records intentionally omit all target-bearing fields:

```json
{
  "sample_id": "synthetic_event:synthetic_track:100",
  "event_id": "synthetic_event",
  "source_split": "test",
  "system_prompt": "<PAPER-BASED SYSTEM INSTRUCTION>",
  "user_prompt": "<OBSERVATION-ONLY SCENE DESCRIPTION>",
  "prompt_text": "<FORMATTED LLAMA-2 INSTRUCTION>"
}
```

They contain no `answer`, `intention`, `future_local`, or `y_future` field.

## Prediction JSONL

The inference script writes one top-level object per attempted sample. A valid
record has:

```json
{
  "sample_id": "synthetic_event:synthetic_track:100",
  "status": "ok",
  "raw_output": "<GENERATED TEXT>",
  "intention": 1,
  "trajectory": "float[50][2]"
}
```

A failed parse retains the `sample_id`, `status: "parse_error"`, raw output,
and error message. It is counted as invalid coverage and cannot disappear from
the reported test set.

## Leakage Boundary

The prompt builder accepts an `ObservationSample`, which has no future field.
The following are prohibited from test prompts and model inputs:

- `y_future` and `future_local`;
- ground-truth intention;
- `lane_status` and `time_since_crossing`; and
- any explanation label derived from future motion.

The labelled test future is loaded only by `scripts/evaluate.py`, after the
prediction JSONL already exists, to calculate intention accuracy, macro-F1,
ADE, and FDE.
