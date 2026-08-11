# Synthetic Data Schema for LLC-PC

No trajectory data are distributed with this repository. This document uses
synthetic identifiers and array shapes only to describe the local MATLAB input
expected by the adapter.

## MATLAB Container

The training and test files each contain a configurable top-level struct array.
The example configuration uses `train_data` and `test_data`. Each element is one
trajectory sample with the following fields:

| Field | Shape | Description |
| --- | --- | --- |
| `scenario_id` | scalar | Event or crash-scenario identifier. |
| `traj_id` | scalar | Trajectory identifier within the event. |
| `lane_status` | scalar | Lane-change status supplied by preprocessing. |
| `time_since_crossing` | scalar | Time metadata supplied by preprocessing. |
| `x_hist` | `(T, 6)` | Observed ego history. |
| `y_future` | `(H, 2)` | Future ego coordinates used only for supervised training and evaluation. |
| `ctx` | struct | Ego and neighboring-vehicle histories. |

The paper-aligned defaults are `T = 10` at 10 Hz and `H = 50` for the maximum
5 s prediction horizon.

## Array Layout

`x_hist` uses the following columns:

```text
[X, Y, VX, VY, ACC, YAW]
```

`y_future` uses:

```text
[X, Y]
```

`ctx` contains:

```text
ctx.ego       # (T, 19), the full observed ego record
ctx.phys_tl   # (T, 8), target-lane leader
ctx.phys_tf   # (T, 8), target-lane follower
ctx.phys_tff  # (T, 8), second target-lane follower
ctx.phys_ol   # (T, 8), original-lane leader
ctx.phys_of   # (T, 8), original-lane follower
ctx.phys_off  # (T, 8), second original-lane follower
```

The default `ctx.ego` layout uses zero-based column indices:

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

Each neighbor matrix uses:

```text
[ID, LANE_ID, X, Y, VX, VY, ACC, YAW]
```

The LLC-PC adapter reads the kinematic values and lane-boundary information
through an explicit `DatasetLayout`. If a private dataset uses different column
positions, update the local layout instead of changing the semantic encoding.
Coordinates are expected in metres, velocities in metres per second,
accelerations in metres per second squared, and yaw in radians.

## Schematic Synthetic Sample

The following notation is documentation only; it is not a JSON input file:

```text
sample = {
  scenario_id: "synthetic_event_0001",
  traj_id: "synthetic_trajectory_0001",
  lane_status: 1,
  time_since_crossing: 0.0,
  x_hist: float[T, 6],
  y_future: float[H, 2],
  ctx: {
    ego: float[T, 19],
    phys_tl: float[T, 8],
    phys_tf: float[T, 8],
    phys_tff: float[T, 8],
    phys_ol: float[T, 8],
    phys_of: float[T, 8],
    phys_off: float[T, 8]
  }
}
```

Missing neighbors are permitted. They must remain marked invalid by the adapter
and must not be interpreted as vehicles located at the coordinate origin.

## Leakage Boundary

Only observed scene fields may be used to render the transportation context map,
construct the text prompt, calculate a retrieval query, or run inference:

```text
lane_status, time_since_crossing, x_hist, ctx
```

`scenario_id` and `traj_id` are bookkeeping fields used for deterministic
alignment and retrieval exclusions. They are not behavioral inputs to the LLM
or trajectory predictor.

`y_future` is prohibited from all LLM, prompt, rendering, retrieval-query, and
inference inputs. It may be used only for:

- fitting intention-point clusters on the training split;
- calculating the supervised trajectory loss on the training split; and
- calculating reported metrics after prediction.

The context database and intention-point clusters must be fitted from training
samples only. Test samples may query the frozen training context database but
must never be inserted into it.
