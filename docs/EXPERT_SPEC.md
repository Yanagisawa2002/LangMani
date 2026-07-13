# LangMani privileged expert specification

This document fixes the M2 demonstration-expert contract used unchanged by M3A.

## Runtime boundary

`PickPlaceExpert` solves only `LangMani-PickPlaceByInstruction-v0`, with `num_envs=1` and
`control_mode="pd_joint_pos"`. It reads the active semantic `TaskSpec` and privileged state only
through the environment's expert-only accessors. It never selects an actor from pixels or actor
ordering, and it does not add privileged fields to policy observations.

The motion-planning boundary is `MplibPandaPlannerAdapter`, a project-owned adapter over mplib
0.1.1 public APIs and ManiSkill's Panda handles. Runtime code does not import ManiSkill example
runners, does not modify upstream packages, and does not vectorize or multiplex mplib.

## Typed interface

The public project types are `ExpertConfig`, `ExpertStatus`, `ExpertPhase`, `PhaseResult`,
`ExpertResult`, and `PickPlaceExpert`. Results are immutable and JSON-ready; they contain compact
semantic IDs, counters, phase evidence, final evaluation, durations, and optional diagnostic paths,
not observations, images, tensors, planned paths, or trajectories.

Default `ExpertConfig` values are:

| Field | Value |
| --- | --- |
| control mode | `pd_joint_pos` |
| maximum episode steps | 200 |
| planning attempts per planned phase | 1 |
| pre-grasp clearance | 0.08 m |
| grasp approach distance | 0.05 m |
| lift clearance | 0.10 m |
| transport clearance | 0.10 m |
| placement clearance | 0.002 m |
| release settling steps | 15 |
| retreat distance | 0.10 m |
| planner timeout / seed | unsupported by the selected mplib screw planner; must be `None` |

Stable statuses are `success`, `invalid_task`, `initialization_failure`, `ik_failure`,
`planning_failure`, `execution_failure`, `grasp_failure`, `transport_failure`,
`placement_failure`, `verification_failure`, `target_off_table`, `timeout`, and
`unexpected_exception`. Unexpected exceptions are formatted only at the outer command/collection
boundary and preserve their type and message.

## Phase machine

The fixed phase order is:

1. `initialize`
2. `move_to_pregrasp`
3. `approach_target`
4. `close_gripper`
5. `verify_grasp`
6. `lift_target`
7. `move_above_destination`
8. `descend_to_place`
9. `open_gripper`
10. `settle_after_release`
11. `retreat`
12. `verify_task`

Every phase has a stable entry condition, command or target pose, bounded attempts, completion
criterion, and classified failure. Initialization validates environment identity, one-environment
execution, control mode, semantic task and handles, terminal state, and planner synchronization.
Planned paths are executed as 8-D Panda joint-position actions. Truncation, off-table state, or an
unexpected terminal condition aborts with a classified result.

## Deterministic grasp and placement

The sole grasp is a top grasp of the active cube. Cube half extent is `0.025 m`. World approach is
`(0,0,-1)`, gripper closing direction is `(0,-1,0)`, and TCP quaternion uses wxyz
`(0,1,0,0)`. Pre-grasp and grasp positions are derived from the target actor center and configured
clearances. Gripper commands use normalized `-1` to close and `+1` to open for six control steps.

Placement uses the selected bin's interior floor center, cube half extent, placement clearance, and
`0.01 m` wall safety margin. It retains the target-specific collision attachment through transport,
releases, settles, retreats, and finally requires the unchanged conservative M1 success oracle.

## Acceptance

`environment/verify_m2.py --target` first runs the M0 and M1 target gates. It then requires two
repeatable six-task smoke passes, a balanced 180-rollout benchmark over seeds 0 through 29 (30 per
TaskSpec), overall success at least 95%, each TaskSpec at least 90%, zero successful wrong-object or
wrong-bin outcomes, zero unclassified failures, zero benchmark crashes, one rendered successful
expert rollout, and all 12 nonempty phase frames.

The Windows review host cannot import the Linux-only mplib package or physically execute this gate.
M2 physical planning and rendering acceptance therefore remains pending until the native Linux RTX
4090 command passes.
