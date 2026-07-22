# Phase 2B.3 simulator state clone audit

## Contract

ManiSkill 3.0.1 `BaseEnv.get_state_dict()` captures actor states as position, quaternion, linear
velocity, and angular velocity, plus articulation state. For the Panda `pd_joint_pos` controller,
`agent.get_controller_state()` is empty. Phase 2B.3 adds an expert-only snapshot for task counters,
latched failures, elapsed steps, seed metadata, and NumPy/ManiSkill RNG state.

The physical audit compares the same valid 12-action contact-bearing sequence after restore and in
isolated main/sandbox environments. Categorical observations, contacts, success, and failure must
agree exactly. Documented tolerances are 0.020 mm for object pose and joint position, 0.2 mm/s for
linear velocity, 0.0002 rad/s for joint velocity, and 0.0002 rad/s for angular velocity.

## Native results

| Attempt | Boundary | Categorical agreement | Maximum live/restored object-pose error | Result |
| --- | --- | ---: | ---: | --- |
| `23319bc` | two actions before first motion | not yet measured against live continuation | 4.367 mm | fail |
| `61c04c6` | same boundary plus live comparison | 100% | 4.373 mm | fail |
| `90cd0da` | three actions before first target contact | 100% | 4.394 mm | fail |

The last sequence begins from a verified target-contact-free state and then contains real target
contact and motion. Two independently cold-restored environments replay it identically, proving
that sandbox construction and configuration are deterministic. Both nevertheless differ from the
live continuation, so the public snapshot is not a faithful clone of the physical state.

ManiSkill exposes actor/articulation state but not complete PhysX solver history. The missing state
is relevant even before target contact because other persistent contacts and solver warm-start
state remain active. The 4.394 mm error is 219 times the object-pose audit tolerance and is material
beside the task's 5 mm containment clearance. Relaxing the tolerance would invalidate the
prediction contract rather than repair it.

Isolation passed: sandbox execution did not mutate the main environment, another sandbox, main
evaluator counters, or the main step budget. Physics-equivalence for simulator MPC did not pass.

Status: failed hard precondition. Phase 2B.3 is `RESULT_C` and no MPC qualification is authorized.
