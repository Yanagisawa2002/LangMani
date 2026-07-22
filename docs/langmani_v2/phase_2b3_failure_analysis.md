# Phase 2B.3 failure analysis

## Observed failure

Complete public state capture includes robot and object pose/velocity, articulation state, empty
`pd_joint_pos` controller state, task-progress tensors, elapsed steps, seed metadata, and NumPy/
ManiSkill RNG state. A fresh environment can restore those values exactly.

Two cold-restored environments then produce identical 12-step trajectories. A live environment
continued from the capture point does not: the final contact-free audit observed 4.394 mm target
pose error, 1.257 mm/s linear-velocity error, 0.0464 rad/s angular-velocity error, and 0.000899
rad/s joint-velocity error. All 192 categorical comparisons still agreed.

## Root cause and classification

The remaining difference is simulator-internal history that is not represented in ManiSkill
3.0.1's public state dictionary. PhysX solver/contact warm-start state is the leading mechanism;
the project does not claim access to or exact reconstruction of that private state.

This is not a failed policy-quality gate. It makes the proposed simulator rollout an invalid model
of the main environment, so the exact classification is `RESULT_C`.

## Rejected workarounds

- Widening the tolerance: rejected because 4.394 mm is material beside 5 mm containment clearance.
- Reusing warm sandboxes: rejected because the state still depends on hidden history.
- Hand-written transition scores: rejected because they are not simulator MPC.
- Teleporting or correcting objects after a rollout: prohibited.
- Running the development/formal schedules anyway: prohibited by the stop rule.

## Unresolved blocker

A future simulator-MPC phase requires a supported snapshot API that preserves complete solver
state, or a redesigned candidate protocol whose prediction equivalence can be proven without
mid-episode cloning. Either path requires a new, separately authorized architecture and evidence
contract; it is not a repair inside Phase 2B.3.
