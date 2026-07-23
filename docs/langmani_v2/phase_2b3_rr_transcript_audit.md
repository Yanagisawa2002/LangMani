# Phase 2B.3.1-RR historical transcript audit

## Required evidence

A valid reset-replay transcript must bind the exact reset seed and options, task/object/direction/
difficulty/distractor identities, controller and physics configuration, every action actually
submitted, exact action dtype/shape/order and step index, termination/truncation state, the
environment call sequence, stored live boundary states, and an action-prefix content hash.

The defining action contract is native `float32[8]` under `pd_joint_pos`. Planner targets,
geometric waypoints, regenerated expert output, rounded arrays, and unrelated collection episodes
cannot substitute for the submitted arrays.

## Sources inspected

The no-GPU evidence node at `/root/autodl-tmp/langmani` was clean at
`cc31ced97e66220626ef39374b3e3acf84ccd463`. Targeted read-only searches covered:

- `outputs/diagnostics/v2/phase2b2`;
- `outputs/diagnostics/v2/phase2b3`;
- `outputs/datasets/langmani_v2`;
- committed MPC artifacts at `cc31ced...`;
- external geometry artifacts at `ae6aae...`.

No relevant Phase 2B.2/2B.3 HDF5, H5, NPZ, NPY, PT, or PTH transcript was present. Recursive JSON
inspection found no `actions`, `executed_actions`, or `action_sequence` numeric array for the
required diagnostic execution. Filename searches for the relevant diagnostic seeds also returned
no raw trace.

The strongest surviving metadata record is
`state-clone-audit-contact-free-90cd0da.json`. It identifies seed 69000,
`blue_cube/left/standard`, 156 total reference actions, a 12-action window beginning at action 88,
first target contact or motion at index 91, and action-sequence SHA-256
`081aec1500ba1e03d13e4c6214af7c75a62a9c809d64766aea8808adc1ba6ef7f`.
It does not contain the 12 numeric actions or the complete prefix from step zero.

Historical Phase 2B v1 collection files are not substitutes. The milestone contract freezes the
v1 bytes as unavailable for the current training identity, and those unrelated collection
episodes do not establish the exact selected diagnostic reset/call/action history.

## Decision

Original executed action prefixes were not available. The phase stopped as
`RESULT_D / HISTORICAL_PREFIX_UNAVAILABLE`, with zero simulator trials.

This is an evidence-availability conclusion only. It does not prove that reset-replay transitions
would pass or fail.
