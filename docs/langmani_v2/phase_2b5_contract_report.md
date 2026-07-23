# Phase 2B.5 Policy Contract Report

The proposed future-student contract is:

- base-camera RGB, `uint8`, HWC, `256 x 256`, no crop;
- `PandaPolicyStateV0`, `float32[9]`, in stable active-joint name order;
- direct official `pd_joint_pos`, `float32[8]`, at 20 Hz;
- one task ID and one canonical natural-language instruction;
- pre-action frame timing, with no synthetic terminal action.

Object poses, goal poses, full environment states, reward fields, and task-success internals are
diagnostic-only and are not student fields. The exact accepted tasks, language, action statistics,
observation reconstruction, and privilege exclusion are content-bound in
`selected_task_decision.json`, `language_contract_manifest.json`, `action_contract_audit.json`,
`observation_contract_audit.json`, and `privilege_exclusion_audit.json`.

Future splits operate at episode identity, never frame identity. Reset, task, scene, source, and
language-template groups must stay disjoint according to `split_design_manifest.json`. An
unseen-task or cross-skill claim is prohibited unless an entire task/skill family is absent from
training.

The realized pilot contains exactly the declared allowlist and no privileged field:

- 15 successful episodes, five per selected task;
- 1,546 pre-action policy frames aligned one-to-one with 1,546 recorded actions;
- 386 Pick, 811 Stack, and 349 Push frames;
- 15 terminal RGB frames retained only as diagnostics and excluded from policy frames;
- no object pose, goal pose, full environment state, reward, or success-internal student field.

No train/validation/test split was materialized. The artifact contains only the future
episode-level split contract, so no unseen-reset, unseen-language, unseen-task, cross-skill, or
visual-shift performance claim exists.
