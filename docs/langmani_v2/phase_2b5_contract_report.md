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
