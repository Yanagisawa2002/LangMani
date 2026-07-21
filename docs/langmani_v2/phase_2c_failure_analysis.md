# Phase 2C learned-policy failure analysis

No learned checkpoint or closed-loop episode exists yet, so there are no measured failure counts or
representative videos to report. The reachable execution host lacks the accepted Phase 2B external
dataset, and the experiment correctly stopped before training.

The frozen primary taxonomy is:

- `no_initial_motion`
- `incorrect_approach_direction`
- `premature_contact`
- `contact_loss`
- `insufficient_push`
- `overshoot`
- `lateral_drift`
- `cylinder_rotation_failure`
- `wrong_object_interaction`
- `workspace_exit`
- `action_saturation`
- `action_oscillation`
- `gripper_misuse`
- `language_target_confusion`
- `timeout`
- `verification_boundary_failure`
- `invalid_policy_output`
- `other`

Every future failure record must retain the sealed episode identity, split, task, instruction, seed,
geometry, direction, progress and action statistics, inference statistics, and representative media
where practical. The final demo set must contain failures as well as cube, cylinder, lateral,
unseen-language, and visual-shift successes. No category may be inferred from training loss alone.
