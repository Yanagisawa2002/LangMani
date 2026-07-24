# Phase 2B.6.1-v2 result

Phase 2B.6.1-v2 is `RESULT_B`: official source episode 938 reproducibly fails the frozen replay
final-success gate.

| Required question | Evidence-based answer |
| --- | --- |
| 1. Did the validated Vulkan launcher pass? | Yes. Explicit ICD/EGL binding, `vulkaninfo`, minimal SAPIEN, and zero-step StackCube construction passed on the RTX 5090. |
| 2. Were source and action identities exact? | Yes. ZIP, HDF5, JSON, reset, trajectory, count, dtype, shape, and full action SHA-256 all match. |
| 3. Did controls 936 and 937 pass? | Yes. Both passed their sole fresh-process Mode A run and final canonical-success rule. |
| 4. Did episode 938 Mode A pass? | No. All three runs executed 105 actions but failed final canonical success. |
| 5. Were Mode A outcomes deterministic? | Yes categorically: `deterministic_failure`; all recorded transition and terminal diagnostics agree. |
| 6. When did canonical success first occur? | Source step 100; every replay step 101. |
| 7. Did the frozen stable-success gate pass? | Not applicable. The producer has no separate stable-success gate; replay had four consecutive successful steps as a diagnostic. |
| 8. Did source and replay outcomes agree? | No. Source final success is true; all replay final successes are false. |
| 9. Did RGB/state acquisition change the result? | Unverified and not run; Result B hard-stopped before Mode B. |
| 10. Was observation/action alignment correct? | Not evaluated; Mode B/C were prohibited. |
| 11. Did serialization and writer finalization pass? | Not evaluated; Mode C was prohibited and no NPZ or LeRobot root was created. |
| 12. What was the exact first failing sub-gate? | `canonical_terminal_success_gate`. |
| 13. Was the original Phase 2B.6 rejection physical or non-physical? | The new explicit replay shows a reproducible frozen physical/final-success failure. The historical wrapper's missing inner record still prevents claiming its exact old sub-gate. |
| 14. Is a clean production reproduction merely eligible? | No. `phase2b6_clean_reproduction_eligible=false`. |
| 15. Does source exclusion require separate authorization? | Yes. `source_episode_exclusion_review_eligible=true` is review eligibility only; no exclusion was enacted. |
| 16. Why do production, package acceptance, and training remain unauthorized? | A required official replay fails, no accepted 3,000-episode package exists, and this forensic phase cannot amend policy or authorize any consumer. |

The terminal state is:

```text
official_source_episode_938_physically_valid=false
source_episode_exclusion_review_eligible=true
phase2b6_clean_reproduction_eligible=false

phase2b6_production_resume_authorized=false
accepted_multiskill_dataset_validated=false

act_training_eligible=false
smolvla_training_eligible=false
vla_jepa_training_eligible=false
act_training_authorized=false
smolvla_training_authorized=false
vla_jepa_training_authorized=false

student_policy_training_started=false
optimizer_created=false
backward_passes=0
optimizer_steps=0
```

The independent compact-evidence verifier passed all 29 checks. The result fingerprint is
`sha256:30b7b6226d2762a248edc1ec9d9f68030f38bd1b1e385f00a3534c7544af84de`.
The 29-file artifact manifest fingerprint is
`sha256:c49558caa9f04d445f8e34acd6e71d17b0339a82c4a5273ac8f8827f35e15a6e`.

The recommended next phase is a separately authorized project-level source-policy decision. It
must choose among preserving the strict 3,000/3,000 replay requirement, introducing a versioned
and audited exclusion policy, reducing the official task/source set, or obtaining observations
through another official route. This phase does not make that choice, exclude episode 938, resume
production, or authorize training.
