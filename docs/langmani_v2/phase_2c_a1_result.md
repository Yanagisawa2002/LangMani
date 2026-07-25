# LangMani 2.0 Phase 2C-A.1 result

## Terminal classification

`RESULT_B`

Phase 2C-A.1 completed the one authorized bounded-ACT repair, full training,
validation selection, physical development, Task-ID intervention, frozen final
evaluation, compact evidence production, and independent verification. The
pipeline is valid, but every selected policy achieved zero closed-loop
successes. ACT is permanently closed.

## Required questions

1. **Did the frozen input identity pass?** Yes. The package fingerprint is
   `sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`;
   the unchanged 168-file payload-tree digest is
   `sha256:b0e255d8b3f7f77fe33538cb2c313620d1d00adf90562b33dad06b10674474b3`.
   The 169th live file is only the authorized control sidecar.
2. **What was repaired?** Only the unbounded ACT action output was replaced by
   `bounded_action_head_v1`, with direct physical-action loss and identity
   action processors.
3. **Were actions corrected after prediction?** No. There was no clipping,
   projection, gripper threshold, rejection sampling, or replacement action.
4. **Did padding behave correctly?** Yes. Padded timesteps contributed zero
   action loss; all-padding action loss was finite zero.
5. **Did the real GPU and micro-overfit gates pass?** Yes. Real LeRobot
   forward/backward/optimizer/save/reload/inference passed, and both Pick and
   shared micro-overfits reduced their frozen errors.
6. **Did full training complete?** Yes. Exactly four seed-0 runs completed:
   Pick 2,680, Stack 3,680, Push 2,380, and shared 8,712 optimizer steps.
7. **Did all retained checkpoints pass screening?** Yes. All 16 completed
   exactly 10,000 ordered policy queries with zero contract failures. One
   validation-ranked checkpoint per model was frozen.
8. **Did a learned action reach the simulator?** Yes. The closed-loop smoke
   executed 50 real actions and 50 environment steps with zero invalid actions
   or simulator errors.
9. **Which execution horizon was selected?** `H_exec=4`, using the frozen
   five-episode Pick validation comparison and smoothness tie-break.
10. **What was development quality?** Zero successes in all six 30-episode
    per-task/shared validation groups.
11. **Did Task ID affect the shared policy?** Yes. Wrong IDs changed actions in
    15/15 cases and behavior in 3/15; shuffled IDs changed actions in 8/15 and
    behavior in 2/15. This is oracle-condition sensitivity, not language
    grounding.
12. **What was final quality?** Zero successes in all 12 final 30-episode
    policy/split/task groups. Every group had Wilson 95% interval
    `[0, 0.1135]`; pooling both splits gives 0/60 and `[0, 0.0602]` per
    policy/task. All 360 final episodes timed out.
13. **Is multi-skill interference established?** No. Shared-minus-per-task
    success is zero only because both sides are at a zero-success floor.
14. **Was this an infrastructure failure?** No. The complete physical path
    passed with zero invalid actions and simulator errors. The observed result
    is weak learned-policy quality.
15. **May ACT continue?** No. `act_phase_closed=true` and
    `further_act_architecture_authorized=false`.
16. **May SmolVLA or VLA-JEPA train now?** No. The result makes a separately
    invoked SmolVLA phase eligible, but
    `smolvla_training_authorized=false` and
    `vla_jepa_training_authorized=false`.

## Final flags

```text
act_baselines_validated=true
act_policy_quality_weak=true
shared_act_failed=false
act_phase_closed=true
further_act_architecture_authorized=false
smolvla_phase_eligible=true
smolvla_training_authorized=false
vla_jepa_training_authorized=false
```

`shared_act_failed=false` means the preregistered Result-C condition was not
met; it does not mean the shared policy succeeded.

## Immutable identities

- source base:
  `135f13d11f0f47173abe42fa19d969e519d7b1d9`;
- training implementation/evidence:
  `852658d60d8924d53fb7eb0d26cccbc4e54424ad`;
- pre-final evaluation implementation/evidence:
  `160d6c31b6e3708ee982edfa0118152cd7489e0f`;
- final evaluation implementation/evidence:
  `a5c3dcc7c944e02114faaf5b304620cfa7b6cd03`;
- verifier implementation/evidence:
  `4fe0757786502d0f810231dc385dfe5f6b127125`;
- final policy lock:
  `sha256:91efc1160638a169697e49957f66937a1c27ae602818bb31f555a7f382fdb33c`;
- result analysis:
  `sha256:b5606a256b857485ffcc55a19fe777b95ea832e7c9fdb9de6c3c2b5d8fd54b48`;
- final verification:
  `sha256:5bd960e7c44705258c5cf6ef7c1ba82089556723fbb698796338cb1e3c5a0a19`;
- compact artifact manifest:
  `sha256:57606d6114f2befe43be85c62ddb075bcf7d1c0eb76503b75c5c74dc0cc88d2e`.

The independent verifier passed all 17 checks and rehashed all 16 checkpoints.
Generated checkpoints, full episode logs, videos, and datasets were not
committed.

## Repository validation

- Ruff format verification covered 436 Python files and passed.
- Ruff lint passed.
- Phase 2C-A/A.1 targeted tests passed: 61.
- The complete unit suite passed: 1,748.
- Targeted typing over the Phase 2C-A.1 implementation and entry points passed.
- Isolated sdist and wheel builds passed.

The repository-wide command `pytest -m "not gpu and not rendering"` reported
1,799 passed, 18 deselected, and two failures. Both failures are unchanged
native physical M2/M3A smoke tests that the historical marker expression does
not exclude: M2 planner initialization and the consequent one-group M3A
collection failure. They are outside the Phase 2C-A.1 path and reproduce the
pre-existing target-runtime baseline; no Phase 2C-A/A.1 test failed.

## Authorization boundary

This result exhausts the authorized ACT path. The only eligible next research
milestone is a separately specified and separately authorized SmolVLA phase
that consumes the same frozen package. Eligibility is not permission: no
SmolVLA dependency, model, optimizer, training run, or evaluation is started by
this result.
