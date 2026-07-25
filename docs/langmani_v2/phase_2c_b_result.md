# LangMani 2.0 Phase 2C-B result

## Result D — Generic SmolVLA competence failure

The official pretrained SmolVLA pipeline was implemented, trained, reconstructed, and executed
successfully, but the Pick-only policy achieved 0/30 validation successes both before and after
the sole permitted bounded execution-horizon repair.

```text
result = RESULT_D
pipeline_valid = true
pick_initial_success = 0/30
pick_repaired_success = 0/30
pick_repaired_success_rate_wilson_95 = [0.0, 0.1135]
invalid_action_episodes = 0
simulator_error_episodes = 0
bounded_repair_used = true
bounded_repair_remaining = false
smolvla_baselines_validated = false
other_full_models_authorized = false
vla_jepa_phase_eligible = false
vla_jepa_training_authorized = false
```

The independent verification artifact has fingerprint
`sha256:9bc6bc202b118adb015a502b1111107b9c8c74e8e5eea92320a71a8ee98a33b21`
and sets `result_d_stop=true`.

## Recruiter-facing answers

1. **Did SmolVLA break the ACT 0% floor?** No. Bounded ACT finished at 0% success, and the
   fully trained Pick SmolVLA also finished at 0/30 before and after repair.
2. **Which skills were learned?** None met a closed-loop success gate. Only Pick was fully trained;
   Stack, Push, and shared full training were stopped by the Pick gate.
3. **Did the shared model outperform or underperform task-specific models?** Unavailable. The
   shared full model and matching Stack/Push references were not trained.
4. **Was positive or negative multi-skill transfer observed?** Unavailable. No valid multi-skill
   comparison ran.
5. **Did the policy react to natural-language instructions?** The shared micro fixture produced
   different actions for different instructions, proving input sensitivity only. Correct
   language-conditioned control was not evaluated.
6. **How did held-out paraphrases affect success?** Unavailable; the final language split was never
   opened.
7. **How did visual shifts affect success?** Unavailable; visual-shift evaluation was never
   authorized.
8. **What were the dominant failures?** After repair: 16 failed grasps, 13 no-motion timeouts, and
   one object drop across 30 episodes.
9. **What was training and inference cost?** Pick training took 10,031.31 seconds on one RTX 5090,
   with 2.35 GB peak allocated GPU memory. Repaired H=8 inference was 123.87 ms P50 and 126.13 ms
   P95, with seven model queries per 50-step episode.
10. **Is VLA-JEPA justified as the next controlled experiment?** The result motivates investigating
    the formulation, but Phase 2C-B does not make VLA-JEPA eligible or authorize training. A new
    project-level decision is required.

## What remains immutable

The dataset, splits, task definitions, success predicates, observation contract, physical action
contract, official base revisions, three Pick checkpoints, horizon-screen evidence, both
30-episode gates, and the independent Result-D verification remain immutable.

No full Shared/Stack/Push run, final test, language intervention, VLA-JEPA, LatentGuard, SARM, PPO,
or new data phase was started.
