# LangMani 2.0 Phase 2 results

## Recruiter summary

LangMani 2.0 added a genuinely different manipulation family: a Panda must push either a cube or a
horizontal rolling cylinder into one of four planar regions without grasping, lifting, toppling, or
moving the distractor. The environment, task taxonomy, vectorized evaluation, deterministic reset,
privileged mplib expert, diagnostics, and unified evaluator compatibility are implemented.

The project then applied its stop condition honestly. A real RTX 5090 validation completed all 80
predeclared episodes, but the expert achieved 41/50 standard successes (82%) and 21/30 hard
successes (70%). Since standard reliability was below the 90% data-generation target, LangMani did
not manufacture a clean-looking dataset by filtering failures and did not start SmolVLA training.

## Raw results

| Evaluation | Cube/regions | Cylinder/regions | Result |
| --- | --- | --- | --- |
| 16-task smoke | all 8 semantic tasks per difficulty | represented | standard 8/8; hard 5/8 |
| 80-task target gate | 50 standard + 30 hard, fixed seeds | represented | standard 41/50; hard 21/30 |

Target-validation status counts:

| Difficulty | Success | Timeout | Planning | Verification | Wrong object | Workspace exit | Action bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 41 | 5 | 1 | 3 | 0 | 0 | 0 |
| Hard | 21 | 2 | 1 | 0 | 3 | 2 | 1 |

The standard forward-left and forward-right tasks succeeded 24/24. Failures concentrated in
lateral pushes, with the rolling cylinder-left task succeeding only 2/6 times. This isolates the
next engineering problem to side-contact recovery and motion-budget efficiency rather than camera,
task identity, success geometry, or forward reachability.

## What exists and what does not

Implemented and physically exercised:

- `LangMani-PushToRegion-v0` with 16 semantic task instances;
- conservative success/failure metrics and no-leakage observations;
- deterministic Panda/mplib pushing expert with explicit phases;
- real all-task smoke and fixed 50/30 target validation;
- preserved machine-readable results and classified failures.

Not started because the expert gate failed:

- pushing demonstration collection;
- multi-skill LeRobot dataset and split manifests;
- SmolVLA dependency/runtime adapter;
- SmolVLA smoke or full training;
- learned push-only or multi-skill closed-loop evaluation.

## Exact next step

Implement one bounded side-contact recovery improvement that retains the fixed success geometry and
250-step limit. Re-run the same all-task smoke and 50/30 target gate. Only a fresh result with
standard at least 90% and hard at least 70% may authorize demonstration collection.
