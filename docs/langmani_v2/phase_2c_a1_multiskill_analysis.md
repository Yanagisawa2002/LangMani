# LangMani 2.0 Phase 2C-A.1 multi-skill analysis

## Scope

This analysis compares one shared, oracle Task-ID-conditioned ACT against the
three separately trained per-task ACT policies. It does not claim language
understanding and does not compare SmolVLA or VLA-JEPA.

## Closed-loop comparison

Across both frozen final splits, every per-task and shared policy achieved
0/60 for its task. Shared-minus-per-task success-rate and timeout-rate
differences are therefore exactly zero for PickCube-v1, StackCube-v1, and
PushCube-v1. Visual-shift success degradation is also numerically zero for all
groups because both the reference and shifted evaluations are at the same
zero-success floor.

Those equalities are uninformative floor effects. They do not prove that
multi-skill interference is absent. In particular, the experiment cannot
distinguish equal architecture weakness from shared-representation
interference using success alone.

## Failure-category comparison

Pooled final failures differ even at the common zero-success floor:

| Task | Per-task ACT | Shared ACT |
| --- | --- | --- |
| Pick | 6 failed grasp, 53 no initial motion, 1 object drop | 60 no initial motion |
| Stack | 2 failed grasp, 58 no initial motion | 60 no initial motion |
| Push | 60 no initial motion | 58 no initial motion, 2 insufficient push |

The shared model is more uniformly static on Pick and Stack, while its Push
policy produced two insufficient-push episodes. These are behavioral
differences, but they do not rescue task success.

## Action-distribution evidence

The compact interference artifact records shared-minus-per-task action
variance for all eight dimensions by task and split. These differences are
descriptive diagnostics, not a quality metric. The action path remained native
and uncorrected in both model families, so differences are policy outputs
rather than evaluator projection artifacts.

## Task-ID sensitivity

The preregistered 45-episode intervention ran 15 correct-ID, 15 cyclic-wrong-ID,
and 15 balanced-shuffled-ID conditions over matched validation states:

- cyclic-wrong IDs changed actions in 15/15 cases and classified behavior in
  3/15;
- shuffled IDs changed actions in 8/15 cases and behavior in 2/15;
- mean first-action L2 change was `0.024224` for wrong IDs and `0.012714` for
  shuffled IDs;
- mean common-horizon L2 change was `0.318686` and `0.167667`;
- all intervention success counts were zero.

The task-ID input is not ignored, but the measured sensitivity did not yield
correct control. Because the condition is an oracle stable task ID, no language
grounding claim is permitted.

## Conclusion

The preregistered definition of `RESULT_C` required the shared ACT to fail while
the per-task baselines remained meaningfully successful. That condition did not
occur: both families failed at the floor. The result is therefore
`RESULT_B`—a valid ACT pipeline with weak ACT policy quality—not evidence for or
against isolated multi-skill interference. ACT is closed without another
architecture, seed, or repair.
