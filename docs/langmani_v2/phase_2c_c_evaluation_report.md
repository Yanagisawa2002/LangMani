# Phase 2C-C evaluation report

Status: complete as Case B. All authorized closed-loop gates ran, and the unseen-reset test
remained sealed because validation did not reach 3/30.

## Frozen selection

Validation-only offline metrics selected the relative step-20,000 checkpoint
`sha256:f42b6cbd7ce798aef969e7c328e1ccfc7dc677fa407498c9b3b0344bc3e621f3`.
The frozen six-reset horizon screen then produced:

| Horizon | Success | Grasp region | Grasp | Lift | Failure summary |
| ---: | ---: | ---: | ---: | ---: | --- |
| H=1 | 0/6 | 2/6 | 0/6 | 0/6 | 2 failed grasp, 4 no motion |
| H=8 | 0/6 | 6/6 | 6/6 | 3/6 | 2 failed grasp, 4 object drop |

The pre-registered lexicographic rule selected H=8 from grasp and lift progress. Neither full
validation nor test results were used for that choice.

## Absolute training-reset diagnostic

The frozen absolute step-20,000 Pick model was evaluated without retraining on the same 30 accepted
training reset identities later used by the relative model. At H=8 it achieved 1/30 success:
9 entered the grasp region, 2 grasped, 1 lifted, 19 were classified failed-grasp, 9 no-motion, and
1 object-drop. There were 29 timeouts, zero invalid actions, and zero simulator errors. This proves
the absolute model can solve at least one demonstrated reset, but not that it generalizes.

## Relative validation

The selected relative checkpoint/H=8 completed all 30 validation episodes:

- 0/30 success and 30 timeouts;
- 30/30 entered the grasp region, 22/30 grasped, and 4/30 lifted;
- 14 failed-grasp, 14 object-drop, and 2 no-motion;
- zero invalid actions and zero simulator errors;
- 50.0 mean episode steps, seven mean policy queries;
- inference latency p50 `119.926 ms`, p95 `121.206 ms`.

The run therefore exercised a valid closed-loop consumer and showed much more contact behavior
than the frozen absolute validation result, but it did not meet the task success predicate.

## Relative training resets

On the same 30 accepted training reset identities used for the absolute diagnostic, the relative
model achieved 1/30 success:

- 25/30 entered the grasp region and 19/30 grasped;
- 0/30 met the diagnostic lift threshold;
- 14 failed-grasp, 12 object-drop, 3 no-motion, and 1 other failure;
- 29 timeouts, zero invalid actions, and zero simulator errors;
- 48.6 mean episode steps;
- inference latency p50 `137.622 ms`, p95 `143.787 ms`.

The single deterministic training-reset success satisfies the pre-registered minimum for
"meaningful" training-reset fit. Validation remained 0/30, so the result is Case B rather than
Case C.

## Unseen-reset status

The 50-episode `test_unseen_reset` schedule was not opened or executed because validation was below
3/30. Its success and failure metrics are unavailable, not zero. No language, visual-shift,
Shared, Stack, Push, or repair evaluation ran.
