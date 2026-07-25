# LangMani 2.0 Phase 2C-B evaluation report

## Offline checkpoint diagnostics

All three retained Pick checkpoints were evaluated on the same deterministic 128-frame validation
view. No diagnostic produced an invalid action, non-finite loss, clipping event, or projection
event.

| Step | Physical MAE | First-action MAE | Arm MAE | Gripper MAE | Flow loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.10415 | 0.08935 | 0.10901 | 0.07013 | 0.13587 |
| 10,000 | 0.08085 | 0.06125 | 0.08214 | 0.07184 | 0.12992 |
| 20,000 | 0.06079 | 0.02818 | 0.06115 | 0.05826 | 0.10066 |

The validation-only selector retained 20k and 10k. It did not select on total training loss alone.

## Frozen execution-horizon screen

The first six frozen Pick validation identities were run for each retained checkpoint and
`H_exec` in `{1,4,8}`.

| Checkpoint | H | Success | Dominant failures | Queries/episode | Smoothness L2 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 10k | 1 | 0/6 | 5 failed grasp, 1 no motion | 50 | 0.22014 |
| 10k | 4 | 0/6 | 5 no motion, 1 failed grasp | 13 | 0.21731 |
| 10k | 8 | 0/6 | 5 failed grasp, 1 no motion | 7 | 0.18248 |
| 20k | 1 | 0/6 | 6 no motion | 50 | 0.08297 |
| 20k | 4 | 0/6 | 6 no motion | 13 | 0.11193 |
| 20k | 8 | 0/6 | 5 failed grasp, 1 no motion | 7 | 0.11896 |

All six cells had zero invalid actions and zero simulator errors. Since success and timeout counts
tied, the frozen ranking selected 20k/H=1 on action smoothness.

## Initial Pick gate

The selected 20k/H=1 policy completed 30 validation episodes:

- 0 successes and 30 timeouts;
- success rate 0.0, with Wilson 95% confidence interval `[0.0, 0.1135]`;
- 26 `no_initial_motion`, 4 `failed_grasp`;
- zero invalid actions and zero simulator errors;
- 50 environment steps and 50 policy queries per episode;
- inference latency P50/P95 of 125.60/135.63 ms.

The independent verifier classified the pipeline as valid, the quality gate as failed, and one
bounded repair as permitted.

## Sole bounded repair

The same 20k checkpoint with the pre-locked H=8 configuration had produced contact-bearing
behavior in five of six screen episodes, whereas H=1 produced six no-motion failures. This
demonstrated one specific execution-horizon defect. The sole repair therefore changed only
`H_exec` from 1 to 8. It did not alter model bytes, data, splits, observations, actions, success,
or training.

The repaired 30-episode gate completed with:

- 0 successes and 30 timeouts;
- success rate 0.0, with Wilson 95% confidence interval `[0.0, 0.1135]`;
- 16 `failed_grasp`, 13 `no_initial_motion`, and 1 `object_drop`;
- zero invalid actions and zero simulator errors;
- 50 environment steps and 7 policy queries per episode;
- inference latency P50/P95 of 123.87/126.13 ms.

H=8 changed behavior and reduced no-motion failures, but did not produce task success.

## Result

The repaired gate remained 0/30. The independent verifier set `result_d_stop=true`,
`bounded_repair_permitted=false`, and `other_full_models_authorized=false`. This is a SmolVLA
competence failure, not an infrastructure or action-contract failure.

The final/test schedules were never opened. Shared, Stack, Push, language intervention, held-out
paraphrase, visual-shift, and multi-skill metrics are unavailable rather than zero.
