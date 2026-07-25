# LangMani 2.0 Phase 2C-A.1 evaluation report

## Result

The bounded action path passed real closed-loop execution with zero invalid
actions and zero simulator errors. The selected policies nevertheless achieved
zero successes in development and final evaluation. This is a valid
model-quality result, classified as `RESULT_B`; it is not an infrastructure
failure.

## Closed-loop infrastructure smoke

The preregistered Pick validation identity
`phase2c-a:validation:pickcube:000` ran at `H_exec=4`.
The selected Pick policy made 13 real policy queries and submitted 50 native
actions to 50 real environment steps. Every action was finite and in bounds;
no clip or projection occurred. The episode timed out with `failed_grasp`.
Success was not a smoke acceptance criterion.

The smoke runtime-manifest fingerprint is
`sha256:a95c324a0f8f4f6d96e6ee8f04d28843055994b8873311d7fcab8b795ef54c77`.

## Execution-horizon selection

The first five immutable Pick validation identities were evaluated once for
each candidate:

| `H_exec` | Success | Timeout | Mean action smoothness L2 | Mean queries | Inference p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0/5 | 5/5 | 0.059565 | 50 | 5.943 ms |
| 4 | 0/5 | 5/5 | 0.057090 | 13 | 6.207 ms |
| 8 | 0/5 | 5/5 | 0.062419 | 7 | 6.695 ms |

All candidates had zero invalid actions and simulator errors. The frozen
tie-break selected `H_exec=4` by lower action smoothness. The selection
fingerprint is
`sha256:4f833e1b78ea64bfdf4799a166946ccbe26a958b6252e6369e0e2746775aa884`.

## Validation development

Each policy/task group ran 30 matching validation episodes:

| Policy | Task | Success | Timeout | Failure categories |
| --- | --- | ---: | ---: | --- |
| Per-task | PickCube-v1 | 0/30 | 30/30 | 14 failed grasp; 16 no initial motion |
| Per-task | StackCube-v1 | 0/30 | 30/30 | 1 failed grasp; 29 no initial motion |
| Per-task | PushCube-v1 | 0/30 | 30/30 | 30 no initial motion |
| Shared | PickCube-v1 | 0/30 | 30/30 | 30 no initial motion |
| Shared | StackCube-v1 | 0/30 | 30/30 | 30 no initial motion |
| Shared | PushCube-v1 | 0/30 | 30/30 | 30 no initial motion |

Every group executed 1,500 actions from 390 policy queries and recorded zero
invalid actions and zero simulator errors.

## Frozen final evaluation

The final lock was created before any final episode and binds exactly 30
identities for each task in each of the `unseen_reset` and `visual_shift`
splits, the four selected checkpoints, processors, `H_exec=4`, 50-step budget,
and evaluator commit. Its fingerprint is
`sha256:91efc1160638a169697e49957f66937a1c27ae602818bb31f555a7f382fdb33c`.

| Policy | Split | Task | Success | Wilson 95% | p50 / p95 latency |
| --- | --- | --- | ---: | --- | ---: |
| Per-task | Unseen reset | Pick | 0/30 | [0, 0.1135] | 5.996 / 6.151 ms |
| Per-task | Unseen reset | Stack | 0/30 | [0, 0.1135] | 6.051 / 6.999 ms |
| Per-task | Unseen reset | Push | 0/30 | [0, 0.1135] | 5.960 / 6.496 ms |
| Per-task | Visual shift | Pick | 0/30 | [0, 0.1135] | 6.666 / 7.614 ms |
| Per-task | Visual shift | Stack | 0/30 | [0, 0.1135] | 6.897 / 7.199 ms |
| Per-task | Visual shift | Push | 0/30 | [0, 0.1135] | 6.511 / 7.275 ms |
| Shared | Unseen reset | Pick | 0/30 | [0, 0.1135] | 6.091 / 6.220 ms |
| Shared | Unseen reset | Stack | 0/30 | [0, 0.1135] | 6.119 / 7.148 ms |
| Shared | Unseen reset | Push | 0/30 | [0, 0.1135] | 6.074 / 6.283 ms |
| Shared | Visual shift | Pick | 0/30 | [0, 0.1135] | 6.653 / 6.915 ms |
| Shared | Visual shift | Stack | 0/30 | [0, 0.1135] | 6.570 / 7.570 ms |
| Shared | Visual shift | Push | 0/30 | [0, 0.1135] | 6.433 / 8.589 ms |

The final evaluation comprises 360 episodes, 18,000 executed native actions,
and 4,680 policy queries. All 360 episodes timed out. There were zero invalid
actions and zero simulator errors.

Pooling both final splits gives 0/60 per policy/task with Wilson 95% interval
`[0, 0.0602]`. Pooled latency was:

| Policy | Task | p50 / p95 |
| --- | --- | ---: |
| Per-task | Pick | 6.159 / 7.581 ms |
| Per-task | Stack | 6.844 / 7.123 ms |
| Per-task | Push | 6.189 / 7.054 ms |
| Shared | Pick | 6.201 / 6.821 ms |
| Shared | Stack | 6.520 / 7.215 ms |
| Shared | Push | 6.192 / 7.394 ms |

## Task-ID intervention

The intervention lock fingerprint is
`sha256:430e04ad21cea3c53df11990b71f7bd84b7bfc2700df7b487d9fc7bf40f57a8e`.
On 15 matched validation identities, cyclic-wrong IDs changed the action in
15/15 cases and the classified behavior in 3/15. Balanced shuffled IDs changed
the action in 8/15 and behavior in 2/15. Mean first-action L2 differences were
`0.024224` and `0.012714`; common-horizon differences were `0.318686` and
`0.167667`. All intervention episodes had zero success.

The diagnostic therefore proves that the oracle task ID can affect actions and
occasionally the observed behavior. It does not establish correct task
semantics or language grounding.

## Interpretation

The old Phase 2C-A result was a consumer/action-contract failure before any
environment step. Phase 2C-A.1 removes that blocker: the complete physical
evaluation path is valid. The new zero-success result is therefore attributable
to weak learned policy behavior under the frozen training and evaluation
contract, not to action-bound rejection, simulator failure, or a disconnected
server.
