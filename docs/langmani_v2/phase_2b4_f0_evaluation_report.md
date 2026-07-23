# Phase 2B.4-F0 evaluation report

## Fixed micro evaluation

The deterministic checkpoint ran on all 48 disjoint micro seeds `800000..800047`. The schedule
equally covered cube/cylinder and left/forward-right Standard tasks.

| Stratum | Success |
| --- | ---: |
| overall | 0/48 (0%) |
| cube | 0/24 (0%) |
| cylinder | 0/24 (0%) |
| left | 0/24 (0%) |
| forward-right | 0/24 (0%) |

Outcomes were 14 timeouts, 22 wrong-object displacements, and 12 target-workspace exits. Average
episode length was 113.792 steps and average deterministic episode latency was 3.566 seconds.

The 5,462 executed actions were finite, within exact native bounds, and nonconstant on every
reported component set. There were zero clipping, projection, invalid-action, out-of-bound-action,
grasp, lift, topple, and robot-collision events. The exact action standard deviations were
`[0.5708, 0.3489, 2.1697, 0.6171, 1.3285, 0.5712, 2.0964, 0.0152]`.

## Gate and hard stop

The episode-count requirement passed. Overall, cube, cylinder, direction coverage, and
zero-tolerance safety requirements failed. The micro gate is therefore false.

The broader 60-episode feasibility-development distribution did not run. Its seeds, all future
full-development seeds, and formal qualification seeds `66300..66399` remain unaccessed. No
inference from micro failure is made about broader performance.
