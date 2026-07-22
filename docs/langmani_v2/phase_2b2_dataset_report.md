# Phase 2B.2 dataset report - Result B

Phase 2B.2 stopped at the mandatory expert gate. No formal v2 episode collection was started, so
there is no raw archive, LeRobot export, split manifest, accepted dataset package, content-addressed
archive, restored replica, or SmolVLA loader claim.

| Metric | Result |
| --- | ---: |
| Formal collection attempts | 0 |
| Accepted successful episodes | 0 |
| Failed collection episodes | 0 |
| Exported frames | 0 |
| Exported bytes | 0 |
| Optimizer steps | 0 |
| Acceptance status | `INCOMPLETE` / Result B |

The first stricter v2 expert gate reached 75/100, and the next complete formal gate reached 11/100.
Six later candidates were screened on disjoint diagnostic-only seeds; none justified consuming the
still-untouched 66300--66399 formal range. Candidate K stopped on a zero-tolerance workspace exit.
The final Candidate L stopped after four timeouts because its best possible 64-episode result fell
to 60/64 (93.75%). The required `expert_success_rate >= 95%` condition therefore fails.

This report deliberately does not fabricate dataset manifests with zero episodes and does not mark
an incomplete directory `ACCEPTED`. The next authorized work is a separately specified expert
architecture milestone. Collection, export, archive, replication, SmolVLA loading, training, and
Phase 2D remain closed.
