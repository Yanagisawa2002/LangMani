# Phase 2B.2 dataset status report

Phase 2B.2 remains blocked at the mandatory expert gate. No formal v2 episode collection has
started, so
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
| Acceptance status | `INCOMPLETE` / expert repair in progress |

The first stricter v2 expert gate reached 75/100, and the next complete formal gate reached 11/100.
Eight later candidates were screened on disjoint diagnostic-only seeds; none justified consuming
the still-untouched 66300--66399 formal range. Candidate L stopped after four timeouts because its
best possible 64-episode result fell to 60/64 (93.75%). Candidate M then stopped on a workspace
exit after 3/5 successes. Candidate N preserved 4/4 cube successes but its first cylinder task also
left the target workspace, so the zero-tolerance stop rejected it after 5/64 scheduled episodes.
The required `expert_success_rate >= 95%` condition is therefore not yet satisfied. Candidate O is
locally validated and bound for a disjoint diagnostic-only probe; it cannot open collection unless
a subsequent complete formal 100-episode gate also passes every zero-tolerance criterion.

This report deliberately does not fabricate dataset manifests with zero episodes and does not mark
an incomplete directory `ACCEPTED`. Collection, export, archive, replication, SmolVLA loading,
training, and Phase 2D remain closed while Candidate O is screened.
