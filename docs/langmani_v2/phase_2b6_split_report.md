# Phase 2B.6 Split Report

## Status boundary

The deterministic source assignment was fully materialized before physical production. It passed
its identity-level overlap audit, but the later replay hard stop means these assignments are a
frozen candidate split manifest, not splits of an accepted derived dataset.

Primary split manifest fingerprint:
`sha256:4367e816bf90ba8ee468bd3a0ad5e8c703565bbb3b1dec4e0c674760f5db1100`.

## Primary assignments

| Split | Per task | Global |
| --- | ---: | ---: |
| train | 700 | 2,100 |
| validation | 100 | 300 |
| test_unseen_reset | 100 | 300 |
| test_unseen_task_language | 50 | 150 |
| test_visual_shift | 50 | 150 |
| total | 1,000 | 3,000 |

Assignment is deterministic over immutable source episode and task-local reset identities. The
official source exposes per-episode reset kwargs but no broader reset-family label, so no broader
reset-family generalization claim is available. The task-local identity prevents identical kwargs
from different environments being falsely treated as one physical reset.

The pre-conversion identity audit found zero overlap in derived episode identities, source
trajectory identities, task-local reset identities, action hashes, initial-state hashes, and
task/split media namespaces. A full physical-media leakage audit did not run because LeRobot media
was never created.

## Language

Language-template manifest fingerprint:
`sha256:5a0774b90847b4ea1a43c4c603d9f3cbd283622f47cb1733880d6356fc457997`.

Each task has six train templates, two validation templates, and four held-out templates.
Assignment is deterministic from source identity and split. Held-out templates do not appear in
the training bank, and trajectories are not duplicated merely to attach another instruction.

## Visual shift

The candidate `test_visual_shift` source identities are disjoint from train and validation.
Only the three-episode physical-equivalence pilot was produced. It passed with no action,
physics, reset, geometry, or timing change. The planned 150 shifted episodes were not
materialized.

## Cross-skill views

Cross-skill manifest fingerprint:
`sha256:fdc436d4bfd70b39e708a5220a8892f4542a3a7f7c78729fb3f0b6a627687d89`.

| Fold | Candidate train skills | Held-out skill | Train refs | Test refs |
| --- | --- | --- | ---: | ---: |
| fold_holdout_pick | stacking, planar pushing | pick and place | 1,400 | 300 |
| fold_holdout_stack | pick and place, planar pushing | stacking | 1,400 | 300 |
| fold_holdout_push | pick and place, stacking | planar pushing | 1,400 | 300 |

These are metadata-only alternate views. They duplicate no media and prove no unseen-skill model
result. Because production failed, they are not currently eligible training views.
