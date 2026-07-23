# Phase 2B.5 Official Source Audit

This report is finalized from immutable source hashes and the native runtime reports under
`artifacts/langmani_v2/phase_2b5/`.

The required candidates are `PickCube-v1`, `StackCube-v1`, `PushCube-v1`, `PokeCube-v1`, and
`PullCube-v1`. The optional insertion/orientation candidates remain registry-only because the
minimum coherent direct-action set can be decided without downloading them.

The final task-by-task byte counts, HDF5 statistics, robots, cameras, reset diversity, action
modes, source types, and compatibility decisions are recorded in `task_candidate_audit.json`,
`source_download_manifest.json`, and `source_schema_statistics.json`. No official source is copied
into either historical custom-dataset identity.

| Task | Decision | Skill family | Official episodes | Transitions | Length min/median/max |
| --- | --- | --- | ---: | ---: | --- |
| `PickCube-v1` | selected, direct | pick and place | 1,000/1,000 successful | 77,976 | 49 / 78 / 103 |
| `StackCube-v1` | selected, direct | stacking | 1,000/1,000 successful | 107,420 | 80 / 106 / 412 |
| `PushCube-v1` | selected, direct | planar pushing | 1,000/1,000 successful | 68,978 | 61 / 69 / 81 |
| `PokeCube-v1` | not selected | tool-mediated poking | delta-action sources only | n/a | n/a |
| `PullCube-v1` | not selected | planar pulling | delta-action sources only | n/a | n/a |

Across the three selected sources, all 3,000 trajectories and 254,374 transitions passed schema,
finite-value, native-bound, terminal-metadata, and `T`/`T+1` alignment checks. There were zero
malformed trajectories, zero failed source trajectories, zero duplicate action-trajectory hashes,
and zero action-bound violations. This audit does not claim that all 3,000 trajectories were
physically replayed.
