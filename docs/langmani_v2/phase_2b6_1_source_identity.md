# Phase 2B.6.1 source identity

The exact official target was proven before any simulator execution:

| Field | Value |
| --- | --- |
| Task | `StackCube-v1` |
| Source episode | `938` |
| Episode seed | `962` |
| Trajectory key | `traj_938` |
| Reset kwargs | `{"options": {}, "seed": 962}` |
| Reset identity | `sha256:e396b1aea2fe017ae24ab29115f5e3702fa1a37d6d34f0f9d0629f188f44410b` |
| Action contract | native `float32[105,8] pd_joint_pos` |
| Action SHA-256 | `sha256:5fc50bc7beeb91e55b2eeccf3f016414811d314142e7a80ad6037d4f83d92eb8` |
| Environment states | `106` (`T+1`) |
| Source trajectory identity | `sha256:f5c25c7b4f1f7f5ed0cd8b3641f7a451f758c1a47eb9aa31bf96b0081848f7a3` |
| Derived identity reserved by Phase 2B.6 | `sha256:1eefa0f33e0763a63960047ef7ee5b343fd4b84f355382d486b5918953de38d1` |
| Source terminal success | `true` |

The action count, dtype, shape, hash, reset identity, trajectory identity, metadata elapsed length,
and frozen source inventory all agree. The action hash is the same identity bound into the failed
producer input. Thus `PRODUCER_INPUT_IDENTITY_UNPROVEN` does not apply.

The official HDF5 success array first becomes true at action index 99 (step 100), remains true for
six trailing steps, and is true at the final source step. These are source metadata facts, not a
fresh replay result.

The accepted controls were also byte-bound:

| Episode | Seed | Actions | Action SHA-256 | Historical accepted replay |
| --- | ---: | ---: | --- | --- |
| 936 | 960 | 82 | `sha256:5df33ca1f1542e2773797aea9e6c0c1e8bda8a100404a608e055d1fe3369e406` | observed |
| 937 | 961 | 92 | `sha256:4ebd85a8a0d68db04af42b29e609795bc942605052fc6cf8c40066a3a813da3d` | observed |

The official StackCube ZIP remained
`sha256:f9b7d34b9aa418a04aa8e4322d4dea5aa27e8ae81757f60b210c1ffc54bf9c1b`.
The immutable Phase 2B.6 artifact manifest and all 21 files rehashed successfully.
