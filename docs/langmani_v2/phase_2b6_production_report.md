# Phase 2B.6 Production Report

## Outcome

Phase 2B.6 closed as `RESULT_C`: production or replay validation failure. It did not create
`LangManiOfficialMultiSkill-v1` or any accepted training package.

The native producer ran from commit `73b2cd1517e7c3d2bfc08bd2898e46a3195eb1c9` on an RTX 5090
using ManiSkill 3.0.1, SAPIEN 3.0.3, Python 3.12.13, and the frozen official demonstration
revision `d674485bbffdd533914e52d272fdda34c0515608`.

## Preflight

- Phase 2B.5 package verification passed all nine checks over its 27-file evidence package.
- All three official ZIP hashes, sizes, selected members, and extracted bytes matched.
- Source schema validation passed for 3,000/3,000 successful trajectories and 254,374
  transitions.
- There were zero malformed trajectories, non-finite actions, action-bound violations, or
  within-task duplicate action hashes.
- Frozen specification SHA-256:
  `sha256:84021d42a33ad939cf4b707eaf75a8b71fcd6c083b88ee2a4aed0b7181e0b232`.
- The post-render visual-shift pilot passed 3/3 with categorical agreement 1.0 and no action or
  physics change.

## Full replay

| Task | Attempted | Accepted replay | Rejected | Accepted frames | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| PickCube-v1 | 1,000 | 1,000 | 0 | 77,976 | complete |
| StackCube-v1 | 939 | 938 | 1 | 100,657 | hard-stopped |
| PushCube-v1 | 0 | 0 | 0 | 0 | not started |
| Total | 1,939 | 1,938 | 1 | 178,633 | incomplete |

The first rejection was `StackCube-v1` source episode 938, source trajectory identity
`sha256:f5c25c7b4f1f7f5ed0cd8b3641f7a451f758c1a47eb9aa31bf96b0081848f7a3`, assigned to
`test_unseen_reset`. Source success was true. The physical replay executed once and returned a
failed replay gate. It was not an infrastructure failure, was not eligible for retry, and was not
retried.

The original producer wrapper replaced the returned detailed replay record with a generic
`episode replay gate failed` row before persistence. Therefore the exact failed inner subgate is
unavailable. The wrapper's recorded simulator-error count is not treated as proof of a simulator
exception. No missing value was reconstructed from inference. A later source fix preserves full
returned subgate diagnostics for future runs, but episode 938 was not rerun.

## Preserved locations

- Partial production root:
  `/root/autodl-tmp/langmani-external/phase2b6/production_v1`
- Partial production bytes at closeout: 20,770,745,988 bytes.
- Accepted raw replay NPZ files: 1,938.
- Compact evidence root:
  `/root/autodl-tmp/langmani-external/phase2b6/evidence`
- Git-tracked compact copy:
  `artifacts/langmani_v2/phase_2b6/`

These bytes are diagnostic partial output. They are not a primary accepted dataset and cannot be
used as an implicit failure-training corpus.

## Stages not started

The hard stop prevented task-specific LeRobot conversion, full visual-shift materialization,
unified multi-root creation, train-only normalization, full LeRobot readback, full
source-to-derived verification, media leakage audit, archive creation, and restore validation.
No model, checkpoint, optimizer, backward pass, LoRA job, or learned-policy evaluation started.
