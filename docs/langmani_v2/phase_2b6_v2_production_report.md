# LangMani 2.0 Phase 2B.6-v2 production report

Phase 2B.6-v2 completed a new source-episode-zero production run under the
pre-registered versioned exclusion policy. It did not resume the Phase 2B.6
partial root and did not reuse any of its 1,938 records. The terminal
classification is `RESULT_A`.

## Provenance and runtime

- Source branch:
  `codex/langmani-v2-phase2b6-1-v2-forensic-replay`
- Source commit: `18c31f8bc6839d80b465ad73894fa886fa6c8dfb`
- Producer branch:
  `codex/langmani-v2-phase2b6-v2-instrumented-production`
- Full replay producer commit:
  `36a50acd07b24257c90330087ed96d3250bce531`
- Final evidence/verification implementation commit:
  `8fc84141aaf137169917286f8ddd6243fc88bcae`
- Server: `autodl-container-676946bb52-28ec5706`
- GPU: NVIDIA GeForce RTX 5090,
  `GPU-27f8a10c-1b18-1109-85da-247ee1927024`, driver `580.76.05`
- Replay environment: `/root/autodl-tmp/conda-envs/langmani`
- LeRobot environment: Python 3.12.13, LeRobot 0.6.0, PyAV 15.1.0,
  Torch 2.11.0+cu128

The accepted launcher used
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json`,
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`,
`CUDA_VISIBLE_DEVICES=0`, and a fresh process-scoped `XDG_RUNTIME_DIR`.
`vulkaninfo`, the minimal SAPIEN probe, and zero-step construction for
PickCube, StackCube, and PushCube passed. The zero-step checks executed no
reset, step, or action.

## Immutable inputs

The official demonstration revision was
`d674485bbffdd533914e52d272fdda34c0515608`; the ManiSkill source revision was
`a4a4f9272ad64b1564035874b605ceb687b63ed8`.

| Task | Source episodes | Source transitions | Source ZIP SHA-256 |
| --- | ---: | ---: | --- |
| PickCube-v1 | 1,000 | 77,976 | `b2d4afb30fa309755862b98c342e6ee18918253c93f3bbac16ed6670748f26d8` |
| StackCube-v1 | 1,000 | 107,420 | `f9b7d34b9aa418a04aa8e4322d4dea5aa27e8ae81757f60b210c1ffc54bf9c1b` |
| PushCube-v1 | 1,000 | 68,978 | `1ce24408bf93658501faaa0ef164a0cdc29539cc412612a413fac746697eeab2` |
| Total | 3,000 | 254,374 | — |

All Phase 2B.5, 2B.6, 2B.6.1, 2B.6.1-R, and 2B.6.1-v2 compact artifact
manifests matched their frozen fingerprints and result classes. Their verified
file counts were 27, 21, 24, 19, and 29 respectively.

## Production outcome

| Task | Accepted | Excluded | Accepted rate | Accepted frames |
| --- | ---: | ---: | ---: | ---: |
| PickCube-v1 | 1,000 | 0 | 100.0% | 77,976 |
| StackCube-v1 | 999 | 1 | 99.9% | 107,315 |
| PushCube-v1 | 999 | 1 | 99.9% | 68,909 |
| Total | 2,998 | 2 | 99.9333% | 254,200 |

All 3,000 source identities have exactly one terminal classification:
2,998 `ACCEPTED_REPLAY` and 2 `EXCLUDED_DETERMINISTIC_PHYSICAL`. There were no
source-contract exclusions, infrastructure retries, or unclassified failures.
Every accepted episode had exact source/replay outcome agreement, exact
observation/action alignment, zero simulator errors, zero invalid actions, and
zero non-finite values.

### Explicit exclusions

`StackCube-v1` source episode 938:

- split: `test_unseen_reset`
- first failed sub-gate: `canonical_terminal_success_gate`
- exact source/replayed action count: 105/105
- exact action SHA-256:
  `5fc50bc7beeb91e55b2eeccf3f016414811d314142e7a80ad6037d4f83d92eb8`
- source success: true; replay final success: false
- transient canonical success indices: 100, 101, 102, 103
- simulator errors, invalid actions, and non-finite values: 0
- retry: false
- forensic comparison: consistent with immutable Phase 2B.6.1-v2 evidence

This is the preregistered deterministic physical incompatibility. The episode
was not shortened at step 103, repaired, replaced, or retried.

`PushCube-v1` source episode 202:

- split: `train`
- first failed sub-gate: `canonical_terminal_success_gate`
- exact source/replayed action count: 69/69
- exact action SHA-256:
  `436bcad43799b2f409799ec7196353b4b2eb307dc72441e1a00263290e4828df`
- source success: true; replay final success: false
- no canonical success index
- simulator errors, invalid actions, and non-finite values: 0
- retry: false

The explicit valid-environment run made this an unambiguous deterministic
physical failure. It was retained in the exclusion manifest and omitted from
all policy roots.

## Student data contract

- `observation.images.base_camera`: uint8 RGB, 256 x 256, 20 Hz, no crop
- `observation.state`: `PandaPolicyStateV0`, float32[9]
- `action`: native `pd_joint_pos`, float32[8], no clipping, interpolation, or
  projection
- timing: `observation[t]` immediately precedes `action[t]`
- no fabricated terminal action or terminal diagnostic frame
- deterministic split-controlled language, six train templates, two
  validation templates, and four held-out templates per task
- task, skill, source, derived episode, frame, timestamp, language-template,
  and split identities retained

Object/goal pose, full simulator state, reward, success internals, and contacts
are excluded from student-policy fields.

## Independent verification

The real LeRobot 0.6 readback loaded all 18 roots, 3,148 generated episodes,
and 267,041 generated frames (2,998/254,200 primary plus 150/12,841
visual-shift). It decoded every frame with zero video, image, state, action,
task/language, timestamp, or metadata failure.

The independent source-to-derived verifier checked all 2,998 accepted episodes
and 2,033,600 action scalar values with exact equality and no approximate
tolerance. Source accounting separately proved `3,000 = 2,998 + 2`, verified
both exclusion references, and proved both exclusions absent from policy
roots.

The authoritative compact evidence is under
`artifacts/langmani_v2/phase_2b6_v2/`.
