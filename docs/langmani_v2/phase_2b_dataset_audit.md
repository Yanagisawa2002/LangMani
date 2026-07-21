# LangMani 2.0 Phase 2B dataset audit

## Decision

Phase 2B produced a real ManiSkill-native `push_to_region` archive with every attempted trajectory
retained, independently action-replayed every generation-accepted trajectory, and exported only
replay-accepted demonstrations through the real LeRobot 0.6 API. The final Phase 2C authorization
decision is **Phase 2C authorized**. The independent final verifier passed every tracked-manifest,
collection, replay, export, split, readback, runtime-lineage, and cross-dataset compatibility gate.
This decision authorizes only the next separately invoked data consumer; no SmolVLA model was
implemented, loaded, or trained in Phase 2B.

## Frozen producer and runtime

- Branch: `codex/langmani-v2-phase2b-data`
- Accepted expert: `PushToRegionExpert/CandidateE`
- Expert implementation ancestor: `59ca88e9f0514187252a6286ab1b8e06c4318fb4`
- Collection producer: `8dc9f4e24598d4a0266d8afe15f50603eae14515`
- Collection fingerprint:
  `sha256:bb274033a2f4922941e466aadae8eaf648c65573d477e6c59219c5e8db9faa95`
- Environment/control: `LangMani-PushToRegion-v0`, Panda, `pd_joint_pos`, 100 Hz simulation,
  20 Hz control, 250-step episode bound
- Runtime: Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8, ManiSkill 3.0.1, NumPy 1.26.4,
  mplib 0.1.1, LeRobot 0.6.0, one NVIDIA GeForce RTX 5090

The first pilot under the superseded fingerprint
`sha256:cacc0abc38c66c983bcbe7ca0b8e278870220a454bbbdc6cd06a5563060a16f1`
is retained as invalid infrastructure evidence. It reused one environment within a native shard
and compared batched state hashes against unbatched slices. No record from that run enters the
accepted dataset. The corrected collection constructs and closes a fresh environment per attempt.

## Frozen collection matrix

The source configuration is `configs/langmani_v2/phase2b_push_collection.yaml`. It uses two
geometries (`blue_cube`, `orange_cylinder`), four regions (`left`, `right`, `forward_left`,
`forward_right`), both `standard` and `hard` difficulty, deterministic unique scene seeds, and
training/validation/held-out-paraphrase template groups. Canonical `PushTaskSpec` metadata is stored
separately from rendered language.

The initial full schedule was exactly 300 standard plus 120 hard attempts. Before top-up, it
produced 329 generation-accepted and 91 generation-rejected trajectories:

| Stratum | Attempted | Generation accepted |
| --- | ---: | ---: |
| Cube | 212 | 173 |
| Horizontal cylinder | 208 | 156 |
| Standard | 300 | 245 |
| Hard | 120 | 84 |
| Left | 105 | 76 |
| Right | 105 | 89 |
| Forward-left | 105 | 79 |
| Forward-right | 105 | 85 |

Because 329 was below the frozen preferred target of 360, one bounded 96-attempt top-up was
executed. It added 68 generation-accepted candidates and preserved 28 failures, using new
identities and seeds without relaxing success, action, schema, or replay requirements. A first
pre-access top-up launch at Git `f58b6b3` was retained as invalid infrastructure evidence: it tried
to reuse the immutable full-run runtime manifest. Git `f081275` gave top-up its own content-bound
runtime manifest while preserving the full producer identity.

## Pilot

The corrected 32-attempt pilot covered both geometries, all regions and difficulties, distractor
scenes, and all three language-template groups. Generation accepted 22 and rejected 10. All 22
accepted trajectories passed independent action replay, categorical outcome and frame-count
checks. The real LeRobot export contained 22 episodes and 2,914 frames, loaded through
`LeRobotDataset`, and occupied 7,773,981 bytes.

The pilot replay was produced before the final report schema added an aggregate per-transition
label field. Its T/T+1 native time contract and generation labels passed, but the immutable old
replay report does not claim the later aggregate field. The full authoritative replay below uses
the enhanced implementation and compares every `terminated`, `truncated`, `success`, and `fail`
label at all T transitions.

## Attempt retention and failure corpus

The final source contains 516 attempts: 397 accepted after replay and 119
rejected. Every attempt keeps its semantic identity, reset seed, task, instruction/template,
expert result and phase trace, action/event contract, native HDF5/JSON paths, state hashes,
terminal evaluation, and failure classification. Rejected attempts remain in a separate failure
corpus and never enter the imitation-learning export.

Final attempt and acceptance counts:

| Stratum | Attempted | Accepted | Rejected |
| --- | ---: | ---: | ---: |
| Cube | 260 | 208 | 52 |
| Horizontal cylinder | 256 | 189 | 67 |
| Standard | 348 | 283 | 65 |
| Hard | 168 | 114 | 54 |
| Left | 129 | 90 | 39 |
| Right | 129 | 107 | 22 |
| Forward-left | 129 | 100 | 29 |
| Forward-right | 129 | 100 | 29 |
| Training templates | 360 | 280 | 80 |
| Validation templates | 104 | 77 | 27 |
| Held-out paraphrases | 52 | 40 | 12 |

Failure categories and generation/replay rejection reasons:

| Root failure category | Count |
| --- | ---: |
| Verification failure | 49 |
| Planning failure | 29 |
| Timeout | 19 |
| Wrong-object interaction | 18 |
| Target outside workspace | 2 |
| Contact failure | 1 |
| Execution failure | 1 |
| Action-replay failure | 0 |

Generation acceptance reasons are multi-label: 119 non-successful expert results, 119 canonical
success failures, 97 invalid time contracts caused by failed rollouts, 18 wrong-object
displacements, three environment-fail terminals, and two workspace exits. They are not summed as
independent episodes.

Episode-length and correction-push distributions:

| Accepted episode property | Count |
| --- | ---: |
| Length 100--149 | 377 |
| Length 150--199 | 5 |
| Length 200--250 | 15 |
| Zero corrective pushes | 381 |
| One corrective push | 16 |

The rejected corpus additionally contains 36 episodes of length 0--49, 11 of length 50--99, 47 of
length 100--149, and 25 of length 150 or more. It retains 99 zero-correction and 20 one-correction
attempts.

No accepted action was silently clipped, normalized into another control mode, or augmented with a
terminal no-op. Raw action T, transition labels T, and native states T+1 remain explicit.

## Independent action replay

The final replay ran in fresh environments and never constructed or invoked the expert. It executed
all recorded actions, including the final recorded action, and recomputed the environment outcome.

| Check | Result |
| --- | ---: |
| Generation-accepted candidates replayed | 397 / 397 |
| Accepted after replay | 397 |
| Categorical outcome match | 397/397 (100%) |
| Per-transition label match | 397/397 (100%) |
| Frame-count match | 397/397 (100%) |
| Terminal position error | max/mean/p95 = 0/0/0 m |
| Terminal orientation error | max 5.96e-8 rad; mean 1.10e-8 rad; p95 4.21e-8 rad |

State replay was not used as an acceptance substitute. A generation success enters the export only
after real action replay passes.

## LeRobot export

The derived export is stored outside Git at
`outputs/datasets/langmani_v2/phase2b-push-lerobot-v1`. It uses LeRobot 0.6.0's v3 layout and the
following policy feature contract:

| Field | Type and shape | Source |
| --- | --- | --- |
| `observation.images.base_camera` | `uint8 [3,256,256]` | state-restored RGB at state[t] |
| `observation.state` | `float32 [9]` | `PandaPolicyStateV0` |
| `action` | `float32 [8]` | original `pd_joint_pos` action[t] |
| `task` | string | rendered push instruction |

Canonical task IDs, task parameters, scene group, seed, template group, and native trajectory hash
remain in sidecars. Privileged simulator state is not a policy feature. RGB is reconstructed from
the stored state before action[t] without calling `env.step`, preserving state/action alignment.

The final export contains 397 episodes, 53,297 frames, and 141,422,866 bytes. Its streaming
image/state/action statistics are stored in
`langmani/dataset_statistics.json`. Every split was reloaded through the real `LeRobotDataset` API,
and at least one item per split matched the declared feature contract.

## Stable splits and leakage audit

| Split | Episodes | Frames |
| --- | ---: | ---: |
| Train | 203 | 26,968 |
| Validation | 77 | 10,425 |
| Test unseen scene | 26 | 3,382 |
| Test unseen language | 40 | 5,277 |
| Test hard | 23 | 3,210 |
| Test visual shift | 28 | 4,035 |

Splits are assigned at stable group level, never frame level. The independent audit checks duplicate
episode IDs, trajectory hashes, initial-state hashes, seeds, scene groups, held-out template-group
placement, and cross-split files. Result: **passed**, with zero duplicate episode IDs, seeds,
trajectory hashes, initial-state hashes, scene overlaps, held-out-language overlaps, source-shard
cross-references, or exported-file overlaps.

## Historical pick-and-place compatibility

The original 360-episode/64,548-frame M3B pick-and-place dataset remains unchanged. Its passing full
validation report and statistics are SHA-256-bound into the compatibility report. The audit loaded
both datasets through the real LeRobot API and observed directly compatible `uint8 [3,256,256]`
RGB, `float32 [9]` Panda state, `float32 [8]` `pd_joint_pos` actions, 20 Hz timing, and task strings.
Skill-qualified TaskSpec sidecars and future train-only multi-skill normalization require only
deterministic view construction; privileged native states remain excluded.

Compatibility result: **passed**. Unified status: **immutable multi-root index ready**. The unified
artifact is an immutable multi-root index; neither source dataset is copied, concatenated, or
rewritten.

## Source and evidence validation

- Ruff format/check: passed across 310 files.
- CPU-safe suite: 1,474 passed, 15 skipped, 16 deselected.
- Dataset-specific suite: 22 passed.
- Push-specific suite: 50 passed.
- Server-side Phase 2B/push contract suite: 72 passed.
- Isolated sdist and wheel build: passed.
- Corrected physical pilot, LeRobot readback, compatibility audit, full replay/export, and final
  independent verifier: passed. The final report set `pipeline_integrity_validated=true`,
  `full_collection_validated=true`, `physical_target_validated=true`, and
  `smolvla_phase2c_authorized=true` with every admission gate true.

Critical content identities are recorded in `phase_2b_result_manifest.json`; exact commands and
their validation status are recorded in `phase_2b_source_validation.json`. Generated trajectories,
videos, datasets, logs, and full raw reports remain under `outputs/` and are not source-controlled.

## Known weaknesses and next boundary

- The accepted expert still rejects real hard, cylinder, planning, verification, timeout, and
  wrong-object-interaction attempts; those outcomes are preserved rather than hidden.
- The pilot's first immutable replay schema predates aggregate transition-label reporting; the full
  production replay closes that reporting gap.
- The dataset is expert-generated and covers one pushing skill family; learned-policy quality is
  not established by dataset acceptance.
- Phase 2B does not implement, load, train, or evaluate SmolVLA.

The exact next stage is a separately invoked Phase 2C SmolVLA implementation/training milestone.
It must consume only the content-bound accepted dataset and requires a new explicit command;
Phase 2B never starts training automatically.
