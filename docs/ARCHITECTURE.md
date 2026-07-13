# Architecture

This document defines current and future ownership boundaries. M0 and M1 established the runtime
foundation and environment boundary, M2 added the privileged expert, M3A implemented the
authoritative ManiSkill-native raw archive, and M3B added deterministic LeRobotDataset v3
derivation and validation. Active M4 owns three reproducible ACT controls, their checkpoints, and
closed-loop evaluation. The root package stays lightweight, and importing
`langmani.environments` is still the explicit registration boundary.

## Dependency direction

The dependency flow is deliberately one-way:

```text
diagnostic commands -> collection / environments / experts / datasets / policies
collection -> environments / experts / datasets
experts -> environments
datasets -> M3A authority plus one-way M3B derived representation
policies -> completed M3B schemas plus installed LeRobot policy interfaces
policy evaluation -> environments and policy interfaces
```

Shared schemas should live at the narrowest neutral boundary. ManiSkill-specific objects must not
leak into stored dataset records or policy APIs.

## Environments

`langmani.environments` owns the one M1 ManiSkill task registration, typed task/episode metadata,
deterministic reset configuration, privileged-observation gating, and batched task evaluation. Its
current structure is:

```text
langmani.environments
├── specs.py                       # frozen specs with JSON-ready dict representations
├── task_logic.py                  # ManiSkill-independent batched Torch logic
├── expert_state.py                # privileged single-environment runtime handles
└── pick_place_by_instruction.py   # scene, reset, cameras, observations, evaluation
```

`TaskSpec` selects semantic object/bin/template IDs; `EpisodeSpec` combines that selection with a
scene seed and stable identifiers. Scene IDs never depend on task selection, and task IDs never
depend on physical layout. Explicit metadata accessors return immutable Python values only when a
caller asks for them. The simulator's per-step observation and evaluation paths retain numeric
batched tensors.

The environment module is the only layer allowed to own ManiSkill `Actor`, `Pose`, robot, sensor, or
scene objects. The pure tensor helper accepts ordinary Torch tensors so success/no-leak logic can be
reviewed without launching SAPIEN. M2 adds `get_expert_task_context()` and
`get_expert_evaluation()` as an explicit exception for the privileged demonstration generator.
Those accessors require `num_envs=1`, bind semantic IDs to actors and bin geometry, and are never
called while constructing policy observations or per-step `info`.

M3A adds `get_expert_initial_scene_state()` to that same privileged boundary. The collector calls it
only after reset and before expert execution to materialize actual cube poses, static-bin poses, and
Panda qpos. The explicit bin values are necessary because ManiSkill's native simulator-state tree
omits static actors; the accessor still exposes nothing through policy observations.

The environment still does not own planning, expert phase behavior, dataset persistence, training,
checkpoint selection, or policy metrics. M4 adds only a narrow policy-rollout evaluation accessor;
it does not change observations or expose privileged values.

## Experts

`langmani.experts` owns the active M2 task-solving interfaces, deterministic Panda planning adapter,
phase execution, and compact expert results:

```text
langmani.experts
├── types.py            # ExpertConfig/Status/Phase, PhaseResult, ExpertResult
├── planner.py          # lazy, direct mplib 0.1.1 Panda adapter
├── pick_place.py       # explicit 12-phase PickPlaceExpert orchestration
└── command_support.py  # output-confined diagnostic helpers
```

The expert is a privileged demonstration generator, not a deployable vision policy. It validates
the environment ID, `num_envs=1`, `pd_joint_pos`, active `TaskSpec`, semantic actor/bin handles,
robot handles, and initial evaluation before planning. It consumes only the explicit expert-only
environment boundary and emits actions plus compact provenance. It does not write LeRobot datasets
or expose its inputs through policy observations.

The planner adapter imports mplib lazily and directly wraps public mplib 0.1.1 APIs. It does not
depend on ManiSkill example runners or copy an upstream solver. Fixed-state `plan_screw` is the only
planner in M2: mplib 0.1.1 offers no planner seed and no screw-planner timeout, so configuration
must leave those capabilities unset and no unseeded RRT fallback exists. The adapter synchronizes
the Panda base pose and nine-joint simulator state, plans seven arm joints, and keeps gripper control
in the expert's `pd_joint_pos` action path.

The stable phase sequence is `initialize`, `move_to_pregrasp`, `approach_target`, `close_gripper`,
`verify_grasp`, `lift_target`, `move_above_destination`, `descend_to_place`, `open_gripper`,
`settle_after_release`, `retreat`, and `verify_task`. Each phase owns entry conditions, command or
target pose, bounded attempts, completion checks, status classification, and timing/accounting.
`ExpertResult` contains compact JSON-ready metadata and summaries only; full planned paths,
observations, frames, tensors, and trajectories remain runtime values.

Grasp and placement generation are deterministic. The top grasp uses world approach `(0, 0, -1)`
and closing `(0, -1, 0)`. Under ManiSkill's Panda convention the TCP orientation columns are
`[closing × approach, closing, approach] = diag(1, -1, -1)`, matching the installed table-scene
Panda's initial TCP orientation, and mplib pose vectors use
`[x, y, z, qw, qx, qy, qz]`. Placement uses the selected bin interior center, bin-floor height,
cube half extent, configured vertical clearance, and the M1 wall-safety margin.

## Collection

`langmani.collection` owns M3A's simulator-facing orchestration:

```text
langmani.collection
├── recording.py       # narrow RecordEpisode 3.0.1 integration
├── replay.py          # single-env action replay and optional state audit
├── collector.py       # retries, all-six admission, sharding, and resume
├── manifest.py        # atomic manifests and JSONL projections
├── inspection.py      # cross-shard corruption/provenance inspection
└── command_support.py # output confinement and atomic command reports
```

The collector runs the existing M2 expert; it does not duplicate phase logic. Candidate scene
groups are processed sequentially. Full failed trajectories are optional and remain under a
separate failure area, while every small attempt result, including the unmodified M2 `ExpertResult`,
is always in the manifest. A successful result reaches replay only after schedule/reset identity,
object/bin semantics, fresh final environment evaluation, false-success labels, and configured
bounds agree. The collection layer never exposes privileged state through policy observations.

Resume is a transaction protocol, not HDF5 append. A per-candidate journal records a started attempt
before expert execution and its compact result afterward. Group bundles journal an open partial
shard. The canonical manifest is the generation commit marker and is written after its deterministic
projections; staging, in-flight journals, and sealed-shard bundles are cleaned only after that commit.
Resume explicitly rejects an interrupted candidate, repairs only unsealed shards, and idempotently
finishes any committed failure-retention move.

## Datasets

`langmani.datasets` owns environment-neutral M3A schemas and archive operations plus the one-way
M3B derived-data boundary:

```text
langmani.datasets
├── types.py      # frozen collection, attempt, replay, group, shard, and summary types
├── identity.py   # canonical JSON and SHA-256 stable IDs
├── schedule.py   # seed-major/object-major six-task schedule
└── archive.py    # strict native HDF5/JSON validation, group bundles, and shards
```

The M3A native archive is authoritative. Source shards retain actions, T+1 environment states,
termination/success labels, exact reset kwargs, semantic metadata, expert/replay provenance, and
checksums. They also retain the explicit initial cube/bin/Panda snapshot; dynamic components are
bound to HDF5 state zero and the complete snapshot is compared across every six-task group. The
fixed recorder schema uses an empty observation group and no reward dataset. Archive inspection
recomputes persistent IDs and cross-checks native episode metadata, manifest records, schedule,
expert results, replay mode/tolerances, group membership, and source-shard identities.
M3B never treats an HDF5 key as acceptance. It first runs the complete M3A inspector and consumes a
content-bound target report, then reads accepted manifest episodes in group/canonical-task order.
Its project-owned modules provide the source gate, export types/fingerprint, deterministic
scene-group splits, PandaPolicyStateV0 extraction, public state-restoration observation path,
LeRobot lifecycle adapter, export orchestration, and independent local validation.

M3B restores state[t] in a fresh M1 RGB environment and writes only base-camera RGB, nine Panda
qpos values, the exact action, and LeRobot task metadata. The LeRobot writer is isolated because
upstream `save_episode` is not transactional and upstream `finalize` does not block later writes.
Fingerprint-owned staging is restarted, never resumed; independent validation precedes atomic
promotion and the completion marker is written last. M3A remains authoritative and M3B never
changes its acceptance. Dataset code does not choose actions or train policies.

M4 consumes only the completed M3B marker, typed manifest/summary/validation report, local
Parquet/video files, and sidecars. Its normal training gate never reopens M3A or recomputes M3B
acceptance. It derives global and per-task views from validated scene-group splits and recomputes
normalization statistics from the selected train frames only; M3B's whole-dataset `meta.stats` is
not a training input.

## Policies

`langmani.policies` owns M4's project-level ACT boundary:

```text
langmani.policies
├── act_types.py          # frozen configs, run/checkpoint/result contracts
├── act_data.py           # completed-M3B gate, episode views, train-only statistics
├── act_conditioning.py   # CanonicalTaskOneHotV0 shared by train and inference
├── act_training.py       # public ACT/processors and bounded optimization loop
├── act_checkpoint.py     # staged local save/reload and resume compatibility
├── act_rollout.py        # ACT queue to single M1 pd_joint_pos environment
├── act_evaluation.py     # schedules, validation ranking, test lock, summaries
├── act_analysis.py       # counterfactual action sensitivity
└── act_runtime.py        # Git/runtime identity and safe immutable output helpers
```

The package wraps installed LeRobot 0.6.0 public interfaces rather than copying ACT. It implements
exactly `per_task`, `mixed_unconditioned`, and `mixed_task_onehot`; it is not a generic future-policy
framework. The first two use image plus 9D Panda state. The third appends a six-way oracle task
condition in memory for a 15D state and does not rewrite M3B. Task text remains metadata and is not
an ACT tensor input. Policies do not define task physics, accept privileged state, invoke M2 during
rollout, upload to Hub, or mutate source datasets.

## Evaluation

M4 keeps its algorithm-specific evaluation in `langmani.policies.act_evaluation` and
`act_rollout`; no generic `langmani.evaluation` package is introduced. The boundary owns exact
scene schedules, Wilson intervals, validation-only checkpoint selection, immutable test
authorization, fresh-seed exclusion, and compact rollout/comparison records. It distinguishes
implementation, fixture training, CUDA training, model quality, and physical target acceptance.
It does not fabricate missing runs, treat offline loss as task success, or use test results to tune
or resume training.

## CLI

M2 uses thin scripts under `environment/`: `run_expert.py` for one rollout,
`benchmark_expert.py` for the explicit six-task matrix, and `verify_m2.py` for contract and target
acceptance. They validate configuration, call expert APIs, preserve unexpected exception type and
message at the outer boundary, surface failures, and write only to ignored diagnostic output paths.
The strict M2 verifier invokes the M0 installation and M1 environment target gates before starting
M2 physical rollouts. It keeps the original repeatable six-task smoke and separately applies the
statistical acceptance policy to a seed-major 180-episode matrix with 30 episodes per TaskSpec;
that verifier policy does not alter expert phase behavior or M1 success geometry. Business logic
remains in `langmani.experts`. A future general `langmani.cli` may replace these milestone commands
without changing subsystem ownership.

M3A adds `collect_raw_demos.py`, `inspect_raw_demos.py`, `replay_raw_demos.py`, and
`verify_m3a.py`. Both target modes first run `verify_m2.py --target`, whose own gate runs M0 and M1
first. `--target-smoke` creates one fresh complete scene group and independently replays its six
episodes. The separate `--target-full` mode validates an existing complete archive, resumes only a
compatible in-progress archive, or creates a missing run only with `--create-new-run`; it never
passes overwrite and never recollects an existing complete 60-group archive. Full mode inspects all
checksums and schemas and independently replays all 360 accepted episodes. Stale command reports are
invalidated before execution; final replay evidence must match the ordered manifest identities and
strictly validate action plus state audit for every episode. These commands are sequential and
introduce no multiprocessing.

M3B adds `scripts/export_lerobot_dataset.py`, `scripts/validate_lerobot_dataset.py`,
`scripts/inspect_lerobot_episode.py`, and `environment/verify_m3b.py`. Export dry-run validates the
content-bound source and prints ordering, split, fingerprint, frame, and feature expectations
without creating Parquet or video. Target smoke runs the ordered prior gates before one real
six-episode export; target full consumes the real 60-group archive and may independently validate
an existing immutable completed output instead of rebuilding it. Fixture video checks remain
separate from physical flags. These commands do not upload, train, or parallelize.

M4 adds `scripts/train_act.py`, `scripts/evaluate_act.py`,
`scripts/compare_act_baselines.py`, `scripts/inspect_act_checkpoint.py`, and
`environment/verify_m4.py`. Training owns the completed-data/Git gates, explicit variant, train-only
statistics, stable run directory, bounded optimization, and atomic checkpoints. Evaluation resets
the ACT and processor queues at every episode, never clips invalid actions, and never invokes the
expert. Full-mode test access requires the immutable validation-selected checkpoint and predeclared
schedule. Target smoke chains the prior target gates before real CUDA/tiny-overfit/rollout work;
full target additionally runs all eight policies, locked test, and the fixed 180-episode fresh
benchmark. Fixture work cannot set physical or model-quality flags.

## Cross-cutting rules

- Project-owned interfaces use static typing and tests.
- Generated datasets, assets, videos, checkpoints, caches, and results stay outside the package.
- Third-party APIs are wrapped at boundaries rather than copied or patched in-tree.
- Dependency and interface changes require an entry in `docs/DECISIONS.md`.
- Every M4 run records the full Git commit; a dirty tree is development-only and never final.
- M4 train statistics and checkpoint selection may use train/validation only; test remains locked.
- Simulator, CUDA, Vulkan, or rendering failures remain visible and cause strict verification to
  fail.
