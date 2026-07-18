# Architecture

This document defines current and future ownership boundaries. M0 and M1 established the runtime
foundation and environment boundary, M2 added the privileged expert, M3A implemented the
authoritative ManiSkill-native raw archive, and M3B added deterministic LeRobotDataset v3
derivation and validation. M4/M4.1 own three reproducible ACT controls, their checkpoints,
closed-loop evaluation, and the explicit action-bound runtime. Completed M4.2 development adds only
committed seed locks, runtime/post-grasp analysis, and one rejected oracle TaskToken ACT. Completed
M4.3a adds a zero-training semantic-audit and immutable evidence boundary. M4.3b added one
FactorFiLM architecture, completed its target-development evidence, and failed its quality gate;
the shared-ACT search is closed. M5A adds a separate `langmani.language` subsystem for a derived
family-split command corpus, three language-to-TaskSpec routers, strict rejection, an immutable
six-PerTask controller registry, and routing-versus-control attribution. It does not change M1,
M3A/M3B, or frozen policy bytes. The root package stays lightweight, and importing
`langmani.environments` remains the explicit task-registration boundary.

## Dependency direction

The dependency flow is deliberately one-way:

```text
diagnostic commands -> collection / environments / experts / datasets / policies
collection -> environments / experts / datasets
experts -> environments
datasets -> M3A authority plus one-way M3B derived representation
policies -> completed M3B schemas plus installed LeRobot policy interfaces
policy evaluation -> environments and policy interfaces
M4.2 -> completed M4 evidence plus completed M3B schemas
M4.3a -> completed M4/M4.2 evidence plus completed M3B validation schemas
M4.3b -> immutable M4.3a evidence plus completed M3B train/validation schemas and installed ACT
M5A language -> M1 TaskSpec schemas plus project-owned corpus/router contracts and Transformers
M5A dispatch -> router decisions plus frozen M4 PerTask/M4.2 runtime evidence and M1 evaluation
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
checkpoint selection, or policy metrics. M4 adds only a narrow policy-rollout evaluation accessor.
The compact expert-only post-step diagnostic now includes all cube positions/orientations and
linear/angular velocities, bin centers, TCP position, and containment facts needed to classify a
newly executed first interaction. Those values are read only after action selection for evidence;
they are absent from policy observations and per-step `info`. This boundary does not change M1's
visual no-leakage contract or expose privileged values as model inputs.

## Experts

`langmani.experts` owns the active M2 task-solving interfaces, deterministic Panda planning adapter,
phase execution, and compact expert results:

```text
langmani.experts
├── types.py            # ExpertConfig/Status/Phase, PhaseResult, ExpertResult
├── runtime.py          # explicit NumPy-1 planner-interpreter selection and pin probe
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

The main environment retains NumPy 2.2.6. `runtime.py` selects an explicit inherited virtual
environment that overlays the NumPy 1.26.4 ABI required by mplib 0.1.1 and validates the complete
effective planner-side module set. M0/M1 and M3A inspection stay in the main interpreter; planner
creation, expert commands, M3A collection, and action replay use the selected interpreter. These are
bounded, sequential command processes. The planner itself remains in-process with `num_envs=1`;
M2/M3A add neither planner multiprocessing nor vectorization.

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

M4.2 consumes the same completed M3B view and immutable M4 checkpoint/evaluation evidence. Its
committed schedule locks are policy-package resources, not a new dataset layer. M4.2 never writes
M3A/M3B, recalculates their admission, or uses historical M4 test/fresh evidence for selection.

M4.3a consumes the completed M3B validation view and the already-open `m42_dev_v0` identities. It
does not reopen M3A, modify M3B, open M3B test/M4 fresh, or materialize the sealed final schedule.
Its portable manifests retain stable source identities and fingerprints, not video frames, action
chunks, machine paths, or a second dataset representation.

M4.3b consumes the unchanged M3B train and validation episode views. It derives two condition
indices from stable TaskSpec metadata in memory and writes no dataset. Semantic-audit observations,
M3B test, M4 fresh, `m42_dev_v0`, and `m42_final_v0` are not training or checkpoint-selection data.

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
├── act_runtime.py        # Git/runtime identity and safe immutable output helpers
├── m42_types.py          # immutable stage/runtime/selection/final-gate contracts
├── m42_schedule.py       # joint leakage-safe development/final schedule locks
├── m42_schedules/        # committed exclusions and exact seed-list resources
├── m42_runtime.py        # execution horizon and explicit binary-gripper runtime
├── m42_analysis.py       # post-grasp phases, rankings, paired metrics, go/no-go
├── m42_evidence.py       # completed M4 evidence and immutable selection gates
├── m42_training.py       # TaskToken run identity, data mapping, and training orchestration
├── m42_evaluation.py     # validation ranking and development/final benchmarks
└── act_task_token.py     # public LeRobot ENV-token integration and API checks
```

M4.3a additionally owns `m43_types.py` for immutable contracts, `act_semantic_audit.py` for pure
ActionChunkDistanceV0/retrieval/confusion/taxonomy calculations, and `m43_evidence.py` for
fingerprint-owned staging, validation, and atomic promotion.

M4.3b adds narrow policy modules: `act_factor_film_types.py` for mappings and portable
architecture/run/checkpoint identities, `act_factor_film_conditioning.py` for stable TaskSpec-to-
index conversion shared by training and inference, `act_factor_film_adapter.py` for the isolated
LeRobot subclass/hooks, `act_factor_film_training.py` for fair data, fixture, resume, and training,
`act_factor_film_evaluation.py` for validation ranking/reload/development quality contracts, and
`act_factor_film_evidence.py` for fingerprint-owned post-training evidence. The separate
`act_factor_film_verification.py` re-derives acceptance from completed roots without mutation.
Evaluation code may be committed after the producer without changing the producer commit stored in
the run/checkpoints. The authorized lineages are architecture/structure `8ee0f1b...`, target
training `0088e293...`, and post-training evaluator/verifier implementation `1bacb66d...`.

The base M4 package wraps installed LeRobot 0.6.0 public interfaces rather than copying ACT. M4
implements exactly `per_task`, `mixed_unconditioned`, and `mixed_task_onehot`; it is not a generic
future-policy framework. The first two use image plus 9D Panda state. The third appends a six-way oracle task
condition in memory for a 15D state and does not rewrite M3B. Task text remains metadata and is not
an ACT tensor input. Policies do not define task physics, accept privileged state, invoke M2 during
rollout, upload to Hub, or mutate source datasets.

M4.2 remains a narrow extension rather than a generic policy framework. `ExecutionHorizonPolicyV0`
calls the installed public chunk-prediction path and owns an independent queue for exactly 1, 5, or
10 actions; reset clears both queues. `BinaryGripperEnvPostprocessorV0` is composed with the
existing M4.1 `project` processor and changes only component 7. It records raw, binary, projected,
and executed values separately and rejects malformed or nonfinite actions.

`ACT-Mixed-TaskToken` uses the installed LeRobot 0.6.0 public `FeatureType.ENV` feature at
`observation.environment_state`. The public `nn.Linear(6, dim_model)` environment projection maps
the canonical one-hot command to one learned hidden-dimensional vector, and ACT inserts it as a
dedicated encoder token between the 9D Panda-state token and image tokens. The Panda state remains
9D; the command is not appended to qpos. `NormalizationMode.IDENTITY` preserves the one-hot. A
project-owned contract validates the ENV feature, projection shape, and encoder-position count so
an upstream change fails explicitly. LangMani does not subclass, fork, patch, or copy ACT.

Historical checkpoint bytes and schemas remain unchanged. The shared atomic checkpoint lifecycle
accepts a narrow structural identity protocol so the independent M4.2 identity can reuse save and
reload without adding a fourth historical `ActVariant` or weakening any identity comparison.

M4.3a compares postprocessed environment-semantic action chunks before runtime action transforms.
Its primary retrieval metric is action-range-normalized arm-only L2 over the locked execution
horizon. Six-way task retrieval, object/bin centroid retrieval, deterministic confusions, first
interaction, and post-grasp classes remain separate evidence. The audit never treats nonzero action
distance as semantic correctness, steps an environment for offline chunk comparison, or supplies
privileged diagnostic values to a policy. Completed audit roots are immutable atomic promotions;
structural fixtures cannot claim real checkpoint inference or physical validation.

The real M4.3a audit completed over 108 observations and showed that State-OneHot and TaskToken
changed outputs without reliable requested semantics. M4.3b therefore adds exactly one
`FactorFiLMACTPolicy(ACTPolicy)`. Target-object indices address a 32D embedding and residual FiLM on
the ResNet-18 layer-4 `[B,512,8,8]` feature map before image projection. Destination-bin indices
address a separate 32D embedding and residual FiLM on the encoded `[B,dim_model]` 9D-state token
before Transformer processing. Neither condition enters the opposite path; no combined task token
or appended state component exists. Identity-like `N(0,1e-5)` projection weights plus zero bias add
exactly 67,744 parameters to the full base ACT.

LeRobot exposes no public extension point at those two representations. One isolated adapter uses
the public `PreTrainedConfig`, `ACTConfig`, `ACTPolicy`, and processor factory, then registers
instance-local hooks on the semi-stable `model.backbone` and
`model.encoder_robot_state_input_proj` outputs. It records and
validates LeRobot 0.6.0 signatures, model type/attributes, tensor shapes, and Transformer token
layout. It does not import/copy the private ACT implementation or monkey-patch global behavior, and
it fails closed rather than falling back to another policy.

## Evaluation

M4 keeps its algorithm-specific evaluation in `langmani.policies.act_evaluation` and
`act_rollout`; no generic `langmani.evaluation` package is introduced. The boundary owns exact
scene schedules, Wilson intervals, validation-only checkpoint selection, immutable test
authorization, fresh-seed exclusion, and compact rollout/comparison records. It distinguishes
implementation, fixture training, CUDA training, model quality, and physical target acceptance.
It does not fabricate missing runs, treat offline loss as task success, or use test results to tune
or resume training.

M4.2 evaluation adds a strict development/final boundary. The exact 12-scene `m42_dev_v0` and
30-scene `m42_final_v0` lists are jointly regenerated from canonical SHA-256 inputs after excluding
125 prior observed or predeclared seeds. A caller may validate the final lock without materializing
its episodes. Final materialization additionally requires explicit authorization, clean Git, a
matching implementation fingerprint, and immutable horizon, gripper, and TaskToken-checkpoint
selections. Post-grasp state is diagnostic only and cannot affect policy input or checkpoint
selection. Only M3B validation selects the TaskToken checkpoint.

M4.3a keeps validation and `m42_dev_v0` audit sections independently identified even when one
combined command produces both. It rejects test, historical fresh, and final identities before
policy loading. M4.3b reuses the M4 training lifecycle but only M3B validation may select among its
20 checkpoints: each receives the same 36 episodes and the fixed seven-key rank. A selected
checkpoint must reproduce one complete postprocessed `[50,8]` chunk in a fresh process at
`atol=rtol=1e-6` before `m42_dev_v0` is available. Development then evaluates PerTask,
State-OneHot, and FactorFiLM on the same 72 identities each, 216 episodes total, and publishes
semantic, first-interaction, post-grasp, runtime-action, and quality-gate evidence. Experiment/
physical completion remains independent from the 16-condition quality result. Final remains sealed.
The legacy queue field `offline_validation_action_loss` retains the same total ACT validation
objective used by historical M4/M4.2 selection (reconstruction plus weighted KL when active); the
name is preserved for schema compatibility and is not interpreted as an arm-only loss.

## CLI

M2 uses thin scripts under `environment/`: `run_expert.py` for one rollout,
`benchmark_expert.py` for the explicit six-task matrix, and `verify_m2.py` for contract and target
acceptance. They validate configuration, call expert APIs, preserve unexpected exception type and
message at the outer boundary, surface failures, and write only to ignored diagnostic output paths.
The strict M2 verifier invokes the M0 installation and M1 environment target gates in the main
runtime, then invokes the planner-runtime gate before starting M2 physical rollouts in the selected
side runtime. It keeps the original repeatable six-task smoke and separately applies the
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
strictly validate action plus state audit for every episode. Collection and replay use the same
fingerprinted planner-side interpreter as the expert; offline inspection remains in the main
runtime. These commands are sequential and introduce no multiprocessing.

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
the ACT and processor queues at every episode, invokes the explicit versioned action-bound
postprocessor directly before `env.step`, and never invokes the expert. Full-mode test access
requires the immutable validation-selected checkpoint and predeclared schedule. Target smoke chains
the prior target gates before real CUDA/tiny-overfit/rollout work. Full target requires explicit
`project` mode and supports a no-training preflight that validates the completed M3B evidence and
computes all eight exact future `full` run fingerprints, split views, train-only statistics,
effective configurations, and schedules without creating a model directory or starting rollout.
The subsequent non-dry command additionally runs all eight policies, locked test, and the fixed
180-episode fresh benchmark. Fixture work cannot set physical or model-quality flags.

M4.2 adds `scripts/run_m42_runtime_ablation.py`, `scripts/train_act_task_token.py`,
`scripts/evaluate_m42.py`, and `environment/verify_m42.py`. Commands accept explicit paths and
dry-run, write machine-readable fingerprint-owned artifacts through owned staging, and fail on
incompatible reuse. `--target-development` verifies completed M4 evidence, locks both schedules,
runs horizon then gripper selection, trains/selects exactly one TaskToken model from M3B validation,
and evaluates development. It cannot materialize `m42_final_v0`. `--target-final` is a later,
separate authorization that verifies all immutable locks, performs the paired final benchmark, and
writes go/no-go without starting M5.

M4.3a adds `scripts/audit_act_semantics.py` and `environment/verify_m43.py`. The audit command has
explicit dataset/checkpoint/evidence roots, named validation/`m42_dev_v0`/combined modes, dry-run,
path-safety and immutable-output checks, and a machine-readable command report. The non-target
verifier exercises portable math, retrieval/confusion, serialization, lifecycle, CLI dry-run, and
the final-access prohibition.

M4.3b adds `scripts/train_act_factor_film.py`. Its dry-run and fixture paths accept explicit dataset,
output, evidence, device, and report paths and never start long training. The authorized explicit
`--target-development` path owns clean-Git/evidence/M3B gates, immutable run identity, 100,000-step
seed-0 training, 20 checkpoints, and a fingerprint-bound validation queue; `--resume-checkpoint`
accepts only the latest declared or sole next atomically promoted checkpoint, and
`--clean-staging` removes only matching unlinked incomplete staging. Source/Git/data/evidence/
historical paths are protected before report or output writes. Post-training work belongs to
`scripts/evaluate_act_factor_film.py`, whose resumable stages write outside the immutable producer
run. `environment/verify_m43b.py` independently reads and re-derives the evidence without training,
selection mutation, or rollout. `environment/verify_m43.py` retains the local architecture/fixture
contracts and cannot claim target training or rollout.

## Language routing

`langmani.language` owns M5A's modular boundary:

```text
langmani.language
├── router_types.py          # immutable decisions, examples, metrics, and identities
├── corpus.py / splits.py    # deterministic generation and family-level isolation
├── _final_language_authority.py # lazy final-only command source; never imported by development
├── schedules.py             # development locks and sealed final locks
├── rule_router.py           # deterministic lexical/conflict/rejection baseline
├── text_classifier.py       # shared encoder plus status/object/bin heads
├── text_training.py         # train-only staged optimization and resumable artifacts
├── text_calibration.py      # validation-only temperature and threshold selection
├── stage_protocol.py        # promotion states, budgets, and bounded checkpoint policy
├── schema_validation.py     # strict structured local-LLM output boundary
├── llm_router.py            # one pinned local instruct model with bounded repair
├── router_evaluation.py     # immutable language-only summaries and quality gates
├── controller_registry.py   # portable six-PerTask controller identity
├── dispatcher.py            # one predicted controller or zero safe dispatch
├── failure_attribution.py   # disjoint routing/control failure classes
├── evaluation.py            # paired control summaries and development gates
└── artifact_validation.py   # independent corpus/router/control evidence revalidation
```

Corpus template-family IDs are not M1 instruction-template IDs. A valid language route always
constructs the existing `canonical_v0` TaskSpec. Text stays out of policy observations, numeric
state, and per-step info. The language subsystem may read M3B sidecar seed identities to exclude
prior layouts, but it does not reopen M3A or load M3B test observations/actions/video.

The controller registry is assembled from a narrow metadata allowlist: the M3B completion marker,
six completed PerTask manifests and validation selections, selected checkpoint artifacts/hashes,
selected validation runtime manifests, and the M4.2 runtime lock. It rejects M4 comparison,
verification, test, historical-fresh, and final result paths and does not copy source evaluation
schedules into the portable registry. Portable fingerprints exclude locator paths but bind all
semantic run, checkpoint, processor, statistics, M3B, Git, and H=10/`project` runtime identities.
The active M1 action-space contract is checked before controller loading. Dispatch uses the
scheduled oracle TaskSpec for M1 reset and evaluation while the predicted TaskSpec selects the
policy. Rejection returns before controller lookup or any environment action. The registry cannot
reselect, overwrite, blend, or retrain a controller.

`scripts/build_language_corpus.py`, `scripts/train_text_router.py`,
`scripts/evaluate_language_routers.py`, and `scripts/run_language_control.py` are thin explicit
command boundaries. `environment/verify_m5a.py` separates portable fixtures from native target
development and reports quality separately from correct experiment execution. Final language and
control locks may be validated but are not materialized by development commands. The ordinary
corpus path keeps only opaque final IDs/counts/digests and never imports
`_final_language_authority.py`; only a separately authorized future final entry point may load it.

## Cross-cutting rules

- Project-owned interfaces use static typing and tests.
- Generated datasets, assets, videos, checkpoints, caches, and results stay outside the package.
- Third-party APIs are wrapped at boundaries rather than copied or patched in-tree.
- Dependency and interface changes require an entry in `docs/DECISIONS.md`.
- Every M4 run records the full Git commit; a dirty tree is development-only and never final.
- M4 train statistics and checkpoint selection may use train/validation only; test remains locked.
- M4.2 runtime selection uses only `m42_dev_v0`; TaskToken checkpoint selection uses only M3B
  validation; the sealed `m42_final_v0` is never a development input.
- M4.3a uses M3B validation and `m42_dev_v0` as separately labeled audit sources, never M3B test,
  M4 fresh, or `m42_final_v0`; its real immutable audit is complete.
- M4.3b uses stable TaskSpec metadata for separate object/visual and bin/state FiLM paths. Training
  uses only M3B train, checkpoint selection uses only M3B validation, state stays 9D, and local
  fixtures cannot set training, selection, rollout, physical, final, or SmolVLA flags. Post-training
  evidence records its own evaluator Git while preserving architecture baseline `8ee0f1b...` and
  exact target-training producer `0088e293...` as separate identities.
- M5A gradients use language train only; classifier selection/calibration uses validation only.
  Classifier development uses one seed and one immutable run: fixture and tiny-overfit precede a
  step-29 pilot, and promotion resumes the same model/optimizer/scheduler/processor/RNG state up to
  at most step 145. Validation patience is one and checkpoint roles are bounded to pilot/latest/best.
  LLM prompt/configuration may be frozen from train/validation evidence, but every few-shot example
  ID remains train-family only and the same immutable prompt artifact is reused in language and
  control evaluation. Language/control development measure generalization only, and all final
  texts/episodes, M3B test frames, historical fresh evaluation, and `m42_final_v0` remain sealed.
- M5A.1 is a read-only decision layer over the immutable selected classifier. Train is used only
  to verify label/class provenance; validation alone supplies logits, temperature fitting, fixed
  decoder-grid selection, confusion matrices, and diagnostics. The classifier checkpoint identity
  is never rewritten. A passing decoder receives a separate runtime fingerprint; otherwise the
  candidate is frozen with no executable runtime and no authorization for another seed or update.
- M5A rejection has no executable TaskSpec and must return before policy or environment execution.
- M5A.4 separates lexical and Qwen semantic facts from the only decision authority. The Qwen frame
  is generated under one cached strict Outlines JSON grammar and contains no status, TaskSpec, or
  reason. `DeterministicSafetyArbiterV0` routes only on exact object/bin agreement and a supported
  action; every disagreement is a non-executable rejection. Historical validation is quarantined,
  and the complete development partition cannot be opened until the prompt/schema/arbiter/runtime
  lock is durably written.
  Expected TaskSpec, router decision, selected controller, and active EpisodeSpec remain separate
  evidence so routing and control failures cannot be collapsed.
- M5A physical development uses disjoint 1/3/6-scene partitions. Oracle runs at every stage,
  failed learned candidates are removed, RuleRouter remains offline, and only one screened router
  reaches the 36-pair full stage. The 12-scene final authority is lazily sealed and separately
  authorized.
- Target development independently revalidates a real rejection no-op probe and retains four action
  streams: raw policy output, optional binary-transformed output, bounds-projected output, and the
  action actually executed. The locked project-only runtime records the binary stream as absent.
- OneHot and TaskToken are oracle discrete-task controls, not language understanding.
- Simulator, CUDA, Vulkan, or rendering failures remain visible and cause strict verification to
  fail.
