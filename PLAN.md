# LangMani roadmap

Milestone M4.2 is active. M0 through M3B, M4 full, and M4.1 target smoke are complete on native
targets. M4 full is experimentally and physically validated, but its declared quality gate is
false. M4.2 may execute development only until its runtime and TaskToken choices are frozen; the
sealed final benchmark remains pending and M5 is not authorized. Later milestones describe intended
sequencing, not authorization to implement those systems now.

## M0 — Reproducible environment foundation (complete)

Establish the src-layout package, dependency decisions, CPU-safe tests, explicit GPU/rendering
markers, and a strict native Linux diagnostic proving that PyTorch CUDA, ManiSkill, Vulkan
rendering, and base LeRobot imports coexist.

## M1 — Environment and language contracts (complete)

Implement exactly one deterministic, vectorization-compatible ManiSkill environment,
`LangMani-PickPlaceByInstruction-v0`, with three colored cubes, two primitive shallow bins, typed
scene/task metadata, canonical language accessors, privileged-state gating, conservative batched
success metrics, separate policy/diagnostic cameras, and explicit CPU/GPU/rendering acceptance
boundaries. M1 does not include task-solving or data-collection behavior.

## M2 — Privileged motion-planning expert (complete)

Implement one deterministic, privileged-state Panda expert for all six semantic object/bin tasks in
`LangMani-PickPlaceByInstruction-v0`. M2 owns the project-level expert types, a lazy adapter over
the public mplib 0.1.1 planner API, explicit phase execution, verification, structured failure
classification, single-rollout diagnostics, and a six-combination benchmark.

The expert runs only with `num_envs=1` and `pd_joint_pos`. It uses the active `TaskSpec` and narrow
expert-only environment interfaces; it never infers targets from image pixels or actor order. The
stable phase sequence is `initialize`, `move_to_pregrasp`, `approach_target`, `close_gripper`,
`verify_grasp`, `lift_target`, `move_above_destination`, `descend_to_place`, `open_gripper`,
`settle_after_release`, `retreat`, and `verify_task`. Each phase records attempts, completion or a
specific failure, environment steps, planning calls, and elapsed durations in a JSON-serializable
result.

M2 uses a deterministic top grasp and bin-interior-center placement. It preserves M1's visual
no-leakage contract and adds no trajectory recorder, dataset export, training, deployable vision
policy, LLM/VLM call, teleoperation, reinforcement learning, or domain randomization. mplib is not
vectorized and no multiprocessing or RRT fallback is introduced. The native RTX 4090 target gate
passed its six-task smoke and 177/180 balanced benchmark (98.33%) with zero wrong-target successes,
unclassified failures, or crashes.

## M3A — ManiSkill-native raw demonstrations (implementation complete)

Collect a deterministic, resumable authoritative source archive from the M2 expert. Candidate
scene seeds are ordered, and every accepted `CounterfactualSceneGroup` contains one physical layout
paired with all six canonical TaskSpecs. The first target archive is exactly 60 complete groups and
360 successful episodes, with 60 per task combination.

M3A owns typed collection/replay contracts, cryptographic stable IDs, bounded retries, ManiSkill
`RecordEpisode` integration, HDF5/JSON shards with environment states and actions, attempt and
episode manifests, action replay, state audit, corruption/schema checks, resume behavior,
inspection commands, and target verification. It does not create LeRobotDataset, Parquet, policy
videos, training data transformations, or policies. One six-task native target smoke group passed
recording and action replay. Native full acceptance then produced 60 complete groups and 360
accepted/replayed episodes, exactly 60 per TaskSpec, from 65 ordered candidate scenes. Five groups
were rejected and all 404 attempts remain accounted for; no partial group, accepted replay failure,
schema/checksum failure, or unclassified failure was admitted.

## M3B — LeRobotDataset v3 export (implementation complete)

Deterministically convert accepted episodes from a content-bound, validated M3A source into one
local LeRobotDataset v3. M3B restores each recorded pre-action state in a fresh M1 environment,
renders only the fixed 256 x 256 `base_camera`, extracts nine Panda joint positions, preserves the
exact eight-dimensional raw action and canonical task string, and emits exactly T frames for T
actions.

M3B owns typed export contracts, stable export fingerprints, scene-group-level 48/6/6 splits,
source-to-derived mapping, a project lifecycle guard around the public LeRobot 0.6.0 writer,
PyAV H.264/yuv444p video, all-or-nothing staging, independent source/action/state/video/Parquet validation,
DataLoader smoke tests, and structural/smoke/full verification modes. M3A remains authoritative.
M3B adds no training, policy configuration, Hub upload, failure trajectories, additional sensors,
language generation, or parallel export. One six-task target export passed real state-restoration
rendering, H.264 decode, source alignment, finalization, and public reload. Native full acceptance
then exported all 360 episodes and 64,548 frames with exact 288/36/36 episode and 48/6/6 scene-group
splits. Every TaskSpec contributes 48/6/6 episodes; all videos decode, provenance and action/state
alignment pass, and no scene group crosses a split. The accepted export fingerprint is
`sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4`.

## M4 — Reproducible ACT baselines (complete)

Train and compare exactly three controls on one completed M3B dataset: six `per_task` ACT policies
with image plus 9D Panda state; one `mixed_unconditioned` ACT with the same 9D input and deliberate
counterfactual ambiguity; and one `mixed_task_onehot` ACT with image plus a 15D state consisting of
the same Panda state and a project-owned canonical six-way oracle command.

M4.1 owns a versioned environment-action postprocessor after the saved LeRobot postprocessor and
before `env.step`. `reject` preserves strict failure; `project` deterministically bounds finite
actions using the actual M1 action space while retaining raw/executed audits and independent
`task_success` versus `strict_unprojected_success`. A separate runtime fingerprint binds checkpoint
and processor fingerprints, bound configuration, environment/action-space contract, task mapping,
rollout configuration, schema, and current code commit. Existing checkpoint fingerprints remain
unchanged.

M4.1 target smoke
validates the completed M0-through-M3B evidence, reuses the existing PerTask and TaskOneHot
checkpoints without training, reproduces strict bound rejection, executes a projected legal step,
then attempts one PerTask and six same-scene TaskOneHot rollouts.

The clean RTX 4090 smoke passed PerTask 1/1 and TaskOneHot 6/6. All seven task successes required
at least one explicit gripper projection, so strict-unprojected success remains 0/7 and raw-action
bounds validity remains false.

M4 full subsequently trained and sealed all eight 100,000-step runs. PerTask achieved 31/36 on the
locked test and 143/180 on historical fresh seeds; Mixed-Unconditioned achieved 5/36 and 18/180;
Mixed-TaskOneHot achieved 27/36 and 101/180. Validation-only selection, locked test access,
fresh-seed evaluation, provenance, and physical execution passed, so
`full_experiment_validated=true` and `physical_target_validated=true`. The intended conditioning and
fresh-seed quality thresholds did not all pass, so `baseline_quality_validated=false` remains an
equally important result.

## M4.2 — Oracle-control robustness (active; development only)

M4.2 first uses frozen Mixed-TaskOneHot and representative green-left PerTask checkpoints to compare
execution horizons 10, 5, and 1 on the committed 12-scene `m42_dev_v0` schedule. After locking one
horizon, it compares the existing explicit `project` action runtime with an explicit component-7
`BinaryGripperEnvPostprocessorV0`. Raw, binary-transformed, projected, and executed actions remain
separate, and privileged post-grasp phases are diagnostics only.

M4.2 then trains exactly one `ACT-Mixed-TaskToken`. Panda policy state remains 9D. The canonical
six-way oracle command uses LeRobot 0.6.0's public `FeatureType.ENV` path, whose linear 6-to-hidden
projection creates a dedicated Transformer environment token. This is discrete oracle conditioning,
not language understanding. Training retains the M4 configuration and train-only statistics; only
M3B validation may select its checkpoint.

Both M4.2 schedules were generated and committed jointly after excluding 125 prior observed or
predeclared seeds. `m42_dev_v0` contains 12 scenes and has fingerprint
`sha256:981547e771b2b5cd3a77e2788bb49d29fc45b3f59c607021a03a4e2ce70b43f1`.
The sealed 30-scene `m42_final_v0` has fingerprint
`sha256:b2aef313e076201f7a94875c835c2d606d8f255e3f75d53f7ac7fadcdbb267fc`.
Development must stop before rendering or resetting a final seed. Final results, paired metrics, and
the SmolVLA go/no-go decision remain pending until a later explicit `--target-final` invocation.
The normative contract is `docs/M42_ORACLE_CONTROL_SPEC.md`.

## M5 — SmolVLA baseline (planned)

Add SmolVLA-specific preprocessing, training, and evaluation only if the immutable M4.2-final
decision is `go_for_smolvla`. M4.2 never starts M5 automatically.

## M6 — Evaluation and release hardening (planned)

Unify policy evaluation, regression thresholds, reproducibility reports, command-line workflows,
and GPU-runner CI when suitable hardware becomes available.
