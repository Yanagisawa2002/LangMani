# LangMani roadmap

Milestone M4 is active. M0 through M3B are implemented. Native target acceptance and the first real
M3A/M3B datasets remain pending; M4 local work may use fixtures but must not treat them as policy
quality or physical evidence. Later milestones describe intended sequencing, not authorization to
implement those systems now.

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
recording and action replay; the authoritative 60-group archive remains pending full collection.

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
rendering, H.264 decode, source alignment, finalization, and public reload; the complete 360-episode
derived dataset remains pending full acceptance.

## M4.1 — Auditable ACT action-bound handling (complete)

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

Local completion means typed/fixture/API tests and truthful pending flags. M4.1 target smoke
validates the completed M0-through-M3B evidence, reuses the existing PerTask and TaskOneHot
checkpoints without training, reproduces strict bound rejection, executes a projected legal step,
then attempts one PerTask and six same-scene TaskOneHot rollouts. It does not start the finalized
60-group dataset, eight full ACT runs, M4 full selection/test/fresh-seed protocol, or M5.

The clean RTX 4090 smoke passed PerTask 1/1 and TaskOneHot 6/6. All seven task successes required
at least one explicit gripper projection, so strict-unprojected success remains 0/7 and raw-action
bounds validity remains false. This closes M4.1 without claiming M4 full acceptance.

## M5 — SmolVLA baseline (planned)

Add SmolVLA-specific preprocessing, training, and evaluation only after the dataset and ACT
baseline contracts are stable.

## M6 — Evaluation and release hardening (planned)

Unify policy evaluation, regression thresholds, reproducibility reports, command-line workflows,
and GPU-runner CI when suitable hardware becomes available.
