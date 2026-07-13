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
vectorized and no multiprocessing or RRT fallback is introduced. Native Linux RTX 4090 execution,
CUDA, Vulkan, PhysX GPU, rendering, and physical expert acceptance remain pending until the ordered
M0/M1/M2 target verification commands all pass. The target gate includes a six-task smoke and a
balanced 180-rollout benchmark with explicit success, wrong-target, classification, and crash
thresholds.

## M3A — ManiSkill-native raw demonstrations (implementation complete)

Collect a deterministic, resumable authoritative source archive from the M2 expert. Candidate
scene seeds are ordered, and every accepted `CounterfactualSceneGroup` contains one physical layout
paired with all six canonical TaskSpecs. The first target archive is exactly 60 complete groups and
360 successful episodes, with 60 per task combination.

M3A owns typed collection/replay contracts, cryptographic stable IDs, bounded retries, ManiSkill
`RecordEpisode` integration, HDF5/JSON shards with environment states and actions, attempt and
episode manifests, action replay, state audit, corruption/schema checks, resume behavior,
inspection commands, and target verification. It does not create LeRobotDataset, Parquet, policy
videos, training data transformations, or policies. Native Linux collection and the first
authoritative 60-group archive remain pending until the target gate passes.

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
language generation, or parallel export. Real state-restoration rendering and the first complete
360-episode derived dataset remain pending target-machine acceptance.

## M4 — Reproducible ACT baselines and closed-loop evaluation (active)

Train and compare exactly three controls on one completed M3B dataset: six `per_task` ACT policies
with image plus 9D Panda state; one `mixed_unconditioned` ACT with the same 9D input and deliberate
counterfactual ambiguity; and one `mixed_task_onehot` ACT with image plus a 15D state consisting of
the same Panda state and a project-owned canonical six-way oracle command.

M4 owns the completed-M3B-only gate, explicit scene-safe episode views, train-only normalization,
stable Git/data/config run identities, installed LeRobot 0.6.0 ACT/processors, deterministic bounded
training, atomic local checkpoints and resume rejection, counterfactual audits, learned-policy M1
rollouts, validation-only checkpoint ranking, immutable test lock, fixed unseen-seed schedule, and
comparison reports. Standard ACT consumes no task text, so M4 does not claim language
understanding. M4 does not reopen M3A in normal training, rewrite M3B, invoke M2 during rollouts,
upload to Hub, or add SmolVLA, RL, DAgger, multi-GPU training, new tasks, or data augmentation.

Local completion means typed/fixture/API tests and truthful pending flags. Target smoke additionally
requires the ordered M0-through-M3B smoke chain, a real six-task derived group, CUDA
forward/backward, declared tiny-overfit controls, local checkpoint reload, and closed-loop M1
execution. Full acceptance requires the finalized 60-group/360-episode M3B dataset, all eight ACT
runs, validation-only selection, locked test evaluation, the fixed 30-seed x six-task benchmark,
and a provenance-complete comparison. Experiment completion and baseline quality are reported
separately.

## M5 — SmolVLA baseline (planned)

Add SmolVLA-specific preprocessing, training, and evaluation only after the dataset and ACT
baseline contracts are stable.

## M6 — Evaluation and release hardening (planned)

Unify policy evaluation, regression thresholds, reproducibility reports, command-line workflows,
and GPU-runner CI when suitable hardware becomes available.
