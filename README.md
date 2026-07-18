# LangMani

LangMani is a language-conditioned robotic manipulation research repository. M0 established the
runtime foundation, M1 added the environment/language contracts, M2 added a deterministic
privileged Panda expert, M3A implemented the authoritative ManiSkill-native raw archive, and M3B
implemented its validated local LeRobotDataset v3 derivation. M4 and M4.1 established reproducible
ACT controls, closed-loop evaluation, and auditable action projection. M4.2 completed its
oracle-conditioned development diagnosis and rejected TaskToken. M4.3a completed the real frozen-
policy semantic audit. M4.3b completed one factorized FiLM ACT target-development run, passed its
experiment/physical verifier, and failed its quality gate. The shared-ACT architecture search is
closed. M5A now implements modular language-to-TaskSpec routing over the six frozen PerTask ACT
controllers; target classifier/LLM training and language/control development results are not yet
claimed.

M4 implements six per-task ACT policies, one mixed unconditioned ACT, and one mixed ACT with an
oracle six-way task one-hot. Standard ACT consumes no natural-language text, so M4 is not a language
understanding milestone. M4.2 added runtime ablations and exactly one oracle TaskToken ACT. M4.3a
added no model and established that output changes are not reliably aligned with requested object/
bin semantics. M4.3b added one oracle `ACT-Mixed-FactorFiLM`, not language understanding. M5A adds
three explicit language routers and strict rejection, but keeps continuous control frozen. It does
not add SmolVLA, rewrite M3B, reopen M3A, invoke M2 during policy rollouts, publish models to Hub,
access any sealed final schedule, or change the M1/M2 task.

## Target platform

The acceptance target is a **native Linux** workstation with a compatible NVIDIA GPU, NVIDIA
driver, CUDA-capable PyTorch wheels, and a working Vulkan installation. Initial acceptance evidence
was produced on an RTX 4090; later milestone-specific target evidence also used RTX 5090 hosts.
Windows-native and WSL execution are not supported target configurations.

## M1 environment

Importing `langmani.environments` registers exactly one environment:
`LangMani-PickPlaceByInstruction-v0`. It always contains a Panda, the movable primitive actors
`red_cube`, `green_cube`, and `blue_cube`, and the fixed primitive compound actors `left_bin` and
`right_bin`. No simulator assets are downloaded; all non-robot scene geometry, including the table,
ground, cubes, and five-box shallow bins, is constructed from boxes.

```python
import gymnasium as gym
import langmani.environments  # registers the environment

env = gym.make(
    "LangMani-PickPlaceByInstruction-v0",
    num_envs=1,
    obs_mode="state_dict",
    reward_mode="sparse",
    control_mode="pd_joint_delta_pos",
)
observation, info = env.reset(
    seed=123,
    options={
        "task_spec": {
            "target_object_id": "red_cube",
            "target_bin_id": "right_bin",
            "instruction_template_id": "canonical_v0",
        }
    },
)
texts = env.unwrapped.get_task_texts()
episodes = env.unwrapped.get_episode_specs()
env.close()
```

The reset `task_spec` mapping must contain exactly those three string fields. Unknown, missing, or
extra fields fail before the simulator is reset. A single override broadcasts to every environment
selected by ManiSkill's optional `env_idx`. That index may be a one-dimensional integer tensor,
NumPy array, or sequence and must contain unique, strictly increasing in-range entries, matching
ManiSkill's reset-mask ordering. Task selection never triggers reconfiguration.
`reset_to_env_states` is deliberately rejected in M1 because the numeric task state is not part of
ManiSkill's simulator-state snapshot. When no override is given, `scene_seed % 6` selects one of the
six object/bin combinations in canonical object-major order.

`get_task_texts()` and `get_episode_specs()` always return immutable tuples, including for one
environment. `EpisodeSpec` is frozen and its `to_dict()` result is JSON-serializable. Its stable `scene_id` depends only on
the scene seed, while its stable `task_id` depends only on `TaskSpec`; therefore the same seed can be
reused with a different task to create a layout counterfactual with identical cube poses. For
example, serialize it with `json.dumps(episodes[0].to_dict())`.

### Reset geometry and robot state

The table surface is `z=0`. Each seed randomizes cube x positions in `[-0.18, -0.07]` m, assigns
the colors to three randomly permuted y cells centered at `-0.12`, `0.0`, and `0.12` m, adds at
most `0.012` m y jitter, and randomizes cube yaw. The disjoint cells guarantee initial cube
separation without a rejection loop. The bins remain at `(0.08, 0.18)` m (left, from the Panda's
view) and `(0.08, -0.18)` m (right), with an interior half-width of `0.085` m. This separates the
source and destination regions for every seed.

Relative to the Panda base at `(-0.615, 0)`, every source-cell center remains within about `0.56` m
horizontally and each bin center is about `0.72` m away. The source region follows the installed
ManiSkill Panda tabletop-task convention, while the bins were moved inward from the first design to
retain reach margin. This is a structural reachability choice; no expert, IK solver, or autonomous
reach trial is introduced in M1, and end-to-end manipulation remains a native-target follow-up.

Every reset explicitly clears cube linear and angular velocities. Panda initialization follows the
installed ManiSkill 3.0.1 table-task qpos and base pose `(-0.615, 0, 0)`, with joint reset noise
fixed at `0.0`. The environment requires ManiSkill enhanced determinism so unseeded resets draw a
fresh episode seed that is retained in the metadata contract. The registered Gym time limit is 200 control
steps, which is 10 seconds at ManiSkill's default 20 Hz control rate.

### Language, observations, and evaluation

M1 recognizes only `canonical_v0`, yielding exactly six English sentences of the form
`Pick up the red cube and place it in the left bin.` No synonym, paraphrase, spatial-reference,
embedding, generated language, or LLM path exists.

Language, scene IDs, and task IDs are never inserted into observations or per-step `info`. Internal
targets are `torch.long` batched object/bin indices. Standard visual-only ManiSkill modes expose
robot proprioception, `tcp_pose`, and requested sensor textures, but not cube poses, bin poses or
centers, target indices, semantic color IDs, task IDs, embeddings, or success-oracle inputs. When
the requested mode includes privileged state (`state`, `state_dict`, or a combined visual+state
mode), `extra` additionally contains flattened object-major `cube_poses` `(N, 21)`, flattened
left/right `bin_centers` `(N, 6)`, `target_object_index` `(N,)`, and `target_bin_index` `(N,)`.

`evaluate()` returns batched boolean tensors named `target_in_target_bin`,
`target_in_wrong_bin`, `wrong_object_in_target_bin`, `target_is_grasped`, `target_is_static`,
`target_off_table`, `success`, and `fail`. Success requires the horizontal projection of a sphere
enclosing the selected cube to fit inside the bin walls, its height to be near the bin floor, no
wrong cube to occupy that target bin, the target to be released and static, and the target not to be
off the table. Thus a held cube hovering above the correct bin, a target in the wrong bin, or a wrong
cube in the target bin is not success. Only a target whose enclosing sphere has fallen entirely
below the tabletop is an immediate `fail`;
wrong placement can still be corrected. Supported reward modes are exactly `sparse` and `none`.
The sparse mode uses ManiSkill's standard `success.float() - fail.float()` behavior; M1 defines no
dense reward.

### Cameras

The policy camera and human diagnostic camera are separate:

| Camera | Eye → target (m) | Resolution | FOV (rad) | Near / far (m) | Shader |
| --- | --- | --- | --- | --- | --- |
| `base_camera` | `(0.65, -0.75, 0.70)` → `(-0.04, 0, 0.08)` | 256×256 | 1.05 | 0.01 / 10.0 | `minimal` |
| `render_camera` | `(0.78, -0.90, 0.82)` → `(-0.04, 0, 0.08)` | 512×512 | 1.00 | 0.01 / 10.0 | `default` |

There is no wrist camera and no domain randomization. The target diagnostic uses actor
segmentation to require all three cubes and both bins to be visible in `base_camera`; this physical
visibility check passed on the native Linux RTX 4090 target together with the separate 512×512
human-render camera. The report remains the authoritative evidence for the exact machine/run.

## M2 privileged expert

M2 adds a project-owned `PickPlaceExpert` boundary for privileged demonstration generation. The
expert accepts only the registered LangMani environment with `num_envs=1` and
`control_mode="pd_joint_pos"`. It resolves the target actor and destination from the active
semantic `TaskSpec` through `get_expert_task_context()` and reads terminal checks through
`get_expert_evaluation()`. Those accessors expose runtime-only simulator handles and geometry to the
expert; they do not add fields to Gym observations or per-step `info`, and the M1 visual no-leakage
tests remain authoritative.

`ExpertConfig` defaults to the 200-step episode limit, one planning attempt per motion phase,
0.08 m pre-grasp clearance, 0.05 m grasp approach distance, 0.10 m lift and transport clearances,
0.002 m placement clearance, 15 release-settling steps, and a 0.10 m retreat. Diagnostic rendering
is opt-in. The adapter rejects any other control mode. mplib 0.1.1 exposes neither a deterministic
seed nor a timeout for `plan_screw`, so `planner_seed` and `planner_timeout_seconds` must remain
`None`; unsupported values fail validation rather than being ignored.

The phase order is stable and externally reportable:

1. `initialize`
2. `move_to_pregrasp`
3. `approach_target`
4. `close_gripper`
5. `verify_grasp`
6. `lift_target`
7. `move_above_destination`
8. `descend_to_place`
9. `open_gripper`
10. `settle_after_release`
11. `retreat`
12. `verify_task`

Each phase has explicit entry checks, attempt accounting, completion criteria, and a phase-specific
failure. Terminal `ExpertStatus` values are exactly `success`, `invalid_task`,
`initialization_failure`, `ik_failure`, `planning_failure`, `execution_failure`, `grasp_failure`,
`transport_failure`, `placement_failure`, `verification_failure`, `target_off_table`, `timeout`, and
`unexpected_exception`. Unexpected exceptions are converted only at the outer diagnostic command
boundary, where both exception type and message are preserved; a single rollout still exits
nonzero, while the explicit benchmark mode may collect failures.

`ExpertResult` and every `PhaseResult` are compact, immutable, and JSON-serializable. The rollout
result records success/status, scene seed and stable scene/task IDs, canonical instruction, semantic
target IDs, total steps/planning calls/replans, completed and failed phases, final environment
evaluation, per-phase results, planning/execution durations, optional artifact paths, and exception
details only for `unexpected_exception`. It never contains observations, images, tensors, planned
joint paths, or trajectories.

### Deterministic motion construction

The expert wraps the public `mplib.Planner` API directly and imports mplib lazily. Runtime code does
not import ManiSkill's example runner or example solver. The adapter requires exactly mplib 0.1.1,
synchronizes the nine-joint Panda state and robot base pose, plans seven arm joints with
`plan_screw`, and emits `pd_joint_pos` actions with a separate gripper command. M2 deliberately has
no RRT fallback: mplib 0.1.1 does not expose a seed for its sampling planners, while fixed-state
screw planning is the deterministic path used here. mplib is not vectorized and M2 adds no
multiprocessing.

The public status model retains a distinct `ik_failure` for adapters that can report it. The chosen
mplib 0.1.1 `plan_screw` call exposes only `screw plan failed` for differential-IK, joint-limit, and
collision failures, so this concrete adapter conservatively reports that upstream-ambiguous case as
`planning_failure` instead of inventing a more specific cause.

The adapter updates mplib's robot qpos and attachment handle when the grasp state changes, but M2
does not add a world point cloud. In mplib 0.1.1 the attached box participates in collision checks
against a world point cloud only when one is enabled, so M2 does not claim planner-side obstacle
avoidance from that handle alone. The phase clearances and PhysX contacts remain the active safety
mechanisms; target acceptance must confirm them physically before this boundary is expanded.

The single grasp is a top grasp of the active target cube. In world coordinates the approach vector
is `(0, 0, -1)` and the gripper closing vector is `(0, -1, 0)`. ManiSkill's Panda convention builds
the TCP rotation columns as `[closing × approach, closing, approach]`, giving
`diag(1, -1, -1)` and wxyz quaternion `(0, 1, 0, 0)`: TCP local `z` points downward and local `y`
is the closing direction. This equals the installed table-scene Panda's initial TCP orientation;
using the opposite closing sign would create an exact 180-degree first screw rotation that mplib
0.1.1 rejects. Planner
poses are seven-vectors `[x, y, z, qw, qx, qy, qz]`. The grasp center is the cube center; the
pre-grasp and approach offsets move only opposite the downward approach direction and retain table
clearance. A planned path is executed exactly once with no added terminal waypoint repetition, then
requires TCP position error at most 0.015 m and quaternion angular error at most 0.08 rad. The
15 mm bound covers the measured `pd_joint_pos` tracking residual without changing the M1 success
geometry. The fixed cube collision size is 0.05 m on each side.

Placement targets the selected bin's interior center, never an actor-order-derived destination. The
bin interior half-width is 0.085 m and the cube half extent is 0.025 m, leaving 0.060 m nominal
center-to-wall room on each axis. The centered placement therefore satisfies the M1 0.002 m wall
containment clearance, and initialization separately enforces the M2 expert's 0.010 m cube-to-wall
fit margin. Its object-center height is derived from the bin floor top plus the cube half extent
and the configured 0.002 m placement clearance; lift, transport, descent, release, settle, and
retreat remain separate verified phases. A wrong object in the selected bin or the target in the
wrong bin cannot be reported as success.

Run one diagnostic rollout, all six semantic combinations, or the M2 verifier with:

```bash
python environment/run_expert.py
python environment/benchmark_expert.py
python environment/verify_m2.py
```

`run_expert.py` defaults to seed `0`, `red_cube` → `left_bin`, `physx_cpu`, and
`outputs/diagnostics/m2/single_result.json`; `--object-id`, `--bin-id`, `--seed`,
`--sim-backend`, and `--diagnostic-rendering` select another explicit rollout.
`benchmark_expert.py` always visits all six combinations sequentially and accepts a comma-separated
unique `--seeds` list; its default report is `outputs/diagnostics/m2/benchmark.json`. Rendering
writes phase PNGs beside those reports. Both commands confine `--output` to the ignored `outputs/`
tree, return zero only when all requested rollout(s) succeed, return one for a classified or
unexpected rollout failure, and let argparse return two for malformed arguments. `verify_m2.py`
writes `outputs/diagnostics/m2/verification.json`.

M1 reports `terminated=True` as soon as conservative success becomes true, which can occur during
release settling. M2 explicitly permits that success terminal while it completes the mandatory
retreat and final verification phases; truncation, off-table failure, or termination without either
success or the known off-table failure still stops execution with a classified error.

These commands are not evidence of physical acceptance merely because they import or perform
structural checks on a non-target host. ManiSkill 3.0.1's Linux mplib wheel is isolated behind the
explicit `LANGMANI_PLANNER_PYTHON` runtime described below; running mplib beside the main NumPy 2
stack is rejected before native planner construction. The clean M0/M1/M2 target gates have passed;
M2 completed its six-task smoke and 177/180 balanced benchmark (98.33%) with zero wrong-target
successes, unclassified failures, or crashes.

## M3A raw demonstrations

The normative archive contract is in [`docs/RAW_DATA_SPEC.md`](docs/RAW_DATA_SPEC.md), with the
required compatibility entry at
[`docs/DATA_COLLECTION_SPEC.md`](docs/DATA_COLLECTION_SPEC.md). The unchanged M1 and M2 inputs are
summarized in [`docs/ENVIRONMENT_SPEC.md`](docs/ENVIRONMENT_SPEC.md) and
[`docs/EXPERT_SPEC.md`](docs/EXPERT_SPEC.md).

The data architecture has two layers: the authoritative ManiSkill-native HDF5/JSON M3A archive and
the reproducible LeRobotDataset v3 representation implemented by M3B. The source retains executed
`pd_joint_pos` actions, T+1 simulator state dictionaries, exact reset seed and `TaskSpec`, terminal
labels, canonical language, expert/replay evidence, and cryptographic provenance. It deliberately
records with `obs_mode="none"` and no video; visual observations can be regenerated later by first
resetting the semantic task and then restoring recorded states.

The atomic semantic unit is a `CounterfactualSceneGroup`: one scene seed paired in fixed
object-major/bin-minor order with red/left, red/right, green/left, green/right, blue/left, and
blue/right. A group is accepted only when all six expert attempts produce valid raw trajectories,
all six pass action replay, all finish with the existing conservative M1 success, the recorded and
active TaskSpecs agree, and their initial simulator states are identical. One failed task rejects
the complete group; no partial group enters an accepted shard.

Every attempt stores its original M2 `ExpertResult`, not just successful episodes. A claimed expert
success is ineligible for replay unless its scene/task IDs, canonical instruction, object/bin,
fresh final M1 evaluation, wrong-target labels, episode step count, planning attempts, and outer
retry index all agree with the scheduled reset and configured bounds. Such a mismatch is recorded
as `expert_contract_failure`; the result is not rewritten and the trajectory never enters action
replay.

The collector also records an explicit initial physical snapshot before the expert acts: all three
cube poses, both actual bin poses, and Panda qpos. This complements the native T+1 state tree because
ManiSkill omits static bins from that tree. Dynamic snapshot values must match HDF5 state zero, and
all six complete snapshots must be identical within the group; equal seeds alone are insufficient.

Default collection bounds are explicit:

| Setting | Default |
| --- | ---: |
| Candidate seed schedule | consecutive seeds starting at 0 |
| Target complete groups | 60 |
| Accepted episodes | 360 |
| Accepted episodes per TaskSpec | 60 |
| Maximum candidate scenes | 120 |
| Maximum expert attempts per task | 3 |
| Maximum episodes per source shard | 60 (10 complete groups) |
| Replay mode | action replay plus state audit |
| Final target position tolerance | 0.005 m |
| Final target orientation tolerance | 0.05 rad |
| Final Panda joint tolerance | 0.05 rad |

Every persistent identity is derived from canonical JSON and SHA-256, never Python `hash()` or
filesystem order. The semantic identity includes the environment/version, scene seed/ID, task
ID/specification, complete expert configuration fingerprint, `pd_joint_pos`, and collection schema.
Native integer `traj_N` IDs are only local shard locations; the manifest maps stable scheduled,
attempt, raw-trajectory, scene-group, and source-shard IDs.

Run, resume, inspect, and independently replay the default archive with:

```bash
python environment/collect_raw_demos.py
python environment/inspect_raw_demos.py
python environment/replay_raw_demos.py
python environment/verify_m3a.py
```

`collect_raw_demos.py` resumes a compatible incomplete manifest by default. `--no-resume` rejects
an existing run; `--overwrite` replaces only a root with the exact LangMani M3A marker and a
parseable manifest that claims that resolved root (or a narrowly recognized interrupted first
commit). Other useful flags include `--candidate-seed-start`,
`--target-complete-scene-count`, `--maximum-candidate-scene-count`,
`--maximum-expert-attempts-per-task`, `--shard-size`, `--sim-backend`,
`--replay-validation-mode`, the three final-state tolerances, and
`--retain-failed-raw-trajectories`. Command roots are confined below the ignored `outputs/` tree.

The output structure is:

```text
outputs/datasets/m3a/langmani-pick-place-raw-v1/
├── .langmani-m3a-root
├── manifests/
│   ├── collection_manifest.json
│   ├── collection_schedule.json
│   ├── attempts.jsonl
│   ├── episodes.jsonl
│   ├── scene_groups.jsonl
│   ├── source_shards.jsonl
│   └── dataset_summary.json
├── accepted/shards/        # authoritative ManiSkill .h5/.json pairs
├── journal/                # in-flight attempt journal + open-shard group bundles
├── .staging/               # one uncommitted candidate or shard build
└── failed/                 # optional full failed candidate files, never accepted data
```

The collector writes each candidate through ManiSkill 3.0.1 `RecordEpisode` with empty observations,
no rewards, and no video; closes the pair; validates every HDF5/JSON leaf and label; compares a fresh
seeded reset with recorded state zero; replays actions in a single environment; and, when configured,
restores and reads back every recorded state. Only then does it build an accepted group bundle and
source shard. Attempt and shard journals are removed only after the authoritative manifest commit.
ManiSkill's stock GPU replay path is not the M3A oracle because version 3.0.1 ignores reset
`options` there, which would drop LangMani's `TaskSpec`. The project replay validator always uses
the exact recorded reset mapping and never substitutes state replay for action replay.

The native time contract is exact: T float32 `pd_joint_pos` actions, T terminated/truncated and
present success/fail labels, and T+1 environment states including state zero. Replay passes each
in-bounds recorded action to `env.step` unchanged exactly once. It never appends a terminal no-op,
duplicates the final transition, normalizes or converts control modes, or clips an out-of-bounds
action. Replay failures carry stable `ReplayFailureCode` values as well as bounded readable reasons.

Target smoke and full-dataset acceptance are deliberately separate. Start with one fresh physical
scene group: the command runs the complete M0/M1/M2 target gate, records all six semantic tasks,
checks the closed native files and identical initial states, and independently action-replays all six
episodes. This diagnostic run is bounded to 20 ordered candidate seeds and three expert attempts per
task:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py --target-smoke
```

Create the first authoritative archive only with explicit authorization:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py \
  --target-full \
  --dataset-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --create-new-run
```

Subsequent full invocations omit `--create-new-run`. They validate an existing complete archive
without recollecting it, or resume only a compatible `in_progress` manifest. The verifier never
passes `--overwrite`; an existing root cannot be replaced by the creation flag. Full acceptance
requires 60 complete groups, 360 accepted episodes, 60 per TaskSpec, no partial groups, accepted
expert/replay failures, checksum/schema failures, false successes, or unclassified validation
failures, followed by an independent ordered replay of all 360 actions.

The report exposes `implementation_validated`, `prior_target_gates_validated`,
`expert_collection_smoke_validated`, `action_replay_validated`, `full_dataset_validated`, and
`physical_target_validated` separately. Its v3 schema also binds collection run ID, source
configuration fingerprint, and a checksum-derived archive digest so M3B cannot reuse stale target
evidence for different files at the same path. Until the relevant command passes on the native Linux RTX
4090 machine, local structural, synthetic-HDF5, serialization, resume, and mocked-execution checks
are not physical target acceptance.

## M3B LeRobotDataset v3 export

The normative derived-dataset contract is
[`docs/LEROBOT_DATASET_SPEC.md`](docs/LEROBOT_DATASET_SPEC.md). M3A remains authoritative: M3B
first requires the complete M3A inspector plus a content-bound v3 target report, then uses accepted
scene groups as its only source index. A smoke source is exactly one six-task group; a full source
is exactly 60 groups and 360 episodes.

For each raw episode, M3B performs the recorded semantic reset in a fresh `obs_mode="rgb"` M1
environment, restores state[t], obtains the public `base_camera` observation and Panda qpos, and
pairs them with the unchanged action[t]. T actions produce exactly T LeRobot frames; terminal
state[T] remains audit-only. The only policy features are:

```text
observation.images.base_camera  video, uint8 HWC writer input, 256 x 256 x 3
observation.state               float32[9], PandaPolicyStateV0 qpos
action                          float32[8], exact pd_joint_pos action
```

The nine state components are `panda_joint1` through `panda_joint7`, then
`panda_finger_joint1` and `panda_finger_joint2`. LeRobot stores the accepted canonical instruction
through task metadata and adds its standard timestamp/frame/episode/index/task-index fields. No
object/bin pose, target index, semantic ID, full simulator state, success oracle, expert/planner
state, depth, segmentation, or human camera enters policy features.

Full splits are assigned only at counterfactual scene-group level: 48 train, 6 validation, and 6
test groups (288/36/36 episodes and 48/6/6 episodes per TaskSpec). One physical LeRobot dataset is
created; sidecar episode-index lists expose the three splits without duplicating MP4 or Parquet
data.

The writer uses installed LeRobot 0.6.0 public `create/add_frame/save_episode/finalize` APIs and an
explicit PyAV/libx264 H.264 yuv444p configuration. It writes under fingerprint-owned staging, finalizes
exactly once, writes provenance sidecars, independently reloads and validates the staged dataset,
rechecks the unchanged M3A source, atomically promotes, and writes the completion marker last. A
partial writer is never resumed; use `--clean-staging` only for an owned incomplete staging root.

Dry-run, export, validation, and inspection commands are:

```bash
python scripts/export_lerobot_dataset.py \
  --source-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --output-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --repo-id langmani/pick-place-by-instruction-v0 --full --dry-run

python scripts/export_lerobot_dataset.py --source-root ... --output-root ... --full
python scripts/validate_lerobot_dataset.py --dataset-root ... --source-root ... --full
python scripts/inspect_lerobot_episode.py --dataset-root ... --episode-index 0
```

The independent validator reloads the local dataset through LeRobot, reads every Parquet file,
decodes every video frame, checks tasks/boundaries/timestamps/splits/no-leakage, compares all raw
actions and reconstructed qpos, recomputes raw RGB digests, performs deterministic lossy-video
quality comparisons, and runs a `DataLoader` batch with `num_workers=0`.

Local generated-array fixture encoding, decoding, Parquet reading, and DataLoader batching pass,
but they are not physical validation. Native Linux target modes remain authoritative:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3b.py --target-smoke
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3b.py --target-full \
  --source-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1
```

## M4 ACT baselines

The normative contract is [`docs/ACT_BASELINE_SPEC.md`](docs/ACT_BASELINE_SPEC.md). M4 accepts only
a completed, locally reloadable M3B root with the exact 60-group/360-episode full contract. Its
normal gate checks the content-bound M3B marker, manifest, summary, validation report, source
mapping, scene-level 48/6/6 splits, local Parquet/video files, 20 FPS, and the exact policy feature
allowlist. It does not rerun M3A or change M3B admission.

The three controls are:

| Variant | Models | Input |
| --- | ---: | --- |
| `per_task` | six | base-camera RGB + `PandaPolicyStateV0[9]` |
| `mixed_unconditioned` | one | the same RGB + 9D state, with no task condition |
| `mixed_task_onehot` | one | RGB + Panda 9D + `CanonicalTaskOneHotV0[6]` = 15D |

The one-hot follows the M3A/M3B red-left, red-right, green-left, green-right, blue-left, blue-right
ordering and is appended by one project-owned component used for both training and inference. It is
an oracle command, not a language embedding. Neither standard ACT variant receives task text, ID,
scene seed, privileged poses, success state, or expert information.

M4 derives train/validation/test and per-task views from the M3B manifest. Normalization is
recomputed from only the 48 per-task train episodes or all 288 mixed train episodes with
deterministic float64 Welford accumulation. Images are float32 [0,1] per-channel statistics; state
and exact eight-dimensional actions use componentwise population statistics. The 15D suffix uses
mean 0/std 1 so the installed mean/std processor preserves the binary one-hot. M3B's whole-dataset
`meta.stats` is never a training fallback.

Installed LeRobot 0.6.0 public `ACTConfig`, `ACTPolicy`, `make_act_pre_post_processors`,
`PolicyProcessorPipeline`, `LeRobotDataset`, and `resolve_delta_timestamps` are the runtime
boundary. The installed processor does not divide uint8 images by 255, so LangMani performs that
explicit conversion before preprocessing. Public delta resolution keeps observations at the
current frame and creates the configured 50-action chunk with episode-end padding. Installed ACT's
queue executes ten actions per query, and the policy plus both processors are reset at every
episode boundary.

The fixed primary model is ResNet-18 without downloaded pretrained weights, model dimension 512,
eight heads, 3200-dimensional feedforward layers, four encoder/one decoder layers, VAE latent 32,
four VAE encoder layers, dropout 0.1, and KL weight 10. Optimization is AdamW at `1e-5` (including
the backbone), weight decay `1e-4`, no scheduler/warmup, global gradient clipping at 10, CUDA
bfloat16 autocast, batch size 32, 100000 steps, and checkpoint/validation every 5000 steps. The
model/source tensors remain float32. The target must support bfloat16; M4 does not silently switch
precision. Actual target memory and throughput remain recorded in the immutable run reports.

Every run records the actual full Git commit. The tracked M3B baseline is
`6920b52c1f48c278e669cd71b69b8949dd900f3a` (`m3b-implementation`). Full and tiny-overfit evidence
requires a clean tree; a dirty run is allowed only as an explicitly labeled, nonfinal development
override. Canonical SHA-256 run identity binds data/split/statistics fingerprints, episode views,
variant/task, effective model/optimization, seed, runtime versions, Git commit, and schema while
excluding paths, timestamps, hostname, and filesystem order.

Train or inspect the command contracts with:

```bash
python scripts/train_act.py --help
python scripts/evaluate_act.py --help
python scripts/compare_act_baselines.py --help
python scripts/inspect_act_checkpoint.py --help
python scripts/benchmark_act_evaluation_workers.py --help
python environment/verify_m4.py
```

The evaluation-worker benchmark is an isolated throughput diagnostic, not M4 acceptance evidence.
It creates minimal per-worker clones with an independent copy of one already selected checkpoint,
executes the same six validation episodes with `num_envs=1`, and ranks physically passing
configurations by the worst GPU's completed episodes per minute. An explicit CPU-demand estimate can exclude worker
counts that exceed the container's cgroup quota; excluded counts are recorded as unmeasured, never
as failed results. If the parent process is interrupted after workers finish, `--summarize-existing`
recovers durations from filesystem timestamps, verifies each clone against the immutable source,
and preserves unknown subprocess return codes as `null`. Recovered artifact measurements,
subprocess-exit validation, source-mutation isolation, and temporal selection stability remain
separate so an actionable recommendation cannot be mistaken for a fully repeated selection:

```bash
python scripts/benchmark_act_evaluation_workers.py \
  --source-run outputs/models/act/<completed-run-fingerprint> \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/benchmarks/m4_eval_worker_scaling/<new-run> \
  --worker-counts 1 2 4 6 --gpu-ids 0 1 \
  --estimated-cpu-cores-per-worker 12

python scripts/benchmark_act_evaluation_workers.py \
  --source-run outputs/models/act/<completed-run-fingerprint> \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/benchmarks/m4_eval_worker_scaling/<interrupted-run> \
  --worker-counts 1 2 4 6 --gpu-ids 0 1 \
  --estimated-cpu-cores-per-worker 12 --summarize-existing
```

For example:

```bash
python scripts/train_act.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --variant mixed_task_onehot --seed 0 --output-root outputs/models/act --dry-run

# Predict the exact immutable identity/configuration of a future full run without training.
python scripts/train_act.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --variant mixed_task_onehot --seed 0 --output-root outputs/models/act \
  --dry-run --planned-mode full

python scripts/train_act.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --variant per_task --task-id red_cube__left_bin --seed 0 \
  --output-root outputs/models/act --dry-run
```

Checkpoints stage and atomically promote the public ACT weights, preprocessor/postprocessor,
optimizer and optional scheduler state, RNG/training state, and content checksums. Processor state
is bound to the train-statistics fingerprint; the run manifest is updated separately with an atomic
file replacement. Reload is local-only and resume rejects any semantic mismatch or completed run.
The sole valid promoted orphan, including the first checkpoint, is recoverable; a pre-checkpoint
interruption is preserved under an ignored diagnostic directory before deterministic restart.
Validation alone ranks checkpoints by success, wrong-object interaction, target-off-table rate,
offline action loss, then earlier step. An immutable selection record must exist before full-mode
test access; test results cannot change selection or resume training.

Closed-loop inference reads only M1 `base_camera` RGB and Panda qpos, optionally appends the oracle
one-hot, calls installed `select_action`, and applies the declared M4.1 action-bound processor after
the LeRobot postprocessor. It does not call M2. `reject` blocks malformed, nonfinite, or
out-of-bounds actions before `env.step`; `project` retains malformed/nonfinite hard failures and
explicitly projects finite values into the actual environment bounds while recording raw and
executed actions. This is auditable projection, not silent clipping, and it does not binarize the
gripper. Validation/test each use the fixed six M3B groups. The fresh benchmark fixes 30
unseen source-excluded seeds and all six tasks, giving 180 episodes per mixed model and 30 per
per-task model. Reports keep implementation completion, experiment validity, model quality, and
physical acceptance separate. Evaluation directories use owned staging and atomic promotion; resume
reconstructs raw rollout records and rechecks aggregates, schedules, Git state, and test
authorization before selection, analysis publication, or idempotent finalization.

The non-target verifier exercises contracts and a small CPU fixture only. It is not real M3B,
CUDA, model-quality, or physical evidence. Native target modes are:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py \
  --target-smoke --action-bound-mode project
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py --target-full \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act --dry-run --action-bound-mode project
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py --target-full \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act --action-bound-mode project

# Audit the exact eight completed runs from the recorded full dry-run plan.
# This command never trains and never executes a rollout.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py --target-full \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act --action-bound-mode project \
  --reuse-completed-evidence
```

M4.1 target smoke validates the completed M0--M3B smoke evidence, reuses the existing PerTask and
Mixed-TaskOneHot checkpoints without training, reproduces strict rejection, executes one projected
legal action, and attempts one plus six learned-policy M1 rollouts. Reports separate raw validity,
projected legality, task success, and strict-unprojected success. The clean RTX 4090 M4.1 run passed
PerTask 1/1 and TaskOneHot 6/6; 834/951 rollout actions projected only the gripper, so raw-bound
validity and strict-unprojected success correctly remain false. The real finalized 360-episode
dataset is available. Full mode first dry-runs and records all eight exact full identities without
training or rollout, then requires all six per-task policies,
both mixed policies, the complete counterfactual audit, validation-only selection, locked test, and
the 180-episode fresh benchmark. Target modes require the current worktree to be clean before any
completed run is reused. Every full evaluation explicitly uses the frozen `project` action-bound
mode; no evaluator default may choose that behavior implicitly. M4.1 does not retrain the smoke
checkpoints.

After an evaluation-only code repair, `--reuse-completed-evidence` is the fail-closed official
summary path. It reads the exact eight immutable run fingerprints from the existing full dry-run
plan, verifies all 160 validation results plus the eight locked tests, eight fresh-seed benchmarks,
runtime manifests, selections, and completion markers, and runs only the final comparison audit.
Training Git remains checkpoint-bound; each evaluation Git is checked independently against its
sibling runtime manifest. Without this flag, target-full refuses to launch a replacement training
run when compatible completed historical evidence already exists.

Evaluation-worker scaling is measured separately from official selection/test/fresh evidence. The
benchmark command hard-links one immutable selected checkpoint into per-worker minimal run clones,
runs only the fixed six-episode validation schedule, keeps every worker at `num_envs=1`, and hashes
the source run before and after. It compares symmetric per-GPU counts by the slower GPU's completed
episodes per minute, chooses fewer workers when throughput is within 5% of the maximum, and retests
the winner and runner-up. The benchmark output lives under `outputs/benchmarks`; it cannot publish
selection, test, fresh-seed, or final M4 evidence.

M4 full is complete. The six PerTask policies achieved 31/36 (86.1%) on locked test and 143/180
(79.4%) on historical fresh seeds; Mixed-Unconditioned achieved 5/36 and 18/180; and
Mixed-TaskOneHot achieved 27/36 and 101/180. The full experiment, provenance, and native physical
execution are validated, while `baseline_quality_validated=false`. That distinction is intentional:
the experiment is sound, but the oracle-conditioned mixed control remains below its quality target.

## M4.2 oracle-control robustness

The normative contract is
[`docs/M42_ORACLE_CONTROL_SPEC.md`](docs/M42_ORACLE_CONTROL_SPEC.md). M4.2 uses the completed M4
evidence without changing any of its eight checkpoints. It first compares execution horizons 10,
5, and 1 using the frozen Mixed-TaskOneHot checkpoint and the fixed representative green-left
PerTask checkpoint. After one horizon is locked, it compares the existing `project` runtime with an
explicit `BinaryGripperEnvPostprocessorV0` that changes only action component 7. Raw commands,
binary changes, projections, and executed actions remain separately auditable. Any Panda arm
projection or worsened wrong-object/wrong-bin safety count fails development admission.

Two schedules were generated and committed together before rollout:

| Lock | Scenes x tasks | Use | SHA-256 fingerprint |
| --- | ---: | --- | --- |
| `m42_dev_v0` | 12 x 6 = 72 | runtime selection and development comparison | `sha256:981547e771b2b5cd3a77e2788bb49d29fc45b3f59c607021a03a4e2ce70b43f1` |
| `m42_final_v0` | 30 x 6 = 180 | one later sealed paired benchmark | `sha256:b2aef313e076201f7a94875c835c2d606d8f255e3f75d53f7ac7fadcdbb267fc` |

The locks exclude the exact 125-seed union of M3A accepted/rejected candidates, every M3B split,
observed M4 fresh seeds, smoke/tiny and M4.1 diagnostic seeds, and predeclared tiny fresh seeds. The
final list is source-visible for audit but cannot be materialized as rollout episodes without an
explicit final authorization and immutable development selections. Development must not render or
reset a final scene.

M4.2 then trains exactly one `ACT-Mixed-TaskToken` with the same 288 M3B train episodes, 100,000-step
configuration, and validation-only selection used by Mixed-TaskOneHot. `PandaPolicyStateV0` remains
9D. The six-way canonical oracle command uses LeRobot 0.6.0's public `FeatureType.ENV` feature:
ACT's public linear 6-to-512 projection creates a dedicated Transformer environment token between
the Panda-state and image tokens. It is `CanonicalTaskTokenV0`, not instruction tokenization or
language understanding.
Before training, all eight historical selected checkpoints are strictly revalidated from disk and
the effective TaskToken configuration is compared with the frozen Mixed-TaskOneHot run manifest;
an extra complete or incomplete TaskToken run identity is rejected as ambiguous.

The completed target boundary stopped at development:

```bash
python scripts/run_m42_runtime_ablation.py --help
python scripts/train_act_task_token.py --help
python scripts/evaluate_m42.py --help
python environment/verify_m42.py

CUDA_VISIBLE_DEVICES=0 python environment/verify_m42.py --target-development
```

`--target-development` validated prior M4 evidence and both schedule locks, selected horizon 10 and
the existing `project` runtime, trained all 100,000 TaskToken steps, selected checkpoint 90,000 from
M3B validation only, evaluated `m42_dev_v0`, and stopped. PerTask, State-OneHot, and TaskToken
achieved 56/72, 35/72, and 15/72 development successes; TaskToken was rejected.
`python environment/verify_m42.py --target-final` remains a distinct command and was not run.
Final paired results and a SmolVLA go/no-go decision therefore do not exist.

If the runtime command report and its evidence already exist, the verifier never launches the
336 physical episodes again. It first audits all eight immutable benchmark directories, their
identity/artifact/completion fingerprints, exact 336-unique/420-logical episode accounting,
development selections, post-grasp report, M3B/M4 provenance, and experiment manifest. Cross-commit
reuse is accepted only when the historical and current tracked path sets, regular-file modes, and
bytes match for the runtime command, environment/dependency locks, package root, and every policy,
environment, dataset, and expert source. The legacy five-file digest is used only to reproduce the
old report's implementation fingerprint. The TaskToken command thaws internally frozen JSON tuples
at its consumer boundary without changing the list-only public parser or the byte-exact runtime
closure. This repair is recorded in a separate content-addressed `runtime_repair_lineage` file and
never rewrites the original evidence. Partial, linked, untracked, extra, missing, tampered, or
final-access evidence is a hard failure rather than a rerun.

## M4.3a semantic alignment audit

The normative zero-training contract is
[`docs/M43_SHARED_POLICY_REPAIR_SPEC.md`](docs/M43_SHARED_POLICY_REPAIR_SPEC.md). M4.3a uses only
the frozen six PerTask, State-OneHot, and rejected TaskToken checkpoints. It compares one complete
postprocessed 50-action chunk from every policy while holding the base-camera RGB and
`PandaPolicyStateV0[9]` observation fixed. No offline comparison steps an environment, and all
actions are measured before binary-gripper or environment-bound projection.

The primary retrieval metric is `ActionChunkDistanceV0` action-range-normalized L2 over arm
components 0 through 6 and the first locked execution-horizon actions. Raw, train-standard-
deviation-normalized, cosine, gripper-only, first-action, first-five, and full-chunk distances remain
reported secondary evidence. Full-task top-1/top-2/MRR, target-object centroids, global and
object-conditional destination-bin retrieval, deterministic confusion matrices, first interaction,
and post-grasp failure classes establish semantic alignment. A merely nonzero distance does not.

The completed M4.2 development artifacts contain real aggregate post-grasp distributions but not
the complete per-episode fields needed to reconstruct first object displaced, first bin approached,
first object-entry bin, and final per-object/bin relationships. M4.3a reports those aggregate
post-grasp results and marks first-interaction records/confusions unavailable. It never invents
first-interaction values from aggregate wrong-object counts.

Audit M3B validation, `m42_dev_v0`, or both explicitly:

```bash
python scripts/audit_act_semantics.py --help
python scripts/audit_act_semantics \
  --mode combined \
  --device cpu \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --m4-checkpoint-root outputs/models/act \
  --task-token-checkpoint-root outputs/models/act-task-token \
  --m42-diagnostics-root outputs/diagnostics/m42 \
  --runtime-selection outputs/diagnostics/m42/runtime_ablation/runtime_selection.json \
  --output-root outputs/diagnostics/m43/semantic-audit \
  --report outputs/diagnostics/m43/audit-command.json \
  --dry-run
python environment/verify_m43.py
```

`validation`, `m42_dev_v0`, and `combined` remain distinct source modes. The command rejects M3B
test, M4 fresh, and `m42_final_v0` identities; it writes fingerprint-owned evidence through staging,
independent checksum validation, atomic promotion, and a last completion marker. A completed
fingerprint directory is immutable. A non-target verifier without the real evidence checks
contracts, fixture math, serialization, lifecycle, path safety, CLI dry-run, and truthful flags
only; it may discover and set `semantic_audit_completed=true` only after independently validating
the promoted real evidence.

The real clean-Git RTX 5090 combined audit completed at implementation commit
`dfea8b3d7d28274909ff178cb9087a9a90e17ee7` over 36 M3B-validation and 72 `m42_dev_v0`
observations. The completed evidence fingerprint is
`sha256:6342bdf4b019df203e6021947cbb39deacac2ea78cc585b5062091ed1d228671`. Its dataset,
split, and development-schedule fingerprints are respectively
`sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4`,
`sha256:d86d29374ef7956a4ad8a1d9a111ed0924387b5b456d86e8c9ad5a31752f65e5`, and
`sha256:981547e771b2b5cd3a77e2788bb49d29fc45b3f59c607021a03a4e2ce70b43f1`.

Across all 108 observations, State-OneHot reached 37.96% full-task top-1, 70.37% top-2,
MRR 0.6289, 76.85% target-object retrieval, and 50.93% destination-bin retrieval. TaskToken
reached 24.07%, 50.00%, MRR 0.4887, 50.00%, and 49.07% respectively. Both policies changed
outputs without reliably following requested semantics; object confusion is present and bin
retrieval is approximately random. The development evidence also retains shared-policy post-grasp
failures. TaskToken remains rejected. The promoted audit sets `semantic_audit_completed=true` and
keeps test, historical-fresh, final-schedule access, final authorization, and SmolVLA false.

## M4.3b factorized FiLM target development

M4.3b implements exactly one oracle-conditioned policy, `ACT-Mixed-FactorFiLM`. Stable TaskSpec
metadata is decomposed by one shared train/inference component into canonical target-object indices
`red_cube`, `green_cube`, `blue_cube` and destination-bin indices `left_bin`, `right_bin`. It never
parses instruction text. `PandaPolicyStateV0` remains exactly nine dimensions and there is no
appended one-hot or combined task token.

The 32D target-object embedding produces per-channel gamma/beta for the ResNet-18 layer-4 feature
map `[B,512,8,8]`, before ACT's image projection. The 32D destination-bin embedding separately
produces gamma/beta for the encoded Panda-state token `[B,512]`, before Transformer processing.
The two paths use residual FiLM, `x * (1 + gamma) + beta`, with projection weights initialized from
`N(0,1e-5)` and zero bias. Object identity never conditions the state path and destination identity
never conditions the image path. The full primary model keeps the 51,576,712 base ACT parameters
and adds exactly 67,744 conditioning parameters, for 51,644,456 total.

The adapter subclasses the public LeRobot 0.6.0 `ACTPolicy` and uses public `PreTrainedConfig`,
`ACTConfig`, `ACTPolicy.save_pretrained`/`from_pretrained`, and
`make_act_pre_post_processors`. LeRobot exposes no public hook for the required two
intermediate representations, so one isolated module registers instance-local forward hooks on the
semi-stable `model.backbone` output and `model.encoder_robot_state_input_proj` output. It records
and validates the ACT symbols/signatures, `[B,512,8,8]` backbone map, `[B,dim_model]` state token,
latent/state/64-image-token encoder layout, and `[B,50,8]` output. Version, signature, attribute,
shape, or token-layout drift fails rather than falling back to State-OneHot or TaskToken. Direct
FactorFiLM saves require a new or empty real directory, so a refused overwrite cannot mutate
existing weights.

The local non-target structure and fixture commands remain available:

```bash
python scripts/train_act_factor_film.py --help
python scripts/evaluate_act_factor_film.py --help
python scripts/train_act_factor_film.py \
  --dataset-root outputs/fixtures/m43/factor-film-dataset-contract \
  --output-root outputs/fixtures/m43/factor-film-dry-run \
  --device cpu \
  --report outputs/diagnostics/m43/factor-film-fixture-contract.json \
  --dry-run \
  --fixture-contract
python scripts/train_act_factor_film.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act-factor-film \
  --evidence-root outputs/diagnostics/m43/semantic-audit/evidence/930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2 \
  --device cpu \
  --report outputs/diagnostics/m43/factor-film-dry-run.json \
  --dry-run
python scripts/train_act_factor_film.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act-factor-film \
  --evidence-root outputs/diagnostics/m43/semantic-audit/evidence/930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2 \
  --device cpu \
  --report outputs/diagnostics/m43/factor-film-fixture.json \
  --fixture
python environment/verify_m43.py
python environment/verify_m43b.py --help
```

The dry-run and fixture validate model construction, separated conditioning, finite forward/
backward, gradients in the base model and both FiLM paths, one optimizer step, processor and
checkpoint persistence, fresh-instance reload, and deterministic output equivalence. They do not
use a tolerance larger than `atol=1e-6`, `rtol=1e-6` and do not produce a target checkpoint or
quality result. The architecture/structural baseline remains
`8ee0f1babf36b91d1ee2a39701e4a6db6003b660`, and the authorized CUDA training producer is explicitly
pinned to clean compatibility commit `0088e2937556c123c37c2dbe69f73301b1eebfd0`. The target server
must fetch and check out that exact commit before running:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_act_factor_film.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act-factor-film \
  --evidence-root outputs/diagnostics/m43/semantic-audit/evidence/930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2 \
  --device cuda \
  --report outputs/diagnostics/m43/factor-film-target-development.json \
  --clean-staging \
  --target-development
```

That run must reuse the exact 288 M3B train and 36 validation episodes, train-only 9D statistics,
100,000 steps, checkpoint interval 5,000, expected 20 checkpoints, batch size 32, locked seed 0,
action chunk 50, bfloat16 CUDA path, horizon 10, and `project` runtime. Compatible resume accepts
only the latest declared checkpoint or the sole next 5,000-step atomically promoted orphan. Unsafe
or linked output, report, and staging paths are rejected before any write.

Selection and development are separate resumable stages implemented at
`1bacb66d2a6f7c3f2d18d6f65ad7865af9a12cd6`. The target server checks out the current clean
authorization commit containing that implementation; the stages write fingerprint-owned evidence
outside the immutable training run and explicitly bind the producer commit:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_act_factor_film.py \
  --target-development \
  --training-git-commit 0088e2937556c123c37c2dbe69f73301b1eebfd0
CUDA_VISIBLE_DEVICES=0 python environment/verify_m43b.py \
  --training-git-commit 0088e2937556c123c37c2dbe69f73301b1eebfd0 \
  --training-run-root outputs/models/act-factor-film/<run-fingerprint> \
  --evaluation-evidence-root outputs/diagnostics/m43/<evaluation-evidence-root> \
  --structural-verification outputs/diagnostics/m43/target-development-preflight-verification/verification.json
```

All source changes are made and validated in the local repository, then committed and pushed. The
target server only fetches/checks out/pulls those commits; it must not be hot-patched. Generated
training/evaluation artifacts stay under ignored server output roots and are never committed.
The independent verifier requires that explicit structural report so its final `passed` result is
also gated by `factor_film_implementation_validated=true` and
`factor_film_fixture_training_validated=true`. Supplying the report is a prerequisite check, not a
claim that CUDA training or physical rollout succeeded.

All 20 checkpoints run the same 36 M3B-validation episodes. The fixed rank is highest success,
lowest wrong-object interaction, lowest wrong object in target bin, lowest off-table, lowest
timeout, lower checkpoint-bound validation loss, then earlier step. The historical JSON field named
`offline_validation_action_loss` contains the established total ACT validation objective, including
the weighted KL term when VAE is active; its value is retained for fair historical ranking rather
than renamed or recomputed. Selection is immutable before development access.

The selected checkpoint, preprocessor, policy postprocessor, FactorFiLM mapping, and explicit
action runtime are then loaded in a fresh operating-system process. One fixed validation
observation must reproduce the complete postprocessed `[50,8]` environment-action chunk at
`atol=1e-6`, `rtol=1e-6`. Only after that proof may the evaluator run the exact paired
`m42_dev_v0` matrix: 72 PerTask, 72 State-OneHot, and 72 FactorFiLM episodes, 216 total, all with
the same scene/task identities, horizon 10, `project` processor, M1 success geometry, and no M2
expert call. Validation and development semantic audits remain separate; new development episodes
also retain exact first-interaction and post-grasp diagnostics.

New `m42_dev_v0` episode reports use the explicit `development` evaluation split rather than the
legacy `fresh_seed` label. Existing M4.2 evidence is not rewritten and remains readable only through
its narrow compatibility rule. The final path keeps its prior label/behavior and is disabled here,
so new development records cannot be mistaken for historical fresh-seed or final access.

The development decision is a conjunction of the 16 predeclared success, per-task, safety,
projection, numerical-validity, retrieval, and task-sensitivity thresholds in
[`docs/M43_SHARED_POLICY_REPAIR_SPEC.md`](docs/M43_SHARED_POLICY_REPAIR_SPEC.md). A correctly
completed physical experiment may have `passed=true` and `physical_target_validated=true` while
`development_quality_gate_passed=false`. That is the observed result: validation selected step
70,000, fresh reload matched exactly, and the paired scores were PerTask 56/72, State-OneHot 36/72,
and FactorFiLM 38/72. FactorFiLM had six wrong-object grasps, two wrong objects in a target bin, 34
timeouts, zero wrong-bin target placements, zero off-table targets, zero arm projections, and no
non-finite or malformed action. M3B test, historical M4 fresh seeds, and `m42_final_v0` remained
inaccessible. The quality gate failed and the shared-ACT search is closed.

## M5A modular language routing

M5A treats the six selected PerTask ACT policies as an immutable skill library:

```text
natural-language command
    -> RuleRouterV0 | FactorizedTextClassifierV0 | StructuredLocalLLMRouterV0
    -> strict RouterDecision (one canonical TaskSpec or explicit rejection)
    -> exactly one frozen PerTask ACT controller, or no execution
    -> routing-versus-control failure attribution
```

The project-owned language corpus is generated deterministically, balanced across the six tasks,
and split by template family rather than shuffled sentence. Train/validation/development/final
families and near-duplicate structures cannot overlap. Ambiguous, contradictory, unsupported, and
malformed commands have explicit rejection labels. Final command texts and final control episodes
remain sealed during development.

The classifier uses one shared pretrained encoder with separate status, object, and bin heads.
Only train supplies gradients; validation alone selects a checkpoint, calibrates status
temperature, and chooses the selective-routing threshold. Development uses exactly seed 0, at most
five epochs/145 optimizer steps, a step-29 authoritative pilot, patience-one validation early
stopping, and only `pilot`, `latest`, and `validation_best` checkpoint roles. Promotion resumes the
same run with model/optimizer/scheduler/processor/RNG/data-progression state; no seed or encoder
sweep is allowed. The pilot persists per-step loss/gradient/throughput/CUDA telemetry and a pinned
checkpoint contract for corpus, split, labels, Git, and dependencies. A fresh-process stage
verifier reloads that checkpoint and recomputes its validation-only promotion gate before any
later resume can be authorized.
The observed step-29 pilot was correctly rejected. A documented post-pilot exception now permits
exactly that run and `pilot.pt` fingerprint to use `--target-pilot-recovery-resume`; this does not
change ordinary `--target-resume`. The recovery starts at step 30, audits the frozen data/loss
contract, keeps the five-epoch/patience-one/three-role bounds, selects by the versioned seven-key
validation ranking, and calibrates only if the unchanged full-quality gate passes.
That continuation completed 145 steps and selected the immutable epoch-4/step-116 checkpoint.
Routeable TaskSpec/object/bin accuracy reached 100%, but rejection generalization still missed the
unchanged quality gate. M5A.1 now performs one CPU validation-only analysis of that checkpoint. It
evaluates only `BaselineFourWayArgmaxV0`, `AggregatedRejectThresholdV0`,
`HierarchicalStatusDecoderV0`, and `ConservativeRouteDecoderV0` over the committed finite grid,
compares identity versus one validation-fitted status temperature, and never constructs an
optimizer or opens development/final/control inputs.
The local LLM is one explicitly supplied
0.5B-3B instruct model with a pinned revision, deterministic greedy generation, strict JSON parsing,
and at most one format repair. It never calls a hosted API and does not invent confidence. Exactly
one versioned prompt artifact is frozen before development. Every few-shot example ID belongs to a
train family; validation may freeze the prompt/configuration but is never used as in-context text.
Language-only evaluation and control reuse the byte-identical prompt template, ordered example IDs,
generation configuration, and prompt fingerprint.

The target local-LLM default is `bfloat16`; the only supported quantization mode is `none`. Both are
recorded in the router identity, and the runtime fails instead of silently changing dtype,
quantizing, or selecting another model.

Control development uses ten new scenes partitioned before execution into disjoint 1-scene smoke,
3-scene screen, and 6-scene full-development stages. The environment is reset with the schedule's
oracle TaskSpec while the predicted TaskSpec selects the controller, so wrong routing cannot change
the M1 success oracle. Oracle is retained at every physical stage; only language-promoted learned
routers proceed, RuleRouter stays offline, and exactly one screened router reaches the full stage.
The maximum routed physical cost before final is 18 + 54 + 72 = 144 episodes. Rejection returns before controller lookup, policy reset, environment reset,
or `env.step`. Registry validation may read finalized M3B sidecar identities and frozen M4
provenance, but it uses a metadata-only allowlist: the M3B completion marker, six completed PerTask
manifests and validation selections, selected checkpoint manifests/hashes, selected validation
runtime manifests, and the M4.2 runtime lock. It never opens M4 comparison/test/fresh-result
artifacts or M3B test observations, actions, frames, Parquet content, or videos. Before loading a
controller, the active M1 action-space contract must reproduce the frozen validation bounds.

The four parent locks are `m5a_language_dev_v0`, `m5a_language_final_v0`, `m5a_control_dev_v0`, and
`m5a_control_final_v0`; each physical substage is additionally content-bound to its parent and
predecessor promotion report. The final lock contains 12 new scenes x 6 tasks. A later separately
authorized final runs Oracle 72 plus one locked router 72, exactly 144 episodes. Development
materializes only development payloads and retains only sealed final identities/fingerprints.
`passed=true` means the requested stage completed correctly, not that the next stage was promoted.
All final, M3B-test-content, `m42_final_v0`, and SmolVLA access flags remain false.

After the implementation commit is pushed, the first separately authorized target stage is the
classifier fixture:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_text_router.py --fixture \
  --output-root outputs/models/text-router \
  --report outputs/diagnostics/m5a/stages/classifier-fixture.json
python environment/verify_m5a.py --verify-stage classifier_fixture \
  --stage-report outputs/diagnostics/m5a/stages/classifier-fixture.json
```

The one authorized rejected-pilot continuation is:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_text_router.py \
  --target-pilot-recovery-resume \
  --recovery-run-fingerprint sha256:9e3ac659fa2b695c843650df35e3779741d94b3dd70b2aec52a429bc4b2edf49 \
  --recovery-pilot-checkpoint-sha256 sha256:a892c2b73c87884b2b2acf843d22ed9318e6b661640eea6a4964ff8d739b5758 \
  --recovery-maximum-total-epochs 5 \
  --recovery-early-stopping-patience 1 \
  --output-root outputs/models/text-router \
  --report outputs/diagnostics/m5a/stages/classifier-recovery-resume.json
python environment/verify_m5a.py --verify-stage classifier_recovery_training \
  --stage-report outputs/diagnostics/m5a/stages/classifier-recovery-resume.json
```

The bounded no-training M5A.1 sequence is:

```bash
python scripts/analyze_classifier_rejection.py --dry-run \
  --report outputs/diagnostics/m5a/stages/classifier-rejection-dry-run.json
python scripts/analyze_classifier_rejection.py --validation-only --local-files-only \
  --output-root outputs/diagnostics/m5a/rejection-analysis \
  --report outputs/diagnostics/m5a/stages/classifier-rejection-analysis.json
python scripts/analyze_classifier_rejection.py --independent-verify \
  --output-root outputs/diagnostics/m5a/rejection-analysis \
  --report outputs/diagnostics/m5a/stages/classifier-rejection-independent-verification.json
python environment/verify_m5a.py --verify-stage classifier_rejection_analysis \
  --stage-report outputs/diagnostics/m5a/stages/classifier-rejection-independent-verification.json
```

A quality rejection is a successful analysis result: it freezes the classifier, authorizes no
additional training or seed, and produces no promoted runtime. Infrastructure or contract failures
remain nonzero command exits.

The real CPU run selected `ConservativeRouteDecoderV0` with identity temperature, route threshold
0.90, route margin 0.00, object confidence 0.75, and bin confidence 0.85. It eliminated false routes
and retained 179/180 routeable commands, but its 71.21%/80.56%/94.44%
ambiguous/unsupported/malformed recalls still failed the original conjunctive gate. Immutable
evidence `sha256:f2401fcb5b054c79c3b7e9674321eefcf9576dc4dcc5407bd96151db9e9b518d`
passed both independent verifiers. The classifier is frozen, no runtime was created, and no later
M5A stage was opened.

M5A.2 keeps that classifier as an offline negative baseline and compares it with the deterministic
RuleRouter and exactly one learned candidate: `Qwen/Qwen3-1.7B`. Model and tokenizer both pin
revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; target inference uses BF16, no quantization,
the official non-thinking chat-template path, greedy decoding, strict four-field JSON, and at most
one repair. The fixture only validates structure and immutable evidence. It cannot create real
validation/development metrics or authorize control.

```bash
python scripts/evaluate_language_routers.py --fixture \
  --device cpu \
  --output-root outputs/diagnostics/m5a/language-development-fixture \
  --report outputs/diagnostics/m5a/stages/language-development-fixture.json

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_language_routers.py --target-development \
  --device cuda \
  --llm-license-reviewed \
  --llm-model-card-reviewed \
  --output-root outputs/diagnostics/m5a/language-development \
  --report outputs/diagnostics/m5a/stages/language-development.json
python environment/verify_m5a.py --verify-stage language_development \
  --stage-report outputs/diagnostics/m5a/stages/language-development.json
```

The target command loads the model once, runs the fixed train-only smoke, evaluates all three
routers on the complete validation split, and opens `m5a_language_dev_v0` only after the LLM
validation gate passes. It stops before controller lookup or environment creation. Passing the
command means the requested offline protocol completed correctly; the separate quality and
`one_scene_control_smoke_authorized` flags may remain false.

The real M5A.2 target run passed its 20-example smoke but failed the unchanged validation gate:
85.56% full TaskSpec/object/bin accuracy, 14.17% false-route, 92.67% final schema validity,
7.33% malformed after repair, and 59.09%/52.78%/50.00%
ambiguous/unsupported/malformed rejection recall. It is now a frozen negative LLM baseline;
language development was not accessed.

M5A.3 permits one capacity comparison only, using the exact same semantic prompt, few-shot IDs,
strict parser, single repair, validation records, metrics, and gates with
`Qwen/Qwen3-4B-Instruct-2507` revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Local fixture and structural verification do not
download or run the model:

```bash
python scripts/evaluate_qwen_scale_escalation.py --fixture \
  --device cpu \
  --output-root outputs/diagnostics/m5a/qwen4b-escalation-fixture \
  --report outputs/diagnostics/m5a/stages/qwen4b-escalation-fixture.json
python environment/verify_m5a3.py
```

The real command uses one visible GPU, BF16, no quantization, and no fallback. It stops after a
failed smoke or validation gate; only a passing validation may open language development. The
independent verifier takes the emitted immutable evidence root. Neither command loads ACT or a
robot environment, and both leave physical target validation false:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qwen_scale_escalation.py \
  --target-development --device cuda \
  --license-reviewed --model-card-reviewed \
  --output-root outputs/diagnostics/m5a/qwen4b-escalation \
  --report outputs/diagnostics/m5a/stages/qwen4b-escalation.json

CUDA_VISIBLE_DEVICES=0 python environment/verify_m5a3.py \
  --evidence-root outputs/diagnostics/m5a/qwen4b-escalation/<runtime-fingerprint> \
  --report outputs/diagnostics/m5a/qwen4b-escalation-verification.json
```

M5A.4 freezes one neuro-symbolic candidate after the direct Qwen3-4B rejection. It combines a
deterministic lexical fact frame with a grammar-constrained Qwen semantic frame; only the fixed
safety arbiter may emit `RouterDecision`. The already observed validation split is diagnostic only.
An immutable prompt/runtime lock must exist before the command opens `m5a_language_dev_v0` once.
Local structural work and target execution are:

```bash
python scripts/evaluate_neuro_symbolic_router.py --dry-run \
  --report outputs/diagnostics/m5a/stages/neuro-symbolic-router-dry-run.json

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_neuro_symbolic_router.py \
  --target-development --local-files-only \
  --output-root outputs/diagnostics/m5a/neuro-symbolic-router \
  --report outputs/diagnostics/m5a/stages/neuro-symbolic-router.json

python environment/verify_m5a4.py \
  --evidence-root outputs/diagnostics/m5a/neuro-symbolic-router/<runtime-fingerprint>
python environment/verify_m5a.py \
  --verify-stage neuro_symbolic_language_development \
  --stage-report outputs/diagnostics/m5a/stages/neuro-symbolic-router.json
```

The target command uses the cached exact Qwen3-4B revision, BF16 on one GPU, and the exact
Apache-2.0 `outlines[transformers]==1.3.1` decoder. It has no free-form fallback, optimizer,
controller, environment, final access, or automatic control continuation.

M5A.4.1 preserves that failed exact-category smoke and separates no-dispatch safety from diagnostic
rejection taxonomy before any development access. The 20 source outputs are reused unchanged; the
new command writes the router lock before historical validation and the untouched 420-example
development pass:

```bash
python scripts/evaluate_neuro_symbolic_safety_gate.py --dry-run \
  --report outputs/diagnostics/m5a/stages/neuro-symbolic-safety-gate-dry-run.json

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_neuro_symbolic_safety_gate.py \
  --target-development --local-files-only \
  --source-evidence-root outputs/diagnostics/m5a/neuro-symbolic-router/\
183bc09b0ec7a30dbc50c349bab1a1a1d9368392b3f52551f509b716c83bbf7c \
  --classifier-rejection-evidence outputs/diagnostics/m5a/rejection-analysis \
  --output-root outputs/diagnostics/m5a/neuro-symbolic-safety-gate \
  --report outputs/diagnostics/m5a/stages/neuro-symbolic-safety-gate.json

python environment/verify_m5a41.py \
  --evidence-root outputs/diagnostics/m5a/neuro-symbolic-safety-gate/<runtime-fingerprint>
```

The exact rejection status/reason metrics remain mandatory report fields and explicit limitations.
Cloned or migrated target machines must pass the actual immutable M5A.1 analysis root explicitly;
the command validates that artifact and the selected classifier checkpoint before GPU inference.
Only the unchanged hybrid is promotion-eligible, and only the execution-safety/TaskSpec gate can
authorize a later one-scene control smoke. M5A.4.1 itself stops before all robot control.

Other later stages still require separate authorization and use ordinary `--target-resume` only
for an originally promoted pilot, followed by offline language evaluation and
`run_language_control.py --stage {one_scene_control_smoke,three_scene_control_screen,full_control_development}`.
The retired monolithic `verify_m5a.py --target-development` entry point fails without starting
work. No stage starts final or SmolVLA. The complete contract and gates are in
[`docs/M5A_LANGUAGE_ROUTING_SPEC.md`](docs/M5A_LANGUAGE_ROUTING_SPEC.md).

## Target-machine setup

Install [Miniforge](https://github.com/conda-forge/miniforge) first so `conda` is available. Then
run these commands from the repository root. The package names below are for Ubuntu/Debian; on
another native Linux distribution, install the equivalent Vulkan loader and tools explicitly.
System package installation remains an administrator action; the Python diagnostic never changes
the host.

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 vulkan-tools
nvidia-smi
vulkaninfo --summary

conda env create -f environment/environment.yml
conda activate langmani
python -m pip install --no-deps -e .
python -m pip check

# Isolate the NumPy-1 ABI required by mplib 0.1.1 while inheriting the exact base runtime.
PLANNER_VENV="$HOME/.venvs/langmani-planner"
python -m venv --system-site-packages "$PLANNER_VENV"
"$PLANNER_VENV/bin/python" -m pip install -r environment/planner-runtime.txt
conda env config vars set LANGMANI_PLANNER_PYTHON="$PLANNER_VENV/bin/python"
conda deactivate
conda activate langmani
"$LANGMANI_PLANNER_PYTHON" environment/verify_planner_runtime.py
```

M3B activates `lerobot[dataset]==0.6.0`; the base package alone deliberately refuses
`lerobot.datasets` imports. The reviewed environment resolved datasets 4.8.5, pandas 2.3.3,
PyArrow 25.0.0, PyAV 15.1.0, TorchCodec 0.11.1, and jsonlines 4.0.0. M3B explicitly uses PyAV for
writing and reading because the installed Windows TorchCodec DLL chain is not loadable. Exact
LeRobot/PyAV/libavcodec values are checked and stored in every export fingerprint and manifest.

M5A directly pins `transformers==5.4.0`, `tokenizers==0.22.2`,
`outlines[transformers]==1.3.1`, and the directly imported `safetensors==0.8.0`, matching the
inspected local classifier/checkpoint stack and LeRobot 0.6.0's
declared Transformers 5 compatibility. The unrelated system Python's older Transformers
installation is not part of the project runtime. Real classifier and local-LLM commands require
explicit model and tokenizer revisions and fail instead of selecting or downloading a different
model silently.

On native Linux, ManiSkill 3.0.1's package metadata installs `mplib==0.1.1`; Windows does not receive
that conditional dependency. LangMani does not change or duplicate the upstream pin. The main
environment retains its pinned NumPy 2 stack. The planner virtual environment inherits that exact
environment and overlays only NumPy 1.26.4, SciPy 1.15.3, and OpenCV 4.11.0.86. Do not run
`pip check` inside this intentional overlay: inherited LeRobot metadata describes the main runtime.
Instead, `verify_planner_runtime.py` imports and checks the complete effective planner module set,
then constructs and synchronizes the installed Panda planner. Import checks avoid mistaking inherited
main-environment distribution metadata for the overlay that Python actually loads. The adapter also
checks mplib and NumPy before entering the native constructor and fails clearly when either version
changes.

`environment/environment.yml` selects PyTorch 2.11.0 and torchvision 0.26.0 from the official
CUDA 12.8 wheel index. That path requires an NVIDIA driver new enough for CUDA 12.8; the current
LeRobot installation guide gives `570.86` as the minimum driver version.

Important: SAPIEN currently requires `opencv-python`, while LeRobot requires
`opencv-python-headless`. The environment installs both at the same 4.13.0.92 version to satisfy
their metadata, even though OpenCV's publishers do not support installing multiple `cv2` wheels in
one environment. Do not uninstall either wheel in place; this remains an explicit target-machine
risk, and the strict import/render gate is required. See `docs/DECISIONS.md` for the rationale.

## Validation

Formatting, linting, building, and CPU-safe tests do not require GPU simulation or rendering:

```bash
ruff format --check .
ruff check .
python -m build
pytest -m "not gpu and not rendering"
python environment/verify_install.py
python environment/verify_m1.py
python environment/verify_m2.py
python environment/verify_m3a.py
python environment/verify_m3b.py
python environment/verify_m4.py
python environment/verify_m42.py
python environment/verify_m43.py
python environment/verify_m43b.py --help
python environment/verify_m5a.py
```

The actual target-machine gate is stricter. The M2 command itself invokes the M0 installation gate
and M1 environment gate before any M2 rollout, in exactly that order, so an isolated expert run
cannot bypass prerequisites:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/verify_m2.py --target
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py --target-smoke
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py --target-full \
  --dataset-root outputs/datasets/m3a/langmani-pick-place-raw-v1
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3b.py --target-smoke
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3b.py --target-full \
  --source-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py \
  --target-smoke --action-bound-mode project
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py --target-full \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act --dry-run --action-bound-mode project
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py --target-full \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act --action-bound-mode project
CUDA_VISIBLE_DEVICES=0 python environment/verify_m42.py --target-development
# M4.3b uses the exact producer and post-training commands documented above; they are not folded
# into the historical M0--M4.2 chain and never open the sealed final schedule.
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_act_factor_film.py \
  --target-development \
  --training-git-commit 0088e2937556c123c37c2dbe69f73301b1eebfd0
CUDA_VISIBLE_DEVICES=0 python environment/verify_m43b.py \
  --training-git-commit 0088e2937556c123c37c2dbe69f73301b1eebfd0 \
  --training-run-root outputs/models/act-factor-film/<run-fingerprint> \
  --evaluation-evidence-root outputs/diagnostics/m43/<evaluation-evidence-root> \
  --structural-verification outputs/diagnostics/m43/target-development-preflight-verification/verification.json
pytest
```

For auditability, the orchestrated command prints and records its seven subprocesses:
`verify_install.py --target`, then `verify_m1.py --target`, the isolated planner-runtime gate, two
repeatability runs of the M2 six-episode all-task smoke, one balanced 180-episode benchmark, and one
rendered expert rollout whose 12 phase PNGs are checked. M0/M1 always use the main interpreter;
planner construction and every expert rollout use `LANGMANI_PLANNER_PYTHON`. A failed prerequisite
stops the later stages. The balanced run uses ordered scene seeds `0..29`, visits the canonical six
TaskSpecs for every seed, and thus contains exactly 30 episodes per TaskSpec.

`--target` returns a nonzero exit code if any of these are unavailable or invalid: native Linux,
PyTorch CUDA, the CUDA runtime reported by PyTorch, an NVIDIA GPU, `vulkaninfo`, the Vulkan probe,
explicit ManiSkill PhysX CUDA simulation, an RGB policy observation, `rgb_array` rendering, a
single environment step, or writing `outputs/diagnostics/pickcube_rgb.png`.

The M1 target diagnostic separately verifies the custom environment's seeded counterfactual reset,
positive and rejected placements in the real CPU scene, sparse reward and a real step,
six-environment GPU vectorization plus partial reset masks, all numeric evaluation fields, visual
no-leakage, per-actor policy-camera visibility, Panda-hand visibility, and the separate human camera.
CPU and GPU PhysX checks run in separate child processes because SAPIEN does not permit enabling
GPU PhysX after another PhysX backend has initialized in the same process. The parent merges their
validated JSON check records into one machine-readable report and writes two frames under
`outputs/diagnostics/m1/`. A worker crash, missing result, nonzero exit, or ten-minute timeout is a
required failure. Outside native Linux, simulator work is explicitly skipped; that is a contract
review, not physical task or camera validation.

On native Linux, the plain diagnostic uses `physx_cpu` and `render_backend="none"` for its required
simulator check. It also attempts CPU rendering when a Vulkan probe succeeds, but reports that
attempt as a warning rather than pretending it validates the target GPU path. Outside native Linux,
simulator execution is explicitly reported as not attempted so a C-level SAPIEN failure cannot
terminate a review run on an out-of-scope platform.

The M2 verifier checks project-owned expert contracts and, on native Linux, runs the privileged
`pd_joint_pos` expert across all three cubes and both bins with explicit seeded task overrides. In
target mode it first runs the M0 and M1 target gates in the main runtime, then verifies the isolated
planner runtime before any expert command. It
requires every rollout to complete all 12 phases and finish with the environment's conservative
`success` evaluation during the six-episode smoke. The 180-episode balanced benchmark then requires
at least 95% overall success, at least 90% success for each TaskSpec, zero successful completions
with a wrong object in the target bin, zero successful completions with the target in the wrong
bin, zero unclassified failures, and zero benchmark crashes. Planning, execution, grasp, transport,
placement, timeout, off-table, and verification failures remain separately visible in JSON
diagnostics. Outside the native Linux target, missing mplib and skipped physical execution are
reported as pending, never as successful expert validation.

## CPU-only review setup

A Linux reviewer without an NVIDIA GPU can exercise every CPU-safe test with a CPU PyTorch wheel:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install opencv-python==4.13.0.92 opencv-python-headless==4.13.0.92
python -m pip install -e ".[dev]"
pytest -m "not gpu and not rendering"
python environment/verify_install.py
python environment/verify_m1.py
python environment/verify_m2.py
python environment/verify_m3a.py
python environment/verify_m3b.py
python environment/verify_m4.py
python environment/verify_m42.py
python environment/verify_m43.py
python environment/verify_m5a.py
```

The same documented OpenCV dual-wheel risk applies to this review environment.

The rendering test has explicit `gpu`, `rendering`, and `integration` markers. It skips with a
specific reason when native Linux, CUDA, or `vulkaninfo` is absent. A skip is never accepted as the
target-machine GPU/rendering verification; the `--target` command is the authoritative gate.

## Hardware requirements by command

| Command | Native Linux | NVIDIA GPU / CUDA | Vulkan |
| --- | --- | --- | --- |
| `ruff format --check .`, `ruff check .`, `python -m build` | No | No | No |
| `pytest -m "not gpu and not rendering"` | No | No | No |
| `python environment/verify_install.py` | No | No | Optional attempt |
| `python environment/verify_m1.py` | Simulator only on native Linux | No | No |
| `python environment/verify_planner_runtime.py` | Yes | No | No |
| `python environment/verify_m2.py` | Physical expert on native Linux | Target mode | Target mode |
| `python environment/verify_m3a.py` | Structural only | No | No |
| `python environment/verify_m3a.py --target-smoke` | Yes | Yes | Via prior gates |
| `python environment/verify_m3a.py --target-full` | Yes | Yes | Via prior gates |
| `python environment/verify_m3b.py` | No | No | No; generated-array video fixture only |
| `python environment/verify_m3b.py --target-smoke` | Yes | Yes | Yes |
| `python environment/verify_m3b.py --target-full` | Yes | Yes | Yes |
| `python environment/verify_m4.py` | No | No | No; CPU ACT fixture only |
| `python environment/verify_m4.py --target-smoke --action-bound-mode project` | Yes | Yes | Validates completed prior M3B smoke evidence |
| `python environment/verify_m4.py --target-full --dry-run --action-bound-mode project` | Metadata only | No | Existing M3B full gate plus exact eight-run plan |
| `python environment/verify_m4.py --target-full --action-bound-mode project` | Yes | Yes | Via dataset and rollout gates |
| `python environment/verify_m42.py` | No | No | No; contracts and CPU fixture only |
| `python environment/verify_m42.py --target-development` | Yes | Yes | Yes; verifies the final lock but never materializes or executes sealed final episodes |
| `python environment/verify_m42.py --target-final` | Yes | Yes | Yes; separate explicit authorization |
| `python environment/verify_m43.py` | No | No | No; M4.3 audit plus FactorFiLM contracts/CPU fixture only |
| `python environment/verify_m43b.py` | No | No | No; read-only structural/evidence audit unless target-development evidence is supplied |
| `scripts/audit_act_semantics.py --dry-run` | No | No | No; validates an explicit audit plan only |
| `scripts/audit_act_semantics.py` | Real audit host | According to checkpoint device | No environment rollout |
| `scripts/train_act_factor_film.py --dry-run` | No | No | No; validates one immutable training identity only |
| `scripts/train_act_factor_film.py --fixture` | No | No | No; local forward/backward and save/reload only |
| `scripts/train_act_factor_film.py --target-development` | Yes | Yes | No rendering during offline training; no rollout or final access |
| `scripts/evaluate_act_factor_film.py` target stages | Yes | Yes | Yes; validation selection, fresh reload, and `m42_dev_v0` only |
| `python environment/verify_m43b.py` with completed target roots | Yes | No new training | No new rollout; independently audits completed physical evidence |
| `python environment/verify_m5a.py` | No | No | No; deterministic corpus/router/dispatch CPU fixtures only |
| `python environment/verify_m5a.py --verify-stage <stage>` | Stage-dependent | Stage-dependent | Independently validates exactly one immutable M5A stage report; final stays sealed |
| `scripts/export_lerobot_dataset.py` | Real export: yes | According to source/render backend | Yes |
| `scripts/validate_lerobot_dataset.py` | Full source alignment: yes | According to source/render backend | Yes |
| `scripts/inspect_lerobot_episode.py` | No | No | No |
| `scripts/train_act.py --dry-run` | No | No | No |
| `scripts/train_act.py --full` | Yes | Yes | No rendering during offline training |
| `scripts/evaluate_act.py` closed loop | Yes | According to checkpoint/device | Yes |
| `scripts/compare_act_baselines.py`, `scripts/inspect_act_checkpoint.py` | No | No | No |
| `python environment/run_expert.py`, `python environment/benchmark_expert.py` | Yes | As required by configured backend | As required by rendering mode |
| `collect_raw_demos.py`, `replay_raw_demos.py` | Yes | According to stored backend | No rendering |
| `inspect_raw_demos.py` | No | No | No |
| `pytest` GPU/rendering case | Yes | Yes | Yes |
| `python environment/verify_install.py --target` | Yes | Yes | Yes |
| `python environment/verify_m1.py --target` | Yes | Yes | Yes |
| `python environment/verify_m2.py --target` | Yes | Yes | Yes |
| `python environment/verify_m3a.py --target-smoke` | Yes | Yes | Yes |
| `python environment/verify_m3a.py --target-full` | Yes | Yes | Yes |

## Repository map

- `src/langmani/environments/`: M1 registration, typed metadata, environment, pure tensor task logic, and narrow expert-only state accessors.
- `src/langmani/experts/`: M2 types, isolated-runtime selection, direct lazy mplib adapter, and phase-based Panda expert.
- `src/langmani/collection/`: M3A recorder, collector, replay, manifests, and inspection.
- `src/langmani/datasets/`: M3A immutable types, stable IDs, schedules, and native archive checks.
- `src/langmani/datasets/lerobot_*.py`: M3B source gate, contracts, export, and validation.
- `src/langmani/policies/`: M4 ACT, M4.2 runtime/TaskToken, M4.3a semantic audit, and M4.3b FactorFiLM boundaries.
- `src/langmani/language/`: M5A corpus/split/schedule locks, routers, strict schemas, metadata-only controller registry, dispatch, evaluation, artifact validation, and failure attribution.
- `scripts/`: M3B data commands; M4/M4.2/M4.3 training and evaluation; and M5A corpus, classifier, router-evaluation, and control commands.
- `environment/`: reproducible declaration plus M0 through M5A diagnostics, including independent M4.3b evidence and M5A target-development verification.
- `tests/unit/`: environment/expert/data, ACT contracts, and M5A corpus/router/registry/dispatch/evidence checks.
- `tests/integration/`: LeRobot/ACT boundaries plus M5A classifier training, local-LLM inference, and frozen-controller dispatch.
- `tests/smoke/`: dependency, simulator, vectorization, rendering, expert, and raw collection acceptance.
- `docs/`: subsystem boundaries and decision records.
- `outputs/`: ignored generated diagnostics, datasets, checkpoints, and experiment outputs.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for planned boundaries and
[docs/DECISIONS.md](docs/DECISIONS.md) for tested assumptions and unresolved risks.
