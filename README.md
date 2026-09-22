# LangMani

LangMani is a language-conditioned robotic manipulation research repository. M0 established the
runtime foundation, M1 added the environment/language contracts, M2 added a deterministic
privileged Panda expert, and M3A implemented the authoritative ManiSkill-native raw archive.
M3B deterministically derives accepted episodes into a validated local LeRobotDataset v3.
M4A closes the single-task upstream ACT baseline on native Linux: formal M3A/M3B validation,
100,000-step training and paired closed-loop evaluation are complete. On 20 fresh red-cube/left-bin
scenes, ACT succeeded 10/20 and the privileged expert 20/20. Generated-array fixtures are not benchmarks.

M3B does not train ACT or SmolVLA, publish to the Hub, change M3A acceptance, export failure
trajectories, add sensors or language paraphrases, use multiprocessing, or change the M1/M2 task.

## Target platform

The acceptance target is a **native Linux** workstation with an NVIDIA RTX 4090, a compatible
NVIDIA driver, CUDA-capable PyTorch wheels, and a working Vulkan installation. Windows-native and
WSL execution are not supported target configurations.

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
segmentation to require all three cubes and both bins to be visible in `base_camera`. The native
M1 target gate passed during the 2026-09-21 acceptance run documented in D-030.

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
clearance. Every planned phase repeats the final joint target for two deterministic control steps,
then requires TCP position error at most 0.01 m and quaternion angular error at most 0.08 rad. The
fixed cube collision size is 0.05 m on each side.

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
structural checks on a non-target host. The current Windows review environment has no mplib
installation because ManiSkill 3.0.1 declares `mplib==0.1.1` only on Linux. Separate native Linux
RTX 4090 acceptance passed the ordered target gates and expert execution; see D-029/D-030 and
the M4A delivery record for the effective planner overlay and retained evidence.

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

## M4A — ACT Baseline

ACT provides a small visual imitation baseline for **red cube → left bin** in the existing M1
environment. It observes only `observation.images.base_camera` (RGB 256×256) and
`observation.state` (the nine measured Panda joint positions). It predicts 50 absolute
`pd_joint_pos` actions of eight components; upstream ACT executes ten queued actions before
querying again. The expert additionally sees object poses, bin geometry, grasp state, semantic
target identity, and planner state. None of these expert-only inputs reaches ACT.

ACT does **not** test language understanding. The dataset retains all six `canonical_v0` sentences
and TaskSpec metadata. This baseline selects one task by metadata and supplies no sentence,
task token, oracle one-hot feature, or custom language encoder to the policy.

The validated source is a **new, real, complete M3B export** at
`outputs/datasets/m3b/langmani-pick-place-lerobot-v1`, derived from the separately regenerated M3A
archive. Its repo ID, export fingerprint, exact file checksums and frame counts are read from
the actual source, never inferred from historical runs. The existing 48/6/6 scene split yields
48 training, 6 held-out validation and 6 excluded test episodes for the selected task. The seed
and every ID are saved in `split.json`. A one-scene M3B smoke export is insufficient for this split.

After the existing full M3A/M3B target gates pass, run from this repository root:

```bash
DATA=outputs/datasets/m3b/langmani-pick-place-lerobot-v1
python scripts/act_baseline.py validate --dataset-root "$DATA" \
  --report results/act_baseline/validation.json
python scripts/act_baseline.py split --dataset-root "$DATA" \
  --output results/act_baseline/split.json

# Actual-data smoke: official dataloading, forward/backward, optimizer, save and reload.
python scripts/act_baseline.py train --dataset-root "$DATA" \
  --split results/act_baseline/split.json --output results/act_baseline/smoke-seed0 \
  --mode smoke --device cuda --batch-size 8 --steps 3 --seed 0

# One full baseline; a clean Git commit is required. Optional W&B: add --wandb.
python scripts/act_baseline.py train --dataset-root "$DATA" \
  --split results/act_baseline/split.json --output results/act_baseline/full-seed0 \
  --mode full --device cuda --batch-size 8 --steps 100000 --seed 0

# Fresh, paired ACT/expert reset distribution. No held-out checkpoint tuning here.
python scripts/act_baseline.py evaluate --dataset-root "$DATA" \
  --split results/act_baseline/split.json \
  --checkpoint results/act_baseline/full-seed0/training/checkpoints/100000/pretrained_model \
  --output results/act_baseline/full-seed0/evaluation-seed42000-native-gripper-v2 \
  --device cuda --episodes 20 --seed 42000 --sim-backend physx_cpu
```

Validation decodes every row and rejects invalid schema, nonfinite values, ordering, FPS,
episode boundaries, task metadata and extra/privileged features without repair. Training uses
LeRobot **0.6.0's official trainer**, optimizer and checkpoint/resume stack. The narrow dataset
factory adapter replaces whole-dataset state/action statistics with exact train-view statistics;
RGB uses fixed ImageNet statistics. Resume, CPU fixture commands, architecture mapping, workload
estimation and the result schema are in [the M4A guide](docs/M4A_ACT_BASELINE.md).

Run directories contain `config.json`, `split.json`, `validation.json`, `normalization.json`,
`checkpoint_metadata.json`, `status.json` and upstream `training/checkpoints/`. Evaluation adds
`metrics.json`, `episodes.csv`, config and split copies. Success rate, length mean/population
standard deviation, termination reasons, paired initial-state hashes, expert rate and absolute
rate gap are recorded. Invalid/nonfinite predictions and out-of-bounds arm joints fail before
stepping. The normalized gripper saturates to `[-1, 1]`, matching the native controller exactly;
saturation counts and maximum overshoot are recorded. A zero-ACT-step evaluation fails acceptance.
All `results/` artifacts, datasets, videos and checkpoints are ignored by Git.

| Formal benchmark | Expert | ACT | Absolute gap |
| --- | --- | --- | --- |
| Regenerated real M3B, red cube → left bin; 20 paired fresh scenes | 20/20 (100%) | 10/20 (50%) | 50 percentage points |

On 2026-09-21, execution commit `915d823` passed 411 native Linux regression tests,
the ordered M0/M1/M2 target gates, and real six-episode M3A collection/replay smoke.
The M2 gate scored 177/180 across its six tasks; this is an expert prerequisite,
not the later ACT/expert comparison. Formal M3A collection and independent replay of all 360 episodes,
full M3B validation (64,548 frames), and exhaustive M4A validation passed. The selected task has
8,588 training frames, 1,073 held-out frames and 1,077 excluded test frames. Actual-data CUDA smoke
passed, including three paired 200-step ACT rollouts (0/3 success; expert 3/3).
Full ACT training completed 100,000 steps with batch 8, seed 0 and checkpoint reload: 800,000
sample presentations, 93.153 equivalent train-frame passes, 17,200.83 seconds of training.
The original final evaluation rejected the first gripper action in every scene and executed zero
ACT steps, so it is not an executed closed-loop baseline. Evaluation commit `0dafc31` repairs that
native-controller mismatch. The same checkpoint and exact 20-scene schedule completed evaluation
at 2026-09-21 15:56:06 UTC: ACT executed 3,311 steps, with 10 successes and 10 timeouts at 200 steps;
the expert succeeded in all 20 scenes. Every initial-state hash pair matched, source scenes were
excluded, and there were no infrastructure errors. ACT mean/population-standard-deviation episode
length was 165.55/34.54 steps; expert 178.00/5.83. The native gripper saturation audit counted
464 commands, maximum overshoot 0.0211761; arm commands remained strictly checked.
This is one training seed and 20 fresh reset scenes, not evidence of language generalization.
See [the A–H delivery record](docs/M4A_DELIVERY.md) for measured workload, local recovery archives,
all 32 changed files, exact commands, test results and preserved failure details.

Track **implementation complete**, **smoke tested**, **full dataset validated**, **full ACT trained**
and **closed-loop evaluated** separately. Generated fixtures test code and upstream compatibility;
all five are now complete for this recorded run. Dataset, final checkpoint and evaluation evidence
are backed up locally with archive and per-file SHA-256 verification.
SmolVLA is outside M4A.

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
```

The native mplib 0.1.1 expert needs a NumPy 1 overlay, while LeRobot retains the
declared NumPy 2 environment. Create it once from the activated base environment:

```bash
python -m venv --system-site-packages .venv-planner
.venv-planner/bin/python -m pip install --no-deps -r environment/planner-runtime.txt
export LANGMANI_PLANNER_PYTHON="$PWD/.venv-planner/bin/python"
"$LANGMANI_PLANNER_PYTHON" environment/verify_planner_runtime.py
```

Keep that variable exported for M2/M3A gates and M4A evaluation. M3A records the
effective imported planner versions. M4A executes the reference expert in that
interpreter and checks its reset-state hash against ACT's independent reset;
worker JSON, runtime pins, and logs are retained alongside evaluation metrics.
M1 CPU and GPU PhysX diagnostics run in isolated subprocesses.

M3B activates the dataset extra; M4A additionally activates `lerobot[dataset,training]==0.6.0`
for the official trainer and optional W&B. The base package alone deliberately refuses
`lerobot.datasets` imports. The reviewed environment resolved datasets 4.8.5, pandas 2.3.3,
PyArrow 25.0.0, PyAV 15.1.0, TorchCodec 0.11.1, and jsonlines 4.0.0. M3B explicitly uses PyAV for
writing and reading because the installed Windows TorchCodec DLL chain is not loadable. Exact
LeRobot/PyAV/libavcodec values are checked and stored in every export fingerprint and manifest.

On native Linux, ManiSkill 3.0.1's package metadata installs `mplib==0.1.1`; Windows does not receive
that conditional dependency. LangMani does not change or duplicate the upstream pin. The M2 adapter
checks the exact installed version at runtime and fails clearly when it is missing or changed.

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
pytest
```

For auditability, the orchestrated command prints and records its six subprocesses:
`verify_install.py --target`, then `verify_m1.py --target`, two repeatability runs of the M2
six-episode all-task smoke, one balanced 180-episode benchmark, and one rendered expert rollout
whose 12 phase PNGs are checked. A failed prerequisite stops the later stages. The balanced run
uses the ordered scene seeds `0..29`, visits the canonical six TaskSpecs for every seed, and thus
contains exactly 30 episodes per TaskSpec.

`--target` returns a nonzero exit code if any of these are unavailable or invalid: native Linux,
PyTorch CUDA, the CUDA runtime reported by PyTorch, an NVIDIA GPU, `vulkaninfo`, the Vulkan probe,
explicit ManiSkill PhysX CUDA simulation, an RGB policy observation, `rgb_array` rendering, a
single environment step, or writing `outputs/diagnostics/pickcube_rgb.png`.

The M1 target diagnostic separately verifies the custom environment's seeded counterfactual reset,
positive and rejected placements in the real CPU scene, sparse reward and a real step,
six-environment GPU vectorization plus partial reset masks, all numeric evaluation fields, visual
no-leakage, per-actor policy-camera visibility, Panda-hand visibility, and the separate human camera.
It writes a machine-readable report and two
frames under `outputs/diagnostics/m1/`. Outside native Linux, its simulator work is explicitly
skipped; that is a contract review, not physical task or camera validation.

On native Linux, the plain diagnostic uses `physx_cpu` and `render_backend="none"` for its required
simulator check. It also attempts CPU rendering when a Vulkan probe succeeds, but reports that
attempt as a warning rather than pretending it validates the target GPU path. Outside native Linux,
simulator execution is explicitly reported as not attempted so a C-level SAPIEN failure cannot
terminate a review run on an out-of-scope platform.

The M2 verifier checks project-owned expert contracts and, on native Linux, runs the privileged
`pd_joint_pos` expert across all three cubes and both bins with explicit seeded task overrides. In
target mode it first runs the M0 and M1 target gates. It
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
| `python environment/verify_m2.py` | Physical expert on native Linux | Target mode | Target mode |
| `python environment/verify_m3a.py` | Structural only | No | No |
| `python environment/verify_m3a.py --target-smoke` | Yes | Yes | Via prior gates |
| `python environment/verify_m3a.py --target-full` | Yes | Yes | Via prior gates |
| `python environment/verify_m3b.py` | No | No | No; generated-array video fixture only |
| `python environment/verify_m3b.py --target-smoke` | Yes | Yes | Yes |
| `python environment/verify_m3b.py --target-full` | Yes | Yes | Yes |
| `scripts/export_lerobot_dataset.py` | Real export: yes | According to source/render backend | Yes |
| `scripts/validate_lerobot_dataset.py` | Full source alignment: yes | According to source/render backend | Yes |
| `scripts/inspect_lerobot_episode.py` | No | No | No |
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
- `src/langmani/experts/`: M2 types, direct lazy mplib adapter, and phase-based Panda expert.
- `src/langmani/collection/`: M3A recorder, collector, replay, manifests, and inspection.
- `src/langmani/datasets/`: M3A immutable types, stable IDs, schedules, and native archive checks.
- `src/langmani/datasets/lerobot_*.py`: M3B source gate, contracts, export, and validation.
- `scripts/`: M3B export, validation, and read-only episode inspection.
- `environment/`: reproducible declaration plus M0/M1/M2/M3A/M3B diagnostics and commands.
- `tests/unit/`: metadata, no-leakage, expert, archive, corruption, replay, resume, and CLI checks.
- `tests/smoke/`: dependency, simulator, vectorization, rendering, expert, and raw collection acceptance.
- `docs/`: subsystem boundaries and decision records.
- `outputs/`: ignored generated diagnostics and future experiment outputs.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for planned boundaries and
[docs/DECISIONS.md](docs/DECISIONS.md) for tested assumptions and unresolved risks.

## M4B: paired destination selection with SmolVLA

M4B is implemented separately from the frozen M4A ACT baseline. It uses the red cube and two
existing destinations to test whether changing only text changes goal choice. See
[the frozen protocol](docs/M4B_PROTOCOL.md) and [commands and evidence status](docs/M4B_DELIVERY.md).
Use the optional `m4b` extra only in an isolated environment. The full dataset, native input pairing,
real-data smoke and checkpoint resume have passed validation. Full training (20,000 steps/batch 8/
seed 0), paired language evaluation and local recovery verification are complete.

| Language condition | Original requested-goal success |
| --- | ---: |
| Correct canonical instruction | 15/40 (37.5%) |
| Swapped canonical instruction | 0/40 |
| Blank | 0/40 |
| Held-out paraphrase | 0/40 |

In **6/20 identical-scene pairs**, changing only the canonical instruction caused the corresponding
successful left/right goal change. Swapped text reached its supplied goal in 15/40 scored rows;
the 0/40 figure above scores the original opposite request. There are 100 physical policy rollouts
and 160 explicitly linked scoring rows, with 85 physical timeouts and no infrastructure errors.
This supports limited template-dependent goal selection; 0/40 paraphrase success establishes no
robustness to the tested reformulations. M4A remains frozen and is not a direct architecture comparison.
