# Environment and dependency decisions

This file records decisions through the active M5A milestone as of 2026-07-19.
“Metadata-compatible” means official package requirements have a non-empty version intersection;
it is not a claim of native Linux GPU or rendering success.

## D-001 — Platform boundary

The acceptance platform is native Linux with a compatible NVIDIA GPU. Initial acceptance evidence
used an RTX 4090, and later milestone-specific target evidence also used RTX 5090 hosts. A GPU model
name alone is never acceptance: each target command must still prove its required CUDA, Vulkan,
PhysX, rendering, and dependency contracts. Windows-native and WSL-specific workarounds are
intentionally out of scope because ManiSkill documents GPU simulation and rendering support on
native Linux/NVIDIA, while WSL lacks those paths.

NVIDIA drivers, the Vulkan loader/ICD, and `vulkaninfo` are system responsibilities. The diagnostic
reports them but never installs or changes system packages.

## D-002 — Python and environment manager

Use Miniforge/conda to create Python `3.12.13`; use pip inside that environment for Python
packages. LeRobot 0.6.0 requires Python 3.12 or newer, while ManiSkill 3.0.1 pins Linux
`mplib==0.1.1`, whose published wheels stop at CPython 3.12. The project therefore declares
`>=3.12,<3.13` rather than claiming Python 3.13 support.

PyTorch is not installed through conda. The CUDA wheel and its bundled runtime are selected from
PyTorch's official wheel index, avoiding a second conda-managed PyTorch/CUDA stack.

## D-003 — Candidate M0 version set

Direct project dependencies were pinned after checking published metadata intersections, then
installing the CPU variants together locally and passing `pip check` plus import tests. The target
CUDA wheel choices also match LeRobot 0.6.0's published constraints and PyTorch's official version
pairing. This establishes resolver/import compatibility only; the CUDA/renderer combination remains
a candidate until the native Linux gate passes. The table records the original bootstrap status;
subsequent native-target evidence and the planner ABI split are recorded in D-032.

| Component | Selected version | Decision status |
| --- | --- | --- |
| Python | 3.12.13 | Available for Linux; local review used 3.12.10; target pending |
| PyTorch | 2.11.0+cu128 | Local 2.11.0+cpu import passed; target CUDA pending |
| torchvision | 0.26.0+cu128 | Local 0.26.0+cpu import resolved; target pending |
| ManiSkill | 3.0.1 | Local import passed; target GPU/render execution pending |
| SAPIEN | 3.0.3 | Local import path loaded; target simulator execution pending |
| LeRobot | 0.6.0 + `dataset` extra | Local writer/reader fixture passed; target export pending |
| PyAV / PyArrow | 15.1.0 / 25.0.0 | Local H.264/yuv444p decode and Parquet reads passed; target pending |
| Gymnasium | 1.2.3 | Local import passed; target simulator execution pending |
| h5py | 3.16.0 | Local M3A synthetic archive checks passed; target native recording pending |
| NumPy | 2.2.6 | Local import passed; target simulator execution pending |
| Pillow | 12.3.0 | Local import passed; target frame persistence pending |
| OpenCV wheels | 4.13.0.92 | Local `cv2` import passed; unresolved collision below |
| pip | 26.1.2 | Environment installer pin; Linux conda solve passed in dry-run |
| setuptools | 80.9.0 | Satisfies LeRobot `>=71,<81`; also pins the PEP 517 backend |
| pytest | 8.4.2 | Local test run passed with documented hardware/platform skips |
| Ruff | 0.15.21 | Local format and lint checks passed |
| build | 1.3.0 | Local sdist and wheel builds passed |

The `+cu128` local label is selected in `environment/environment.yml`; the public project
dependency remains `torch==2.11.0`, which accepts CPU and CUDA local builds for CPU-only review.
The CUDA 12.8 path requires a sufficiently new NVIDIA driver; LeRobot's current installation guide
states a minimum of 570.86.

Official sources consulted:

- [ManiSkill installation and system support](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [ManiSkill environment API](https://maniskill.readthedocs.io/en/latest/api/mani_skill/envs/sapien_env/)
- [ManiSkill 3.0.1 package metadata](https://pypi.org/pypi/mani-skill/3.0.1/json)
- [LeRobot 0.6.0 package metadata](https://pypi.org/pypi/lerobot/0.6.0/json)
- [LeRobot installation guide](https://huggingface.co/docs/lerobot/installation)
- [PyTorch version pairing](https://pytorch.org/get-started/previous-versions/)

## D-004 — Base LeRobot only

M0 installs `lerobot==0.6.0` without `dataset`, `training`, `smolvla`, or `all` extras. Those extras
would prematurely introduce dataset codecs and policy dependencies. FFmpeg is deferred until the
dataset milestone for the same reason.

D-024 supersedes the `dataset` part of this decision now that M3B is active; training, SmolVLA, and
`all` extras remain excluded.

## D-005 — Explicit simulator and renderer backends

CPU-safe tests use one `PickCube-v1` environment with `obs_mode="state"`,
`sim_backend="physx_cpu"`, and `render_backend="none"`.

The target verifier uses `obs_mode="rgb"`, `render_mode="rgb_array"`,
`sim_backend="physx_cuda"`, and `render_backend="sapien_cuda"`. It separately validates RGB leaves
in the policy observation and the viewer frame returned by `env.render()`. This distinction matters:
`render_mode="rgb_array"` does not prove that the policy observation contains RGB.

The verifier checks `env.unwrapped.gpu_sim_enabled`; it never relies on the `auto` backend, which
would choose CPU physics for a single environment.

## D-006 — Test and failure policy

`gpu`, `rendering`, and `integration` are strict pytest markers. CPU-safe review excludes the first
two. Hardware tests have explicit platform/tool skips, but any failure after prerequisites exist is
a test failure rather than a skip.

`environment/verify_install.py --target` is the authoritative native Linux gate. It exits nonzero
for missing or failed CUDA, Vulkan, GPU physics, RGB observation, rendering, step, or diagnostic
frame persistence. The default diagnostic still exits nonzero for missing pinned dependencies or a
failed native-Linux CPU environment reset/step, while unavailable rendering and simulator checks on
out-of-scope operating systems are reported as explicit warnings.

The verifier enables Python's fault handler so a native SAPIEN abort or segmentation fault emits a
Python stack before the process exits nonzero. Such a native crash cannot be folded into the normal
summary, but it remains visible rather than being silently converted into a skip or success.

## D-007 — No Docker or CI in M0

Docker is deferred. CI is also deferred because no required GPU runner is available. Adding a CPU
CI job alone would not satisfy the GPU/rendering definition of done and is outside this task.

## D-008 — M1 registration and project-owned metadata

M1 registers exactly `LangMani-PickPlaceByInstruction-v0` with Gym/ManiSkill's
`register_env(..., max_episode_steps=200)`. Importing `langmani.environments` is the explicit
registration action; the root package does not eagerly import SAPIEN. The environment rejects every
robot UID except `panda` because ManiSkill 3.0.1's `SUPPORTED_ROBOTS` check only warns. Supported
reward modes are exactly `sparse` and `none`; no dense or normalized-dense implementation exists.

`TaskSpec` and `EpisodeSpec` are frozen project-owned dataclasses with fresh dictionary serializers.
The reset override accepts one strict JSON-shaped `options["task_spec"]` mapping and broadcasts it to
the environments being reset. It validates before delegating to `BaseEnv.reset`, so malformed IDs or
fields cannot mutate simulation state. `reset_to_env_states` is rejected until a later milestone
defines how non-simulator task state participates in state restoration.

Scene and task identity are intentionally orthogonal. `scene_id` is a versioned literal plus the
actual episode seed; `task_id` is a versioned literal plus semantic IDs. Neither uses Python's
process-randomized `hash()`. Without an override, seed modulo six selects the task after all layout
randomness has been consumed.

## D-009 — Primitive scene and deterministic reset

All non-robot M1 scene geometry is project-owned primitive box geometry. This includes the table,
ground, three dynamic cubes, and two static five-box shallow bins. The installed ManiSkill
`TableSceneBuilder` is not used directly for geometry because its visual table is a GLB; a small
subclass supplies primitive geometry while calling its installed Panda initialization method. This
preserves the official Panda base pose and qpos convention without copying ManiSkill source.

The environment requires `enhanced_determinism=True` and fixes robot joint reset noise to zero.
Cube placement uses ManiSkill's per-environment `_batched_episode_rng`: three disjoint y cells receive
bounded jitter and are randomly permuted across colors. This rejection-free construction guarantees
cube separation and source/bin separation for every seed. Task selection happens only after scene
layout. Cube linear and angular velocities are explicitly reset even though BaseEnv also clears
dynamic state.

The numeric layout constants are source x `[-0.18, -0.07]` m; y centers `[-0.12, 0, 0.12]` m with
`±0.012` m jitter; cube half-size `0.025` m; bin centers `(0.08, ±0.18)` m; and bin interior
half-width `0.085` m. Left is positive y from the Panda's view.

The source region is at most about 0.56 m horizontally from the Panda base, and the bin centers are
about 0.72 m away. The bins were deliberately moved inward from `(0.16, ±0.24)` after API review
identified that placement as too close to the Franka workspace edge. This is a conservative layout
decision, not a claim that an expert or IK trajectory has been validated in M1.

## D-010 — Language and observation leakage boundary

M1 supports one template ID, `canonical_v0`, and six exact English object/bin combinations. Strings,
scene IDs, and task IDs are materialized only by `get_task_texts()` and `get_episode_specs()`. Those
accessors return tuples and frozen dataclasses for both single and batched environments. They are not
called by `step()`, `evaluate()`, or observation construction.

ManiSkill 3.0.1 invokes `_get_obs_extra()` for visual modes as well as state modes. Therefore the
environment gates every privileged value with `self.obs_mode_struct.use_state`. Visual-only modes
receive only realistic robot proprioception, TCP pose, and requested camera textures. Privileged
state adds all three cube poses, both bin floor centers, and numeric target object/bin indices. Pose
and center fields are flattened to `(N, 21)` and `(N, 6)` because ManiSkill 3.0.1's standard `state`
and combined `state+visual` modes horizontally concatenate every numeric leaf. A pure Torch unit test
and a renderer smoke test enforce this boundary; a combined state-plus-visual mode is allowed to
contain the privileged state by definition.

## D-011 — Conservative evaluation, failure, and reward semantics

`evaluate()` computes one `(N, 3, 2)` cube/bin containment matrix and returns eight `(N,)` boolean
tensors: `target_in_target_bin`, `target_in_wrong_bin`, `wrong_object_in_target_bin`,
`target_is_grasped`, `target_is_static`, `target_off_table`, `success`, and `fail`. The per-step path
uses Torch stack/gather/index operations and performs no host conversion.

Containment uses a sphere enclosing the cube, an additional 2 mm wall clearance, and a 5 mm
resting-height tolerance. This deliberately rejects hovering above a bin and is conservative for a
rotated cube. Success additionally requires release, static thresholds of 0.01 m/s and 0.1 rad/s,
no wrong object in the selected bin, and no off-table condition. Wrong placement remains recoverable;
only falling until the cube's enclosing sphere is below the tabletop is `fail`. The sparse reward
inherits ManiSkill's `success.float() - fail.float()` and `none` remains zero.

## D-012 — M1 cameras and verification boundary

`base_camera` is a fixed 256×256 policy sensor using ManiSkill's fastest `minimal` shader. Its eye,
target, vertical FOV, near, and far values are `(0.65, -0.75, 0.70)`, `(-0.04, 0, 0.08)`, `1.05`,
`0.01`, and `10.0`. The separate `render_camera` is 512×512 with eye `(0.78, -0.90, 0.82)`, the same
target, FOV `1.0`, planes `0.01/10.0`, and the `default` shader. M1 adds neither a wrist camera nor
domain randomization.

`environment/verify_m1.py --target` is the M1 physical gate. In addition to native Linux, CUDA, and
Vulkan prerequisites, it requires positive and rejected placements in the real CPU scene, sparse
reward and real environment steps, a six-environment GPU reset, a two-slot partial reset mask, all
six default tasks, policy RGB, visual no-leakage, per-actor segmentation pixel evidence, Panda-hand
visibility, a non-flat separate human frame, and persisted diagnostic artifacts. The non-target
command performs contract checks everywhere and CPU simulation only on native Linux. A skip outside
that platform is not physical validation.

M1 implementation choices were checked against the locally installed ManiSkill 3.0.1 source and
the matching official examples rather than recalled APIs:

- [ManiSkill task-building introduction](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/intro.html)
- [ManiSkill advanced task-building guide](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html)
- [PlaceSphere 3.0.1 shallow-bin implementation](https://github.com/haosulab/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/place_sphere.py)
- [StackCube 3.0.1 batched evaluation](https://github.com/haosulab/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/stack_cube.py)
- [ManiSkill RNG guidance](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html)

## D-013 — M2 privileged expert boundary and control contract

M2 implements a project-owned `PickPlaceExpert` only for
`LangMani-PickPlaceByInstruction-v0`. It is a privileged demonstration generator and must reject an
incorrect environment ID, `num_envs != 1`, a control mode other than `pd_joint_pos`, an invalid or
missing active `TaskSpec`, absent semantic actors/bins, missing Panda handles, or an initially
terminal target. M2 does not vectorize mplib or introduce multiprocessing.

The environment remains the owner of ManiSkill handles. Its explicit
`get_expert_task_context()` boundary associates stable semantic IDs with the target object, target
bin, all object/bin handles, bin floor center, geometry constants, Panda agent, and robot. Its
`get_expert_evaluation()` boundary exposes the existing conservative task diagnostics. Both require
a single reset environment and are expert-only runtime interfaces. They add nothing to Gym
observations or per-step `info`, so D-010 and every M1 visual no-leakage test remain unchanged.

`ExpertConfig` fixes `pd_joint_pos` and defaults to a 200-step rollout, one planning attempt per
motion phase, 0.08 m pre-grasp clearance, 0.05 m grasp approach distance, 0.10 m lift clearance,
0.10 m transport clearance, 0.002 m placement clearance, 15 settling steps, and a 0.10 m retreat.
Diagnostic rendering is opt-in. Configuration validation rejects unsupported planner seeds and
timeouts rather than silently ignoring them.

## D-014 — Direct lazy mplib 0.1.1 adapter and deterministic geometry

The M2 planner is a thin project-owned adapter over public mplib 0.1.1 APIs. It imports mplib lazily
because ManiSkill 3.0.1 declares `mplib==0.1.1` only on Linux, and the Windows review environment
therefore has no mplib installation. The adapter checks the exact upstream version, builds
`mplib.Planner` from the installed Panda URDF/SRDF and public robot handles, sets the robot base pose,
synchronizes the current nine-joint simulator qpos, plans seven arm joints, and manages the attached
0.05 m cube collision box. Runtime code does not import ManiSkill's example runner or example
planning solver, and no upstream implementation is copied into LangMani.

M2 does not publish a world point cloud to mplib. The 0.1.1 collision implementation only checks an
attached tool against world points when point-cloud collision is enabled, so updating the attached
box is state synchronization rather than a claim of planner-side obstacle avoidance in M2. Explicit
phase clearances and PhysX contacts are the active safeguards; adding deterministic world collision
geometry is deferred until target evidence shows it is needed.

M2 calls `plan_screw` only. mplib 0.1.1 exposes neither a public planner seed nor a timeout for its
screw planner. Its RRT planning path has a planning-time argument but no seed, so an RRT fallback
would violate the deterministic M2 contract and is intentionally absent. Fixed simulator state plus
fixed target poses are the deterministic planning inputs. Upstream status strings beginning with
`IK Failed` map to `ik_failure`; other unsuccessful planner statuses map to `planning_failure`.
The selected 0.1.1 `plan_screw` implementation actually collapses differential-IK, joint-limit, and
collision failures into `screw plan failed`, so its real default path produces `planning_failure`.
The distinct `ik_failure` remains a project interface value for an adapter that can support that
diagnosis; M2 does not infer it without evidence.

The sole grasp is a top grasp of the semantic target cube. The world approach vector is
`(0, 0, -1)` and closing vector is `(0, -1, 0)`. ManiSkill's Panda convention constructs TCP
orientation columns as `[closing × approach, closing, approach]`, yielding `diag(1, -1, -1)` and
wxyz quaternion `(0, 1, 0, 0)`; TCP local `z` points down and local `y` is the finger-closing
direction. This matches the installed table-scene Panda's initial TCP orientation. The opposite
closing sign would create an exact pi relative rotation on the first motion, which mplib 0.1.1's
screw planner explicitly rejects. The planner pose convention is
`[x, y, z, qw, qx, qy, qz]`. The TCP grasp center is the cube center, and the fixed cube assumption
is a 0.025 m half extent on each axis. Pre-grasp/approach offsets are along the opposite of the
downward approach and retain table clearance. Each planned motion repeats its final joint target
for two deterministic control steps before checking 0.01 m TCP position and 0.08 rad sign-invariant
quaternion-angle tolerances.

D-032 supersedes only those final-waypoint repetition and TCP-position constants after native target
evidence; the grasp/orientation construction and every M1 success threshold remain unchanged.

Placement uses the selected bin's interior center and floor height, never actor ordering. The
0.085 m bin interior half-width minus the 0.025 m cube half extent leaves 0.060 m nominal clearance
to every centered wall. M2 validates an explicit 0.010 m expert fit margin, and the centered pose
also exceeds the existing 0.002 m M1 evaluator containment clearance. The
placement center height adds the cube half extent and configured 0.002 m placement clearance to the
bin floor top. The target must still pass the existing release, static, wrong-object, wrong-bin, and
off-table checks; construction of a centered target pose is not itself success.

The installed-source review used ManiSkill 3.0.1's Panda planning examples as behavioral references
for `pd_joint_pos`, Panda joint/action layout, grasp orientation, and planner synchronization, but
the runtime dependency is only the direct mplib adapter. The public mplib 0.1.1 interfaces were
checked against its installed/distributed source and reference documentation:

- [mplib 0.1.1 Planner reference](https://motion-planning-lib.readthedocs.io/v0.1.1/reference/Planner.html)
- [mplib 0.1.1 path-planning tutorial](https://motion-planning-lib.readthedocs.io/v0.1.1/tutorials/plan_a_path.html)

## D-015 — Explicit phases, failure/result contract, and ordered acceptance

The expert executes exactly these stable phases: `initialize`, `move_to_pregrasp`,
`approach_target`, `close_gripper`, `verify_grasp`, `lift_target`, `move_above_destination`,
`descend_to_place`, `open_gripper`, `settle_after_release`, `retreat`, and `verify_task`. Each phase
has explicit entry conditions, a pose or command, bounded attempts, completion criteria, a stable
failure status, planning/execution timing, and step/call/replan accounting. A rollout never hides
the solution inside one opaque planner call.

Terminal statuses are exactly `success`, `invalid_task`, `initialization_failure`, `ik_failure`,
`planning_failure`, `execution_failure`, `grasp_failure`, `transport_failure`,
`placement_failure`, `verification_failure`, `target_off_table`, `timeout`, and
`unexpected_exception`. Unexpected exceptions may be translated only by the outer command boundary,
which preserves exception type and message and exits nonzero for a single rollout. Only explicit
batch benchmark mode may continue to collect later failures.

`ExpertResult` is JSON-serializable and contains success/status, scene seed, stable scene/task IDs,
canonical instruction, semantic target IDs, total environment steps, planning calls, replans,
completed phases, failed phase, final environment evaluation, ordered per-phase results, planning
and execution durations, optional diagnostic paths, and exception type/message only when applicable.
It excludes tensors, observations, frames, complete planned paths, and trajectories.

The M2 command boundary consists of `environment/run_expert.py`,
`environment/benchmark_expert.py`, and `environment/verify_m2.py`. M2 target acceptance cannot be
claimed from an isolated expert run: the native Linux RTX 4090 gate must run, in order,
`verify_install.py --target`, `verify_m1.py --target`, and `verify_m2.py --target`. The final command
orchestrates the first two target gates before it starts M2, so the sequence cannot be bypassed, and
must physically solve all six semantic object/bin combinations twice with matching non-timing
signatures. It then requires one successful rendered expert rollout and 12 nonempty phase PNGs.
Windows structural checks, missing
mplib, imports, mocked planner tests, skipped simulator work, or diagnostic JSON generation are not
physical expert acceptance.

ManiSkill reports `terminated=True` once M1 success becomes true, potentially during settling. The
fixed M2 sequence intentionally allows only that known success terminal to continue through retreat
and final verification. Truncation, off-table failure, and any termination without conservative
success still stop with a classified failure. This post-success continuation is phase-completion
behavior, not trajectory collection.

## D-016 — Statistical M2 target acceptance before demonstration collection

The two identical six-task runs from D-015 remain a deterministic all-task smoke and non-timing
signature check, but they are not by themselves sufficient evidence for beginning M3A collection.
The native target gate now additionally runs `benchmark_expert.py` over the ordered scene seeds
`0..29`. Because the benchmark visits the canonical object-major, bin-minor six-task matrix for
every seed, this produces exactly 180 rollouts and exactly 30 episodes for each TaskSpec.

The 180-rollout report is accepted only when overall success is at least 95%, every TaskSpec success
rate is at least 90%, no successful result reports `wrong_object_in_target_bin=True`, no successful
result reports `target_in_wrong_bin=True`, every failure has a known classified M2 status, and there
are no command-level or caught unexpected-exception crashes. The verifier also requires the report
to contain the exact seed-major semantic row order, complete counts, and agreement between every
successful `ExpertResult` and M1's final `success` evaluation.

`benchmark_expert.py` still exits nonzero when any individual rollout fails. For the statistical
run, `verify_m2.py` therefore treats exit code zero or one as a completed command only when a fresh
benchmark JSON artifact exists, then applies the stricter explicit criteria above. A missing report,
another exit code, malformed matrix, threshold miss, safety-label violation, unclassified failure,
or crash fails target acceptance. This changes only the acceptance orchestration: M2 expert phase
logic, planner behavior, dependency versions, and M1 conservative success geometry are unchanged.

## D-017 — ManiSkill-native archives are the authoritative M3A source

M3A records each expert attempt through ManiSkill 3.0.1 `RecordEpisode` with actions and environment
states enabled, `obs_mode="none"`, rewards and videos disabled. The accepted native schema requires
an empty observation group and forbids a rewards dataset. The resulting HDF5/JSON pair is the
authoritative raw source; M3A does not introduce Parquet, MP4 policy-camera exports, or a
LeRobotDataset. Training-oriented conversion remains an M3B responsibility.

The installed wrapper opens a trajectory pair in write mode and does not provide append-safe resume.
M3A therefore records bounded per-candidate files, closes them before validation, and publishes only
new immutable group bundles and source shards. `h5py==3.16.0` is now an explicit direct dependency
because project-owned corruption, schema, checksum, and shard validation imports its public API.
This is a dependency-interface change required by M3A rather than a change to the established
ManiSkill, mplib, CUDA, or rendering versions.

## D-018 — Complete counterfactual scene groups are the admission boundary

The collection schedule is seed-major and uses the canonical object-major, bin-minor TaskSpec order:
red-left, red-right, green-left, green-right, blue-left, blue-right. Persistent identities use
canonical JSON plus SHA-256 digests; Python's randomized `hash()` and filesystem enumeration order
are never identity inputs.

A candidate seed is admitted only as an atomic six-episode `CounterfactualSceneGroup`. Every member
must have an M2 success result, a closed and structurally valid native trajectory, successful action
replay, final M1 success, and exact TaskSpec/provenance agreement. One failed task rejects the whole
candidate. Successful siblings are retained only as rejected-attempt diagnostics and are never
mixed into accepted shards. The default target is 60 complete groups, hence exactly 360 accepted
episodes and 60 per TaskSpec; bounded retries and candidate limits may not weaken this rule.

## D-019 — Replay is a project-owned, single-environment audit

The installed ManiSkill replay utilities are behavioral references but do not preserve LangMani's
TaskSpec reset option as required and are not used as the M3A authority. M3A supplies a small
single-environment replay path that resets with the recorded scene seed and exact task override,
executes every recorded action, verifies the final M1 evaluation, rejects wrong-object and wrong-bin
outcomes, compares the complete fresh-reset state tree with recorded state zero, and compares final
target pose and robot configuration against explicit tolerances.

When configured, a state-replay audit additionally restores every recorded environment state through
the environment state API after the semantic reset and immediately reads back every numeric leaf
within an absolute `1e-6` tolerance. State restoration is an audit aid, not a substitute for action
replay. It does not add privileged fields to visual policy observations and does not change the M1
leakage contract.

## D-020 — Resume is manifest-driven and sealed data is immutable

M3A writes deterministic JSON/JSONL projections for attempts, episodes, scene groups, and source
shards, then writes the atomic canonical manifest last as the generation commit marker. A per-
candidate transaction journal is written before an expert attempt begins and after its compact
result is available. A resume validates the configuration fingerprint and every sealed shard,
rejects an interrupted candidate exactly once without reusing its attempt IDs, repairs only an open
partial shard from validated group bundles, and completes any post-commit failure-retention move
before clearing its journal. Each accepted group commits its bundle-pair digests in the manifest;
repair refuses a missing or changed bundle instead of recalculating authority from potentially
tampered actions. Corruption of a sealed source pair is a hard failure rather than an invitation to
recollect or silently overwrite evidence.

Inspection treats the manifest, its projections, and native pairs as one provenance graph. It
recomputes collection, candidate, attempt, raw, group, and shard IDs; requires contiguous candidate
and shard indices; checks native and manifest task/expert/replay/step evidence; enforces the stored
replay mode and configured tolerances; and rejects orphan HDF5 or JSON files. Overwrite requires an
exact marker plus a parseable manifest owning the resolved root, except for a narrowly recognized
interrupted first commit. Command reports are invalidated before execution, and target replay
acceptance requires 360 strictly parsed, state-audited results in exact manifest order.

Overwrite is explicit and confined to a marked dataset directory under the repository's ignored
`outputs/` tree. Failed attempts always receive small JSON diagnostics; their full native
trajectories are retained only when the configured failure-retention flag is enabled, under a
separate failure area. The target verifier must first run the ordered M0, M1, and M2 target gates,
then collect/resume, inspect all accepted shards, and independently action-replay all 360 episodes.
Non-Linux structural checks cannot establish this physical acceptance.

## D-021 — Separate M3A target smoke from full-dataset acceptance

The earlier combined `verify_m3a.py --target` interface always attempted the full 60-group archive,
which made first-line target debugging unnecessarily expensive and conflated implementation, prior
gates, expert collection, independent replay, and completed-dataset evidence. M3A now has two
mutually exclusive physical modes. `--target-smoke` creates a fresh diagnostic run with one complete
counterfactual scene group, checks the six closed native trajectories, and independently replays all
six actions. Its automatically selected root is unique per invocation; a caller-supplied root must
not exist, so smoke never replaces prior accepted evidence.

`--target-full` is the authoritative 60-group/360-episode gate. It validates a complete existing
archive without invoking the collector, resumes only a compatible `in_progress` manifest, and
requires `--create-new-run` before creating a missing archive. Explicit creation is rejected when
the selected root already exists. The verifier never passes the collector's overwrite flag, so
accepted scene groups remain immutable. The complete `CollectionConfig` must match authoritative
defaults before reuse, including the resolved owning root, overwrite/resume policies, collection
schema, control mode, expert fingerprint, tolerances, and runtime versions. Copying a manifest to a
different root or changing any policy is therefore rejected before inspection or replay.

Both modes still run `verify_m2.py --target` first and invalidate subordinate command reports before
execution. Smoke requires one group, six balanced tasks, structural/checksum validation, and six
fresh action-plus-state-audit replays. Full acceptance requires exactly 60 groups, 360 episodes, 60
per TaskSpec, zero partial accepted groups, zero accepted expert or replay failures, zero
checksum/schema failures, zero wrong-object/wrong-bin false successes, and zero unclassified
validation failures, followed by 360 independent action replays in manifest order.

The verification schema was introduced as v2 and is superseded by the content-bound
`langmani-m3a-verification-v3` decision in D-023. It reports
`implementation_validated`, `prior_target_gates_validated`,
`expert_collection_smoke_validated`, `action_replay_validated`, `full_dataset_validated`, and
`physical_target_validated` independently. `physical_acceptance` remains a compatibility alias only.
This is a command/report interface decision; it changes no dependency, expert phase behavior,
success geometry, recorder schema, replay oracle, or accepted dataset content.

## D-022 — Pre-replay expert admission, exact transitions, and explicit static-bin state

M3A now treats an M2 success claim as evidence to validate, not as sufficient replay admission.
Every bounded attempt persists the original `ExpertResult`. Before a successful attempt can reach
action replay, the collector checks its scene/task IDs against both schedule and reset
`EpisodeSpec`, its canonical instruction and semantic object/bin, exact agreement between its final
evaluation and a fresh M1 environment evaluation, absence of wrong-object/wrong-bin false success,
and every configured episode/planning/retry bound. A mismatch is classified as
`expert_contract_failure`; the result is deliberately not normalized or rewritten. `AttemptRecord`
therefore enforces semantic agreement only for accepted attempts, allowing rejected records to
preserve the very upstream drift that caused rejection.

The raw transition contract is exact: T float32 `(8,)` `pd_joint_pos` actions, T terminated and
truncated labels, T success/fail labels when present (required by the fixed accepted schema), and
T+1 recursive environment-state leaves including state zero. The project replay path executes each
in-bounds action unchanged exactly once. It adds no terminal no-op, duplicates no transition,
performs no action normalization or control-mode conversion, and rejects rather than clips an
out-of-bounds action. `ReplayValidationResult` now exposes stable `ReplayFailureCode` values paired
with bounded human-readable reasons; passing archive metadata requires both collections to be empty.
This is a public result-interface extension but introduces no dependency or version change.
No authoritative or physically accepted `langmani-m3a-raw-v1` archive exists yet, so these fixes
finalize the v1 contract before its first target production rather than introducing a v2 archive.
Any pair produced by the earlier incomplete draft lacks required fields and fails strict validation;
it cannot be resumed, promoted, or reused as accepted data.

Installed ManiSkill 3.0.1 `ManiSkillScene.get_sim_state()` explicitly skips static actors. Both
LangMani bins are static, so comparing only HDF5 state zero could prove cube/Panda equivalence but
could not prove the required bin-pose equivalence. The privileged environment boundary therefore
adds `get_expert_initial_scene_state()`, called once immediately after reset and before expert
execution. It materializes actual seven-value poses for all three cubes and both bins plus the
nine-value Panda qpos. Accepted JSON sidecars persist this snapshot; archive validation binds cube
poses and Panda qpos to HDF5 state zero and compares the entire snapshot across all six tasks in
both group bundles and source shards. This accessor never participates in policy observations,
rewards, evaluation, or per-step info, so M1's visual no-leakage contract is unchanged.

## D-023 — Bind M3A target evidence to exact archive content

M3B exposed a provenance defect in the M3A verification interface: v2 recorded target flags and a
dataset path but did not cryptographically bind those flags to the manifest and shard content that
was validated. Replacing a legal archive at the same path could therefore leave a stale-looking
success report. This did not weaken M3A's in-run validation, but it was insufficient as a durable
input capability for a derived dataset.

`langmani-m3a-verification-v3` now reruns the complete inspector when writing the report and stores
`collection_run_id`, schedule configuration fingerprint, and a canonical source archive digest.
The digest covers ordered accepted groups/episodes, group bundle checksums, and source-shard
checksums/sizes/ordered raw IDs while excluding paths and timestamps. M3B requires these values plus
the independent mode, prior-gate, replay, full/smoke, and physical flags. This is a provenance-only
M1-exposed defect fix: no collection decision, expert behavior, success geometry, raw schema, or
accepted episode changes.

## D-024 — Activate the LeRobot dataset extra and explicit PyAV backend

D-004 deliberately installed only base `lerobot==0.6.0` before a dataset milestone. M3B is that
milestone. Inspection of the installed package showed that importing `lerobot.datasets` explicitly
requires its `dataset` extra, including datasets, pandas, PyArrow, PyAV, TorchCodec, and jsonlines.
The project dependency is therefore `lerobot[dataset]==0.6.0`; the LeRobot version itself is
unchanged.

The reviewed resolution is datasets 4.8.5, pandas 2.3.3, PyArrow 25.0.0, PyAV 15.1.0, TorchCodec
0.11.1, and jsonlines 4.0.0. PyAV links libavcodec 61.19.101, libavformat 61.7.100, and libavutil
59.39.100; encoder logs identify SVT-AV1 3.0.0. `pip check` passes. The installed TorchCodec package
is discoverable but cannot load its FFmpeg/shared DLL chain on this Windows host, while LeRobot's
default selection checks only discoverability. Every M3B create/load operation therefore passes
`video_backend="pyav"` explicitly. M3B does not import OpenCV for encoding and does not change the
existing dual-`cv2` risk.

Canonical public imports are `LeRobotDataset` from `lerobot.datasets` and `RGBEncoderConfig` from
`lerobot.configs`. Installed `LeRobotDataset.create` supports public data/video file-size settings
but no public `chunks_size` argument. `add_frame` requires a special task string and mutates its
input by removing that field, so M3B passes a new dictionary per frame. `save_episode` owns Parquet,
task metadata, video encoding, and episode tables; project code never edits those generated files.

## D-025 — M3B is a deterministic, policy-only derived representation

M3A remains authoritative. M3B accepts only its content-bound manifest index and restores each
state[t] through public ManiSkill 3.0.1 APIs after an exact scene-seed/TaskSpec reset. Public
`get_obs()` performs render synchronization and sensor capture, so no no-op action, private render
method, direct camera capture, expert rerun, or action-replay regeneration is used.

The exact policy allowlist is base-camera RGB, `PandaPolicyStateV0`, and the raw action. The nine
state components use installed active-joint names (`panda_joint1..7`, then two finger joints) and
are reordered by name. The eight action values remain the stored `pd_joint_pos` action. T actions
produce T frames from states 0 through T-1; state T is audit-only. Canonical task strings use
LeRobot task metadata. All scene/task IDs, checksums, expert/replay evidence, and raw render digests
remain sidecar provenance and never become policy inputs.

Full splits are digest-ranked at complete counterfactual scene-group level: 48 train, 6 validation,
and 6 test groups, giving 288/36/36 episodes and perfect 48/6/6 per-task balance. Split changes,
camera/feature/state/action/codec/FPS changes, source archive changes, LeRobot/PyAV/libavcodec
changes, and ordered source episode changes alter the canonical export fingerprint. Machine paths,
timestamps, filesystem enumeration, and Python `hash()` do not.

## D-026 — Guard LeRobot finalization and use all-or-nothing staging

Installed LeRobot 0.6.0 does not prevent `add_frame` or `save_episode` after `finalize`, and
`save_episode` can write Parquet before video encoding fails. M3B therefore wraps the public writer
with `OPEN -> FINALIZED | FAILED`, saves each episode once, calls successful finalize exactly once,
and never resumes or repairs a partial derived dataset. A fingerprint-owned staging child must not
exist when passed to `create`. Sidecars and an independent local validation are complete before a
same-filesystem atomic rename; `langmani/complete.json` is written last. Completed destinations are
immutable, while owned incomplete staging requires explicit `--clean-staging`. An interruption in
the narrow post-rename/pre-marker window is also treated as an incomplete promoted staging result:
explicit cleanup is allowed only when its sidecar manifest proves the same fingerprint.

The selected encoder is software H.264 (PyAV resolves `h264` to libx264), yuv444p, CRF 18, GOP 2,
preset `medium`, fast-decode 0, one thread, and 20 FPS. Initial libsvtav1/yuv420p CRF 30 calibration
measured MAE 1.285-1.686 and PSNR 39.29-44.09 dB on a smooth pattern, but a second high-chroma
fixture reached only 29.703 dB. AV1 CRF 25 still reached only 29.782 dB because 4:2:0 chroma
subsampling, rather than quantizer alone, dominated that pattern. The supported H.264/yuv444p
configuration measured MAE 0.773-1.069 and PSNR 44.13-47.63 dB; local encoder logs identify libx264
core 165. The fixture successfully wrote Parquet and MP4, finalized, reloaded, decoded, preserved
state/action/task values, and produced a DataLoader batch. Before any target data was examined,
M3B retained acceptance thresholds at MAE <= 5 and PSNR >= 30 dB.

LeRobot/PyAV ffconcat failed when joining the second episode under this Windows repository's
non-ASCII path, but the same operation passed under an ASCII path and the promoted result loaded
from the Unicode final path. Windows therefore stages at a documented ASCII directory on the same
volume; Linux uses destination-adjacent staging. No LeRobot private implementation is copied or
patched. This is a host-path workaround for derived temporary files, not target acceptance.

## D-027 — Bind every M4 run to a tracked Git baseline and completed M3B evidence

The repository now has one honest implementation baseline through M3B rather than a fabricated
per-milestone history: commit `6920b52c1f48c278e669cd71b69b8949dd900f3a`, tagged
`m3b-implementation`. M4 records the actual full `HEAD` in every training/evaluation identity and
checkpoint. Full and tiny-overfit evidence requires a clean worktree. Only an explicitly labeled
development run may record a dirty-tree override; that run is ineligible for final experiment,
quality, or physical flags. Canonical run fingerprints include the commit but exclude paths,
timestamps, hostname, and filesystem order.

Normal M4 training accepts only a completed M3B root. It validates the typed completion marker,
export manifest, summary, independent validation report, export/source fingerprints, full `v3.0`
mode, exact local feature contract, source mapping and split sidecars, all stored alignment/video
results, and local Parquet/video frame totals. It requires 60 complete groups, 360 episodes, 60 per
TaskSpec, and 48/6/6 scene-group splits (288/36/36 episodes). This is deliberately narrower than
rerunning the M3B exporter/validator: M3B has already bound its M3A provenance and normal M4 does not
reopen M3A, recalculate admission, or mutate the derived dataset. Real target M3B validation remains
a prerequisite; an M4 fixture can test code only.

No dependency version changes in M4. LeRobot remains `lerobot[dataset]==0.6.0`; M4 uses the policy
code already present in that installed package. The only test-metadata change adds explicit
`linux`, `training`, and `evaluation` markers so platform and execution scope are reported honestly.

## D-028 — Use installed LeRobot 0.6.0 ACT/processors with one fixed primary configuration

Installed-source inspection established the public M4 boundary:

```text
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
from lerobot.policies import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy, make_act_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
```

The inspected factory signatures are `make_policy_config(policy_type, **kwargs)`,
`make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None)`, and
`make_pre_post_processors(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs)`.
`ACTPolicy(config, **kwargs)` exposes `forward(batch)`, `select_action(batch)`, `reset()`,
`save_pretrained(...)`, and local `from_pretrained(...)`.
`PolicyProcessorPipeline` exposes `__call__`, `reset`, `save_pretrained`, and `from_pretrained`.
The installed ACT-specific factory accepts `make_act_pre_post_processors(config,
dataset_stats=...)`. Runtime code uses these public APIs and does not import example runners, copy
ACT source, edit LeRobot files, authenticate, or push to Hub.

Inspection found three relevant deviations from unsafe assumptions. First, `ACTConfig` defaults
`push_to_hub=True`, so M4 always pins it false. Second, the ACT processor renames, batches, moves,
normalizes, unnormalizes, and returns actions to CPU but does not convert uint8 RGB to float [0,1];
the official trainer does that before preprocessing. M4 therefore performs and validates the same
explicit conversion. Third, `use_amp` does not itself own the complete training autocast/gradient
path, so the project loop owns it and records it.

Installed default ACT architecture is ResNet-18 with ImageNet weights, 100/100 action chunk/query,
model dimension 512, eight heads, feedforward 3200, four encoder/one decoder layers, VAE latent 32
with four VAE encoder layers, dropout 0.1, KL weight 10, AdamW `1e-5`/`1e-4`, and no scheduler. M4
retains those architecture dimensions but pins no pretrained weights, chunk 50, ten actions per
query, one observation step, AdamW learning/backbone rates `1e-5`, weight decay `1e-4`, no
scheduler/warmup, global clipping 10, batch 32, 100000 steps, and checkpoint/validation every 5000
steps. It also pins visual/state/action `MEAN_STD`, no final-stride dilation, ReLU feedforward,
current-only observations, action deltas `range(50)`, PEFT off, and every Hub/pretrained reference
null. All variants share the configuration except the required 9D/15D state width.

The primary target precision is bfloat16 autocast on CUDA with float32 parameters/source tensors.
M4 deliberately avoids float16 and therefore does not carry a `GradScaler`; a GPU without bfloat16
support fails rather than changing precision. Strict Torch deterministic algorithms, deterministic
cuDNN, disabled cuDNN benchmarking/TF32, and seeded Python/NumPy/Torch/workers are recorded. CUDA
kernels can still expose unsupported deterministic operations, which are failures rather than
silently accepted nondeterminism. Actual peak GPU memory and training throughput are unknown until
the native target run and must come from structured metrics.

## D-029 — Train-only normalization, public temporal sampling, and oracle task conditioning

LeRobot's episode-filtered dataset retains full-dataset `meta.stats`; M4 therefore never uses it for
training. It constructs explicit scene-safe global and per-task views from M3B stable sidecars and
checks the actual episode set returned by public `LeRobotDataset(episodes=...)`. Per-task statistics
use the selected 48 train episodes; mixed statistics use all 288 train episodes. Deterministic
float64 batched Welford accumulation produces population min/max/mean/std for image channels on
float [0,1], nine Panda qpos components, and eight actions. Saved indices and an independent set
audit prove validation/test contributed no frame.

`CanonicalTaskOneHotV0` uses the existing M3A object-major/bin-minor order and stable TaskSpec IDs.
One shared project component appends it during training from M3B episode provenance and during
inference from the active M1 command. It never parses language or reads physical target state. The
15D normalizer uses the train-only nine Panda statistics plus synthetic one-hot mean 0/std 1, so
conditioning before the installed processor preserves exactly one binary active component. M3B is
not rewritten and standard ACT receives no text or task condition.

The public `resolve_delta_timestamps(ACTConfig, LeRobotDatasetMetadata)` helper is authoritative.
Installed ACT reports no future observation deltas and action indices `range(chunk_size)`. At 20
FPS the primary 50-action chunk is `0.00..2.45` seconds; LeRobot clamps at an episode end and emits
`action_is_pad` rather than crossing the boundary. M4 tests first/final samples and `[50,8]` plus
`[50]` padding shapes instead of manually recreating sampling. Installed `select_action` queries a
new chunk only when its queue is empty and executes `n_action_steps`; M4 resets policy and both
processors at every episode boundary.

## D-030 — Atomic ACT evidence, validation-only selection, and separately locked test

M4 owns semantic compatibility around LeRobot's public serializers. Each checkpoint and its
completion marker are fully written and flushed in same-filesystem staging, checksummed, then made
visible by one atomic directory promotion. It contains or binds
the policy/config, preprocess/postprocess pipelines, train-only statistics, optimizer/optional
scheduler and RNG/training state, data/split/run fingerprints, seed, Git commit, and versions.
Reload is local-only and revalidates all semantic inputs. Resume rejects any mismatch and any
completed immutable run. Target-full reuse and every evaluation also require the current LeRobot,
PyTorch, and CUDA versions to exactly match the versions bound into the training identity; matching
Git and dataset fingerprints alone is insufficient.

Validation checkpoints are ranked before test access by highest task success, lowest wrong-object
interaction, lowest off-table rate, lower offline action loss, then earlier step. The selection
record is atomically published and immutable. Full-mode test evaluation requires its exact selected
checkpoint and exact predeclared schedule; test evidence is separate and cannot change selection or
resume training. Development bypasses are labeled nonfinal.

The checkpoint fingerprint also binds the structured metric for that step. Selection cross-checks
the external JSONL record against this checkpoint copy before reading offline validation loss, so a
mutable metrics file cannot alter ranking evidence.

Scheduled training metrics include the promoted checkpoint path. Uninterrupted summaries measure
wall time across optimization, validation, and checkpoint serialization; resumed summaries leave
full wall time null and retain a clearly named measured lower bound. Resume restores the latest
checkpoint RNG/optimizer state and uses an addressable deterministic batch sampler; regression tests
compare uninterrupted and resumed batch order, parameters, optimizer state, and RNG continuations.
The sole next-step checkpoint promoted before a process interruption may be adopted on resume;
this includes the first scheduled checkpoint when the manifest still has an empty checkpoint list.
Multiple, stale, or out-of-order orphan directories are rejected. If interruption occurs before any
checkpoint is promoted, target orchestration atomically preserves the incomplete directory under an
ignored diagnostic area and restarts the deterministic run from step zero.

Evaluation evidence uses the same transactional principle. An identity owner file guards a fixed
same-filesystem staging directory. The benchmark, raw episode records, validation record, and any
counterfactual analysis are fully written there before the evaluation directory is promoted. Raw
episodes are reconstructed and all aggregates, schedule identities, Git evidence, authorization,
and action-execution status are revalidated on recovery. Only after promotion may the command lock
selection, publish the global analysis report, or idempotently finalize the run. This prevents an
interruption from publishing a selection or completion marker backed by partial evaluation data.

Closed-loop ACT rollout uses only M1 base-camera RGB and Panda qpos, plus the active command one-hot
for the oracle variant; it never calls M2. Installed policy queue semantics are preserved. Invalid
or out-of-bounds actions are classified and terminate instead of being clipped. Validation/test use
the exact M3B scene groups, while a canonical digest fixes 30 unseen source-excluded seeds paired
with all six tasks. Counterfactual audits hold physical observations fixed, compare every predicted
chunk with its task-matched expert chunk, and separately summarize target-object and destination-bin
effects. The final report distinguishes control learning, unconditioned one-to-many ambiguity, and
oracle discrete conditioning. None of these is a claim of language understanding.

Non-target verification may prove contracts and a small real CPU ACT fixture, including
forward/backward and local reload. Target smoke must first pass M0/M1/M2, M3A smoke, and M3B smoke,
then use real M3B data and CUDA for tiny-overfit and learned-policy M1 rollout. Full target requires
the real 360-episode M3B dataset, all six per-task plus two mixed runs, validation selection, locked
test, the 180-episode fresh benchmark, and final comparison. Target orchestration checks the current
worktree is clean before reusing earlier evidence, resumes the latest compatible checkpoint, and
revalidates promoted evaluations rather than skipping them based on a shallow success flag. As of
this decision, no real M3B
episode, CUDA training, tiny-overfit 6/6, learned-policy closed-loop rollout, locked test/fresh
result, measured GPU memory/throughput, baseline-quality acceptance, or physical M4 acceptance has
been produced.

## D-031 — Isolate M1 CPU and GPU PhysX acceptance processes

The first native Linux RTX 4090 run exposed a verifier defect rather than an environment defect:
`environment/verify_m1.py --target` created its required CPU PhysX scene and then attempted to
enable GPU PhysX in the same Python process. SAPIEN 3.0.3 rejects that sequence with `GPU PhysX can
only be enabled once before any other code involving PhysX`.

M1 now executes its CPU simulation checks and its GPU vectorization/rendering checks in two fresh
child processes. Each worker writes a compact JSON check record to a temporary path; the parent
validates and merges those checks into the existing authoritative M1 report. Missing output,
malformed output, a nonzero exit, or the fixed ten-minute worker timeout is a required failure, and
child stdout/stderr remains visible. Contract-only checks and target prerequisites remain in the
parent. This changes only the diagnostic process boundary: environment geometry, observations,
success evaluation, control modes, dependency versions, and acceptance thresholds are unchanged.

## D-032 — Isolate mplib 0.1.1 behind a NumPy 1.26.4 planner runtime

The first native target run used Ubuntu 22.04.5, an RTX 4090, NVIDIA driver 570.124.04, Python
3.12.13, PyTorch 2.11.0+cu128, ManiSkill 3.0.1, SAPIEN 3.0.3, mplib 0.1.1, and main-environment
NumPy 2.2.6. M0 passed CUDA, Vulkan, PhysX GPU, and RGB-render acceptance. After D-031 separated
PhysX backends, M1 passed CPU scene/evaluation checks, six-environment GPU vectorization and partial
reset, visual no-leakage, actor/Panda camera visibility, and both policy and human rendering.

M2 then terminated with a native segmentation fault while constructing
`mplib.pymp.ArticulatedModel`. The same crash reproduced through ManiSkill's installed official
Panda motion-planning solver, so it was not caused by LangMani actor geometry or adapter ordering.
On the same machine, both mplib 0.1.1 and 0.2.1 crashed beside NumPy 2.2.6, while the unchanged mplib
0.1.1 extension constructed the Panda model successfully beside NumPy 1.26.4 with the explicit
installed link/joint lists. This agrees with the upstream reports for the same silent planner exit
under NumPy 2 and the ManiSkill maintainer recommendation to use NumPy 1.26.4:

- [ManiSkill issue #1100](https://github.com/haosulab/ManiSkill/issues/1100)
- [ManiSkill issue #426](https://github.com/haosulab/ManiSkill/issues/426)
- [mplib published releases](https://pypi.org/project/mplib/)

The main environment is not downgraded: doing so would change the already pinned and M1-validated
M0/M3B/M4 dependency surface. Instead, native planner construction uses an explicit Python 3.12.13
virtual environment created with `--system-site-packages`. It inherits the exact main runtime and
overlays only NumPy 1.26.4, SciPy 1.15.3, and OpenCV 4.11.0.86. The selected interpreter is stored in
`LANGMANI_PLANNER_PYTHON`; `environment/verify_planner_runtime.py` imports Gymnasium, h5py,
ManiSkill, mplib, NumPy, OpenCV, Pillow, SAPIEN, SciPy, and PyTorch and checks their effective module
versions before constructing and synchronizing the Panda planner. Importing modules is deliberate:
`importlib.metadata` can select inherited main-environment distribution metadata instead of the
overlay actually used by Python. `MplibPandaPlannerAdapter` separately rejects a non-1.26.4 NumPy
runtime before the unsafe native constructor. Interpreter selection preserves the virtual
environment launcher path instead of resolving its `bin/python` symlink back to the main Conda
binary, because that launcher path determines Python's overlay prefix and import search order.

`verify_m2.py --target` keeps M0 and M1 in the main runtime, then runs the planner gate and all M2
expert commands in the side runtime. M3A uses that same interpreter for expert collection and real
action replay, while its offline archive inspection remains in the main runtime. This is sequential
command orchestration, not an expert architecture change: mplib remains in-process with
`num_envs=1`, and no planner multiprocessing or vectorization is introduced. M3A manifests now bind
all ten effective planner-side module versions. Because no authoritative M3A archive existed before
this change, the v1 schema is retained; older/incomplete mappings fail strict parsing and cannot be
silently resumed.

Target execution also showed that two repeated final-waypoint control steps per planned phase spent
12 of the fixed 200-step episode budget without adding a new path waypoint. They are removed, so
every planned path is executed exactly once and M3A cannot inherit artificial terminal repetitions.
The TCP completion tolerance changes from 10 mm to 15 mm because successful `pd_joint_pos` motions
showed 11.7–12.1 mm residual tracking error at the prior bound. This does not alter object/bin
geometry, release/static checks, false-success checks, or any M1 success threshold.

A bounded pre-commit trial over the exact 180-rollout M2 matrix produced 177 classified successes
(98.3% overall): red-left, green-left, and green-right were 29/30; the other three TaskSpecs were
30/30. The three failures were classified `planning_failure`; wrong-object successes, wrong-bin
successes, unexpected exceptions, and unclassified failures were all zero. This meets the numeric
M2 benchmark thresholds and justifies committing the constants, but it is diagnostic evidence only.
Formal M2 acceptance still requires the clean committed `verify_m2.py --target` run and rendered
12-phase artifact check.

## D-033 — Distinguish tiny-overfit diagnostics from held-out validation identity

The first clean M4 target-smoke attempt reached the real M3B dataset after the ordered M0 through
M3B gates passed, then both CUDA training commands stopped before optimization. Tiny-overfit
correctly evaluates loss on the same deterministic train view it is intended to memorize, but the
training command had also copied that view into `ordered_validation_episode_indices`.
`ActRunIdentity` correctly rejected the overlap instead of allowing evidence to mislabel training
episodes as held-out validation.

M4 now keeps formal validation indices empty for tiny-overfit and records the reused episode indices
under an explicit `train_tiny_overfit_diagnostic` role in the fingerprinted data contract. The same
explicit rule covers development-mode fallback when a small dataset has no validation episodes.
Partial overlaps, full-run overlaps, and every unclassified reuse remain errors. Full experiments
still use disjoint M3B train, validation, and locked-test scene-group splits; validation-only
checkpoint selection and test locking are unchanged. The target smoke must be rerun from this clean
commit before any CUDA training, learned rollout, or physical M4 acceptance is claimed.

## D-034 — Explicitly migrate direct ACT construction to the configured device

After D-033 allowed target smoke to reach the first real offline-loss forward pass, both tiny ACT
runs failed with CUDA inputs and CPU policy weights. Inspection of installed LeRobot 0.6.0 showed
that `ACTPolicy(config)` only constructs `self.model = ACT(config)` and does not migrate the module.
The public LeRobot `make_policy` factory separately calls `policy.to(cfg.device)` after direct or
pretrained construction. LangMani intentionally constructs ACT directly to bind its fixed project
configuration and therefore must perform that public migration step itself.

The project-owned builder now moves the complete policy to the effective configured device before
constructing the optimizer and verifies every parameter and buffer is on exactly that device. The
preprocessor continues to move input tensors independently. CPU fixture coverage tracks the
explicit module transfer call; the native target smoke remains the required CUDA forward/backward
and checkpoint evidence. No dependency, model architecture, precision, dataset, split, loss, or
quality threshold changes.

## D-035 — Resolve policy action bounds through the ManiSkill wrapper boundary

The first fully trained target-smoke models reached closed-loop evaluation, but both rollout
commands stopped before their first policy action because ManiSkill 3.0.1 returns a
`TimeLimitWrapper` that has no direct `single_action_space` attribute. The unwrapped M1 environment
does expose the single-action bounds, and Gymnasium wrappers expose wrapped attributes through
`get_wrapper_attr`. M3A replay had already validated this installed wrapper behavior through its
equivalent action-space resolution path.

The M4 rollout adapter now resolves `single_action_space` through `get_wrapper_attr`, then explicit
wrapped/unwrapped single-space and action-space fallbacks. A leading singleton batch dimension is
removed only from bounds with exact shape `[1,8]`; the final bounds must be exactly `[8]`. Predicted
actions remain float32 `[1,8]`, are reduced to one raw `[8]` environment action, and are rejected
rather than clipped when out of bounds. No action scale, control mode, environment wrapper, success
criterion, model, dataset, or quality threshold changes.

## D-036 — Separate raw ACT output from explicit environment-action projection

The completed RTX 4090 target smoke produced immutable 5000-step PerTask and 10000-step
Mixed-TaskOneHot checkpoints. Their validation losses fell from 0.9983 to 0.0590 and from 0.9899
to 0.0414, and changing only the canonical one-hot changed predictions. The 1068 M3B source
actions were independently audited: the gripper component is exactly `-1` or `+1`, every episode
begins at `+1`, state/action indexing is aligned, and saved train-only processor statistics reload
correctly. The first closed-loop outputs nevertheless overshot the finite gripper upper bound:
Mixed-TaskOneHot was approximately `1.0508`--`1.0684`, and PerTask approximately `1.1280`.
Therefore the failure is continuous regression overshoot, not M3B corruption, temporal shift, or
an invalid source-action contract.

LangMani adds project-owned `BoundedActionEnvPostprocessorV0` after the installed LeRobot policy
postprocessor and directly before `env.step`. Its explicit `reject` mode retains D-035 strict
behavior. Its explicit `project` mode rejects malformed/nonfinite values but computes finite
componentwise projection from the active environment action space as
`min(max(raw, low), high)`. The observed eight-dimensional M1 bounds are
`[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0]` through
`[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0]`. These values are evidence, not
hard-coded processor constants. Raw and executed actions, violation masks, excesses, corrections,
and aggregate projection rates remain separately auditable. Binary gripper conversion is deferred
as a distinct future ablation.

Checkpoint files and fingerprints are immutable. Evaluation instead receives a new canonical
runtime fingerprint covering checkpoint, saved preprocessor/postprocessor, action-bound config,
environment/action-space contract, task mapping, rollout configuration, code commit, and M4.1
schema. Runtime manifests and checksummed per-step audits live beside evaluation results. Target
smoke reuses the two existing checkpoints and completed M3B smoke dataset; it may not train or
rewrite earlier artifacts. The clean RTX 4090 run at commit
`f144a9485717dc38c8df74832c694ec8c1534a42` passed. Its strict probe reproduced raw one-hot
gripper `1.0507922172546387`, rejected it before `env.step`, projected it to exactly `1.0`, and
executed one real M1 step with L1/L2/L-infinity correction `0.05079221725463867`.

The unchanged PerTask checkpoint
`sha256:9dcdbd392673c6fef9796d17db2d05691266fcd14deea2c9cd832fe67ece5a62`
completed red-left in 132 steps. It projected 124/132 actions (93.94%), only in the gripper
dimension, with maximum excess/correction `0.2882833480834961`. The unchanged TaskOneHot checkpoint
`sha256:cee1183ded4d2dde335a139d639cbb57f6362856fcd749be413e9400943ac16f`
completed all six same-scene tasks in 133, 131, 135, 140, 139, and 141 steps. It projected 710/819
actions (86.69%), again only in the gripper dimension, with maximum excess/correction
`0.11678099632263184`. Combined rollout task success was 7/7, projection was 834/951 actions
(87.70%), and strict-unprojected success was honestly 0/7. No malformed/nonfinite action,
infrastructure failure, M2 action, checkpoint modification, retraining, data rewrite, or success-
threshold change occurred. Model, saved processors, and action-bound processor reloaded
deterministically. Consequently raw-bound validity is false while projected-bound validity,
closed-loop inference, the 6/6 task gate, and M4.1 physical target validation are true. M4 full
remains unstarted.

## D-037 — Preflight exact full identities and bind the projected action runtime explicitly

The RTX 4090 target completed the data gates at commit
`29a90aa72468d7ef8dbdf4ecbccdb175d63c21af`. M3A accepted exactly 60 complete counterfactual
groups and 360 action-replayed episodes, 60 per TaskSpec, after considering 65 ordered candidate
scenes. Five groups were rejected; 404 expert attempts were recorded, with zero partial accepted
groups, accepted replay failures, checksum/schema failures, or unclassified validation failures.
M3B then accepted 360 episodes and 64,548 frames with 288/36/36 episodes and 48/6/6 scene groups
across train/validation/test. Every TaskSpec has 48/6/6 episodes, no scene group crosses a split,
all videos decode, and source/action/state alignment passes. Its export fingerprint is
`sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4`.

Before spending the full eight-run training budget, `verify_m4.py --target-full --dry-run` must
validate the existing completed M3B full report and storage, rather than rebuilding the accepted
M3A/M3B artifacts. It invokes `train_act.py --dry-run --planned-mode full` in canonical run order.
The planned mode is semantic: dataset/split/statistics, fixed model and optimization, runtime/Git,
and the resulting SHA-256 run fingerprint are identical to the future non-dry full invocation.
Execution still returns before creating a model directory. The outer verifier requires six
per-task plus two mixed identities, eight unique output directories, exact 48/6 or 288/36
train/validation views, fixed 100,000-step schedules, a clean Git baseline, and no training or
rollout. It records `full_dry_run_validated` and `planned_full_run_count` independently and never
sets `physical_target_validated` from planning evidence.

M4.1 established that finite raw ACT gripper overshoot is expected and that the frozen full runtime
uses the auditable `BoundedActionEnvPostprocessorV0` in `project` mode. The evaluator itself retains
`reject` as its safe default, so relying on that default would silently run a different full
experiment and fail before otherwise valid projected steps. Both target-full preflight and real
execution therefore require explicit `--action-bound-mode project`, and the verifier passes it to
every validation, locked-test, fresh-seed, and completed-run revalidation command. The preflight
plan records the mode. No dependency, ACT weight/loss, data split, train statistic, M1 success
geometry, checkpoint fingerprint, or source dataset changes.

## D-038 — Separate the current-machine smoke gate from checkpoint-bound smoke evaluation data

M4 target smoke checkpoints are immutable and bind the exact M3B export fingerprint used during
their tiny-overfit run. Re-running the ordered M0-through-M3B smoke chain on another machine creates
a new, independently valid six-episode export identity. Using that new path to evaluate an existing
checkpoint correctly fails the evaluator's dataset-identity gate, even when both exports represent
the same six semantic tasks.

Target smoke therefore keeps the latest completed M3B smoke report as the current-machine hardware
and data-pipeline prerequisite, but evaluates each reused checkpoint only against the dataset root
stored in its own run manifest. The PerTask and TaskOneHot checkpoints must name the same root and
SHA-256 export fingerprint. That archive is loaded with storage validation, must contain exactly six
episodes, and must reproduce the saved fingerprint before any environment step. The M4 verification
report records both the current gate dataset and the checkpoint-bound evaluation dataset explicitly.
This permits an auditable cross-machine artifact migration without retraining, editing a checkpoint,
rewriting M3B data, or weakening any dataset, action-bound, rollout, or success contract.

Evaluation results are already namespaced by the action/runtime fingerprint, so numerical
differences across GPU/runtime revisions remain explicit in each output directory. A previously
published clean canonical sensitivity artifact is never overwritten during revalidation. It may be
retained only when its schema, run fingerprint, and checkpoint fingerprint match; the new runtime's
own validated `analysis.json` remains the evidence for that execution. Canonical evidence with a
different identity or dirty Git provenance still fails closed.

## D-039 — Audit decoded counterfactual RGB with the frozen M3B codec thresholds

The first RTX 5090 M4 full attempt at commit
`b67c440e6495f7f52d308781b12a42c60b5b1b56` stopped before GPU training because 5 of 48 train
scene groups did not have byte-identical decoded first frames. The failed audit still showed 48/48
identical Panda states and 48/48 one-to-many expert chunks. A read-only diagnostic measured a worst
pairwise decoded-frame MAE of `0.1492462158203125` and a worst PSNR of
`50.632996173007065` dB. Reconstructing the five affected groups directly from their authoritative
M3A state[0] records produced byte-identical RGB for all six tasks in every group. The difference
therefore comes from separately encoding each episode with the already selected lossy H.264
pipeline, not from scene, task, state, frame, or action misalignment.

M4 keeps exact decoded RGB equality as an explicit diagnostic but admits a train group through
codec equivalence using the M3B export's immutable video thresholds: mean absolute error at most
`5.0` and PSNR at least `30.0` dB across every pair of decoded initial frames. The audit records
both fractions, the observed extrema, and the consumed thresholds. Panda-state equality and
one-to-many expert-action evidence remain exact all-group requirements. Large visual drift still
fails closed, and missing decoded RGB evidence for a non-identical digest is an error. Normal M4
training does not reopen M3A; the completed M3B validation remains responsible for exact M3A action
alignment, reconstructed state/RGB digests, and decoded-video quality. This fixes an audit/model
boundary defect without changing M1, M3A, M3B, splits, train-only statistics, ACT configuration,
checkpoint-selection rules, or action projection.

## D-040 — Parse recursively frozen fresh-seed evidence as JSON-compatible mappings

The first dual-RTX 5090 M4 full evaluation completed all eight 100,000-step training runs, selected
one checkpoint for each of the first two runs from 40 passing validation evaluations, and passed
both locked six-episode tests before both fresh-seed workers stopped with `TypeError: fresh-seed
schedule JSON must be an object`. The stored schedule was present and content-complete. The defect
was at the project-owned serialization boundary: `ActRunIdentity` recursively freezes JSON mappings
and arrays as `_FrozenMapping` and tuple values, while `FreshSeedSchedule.from_dict` required
concrete `dict` and `list` instances.

`FreshSeedSchedule.from_dict` therefore accepts the abstract `Mapping` contract and either list or
tuple for the three JSON-array fields. Exact field membership, schema version, seed counts,
exclusion evidence, and schedule digests remain unchanged and fail closed. A regression test now
constructs a real `ActRunIdentity`, reads its recursively frozen nested schedule, and requires an
exact round trip. This compatibility repair does not alter M3A/M3B data, run identities, existing
checkpoint fingerprints, validation selection, test locks, fresh-seed ordering, model weights,
action projection, or M1 success criteria.

## Local bootstrap evidence

The bootstrap was authored on Windows 11, which is not an acceptance platform. In an isolated
Python 3.12.10 environment using CPU builds of PyTorch 2.11.0 and torchvision 0.26.0:

- `python -m pip check` reported no broken metadata requirements.
- `torch`, `mani_skill`, `lerobot`, and `cv2` imported successfully.
- `ruff format --check .` and `ruff check .` passed.
- `python -m build` produced both an sdist and wheel successfully; the generated artifacts remain
  ignored.
- `pytest` reported 6 passed and 2 skipped. The skips were the non-Linux PickCube execution and the
  GPU/rendering test.
- `pytest -m "not gpu and not rendering"` reported 6 passed, 1 skipped, and 1 deselected.
- The default verifier completed successfully with explicit non-target warnings.
- The target verifier returned exit code 1 for non-Linux and unavailable CUDA, as required.

For M1 on the same non-target Windows host:

- `ruff format --check .` and `ruff check .` passed.
- The full test suite reported 45 passed and 6 explicit native-Linux/CUDA/rendering skips.
- `pytest -m "not gpu and not rendering"` reported 45 passed, 3 native-Linux simulator skips, and
  3 hardware tests deselected.
- `python -m build --no-isolation` produced the sdist and wheel. The exact isolated
  `python -m build` reached its package-install phase but exceeded the 120-second network timeout;
  this did not expose a project build error.
- `environment/verify_m1.py` passed registration, time-limit, asset, and canonical-language checks,
  wrote `outputs/diagnostics/m1/verification.json`, and explicitly skipped simulator execution.
- No native Linux CPU simulation, GPU vectorization, policy rendering, segmentation visibility, or
  human-camera frame was physically validated; `environment/verify_m1.py --target` remains pending.

For M2 on the same non-target Windows host:

- `ruff format --check .` and `ruff check .` passed.
- The full suite reported 122 passed and 7 explicit platform/CUDA/rendering skips.
- `pytest -m "not gpu and not rendering"` reported 122 passed, 4 native-Linux simulator skips,
  and 3 hardware tests deselected.
- `python -m build --no-isolation` built both the sdist and wheel. The exact isolated
  `python -m build` spent 180 seconds installing its pinned setuptools build dependency and timed
  out before project build execution; this is network/isolation evidence, not a source build error.
- `environment/verify_install.py`, `environment/verify_m1.py`, and
  `environment/verify_m2.py` passed their required non-target checks. M2 reported the missing
  Linux-only mplib and physical rollouts as explicit skips and wrote
  `outputs/diagnostics/m2/verification.json` with `physical_acceptance=false`.
- `environment/verify_m2.py --target` returned exit code 1 at the native-Linux precondition and did
  not start M2 checks, as required.

For M3A on the same non-target Windows host:

- `ruff format --check .` covered 51 files and `ruff check .` passed.
- The full suite reported 227 passed and 8 explicit platform/CUDA/rendering skips.
- `pytest -m "not gpu and not rendering"` reported 227 passed, 5 native-Linux simulator skips,
  and 3 hardware tests deselected. The M3A-only unit selection reported 102 passed.
- `python -m pip check` reported no broken requirements.
- Isolated `python -m build --outdir outputs/build-m3a-tail-final-20260713` built both the sdist and
  wheel. The default `dist/` destination contains older artifacts that this Windows identity
  previously could not replace (`WinError 5`); using a fresh ignored output directory proves the
  project build while preserving those pre-existing files.
- `environment/verify_install.py`, `environment/verify_m1.py`, `environment/verify_m2.py`, and
  `environment/verify_m3a.py` passed their non-target checks. M3A wrote
  `outputs/diagnostics/m3a/verification.json` with `validation_scope=non_target_structural` and
  explicitly skipped physical collection and replay.
- The earlier combined `environment/verify_m3a.py --target` command returned exit code 1 at the
  native-Linux precondition and did not start the ordered prior gates or collection, as required at
  the time. D-021 supersedes that command interface; neither new physical mode has been run here.
- Synthetic tests exercised strict ExpertResult pre-replay admission, wrong-target false-success
  rejection, explicit cube/bin/Panda initial-state equivalence, exact T/T+1 timing, unmodified
  in-bounds actions, out-of-bounds rejection, typed replay failures, full-config reuse rejection,
  all-six admission, trusted bundle and shard checksums, HDF5/JSON corruption, projection drift,
  per-state round-trip audit, failed-attempt retention, overwrite safety, and interruptions before
  and after manifest commit. These are contract tests, not a produced or physically replayed
  360-episode archive.

The preceding bootstrap paragraphs are specifically the Windows review record. D-032 supersedes
their former global pending status: a later native RTX 4090 run passed the M0 and M1 target gates,
constructed and executed the planner in the isolated runtime, completed a 177/180 M2 parameter
trial, and established the remaining formal M2/M3A/M3B/M4 gates described below. The parameter trial
is not a substitute for the committed M2 target report.

Before the non-Linux boundary guard was added, directly constructing PickCube on this Windows host
terminated the Python process with a SAPIEN access violation in `actor_builder.py`. This is recorded
as out-of-scope evidence, not a passed CPU simulator check and not a reason to add a Windows
workaround.

## Unresolved risks

1. **FactorFiLM physical training and the semi-stable LeRobot hook boundary remain pending.** The
   real M4.3a audit is complete, but M4.3b currently proves only local architecture/training/
   checkpoint structure. No 100,000-step FactorFiLM run, selected checkpoint, or development rollout
   exists. Its isolated hooks depend on LeRobot 0.6.0 intermediate model attributes, so any package
   upgrade must repeat the fail-closed API, shape, token-layout, fixture, save/reload, and target gates.
2. **ManiSkill does not declare a PyTorch upper bound.** M0 through M4.2 proved the selected PyTorch
   2.11 CUDA build on the accepted targets, but a new driver/container image must repeat the relevant
   install, simulator, rendering, checkpoint, and policy-inference gates.
3. **OpenCV wheel collision.** SAPIEN 3.0.3 requires `opencv-python`, while LeRobot 0.6.0 requires
   `opencv-python-headless`. OpenCV's publishers state that only one wheel sharing the `cv2`
   namespace should be installed. The environment pins both to the same version because both
   upstream metadata requirements must remain satisfied, but this is not an upstream-supported
   resolution. Main-runtime `cv2`, ManiSkill rendering, and the real 360-episode M3B video
   export/decode/reload passed. A new image must still repeat those gates. Uninstalling either wheel
   in-place may damage the other.
4. **System components are not lockable here.** NVIDIA driver, Vulkan ICD, kernel, and distribution
   libraries can still invalidate a correct Python resolution.
5. **No full transitive lockfile yet.** M0 pins the critical direct and renderer packages in the
   environment declaration. Generate and review a platform lock only after the first successful
   target run, so it captures a physically verified rather than merely resolvable environment.
6. **The planner side runtime is an explicit compatibility boundary.** mplib 0.1.1 is incompatible
   with the main NumPy 2 runtime on the target. Exact side-runtime pins, the adapter preflight, and
   manifest fingerprints prevent silent ABI drift, but every new target image must rerun
   `verify_planner_runtime.py`; a future upstream ABI-compatible planner could revisit this split.
7. **M2 planner success is high but not perfect.** The clean target gate passed its six-task smoke
   and 177/180 balanced benchmark (98.33%) with three classified planning failures, zero wrong-target
   successes, zero unclassified failures, and zero crashes. Downstream collection must retain its
   bounded retry and complete-group rules rather than assume every seed succeeds first try.
8. **Dual-runtime replay identity must remain exact.** M3A collection and independent action replay
   must use the same ten-field planner runtime fingerprint. Main-runtime inspection may not be used
   to admit a trajectory whose real action replay was skipped or executed under a different ABI.
9. **M3A remains the immutable raw authority.** The accepted 60-group archive now exists; future
   export or training code must continue to reject partial groups, changed checksums, changed replay
   evidence, or a different collection fingerprint rather than silently repairing source data.
10. **M3B remains a derived immutable view.** The accepted 360-episode export now exists; M4 must
    bind its exact export/split fingerprints and must not recompute splits, normalization from
    validation/test, or derived videos under the same identity.
11. **M4.1 succeeds only under frequent explicit gripper projection.** Target smoke passed 1/1
    PerTask and 6/6 TaskOneHot, but 834/951 rollout actions projected the gripper and strict-
    unprojected success was 0/7. Raw ACT validity must remain reported false; future baselines should
    not confuse projected control success with calibrated raw regression. The later M4 full and M4.2
    development evidence preserve raw/runtime action statistics separately; M4.3 must continue that
    separation.

## D-041 — Audit plan-bound completed M4 evidence without changing training identities

The first dual-RTX 5090 full run trained and sealed all eight 100,000-step ACT models at Git commit
`ceb73db1a0fe88fa7f58f347b51c542aa76663a0`. Evaluation later continued across the D-040
serialization-only repair, so each checkpoint remains bound to its original training commit while
each validation, test, and fresh-seed artifact is independently bound to the exact evaluation code
commit in its `EvaluationRuntimeManifest`. Requiring evaluation Git to equal training Git is
therefore incorrect: it rejects valid, runtime-audited evidence and can cause the outer verifier to
mistake a completed historical experiment for a missing current-commit run.

M4 now exposes explicit `--reuse-completed-evidence` target-full mode. It accepts only the exact
eight run fingerprints and directories recorded by the existing clean, passed full dry-run plan;
requires the same M3B export/split identity, canonical run order, fixed optimization and 5,000-step
checkpoint/evaluation schedules, all 20 sealed checkpoints per run, immutable selection/completion
records, every validation result, locked test, fresh-seed result, and final analysis; then invokes
only the project-owned comparison/provenance audit. It never calls `train_act.py` or
`evaluate_act.py`, and any incompatibility fails rather than falling back to training or rollout.
Normal target-full mode also refuses implicit retraining when a matching completed historical run
exists; an explicitly new experiment must use a new output root.

The comparison audit now validates every benchmark against its sibling evaluation-runtime manifest,
including the selected checkpoint, project action-bound mode, `physx_cpu` environment execution,
action-space contract, task mapping, split/evaluation configuration, deterministic reload flags,
and per-episode runtime fingerprint. It reports the single training Git commit separately from the
set of evaluation Git commits. This changes no M1 success/bounds, M3A/M3B artifact, ACT
weights/loss/configuration, train-only statistics, split, checkpoint fingerprint, selected
checkpoint, or stored evaluation result.

## D-042 — Rank evaluation concurrency by completed episodes, within the CPU quota

M4 policy evaluation is a simulator-and-renderer pipeline whose bottleneck is not represented by
instantaneous GPU utilization. LangMani therefore benchmarks independent `num_envs=1` validation
workers by completed episodes per minute, using the worst GPU as the primary rate. Every worker is
given a minimal clone of one immutable, selection-locked PerTask run; new benchmark runs
independently copy and hash-check the selected checkpoint so a faulty worker cannot mutate the
source. The benchmark never opens test or fresh-seed schedules, never finalizes a clone, and is not
M4 acceptance evidence.

The dual-RTX 5090 target measurement at Git
`c2951fd6657b1f57c6673d4265023babf65d1228` used the fixed six-episode validation schedule and
`project` action handling. The container required
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json`; the default NVIDIA ICD failed before a
rollout with `vk::createInstanceUnique: ErrorIncompatibleDriver`, and that failed warm-up remains
preserved separately as diagnostic evidence. With the working ICD, every executed worker completed
6/6 task-success episodes with zero infrastructure failures and matching schedule/semantic digests.
Recovered wall-clock measurements were:

| workers per GPU | phase | wall time | per-GPU episodes/min | cluster episodes/min |
| --- | --- | ---: | ---: | ---: |
| 1 | warm-up | 76.25 s | 4.721 | 9.442 |
| 1 | timed | 87.63 s | 4.108 | 8.216 |
| 2 | timed | 600.54 s | 1.199 | 2.398 |

The container's cgroup v2 `cpu.max` was `5000000 100000`, an effective 50-core quota. Process
sampling showed approximately 12 CPU cores demanded by each evaluator, so two GPUs at two workers
each already requested about 48 cores and suffered heavy throttling while GPU utilization remained
near idle. Four and six workers per GPU would request approximately 96 and 144 cores; they were
excluded by an explicit quota preflight and are recorded as unmeasured rather than failed.

Among the eligible, physically measured counts, one worker per GPU is the future evaluation-queue
recommendation: its timed per-GPU throughput was 3.43 times the two-worker rate. This is deliberately
not labeled a globally selected optimum. The one-worker warm-up/timed rates differed by more than
the benchmark's 10% temporal-repeat tolerance, and no second two-worker timing was run. Recovered
reports therefore keep `artifact_measurements_validated=true`, `recommendation_supported=true`,
`measurement_validated=false`, `selection_stability_validated=false`,
`recommended_worker_count_per_gpu=1`, and `selected_worker_count_per_gpu=null`. The historical
measurement used hard-linked checkpoint clones. A strict post-run reload verified that the source
checkpoint remained intact, but source-mutation isolation during that run was not guaranteed and is
reported false; future clones are independent copies. Recovery also reports subprocess return codes
as unknown instead of inventing successful exits and does not claim that the interrupted parent's
lost pre-benchmark source snapshot was persisted.

## D-043 — Quarantine observed M4 evidence behind two jointly generated M4.2 seed locks

M4 full is complete and experimentally valid, but its quality gate is false. The six PerTask
policies achieved 31/36 on locked test and 143/180 on historical fresh seeds;
Mixed-Unconditioned achieved 5/36 and 18/180; Mixed-TaskOneHot achieved 27/36 and 101/180. These
results diagnose a useful gap, but their test/fresh scenes have been observed and are permanently
ineligible for runtime, architecture, checkpoint, stopping, or M4.2 go/no-go selection. This
observed completion supersedes the earlier time-local pending-status note in D-040 without changing
that decision's compatibility rationale.

M4.2 therefore commits `m42_dev_v0` and `m42_final_v0` before any new rollout. The locks are
generated together from canonical JSON and a SHA-256 counter stream under
`langmani-m42-joint-seed-generation-v0`. The candidate payload binds the namespace, M3B export
fingerprint, exclusion digest, and integer counter. The first 64 digest bits are reduced to an
int31 seed; excluded/duplicate candidates are skipped. The first 12 accepted values form
development and the next 30 form final. Both schedules expand scenes in the canonical six-task
object-major/bin-minor order, use 200 episode steps, and bind evaluation fingerprint
`sha256:0746ecb022ac67e73bc04d19db39c6e11e76ce264669d5e1dec9474a34cbb6d4`.

The source lock records redundant provenance deliberately. Its union is exactly 125 seeds: all 60
M3A accepted scenes, five rejected candidates (together `0..64`), the exact M3B train/validation/test
partition, 30 observed M4 full fresh seeds, seed 0 used by smoke/tiny/M4.1 diagnostics, and 30
predeclared tiny fresh seeds. The exclusion digest is
`sha256:801e5b9d0595855729a45fa3bab25b85449afab2aec06065dd69233eec0f5610`.
The development fingerprint is
`sha256:981547e771b2b5cd3a77e2788bb49d29fc45b3f59c607021a03a4e2ce70b43f1`; the sealed final
fingerprint is
`sha256:b2aef313e076201f7a94875c835c2d606d8f255e3f75d53f7ac7fadcdbb267fc`.

Loading and regenerating the final lock is permitted for source audit. Materializing final rollout
episodes is a different capability: it requires an explicit final authorization, clean Git, an
exact implementation fingerprint, and locked horizon, gripper, and TaskToken-checkpoint selection
records. `--target-development` cannot supply that authorization. This makes accidental final-seed
reset/render a contract failure rather than a process convention.

## D-044 — Use the public LeRobot ACT ENV feature as the dedicated oracle TaskToken

Installed LeRobot 0.6.0 inspection found that ACT already supports one environment-state feature
through public `lerobot.configs.FeatureType.ENV`, `PolicyFeature`, and `ACTConfig.input_features`.
The model creates `encoder_env_state_input_proj = nn.Linear(env_dim, dim_model)` and inserts that
projected value as a dedicated encoder token in the sequence
`[latent, Panda state, environment state, image tokens...]`. This is the smallest maintainable
integration point and avoids copying, forking, patching, or subclassing the upstream ACT model.

`ACT-Mixed-TaskToken` therefore presents `CanonicalTaskTokenV0` as float32 `[6]` at
`observation.environment_state` with `NormalizationMode.IDENTITY`. The canonical basis vector is
projected by the public 6-to-512 linear layer; its columns plus shared bias act as six learned 512D
task embeddings. PandaPolicyStateV0 stays exactly 9D, action output stays 8D, and image, backbone,
Transformer, VAE, optimizer, loss, and 100,000-step training configuration remain the same as
Mixed-TaskOneHot. The model is an oracle discrete-task policy, not a language or text-token policy.

The public feature types/configuration are the supported dependency boundary. The concrete
`encoder_env_state_input_proj` and `encoder_1d_feature_pos_embed` attributes are additionally
checked to prove the intended injection location and are an upstream structural risk: an installed
ACT change must fail a contract test rather than silently change semantics. LangMani imports no
example-only or private module and modifies no installed file. The M4.2 architecture fingerprint
binds the mapping, feature key, input/hidden dimensions, normalization, and injection location.
Checkpoint resume rejects any mismatch.

The existing atomic checkpoint implementation now types its identity input through the
project-owned `ActCheckpointIdentity` structural protocol. Historical `ActRunIdentity` values
satisfy that protocol unchanged, while the separate M4.2 `TaskTokenRunIdentity` supplies the same
content-bound fields. This is a typing/interface generalization only: the checkpoint schema,
serialized historical identities, component digests, checkpoint fingerprints, and all eight M4
artifacts remain unchanged.

## D-045 — Select M4.2 runtime before training and keep final evaluation separately authorized

M4.2a changes no model weight. It evaluates exactly horizons 10, 5, and 1 from the unchanged
50-action chunk with frozen Mixed-TaskOneHot and green-left PerTask checkpoints on `m42_dev_v0`.
All horizon episodes use the existing explicit `project` runtime. The ranking is declared before
results: mixed success, post-grasp timeout, wrong-object interaction, representative success,
successful median steps, inference cost, then the larger horizon.

Only after horizon lock may `project` be compared with `BinaryGripperEnvPostprocessorV0`. Binary
maps action component 7 by sign (`>=0` to `+1`, `<0` to `-1`), preserves arm values before the
existing project processor, and records raw/binary/projected/executed values separately. It is
eligible only with no safety/error worsening and at least a five-point mixed-success gain, 25%
relative post-grasp-timeout reduction, or two-of-12 representative gain; otherwise project wins.
The transition tie-break counts only per-episode sign changes beyond one close and one release,
rather than penalizing the two transitions required by a normal pick-and-place episode.
This does not redefine M4.1 reject/project and does not add a bounded ACT head.

The selected horizon/runtime then applies to exactly one new TaskToken training run. Only M3B
validation selects its checkpoint. `m42_dev_v0` may compare frozen selected policies afterward but
cannot change the checkpoint. `m42_final_v0` remains inaccessible until every selection is
immutable and an explicit final command is issued. Target-development results, runtime selections,
TaskToken checkpoint identity, final paired metrics, and SmolVLA go/no-go are pending until the
corresponding commands actually complete; no value is inferred from implementation tests. M4.2
never starts M5 automatically.

The runtime-selection transaction also publishes one shared immutable
`experiment_manifest.json`. Its fingerprint binds the exact ordered eight M4 selected checkpoint
fingerprints and an M4.1 processor fingerprint reconstructed from every selected run's validated
`test/evaluation_runtime_manifest.json`: the versioned `project` configuration and actual
environment action-space bounds must agree across all eight runs. The runtime selection then
references this manifest, and the TaskToken run identity, validation queue, development evidence,
and future final evidence must all carry the same fingerprint. TaskToken-specific architecture and
checkpoint identities remain in their own immutable artifacts so that the shared provenance root
has no circular dependency on later outputs.

Target-development acceptance additionally revalidates all eight historical selected checkpoint
directories, atomic markers, component hashes, and strict reload contracts from current disk. It
compares the TaskToken effective training contract with the frozen Mixed-TaskOneHot manifest rather
than trusting reconstructed defaults; the only permitted model-input difference is 9D Panda state
plus the dedicated six-way ENV token in place of state-appended task one-hot. A second complete or
incomplete fingerprint-owned TaskToken run is ambiguous and rejected. Any Panda arm projection,
target-in-wrong-bin regression under binary gripper, nonphysical child command, or development
child that accessed the sealed final schedule invalidates the target-development gate.

## D-046 — Reuse completed runtime ablations only through a semantic-source repair lineage

A verifier or JSON-parser defect discovered after the development runtime sweep must not force a
second execution of the already completed 336 physical episodes, but a Git-commit mismatch alone
is not sufficient authority to reuse them. `verify_m42.py` therefore treats any existing runtime
command report or runtime evidence tree as a fail-closed recovery transaction. If the command
report is absent, partial evidence is rejected rather than resumed; if it is present, the verifier
audits all eight benchmark directories, exact 336-unique/420-logical accounting, ordered episode
sets, identity/artifact/completion fingerprints, development-only flags, selections, post-grasp
analysis, M3B/M4 checkpoint provenance, TaskToken fairness contract, and experiment manifest before
later stages may proceed.

The historical producer commit is read from the evidence. Its original commit-bound implementation
fingerprint is still reproduced from the exact legacy five-file payload used by the sealed report;
that compatibility calculation is not redefined. Reuse authority is deliberately wider: Git must
report exactly the same tracked path set, regular-file modes, and bytes for
`run_m42_runtime_ablation.py`, `environment/environment.yml`, `pyproject.toml`, the package root,
and every tracked file under `policies`, `environments`, `datasets`, and `experts`. Historical paths
come from `git ls-tree`; current paths come from `git ls-files`. Unsafe modes, links, missing files,
untracked non-ignored files, a new tracked semantic file, or any byte change are hard failures. The
committed development and final schedule JSON resources are therefore hashed without materializing
final episodes.

The frozen runtime selection converts nested JSON arrays into immutable tuples internally. The
public `TaskTokenFairComparisonContract.from_dict` remains list-only and is part of the byte-exact
runtime closure. Compatibility is instead isolated at the `train_act_task_token.py` consumer
boundary: `M42TaskTokenRuntimeSelectionFrozenTupleThawV0` recursively converts only Mapping and
tuple/list JSON containers back to canonical dictionaries/lists, rejects non-JSON values, and then
calls the unchanged public parser. The training command is not a producer of the 336 runtime
episodes, so this parser repair neither weakens nor normalizes their semantic source closure.

Successful recovery writes one new content-addressed immutable JSON under
`runtime_repair_lineage/`. It binds producer and consumer commits, the semantic closure, the old
command report, both selections, selected runtime, experiment manifest, post-grasp analysis, and
all eight identity/artifact/benchmark fingerprints. The lineage is a sibling of the old runtime
evidence, declares that old evidence was not rewritten, and is stable for a repeated run at the
same commit. It records the complete semantic path list/count, byte-exact closure and raw
fingerprints, and the consumer repair ID. This recovery mechanism does not materialize
`m42_final_v0`, alter any historical checkpoint or selection, or claim M4.2 development acceptance
by itself.

## D-047 — Audit frozen shared-policy semantics before one factorized repair

M4.2 target-development completed but rejected TaskToken: the frozen PerTask, State-OneHot, and
TaskToken controls achieved 56/72, 35/72, and 15/72 successes respectively, while State-OneHot and
TaskToken grasped wrong objects 13 and 33 times. State-OneHot's action-sensitivity magnitude was
approximately equal to PerTask, so output change alone is not evidence that a shared policy follows
the requested object/bin semantics.

M4.3a therefore introduces a zero-training semantic-alignment audit before any FactorFiLM code or
training. For each fixed base-camera RGB plus `PandaPolicyStateV0[9]` observation, it obtains all six
frozen PerTask reference chunks and the requested State-OneHot/TaskToken chunk after the saved
LeRobot postprocessor but before binary-gripper or bound projection. M3B validation and
`m42_dev_v0` are the only inputs and remain separately identified. M3B test, historical M4 fresh,
and `m42_final_v0` are rejected identities, not optional exclusions.

`ActionChunkDistanceV0` reports raw, action-range-normalized, train-standard-deviation-normalized,
and cosine distances over full/arm/gripper components and first/first-five/locked-horizon/full
windows. The declared primary retrieval metric is action-range-normalized arm-only L2 over the
locked horizon. Deterministic canonical tie-breaking, full-task top-1/top-2/MRR/margin, object and
bin centroids, object-conditional bin retrieval, first-interaction confusions, and an exhaustive
post-grasp taxonomy separate output sensitivity from semantic correctness and later execution
failure.

The historical M4.2 development artifact contains real per-episode post-grasp phases and aggregate
wrong-object metrics, but not the complete first object displaced, first approached bin, first
object-entry bin, and final per-object/bin relationship needed by `FirstInteractionRecord`.
Consequently M4.3a binds and reports the real aggregate post-grasp distribution while declaring
first-interaction evidence/confusions unavailable. Reconstructing those fields from aggregate counts
would fabricate evidence and is prohibited.

M4.3 audit evidence is canonical-JSON/SHA-256 content with a path-independent configuration
fingerprint. A fingerprint-owned sibling staging directory writes configuration, ordered sources,
scope artifacts, machine/human summaries, checksums, manifest, and a last completion marker.
Independent validation precedes atomic promotion. Completed roots are immutable and any linked,
partial, extra, missing, checksum-mismatched, or semantically incompatible reuse fails. Command
reports may contain operator paths, but portable manifests and semantic fingerprints may not.

This adds project-owned M4.3 contracts/evidence and two command boundaries,
`scripts/audit_act_semantics.py` and `environment/verify_m43.py`; it changes no dependency version,
M1/M3A/M3B contract, historical checkpoint, action runtime, or final schedule. Non-target
verification may set `semantic_audit_implementation_validated=true` while keeping
`semantic_audit_completed=false`, `factor_film_training_completed=false`,
`final_schedule_accessed=false`, `smolvla_go=false`, and `physical_target_validated=false`.
M4.3b remains blocked until real validation and development audits complete and validate from a
clean committed boundary. Local structural results and any future target metrics must be recorded
separately; this decision fabricates neither.

The M4.3a input gate treats the runtime lock, complete TaskToken validation queue, all 20 immutable
validation artifacts, and a recomputed validation-only ranking as checkpoint provenance. It loads
the M4.2 development comparison/completion only for `m42_dev_v0` or `combined`, never for
`validation`. The CLI therefore has no M4 diagnostics-root argument: the only necessary historical
M4 inputs are the explicitly allowlisted run manifests, validation selection locks, and train-only
statistics beneath the checkpoint root. Successful runtime reports must explicitly state false for
test, fresh-seed, and final-schedule access; the command layer does not synthesize those claims.

The 2026-07-16 local gate passed Ruff format/lint, `pip check`, sdist/wheel build, and the non-target
verifier. Focused M4.3a tests reported 147 passed and one Windows symlink-privilege skip; the full
CPU-safe suite reported 840 passed, 14 skipped, and 15 hardware/rendering deselected. These results
do not set semantic-audit completion, deterministic real-checkpoint reload, GPU execution, or
physical-target validation. Real inference and GPU availability remain pending.

## D-048 — Keep runtime-selection and TaskToken-validation producer lineages distinct

The frozen M4.2 target-development archive has two independently committed producer lineages. The
runtime lock was produced at `63ae2efbcd849428ccd9ec375bc818872a2073e5`, while all 20 immutable
TaskToken validation artifacts, the TaskToken training manifest, and its completed validation queue
bind `36bd6d81ff5e3b2c74977da1eec0fb0231d2f922`. Reconstructing validation artifact identities from
the runtime producer therefore names 20 nonexistent directories even though the archive is intact.

M4.3a now reconstructs the TaskToken validation evaluator closure from the Git commit jointly bound
by the TaskToken manifest and validation queue. It still validates the runtime fingerprint, runtime
source fingerprint, locked horizon/gripper selection, complete queue, all result fingerprints, and
the validation-only selection independently. It neither enumerates directories to guess identities
nor rewrites historical evidence. Malformed or cross-lineage queue/artifact metadata remains a hard
failure.

This is the verified lineage contract for the frozen M4.2 target-development evidence. If a future
workflow evaluates a TaskToken training run from a different committed evaluator, that workflow
must publish an explicit immutable validation-producer lineage index; it must not infer one from the
runtime lock or silently generalize this compatibility rule.

## D-049 — Interpret the legacy M4.2 rollout split only inside a locked development schedule

The M4.2 producer uses one shared rollout record type for unseen-scene schedules. Consequently its
completed `m42_dev_v0` comparison serializes `rollout.split="fresh_seed"` even though the comparison,
all eight benchmark records, and all 216 episode records bind the locked `m42_dev_v0` schedule and
its development fingerprint. This label does not identify or access the historical M4 fresh-seed
benchmark, but a context-free M4.3 string prohibition cannot distinguish the two meanings.

M4.3a accepts this historical label only at the exact `rollout.split` field of a complete
`m42_dev_v0` episode nested under one of the six PerTask, State-OneHot, or TaskToken development
benchmarks. Before that exception is considered, the comparison and each benchmark must bind the
locked development schedule fingerprint, each episode must declare `m42_dev_v0`, and the exact
12/72/72 episode cardinalities must hold. The committed development lock additionally determines
every episode index, scene seed/ID, task ID, and rollout schedule digest. The same string at any
other JSON path, any mismatched schedule or fingerprint, and every test/final access flag remain
hard failures.

Historical evidence is not rewritten and the M4.2 producer schema is not changed. A future rollout
schema should distinguish development schedules from generic unseen-scene evaluation directly;
this compatibility rule is intentionally limited to the immutable M4.2 development artifact.

## D-050 — Canonicalize saved processor statistics through their public float32 representation

Installed LeRobot 0.6.0 reloads the public `UnnormalizerProcessorStep.stats["action"]["std"]`
record as a NumPy `float32[8]` array. M4.3a originally recognized only Torch tensors and Python
`Sequence` values; NumPy arrays do not implement that sequence abstract base class, so all eight
real checkpoints loaded successfully but the audit rejected the saved action statistics before its
first inference.

M4.3a now accepts only an exact NumPy `float32[8]` array at that public processor boundary, while
retaining exact `float32[8]` Torch and numeric list/tuple fixture paths. Wrong dtype/shape,
unsupported or duplicate records, nonfinite values, and nonpositive values remain hard failures.
The installed-version contract test exercises the real public `UnnormalizerProcessorStep` state
reload and must be rerun before any future LeRobot upgrade. The persisted project-owned train
statistics use JSON floats with more precision than their saved processor representation, so
comparison first canonicalizes the project values to float32 and then requires exact tuple equality
across State-OneHot, TaskToken, and the fingerprint-validated train record. No tolerance is
introduced and no checkpoint or processor artifact is rewritten.

## D-051 — Implement one factorized FiLM ACT through isolated instance-local hooks

The clean RTX 5090 M4.3a audit completed 108 frozen-policy observations at Git
`dfea8b3d7d28274909ff178cb9087a9a90e17ee7` and promoted evidence fingerprint
`sha256:6342bdf4b019df203e6021947cbb39deacac2ea78cc585b5062091ed1d228671`. Under the
locked action-range-normalized arm-only execution-window L2 metric, State-OneHot reached 37.96%
full-task top-1, 76.85% object retrieval, and 50.93% bin retrieval; TaskToken reached 24.07%,
50.00%, and 49.07%. Both changed outputs without reliably following the requested semantics,
object confusion remained, bin retrieval was approximately random, and the development evidence
also retained post-grasp shared-control failures. TaskToken remains rejected. This immutable audit
authorizes one oracle repair, not a language claim or final benchmark.

M4.3b therefore implements exactly one `ACT-Mixed-FactorFiLM` policy. Stable TaskSpec metadata maps
to independent canonical `TargetObjectConditionV0` (`red_cube`, `green_cube`, `blue_cube`) and
`DestinationBinConditionV0` (`left_bin`, `right_bin`) indices. A 32D object embedding projects to
gamma/beta and applies residual FiLM only to the ResNet-18 layer-4 `[B,512,8,8]` feature map before
ACT image projection. A separate 32D bin embedding applies residual FiLM only to the encoded
`[B,512]` Panda-state token before Transformer processing. `PandaPolicyStateV0` remains 9D. There is
no appended one-hot, object/bin vector, ENV input, combined token, instruction parser, or language
embedding, and neither factor is allowed to enter the opposite path.

Installed LeRobot 0.6.0 exposes public `PreTrainedConfig`, `ACTConfig`, `ACTPolicy`, and
`make_act_pre_post_processors`, but no public callback at those intermediate representations. A
single adapter subclasses `ACTPolicy` and registers instance-local Torch forward hooks on the
semi-stable `policy.model.backbone` and `policy.model.encoder_robot_state_input_proj` outputs. It
does not import/copy the private `modeling_act.ACT` implementation, patch installed files, or
monkey-patch global behavior. The adapter binds the exact public signatures, version, underlying
model module/name, feature-map key, image/state projection shapes, latent/state/64-image-token
layout, decoder positions, action head, and one-hook-call-per-path invariant. Any drift fails
clearly; no State-OneHot, TaskToken, or unconditioned fallback is permitted.

Both FiLM projections use `normal(mean=0,std=1e-5)` weights and zero bias. This is close to identity
while preserving a usable first-step gradient to each embedding; exactly zero projection weights
would initially block embedding gradients. The full base ACT has 51,576,712 parameters. The object
path adds 33,888 and the bin path 33,856, so only 67,744 parameters are added and the full policy has
51,644,456. No backbone, hidden size, action head, loss, augmentation, sampler, or optimizer change
is allowed.

FactorFiLM uses independent canonical-JSON/SHA-256 architecture, mapping, run, checkpoint,
manifest, validation-queue, selection, and dry-run identities. These bind the completed audit
evidence, exact M3B/split/train-statistics fingerprints, ordered train/validation episodes, base ACT
configuration, both embedding/injection contracts, optimizer/data/seed/Git/version identity, and
expected 20 checkpoints. Machine paths do not affect portable fingerprints. The existing M4 atomic
checkpoint lifecycle is extended with both FiLM paths and a strict local sidecar/reload; historical
M4/M4.2 identities and checkpoint bytes remain unchanged. Resume rejects incompatible or completed
runs. It binds locked seed-0 model/optimization fingerprints and the H=10/project runtime, accepts
only the latest declared checkpoint or one sole exactly-next atomically promoted orphan, and lets
clean staging remove only a matching owned unlinked incomplete directory. FactorFiLM direct saves
preflight a new/empty destination before the public LeRobot writer runs. Command output/report
paths reject dataset, evidence, source, Git, historical artifacts, and links before any write.

Training is intentionally deferred. The intended target run must match State-OneHot's 288 M3B train
episodes, 36 validation episodes, 9D train-only normalization, ResNet-18, action chunk 50, batch 32,
100,000 steps, checkpoint interval 5,000, AdamW/loss/bfloat16, seed 0, horizon 10, and `project`
runtime. Only M3B validation may rank checkpoints. The exact 20-entry queue has a canonical
fingerprint, and a selection must embed and match that queue's run, schedule, fingerprints, and
steps before ranking in success/wrong-object/wrong-object-in-bin/off-
table/timeout/loss/earlier-step order. M3B test, M4 fresh, `m42_dev_v0`, `m42_final_v0`, and semantic-
audit observations are forbidden training or selection inputs. `m42_dev_v0` may be used only by a
later post-selection development rollout; the final schedule stays sealed.

The structural commands are `scripts/train_act_factor_film.py --dry-run`, `--fixture`, and
`environment/verify_m43.py`. Their local forward/backward, gradient, optimizer, processor,
checkpoint, fresh-instance reload, deterministic inference, path-safety, resume, and selection tests
may set implementation and fixture flags only. They must leave `factor_film_training_completed`,
checkpoint completeness/selection, development completion/quality, physical validation, final
authorization/access, and `smolvla_go` false.

The published validation queue also carries each checkpoint's finite offline validation action
loss. That value is read from the checkpoint's fingerprint-bound training metric through a
metadata-only integrity loader, so the sixth ranking key cannot be replaced independently of the
checkpoint and queue fingerprints. Target identities independently require strictly ordered,
unique 288-train/36-validation episode lists and bind their counts and canonical list
fingerprints into the data contract. The training manifest rechecks the base ACT and H=10/project
runtime contract. Command reports additionally protect every tracked repository-root file (and
`outputs/.gitkeep`) from overwrite; rejected report paths remain byte-identical and unwritten.

The exact later clean-Git target-development command is:

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

An interrupted compatible run may add `--resume-checkpoint`; no implementation task may execute the
command, select a real checkpoint, run `m42_dev_v0`, access `m42_final_v0`, or start SmolVLA.

## D-052 — Separate the immutable FactorFiLM producer from target-development evidence

M4.3b target-development is authorized as one seed-0 experiment, but authorization does not permit
the target evaluator to redefine the architecture after observing results. The architecture and
training producer is therefore fixed at clean Git commit
`8ee0f1babf36b91d1ee2a39701e4a6db6003b660`. Its run manifest, 20 scheduled checkpoints,
processors, metrics, producer Git identity, and fingerprints remain immutable. Selection, fresh-
process reload, development rollout, semantic analysis, and the independent verifier may be added
by a later commit; those artifacts record their evaluator Git separately and live in a
fingerprint-owned evidence tree outside the training run. They cannot rewrite or relabel producer
artifacts. This supersedes only D-051's time-local statement that target execution was deferred;
the D-051 architecture, identity, data, resume, selection, and access constraints remain unchanged.
All source repairs are authored/tested in the local repository and committed/pushed before the GPU
server fetches or checks out them; direct server source edits are not valid experiment provenance.

Only M3B validation selects a FactorFiLM checkpoint. Each of the exact 20 steps
`5000,10000,...,100000` receives the same six scene groups times six tasks, for 36 closed-loop
episodes. The predeclared rank is highest success count, lowest wrong-object interaction count,
lowest wrong-object-in-target-bin count, lowest off-table count, lowest timeout count, lower
checkpoint-bound validation objective, then earlier step. The queue's historical field name
`offline_validation_action_loss` is retained for schema and comparison compatibility, but its value
has always been the existing total ACT validation objective returned by the shared evaluator:
action reconstruction plus the configured weighted KL term when the VAE is active. It is not an
arm-only loss, and changing its name or numeric definition now would break fair comparison with
M4/M4.2 rather than correct those immutable runs.

Development access requires an immutable selection and a fresh operating-system-process reload of
the selected policy, preprocessor, policy postprocessor, mappings, and H=10/`project` runtime. One
fixed validation observation must reproduce the entire postprocessed environment-semantic
`[50,8]` action chunk at `atol=1e-6`, `rtol=1e-6`. The development comparison then executes exactly
72 PerTask, 72 State-OneHot, and 72 FactorFiLM episodes on paired `m42_dev_v0` identities, 216 total.
It recomputes validation/development semantic retrieval separately and records new first-
interaction, post-grasp, raw-action, and runtime-action evidence. M3B test, historical M4 fresh,
`m42_final_v0`, automatic retraining, SmolVLA, and M5 remain prohibited.

The shared rollout contract now includes an explicit `EvaluationSplit.DEVELOPMENT` value.
`run_m42_checkpoint_benchmark` emits that value for newly executed `m42_dev_v0` episodes instead
of mislabeling them as historical `fresh_seed`. This is a forward report-semantics repair only:
D-049 continues to govern immutable M4.2 evidence already serialized with the legacy label, and no
historical file is rewritten. The final-evaluation path retains its existing split behavior and is
unavailable to this target-development workflow. Consequently prohibited-source audits can treat
new `development` and historical fresh-seed evidence as distinct identities without weakening the
compatibility exception for old M4.2 records.

The new interaction audit requires exact cube orientation and angular velocity as well as position
and linear velocity. The environment's narrow expert-only policy-rollout diagnostic accessor is
therefore extended with those batched tensors. The accessor is consumed only after policy action
selection to classify evidence. It does not change `PandaPolicyStateV0[9]`, visual observations,
privileged-state gating, per-step `info`, M1 success geometry, or any policy input. Unit tests retain
the visual no-leakage assertion and independently check the diagnostic shapes.

Target-development evidence uses resumable, checksum-validated, atomically promoted stages and a
separate read-only verifier. Completion/physical validity is independent from the 16-condition
development quality gate: a correctly executed experiment may set `passed=true` and
`physical_target_validated=true` while `development_quality_gate_passed=false` and
`final_benchmark_authorized=false`. If the quality gate passes, authorization is recorded only; the
evaluator still cannot open or run final. This decision records interfaces and evidence ownership,
not a claim that target training, selection, reload, 216 rollouts, quality, or physical validation
has already succeeded.

## D-053 — Stop M4.3b before training when the locked producer fails real preflight

On 2026-07-17, the real target dry-run at the clean locked producer commit
`8ee0f1babf36b91d1ee2a39701e4a6db6003b660` exited with code 90 before policy construction,
optimizer creation, GPU training, model-root creation, or checkpoint promotion. The immutable
command report at
`outputs/diagnostics/m43/factor-film-target-development-preflight.json` records
`AttributeError: 'EpisodeExportRecord' object has no attribute 'episode_index'`. The M3B project
type exposes the derived episode identity as `lerobot_episode_index`; the FactorFiLM validation
schedule used the nonexistent legacy name.

No target training, checkpoint, selection, development rollout, final access, or SmolVLA evidence
was produced by this failed attempt. A local compatibility repair and real-type regression test are
isolated in clean commit `0088e2937556c123c37c2dbe69f73301b1eebfd0`. That patch changes only
the validation schedule's EpisodeExportRecord lookup and exposes its canonical records/digest; it
does not change the FactorFiLM architecture, parameter count, optimization, dataset, split,
train-only statistics, seed, checkpoint cadence, H=10/`project` runtime, selection order, or quality
thresholds.

Commit `0088e2937556c123c37c2dbe69f73301b1eebfd0` is a candidate compatibility producer, not an
implicitly authorized replacement for D-052. Server hot patches, runtime monkeypatches, Git
identity spoofing, or relabeling it as `8ee0f1b...` are prohibited. Until the training-producer lock
is explicitly reauthorized, M4.3b remains stopped after preflight. The post-training evaluator and
independent verifier accept an explicit full training Git commit and record training/evaluation
lineages separately so a future authorization can remain exact and auditable.

## D-054 — Reauthorize the compatibility producer without changing the FactorFiLM experiment

The explicit post-D-053 authorization on 2026-07-17 keeps the M4.3b architecture and structural
baseline at `8ee0f1babf36b91d1ee2a39701e4a6db6003b660` and reauthorizes exactly one target-training
producer at clean commit `0088e2937556c123c37c2dbe69f73301b1eebfd0`. This producer contains only
the compatibility repair already isolated by D-053: it reads the real M3B
`EpisodeExportRecord.lerobot_episode_index` field when constructing the immutable validation
schedule. It does not change FactorFiLM architecture, parameter count, M3B data or fingerprints,
train/validation split, train-only normalization, ACT or optimizer configuration, seed/data order,
100,000-step budget, 5,000-step checkpoint cadence, exact 20-checkpoint queue, H=10/`project`
runtime, seven-key validation-only ranking, semantic metrics, or any of the 16 quality thresholds.

The GPU server must fetch and check out `0088e2937556c123c37c2dbe69f73301b1eebfd0` before training;
server source hot patches, monkeypatches, Git identity spoofing, and relabeling remain prohibited.
The immutable training manifest, checkpoints, processors, metrics, and fingerprints retain that
exact producer identity. Post-training selection, fresh-process reload, `m42_dev_v0` evaluation,
semantic/interaction evidence, and the independent verifier use implementation commit
`1bacb66d2a6f7c3f2d18d6f65ad7865af9a12cd6`. The server checks out the current clean authorization
commit containing that implementation for those stages, writes outside the immutable training run,
and passes `--training-git-commit 0088e2937556c123c37c2dbe69f73301b1eebfd0` explicitly to both the
evaluator and read-only verifier.

This decision authorizes execution, not a completed result. M3B test, historical M4 fresh seeds,
`m42_final_v0`, automatic retraining, final authorization, SmolVLA, and M5 remain inaccessible. No
training, selection, reload, physical rollout, quality, or final flag becomes true until the real
artifacts pass their declared stages and the independent verifier.

## D-055 — Close the shared-ACT search after the completed FactorFiLM development result

The authorized compatibility producer completed its exact 100,000-step seed-0 FactorFiLM run and
all 20 immutable checkpoints. M3B-validation-only selection chose step 70,000. A fresh process
reloaded the selected policy and both processors and reproduced the full postprocessed `[50,8]`
action chunk with maximum absolute and relative error both zero. The paired physical
`m42_dev_v0` comparison completed 72 episodes for each policy: PerTask 56/72 (77.78%),
State-OneHot 36/72 (50.00%), and FactorFiLM 38/72 (52.78%). FactorFiLM recorded six wrong-object
grasps, two wrong objects in a target bin, 34 timeouts, zero target-in-wrong-bin, zero target-off-
table, zero arm projection, and no non-finite or malformed action. The independent verifier
accepted the experiment and physical evidence while the 16-condition development quality gate
remained false.

This is a valid negative architecture result, not an infrastructure failure. FactorFiLM final
authorization is false; M3B test, historical M4 fresh, and `m42_final_v0` were not accessed. The
shared-ACT architecture search stops here. M4 PerTask checkpoints remain the frozen low-level
control ceiling, and no further TaskToken, FactorFiLM, or other shared-ACT tuning is authorized.

## D-056 — Adopt modular language routing over the six frozen PerTask controllers

M5A separates command understanding from continuous control. A project-owned router maps one
out-of-band command to a canonical M1 `TaskSpec`, or rejects it before execution. A portable
registry then selects exactly one of the six already selected PerTask ACT checkpoints. M1 is reset
with the scheduled oracle task even when the predicted route is wrong; only controller selection
uses the predicted task. This preserves the M1 success oracle and makes routing and control failure
independently attributable.

M5A compares a deterministic rule router, one factorized compact text classifier, and one pinned
structured local instruct-model router. The corpus is deterministic, balanced, SHA-256 identified,
and split by template family across train, validation, development, and final. The classifier uses
train only for gradients and validation only for selection, calibration, and threshold selection.
The local LLM uses one versioned prompt artifact frozen before development and strict project-owned
JSON validation with at most one repair. All few-shot example IDs are train-family only; validation
may freeze the prompt/configuration but never supplies in-context examples. Language-only and
control evaluation consume the same prompt template, ordered example IDs, generation configuration,
and prompt fingerprint. Rejection contains no executable task and cannot dispatch, reset, or step.

The four locks `m5a_language_dev_v0`, `m5a_language_final_v0`, `m5a_control_dev_v0`, and
`m5a_control_final_v0` are created before training. Development contains 12 new scenes times six
tasks; final contains 30 new scenes times six tasks. Both exclude prior
M3A/M3B/M4/M4.1/M4.2/M4.3 seed sources. Target-development may validate final fingerprints but
cannot materialize final commands or episodes. M3B test frames/actions/videos, `m42_final_v0`, M2
expert rollout, controller retraining, cloud APIs, SmolVLA, and automatic final execution remain
prohibited. The normative contract is `docs/M5A_LANGUAGE_ROUTING_SPEC.md`.

For M5A, `passed=true` means the declared target-development evidence chain completed correctly;
it does not imply `development_quality_gate_passed=true`. `final_benchmark_authorized` is the OR of
the learned-router gates, records permission only, and never executes final. Final, test-content,
`m42_final_v0`, and SmolVLA access flags remain false throughout development.

## D-057 — Pin the Transformers 5 runtime required by M5A

LeRobot 0.6.0 declares Transformers `>=5.4,<5.6`; the project environment previously obtained no
direct Transformers installation from `lerobot[dataset]`. The unrelated system Python contained
Transformers 4.57.3 with a Hugging Face Hub requirement incompatible with the project environment,
so it is not a valid integration source. M5A therefore adds direct pins
`transformers==5.4.0`, `tokenizers==0.22.2`, and the directly imported
`safetensors==0.8.0` to both package and environment declarations. A local Python 3.12 fixture
using the project Torch 2.11 runtime verified a tiny DistilBERT forward pass,
`save_pretrained`, local-only reload, and exact output/tokenization reproduction. CUDA support and
the real pinned model revisions remain target-development evidence, not a local claim.

Transformers 5.4 exposes deterministic generation but no declared JSON-schema or grammar-guided
generation interface in the installed API. M5A therefore records `do_sample=false` as the effective
deterministic mechanism, parses output through a strict stdlib schema, allows at most one explicit
repair, and rejects malformed output after that bound. A requested temperature of zero is not
misrepresented as an active sampling temperature. No extra JSON-schema framework or grammar
dependency is introduced. Actual encoder/LLM model IDs and immutable revisions are recorded only
when the target command successfully loads the explicitly configured artifacts.

## D-058 — Keep M5A controller discovery metadata-only and final language lazily sealed

M5A must prove that its six frozen PerTask controllers are deployable without reopening historical
evaluation content. `load_controller_registry_metadata` therefore uses an explicit allowlist: the
M3B completion marker, completed PerTask run manifests, validation-only checkpoint selections,
selected checkpoint artifact manifests/hashes, selected validation runtime manifests, and the
M4.2 runtime-selection lock plus its referenced experiment manifest. It rejects comparison and
verification summaries and paths belonging to test, historical fresh, or final results. The
portable registry retains only deployable action/state/image metadata and an opaque fingerprint of
the full source data contract; it does not expand historical evaluation schedules. Active M1 action
bounds are checked against the frozen validation contract before controller loading.

Development corpus construction likewise cannot carry final command strings. It creates the final
manifest from opaque semantic IDs, exact counts, and a sealed content digest without importing
`_final_language_authority.py`. Only a separately authorized future-final entry point may lazily
import that module and pass the explicit final-authorization guard. Target-development may validate
the seal and final schedule identities but cannot materialize raw final text or control episodes.
Structural and near-duplicate identities are computed from split-independent template skeletons
and are themselves bound into the isolation report; split names cannot make isolation pass by
construction. A future authorized final run resolves opaque schedule slots through a deterministic
semantic-slot map after opening the final authority.

Access flags describe content access, not exclusion-only identities. `test_split_accessed=false`
still permits finalized M3B sidecar split/scene-seed IDs and the completion fingerprint solely to
prevent seed reuse; no test rows, observations, actions, frames, video, or results may be opened.
`historical_fresh_accessed=false` permits only the committed exclusion summary's prior seed IDs,
opaque digest, and configuration lock, never the runtime schedule, episode commands, or evaluation
results. `m42_final_accessed=false` and the M5A final-access flags permit sealed lock IDs/digests and
exclusion-only seed identities, but prohibit materialization, reset, render, or rollout. These
semantics keep audit provenance useful without treating an opaque lock as benchmark execution.

Target development also records a dedicated active-M1 rejection probe. It must prove zero
controller lookup, reset, and environment step and is independently rehashed before
`rejection_noop_probe_validated=true`. M5A action evidence has distinct raw,
binary-transformed, projected, and executed streams; under the locked project-only runtime the
binary-transformed entries are explicitly absent rather than copied from another stream.

## D-059 — Gate M5A compute through one authoritative classifier run and progressive control

The router architectures, language metrics, safety contracts, provenance, rejection boundary,
frozen controller registry, and sealed-final rigor from D-056 remain unchanged. Its execution and
budgeting model is superseded. M5A development now advances only through separately authorized
implementation, fixture, tiny-overfit, pilot, resumed training, language-development, one-scene
smoke, three-scene screen, and six-scene full-development stages. A failed candidate is recorded
but does not receive another seed, encoder, model sweep, or larger physical schedule.

`FactorizedTextClassifierV0` uses exactly training seed 0 and reports that initialization
robustness was not evaluated. The locked 900-example train view with batch size 32 has 29 steps per
epoch. The maximum is five epochs/145 steps, with validation at 29-step intervals and patience one.
The one authoritative run pauses at step 29 and retains model, optimizer, constant scheduler,
processor, Python/NumPy/Torch CPU/CUDA RNG state, validation history, and best-state identity.
Promotion resumes that exact run fingerprint rather than restarting. Checkpoint retention has
three atomic roles only: `pilot`, `latest`, and `validation_best`. Fixture, tiny-overfit, pilot
completion, pilot promotion, final training completion, selection, and calibration are distinct
evidence claims.

The parent development authority now contains ten scenes split into disjoint 1/3/6-scene
partitions. Smoke costs Oracle 6 plus primary 6 and optionally a second promoted router 6 (maximum
18). Screening costs Oracle 18 plus 18 per smoke-promoted learned router (maximum 54). Full
development costs Oracle 36 plus exactly one selected router 36 (72). RuleRouter remains a required
offline comparator. Thus the maximum pre-final physical cost is 144 episodes. The separately
authorized final authority contains 12 new scenes times six tasks and runs only Oracle 72 plus the
locked router 72, also 144; development cannot execute it.

The monolithic `verify_m5a.py --target-development` path is retired and fails before work.
`--verify-stage` independently validates one immutable stage report, and its `passed=true` means
that stage completed correctly, not that promotion was granted. Source must be committed and
pushed before target execution. No target stage is authorized by this implementation decision, and
no classifier training, physical rollout, final access, or SmolVLA result is claimed here.

## D-060 — Make the M5A step-29 pilot independently resumable and auditable

The authoritative classifier pilot is not accepted from top-level completion booleans. Its three
atomic checkpoint roles now carry the complete model/optimizer/constant-scheduler state, RNG state,
deterministic data-progression metadata, per-step training telemetry, and a processor contract that
binds model/tokenizer revisions, factorized label mappings, seed, corpus/split fingerprints, Git,
dependencies, boundary definition, and owner/preflight references. The training command disposes
the training instance and performs one local-cache reconstruction and deterministic validation-logit
reload without taking another optimizer step.

The independent `classifier_pilot` verifier is correspondingly deep rather than declarative. In a
fresh command process it reconstructs the pinned model/tokenizer, restores the `latest` checkpoint,
checks optimizer/scheduler/RNG/data progression and next-step identity, compares the fixed logits at
`atol=rtol=1e-6`, validates the exact three-role inventory and telemetry, and recomputes the complete
300-example validation contract and conjunctive promotion gate. A correctly executed rejected pilot
still reports `passed=true` and `classifier_pilot_promoted=false`; it never authorizes a new seed,
restart, automatic resume, development/final access, control rollout, LLM evaluation, or SmolVLA.

## D-061 — Authorize one bounded post-pilot recovery continuation

The authoritative seed-0 `FactorizedTextClassifierV0` pilot at Git
`689918fc3e8736f9d9981591d5904df0afad7031` completed step 29 correctly but failed its original
promotion gate. Its run fingerprint is
`sha256:9e3ac659fa2b695c843650df35e3779741d94b3dd70b2aec52a429bc4b2edf49` and its immutable
`pilot.pt` fingerprint is
`sha256:a892c2b73c87884b2b2acf843d22ed9318e6b661640eea6a4964ff8d739b5758`. Validation showed 85%
full TaskSpec, 85% object, 100% bin, 25% false-route, and 74.24%/2.78%/0% ambiguous,
unsupported, and malformed recall. The candidate remains rejected under the original protocol.

After observing only this validation evidence, and before language development, final sources,
control schedules, LLM evaluation, or controller dispatch, this decision authorizes one explicit
recovery continuation. It is not a new run and is not retroactively part of D-059/D-060. The
versioned amendment binds the exact run/checkpoint, current clean implementation commit, unchanged
training config, five-total-epoch bound, patience one, epoch-end validation schedule, and
`pilot`/`latest`/`validation_best` retention. A 15-item read-only data/loss audit and exact
model/optimizer/scheduler/RNG/data-progression reload must pass before step 30.

Recovery checkpoint ranking is lexicographic: higher full TaskSpec, lower false-route, higher
rejection macro recall, higher object, higher bin, lower validation loss, then earlier epoch. One
subsequent complete non-improving interval stops training, with no extension past step 145. The
original full quality thresholds remain conjunctive and unchanged. Training completion and
checkpoint selection are valid even if quality fails; calibration, threshold selection, and a
runtime artifact are permitted only after a fresh selected-checkpoint reload passes every quality
condition. The independent verifier must accept the bounded evidence while retaining
`classifier_pilot_promoted=false` and all language/control/final/SmolVLA access flags false. This
decision authorizes execution only and does not claim the recovery result.

## D-062 — Diagnose the completed classifier with one fixed validation-only decoder search

The D-061 continuation completed its original seed-0, five-epoch/145-step budget and selected the
epoch-4/step-116 `validation_best` checkpoint. Routeable full TaskSpec, target-object, and bin
accuracy reached 100%, but the original quality gate still failed: false-route was 7.5% and
ambiguous/unsupported/malformed recall was 63.64%/72.22%/94.44%. Near-zero train loss plus the
completed bound does not authorize another seed or more optimizer work.

M5A.1 therefore adds a project-owned, read-only post-hoc decision layer. It predeclares exactly four
decoder candidates, the 0.40-0.95 route/object/bin grids, the -0.20-0.50 route-margin grid, and an
identity-versus-single-validation-temperature comparison before scoring. Train content is opened
only for class, masking, and provenance audits; validation alone fits temperature and selects. No
development/final/control source, lexical rule, LLM, controller, environment, or simulator is used.

The original quality gate and lexicographic safety objective remain unchanged. A passing decoder
gets a new runtime fingerprint without changing the classifier checkpoint identity. If none passes,
the classifier is frozen as an offline rejected baseline, no runtime is published, and additional
training/seed authorization stays false. Immutable checksummed evidence, per-example diagnostics,
and an independent verifier distinguish correct analysis execution from model quality. This is a
public analysis/runtime interface change, not a dependency change.

The authorized CPU execution at implementation Git
`f45115b1d070001e9e82567eed34bb3dcd99149a` produced immutable evidence fingerprint
`sha256:f2401fcb5b054c79c3b7e9674321eefcf9576dc4dcc5407bd96151db9e9b518d`.
The selected conservative decoder used identity temperature and thresholds 0.90 route, 0.00 margin,
0.75 object, and 0.85 bin. It reduced false-route to zero and retained 179/180 routeable commands,
but reached only 71.21%/80.56%/94.44% rejection-class recall. The original conjunctive quality gate
therefore failed. Independent artifact and stage verification passed; the classifier is frozen, no
runtime was published, and no additional training, seed, language development, control, final, or
SmolVLA work was started.

## D-063 — Compare one pinned local LLM while retaining the rejected classifier as negative evidence

The completed M5A.1 decision is permanent: the seed-0 classifier and selected
`ConservativeRouteDecoderV0` remain frozen, descriptive, and ineligible for dispatch, promotion, or
final selection. Language development no longer requires an eligible classifier runtime.
`RuleRouterV0` remains a deterministic offline baseline, and exactly one learned candidate is
permitted: `StructuredLocalLLMRouterV0` with `Qwen/Qwen3-1.7B`.

M5A.2 pins model and tokenizer revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0, BF16, no quantization, and the public
Transformers runtime. It records exact model-file SHA-256 identities and uses the official chat
template with `enable_thinking=False`, greedy generation, a versioned strict reason vocabulary,
and at most one schema repair. Non-schema model text is not persisted. This is a project-owned
interface/evidence change; the already pinned dependency versions do not change.

A fixed train-only smoke precedes one complete validation pass. The complete language-development
split is opened only after the unchanged conjunctive validation thresholds pass. Fixture results
cannot fabricate these metrics or authorize control. Evidence uses staging, checksums, atomic
promotion, and independent raw-record metric/gate recomputation. The comparison loads no ACT
controller or robot environment and never opens language final, control schedules, M3B test,
historical fresh results, `m42_final_v0`, or SmolVLA. A passing offline experiment and a passing
LLM quality gate remain separate; even a selected LLM only authorizes a later separately executed
one-scene smoke.

## D-064 — Permit one fixed Qwen3-4B capacity comparison after the 1.7B rejection

The real M5A.2 Qwen3-1.7B run completed its 20-example smoke and 300-example validation correctly,
but failed the unchanged gate. Full TaskSpec/object/bin accuracy was 85.56%, false-route was
14.17%, final schema validity was 92.67%, malformed-after-repair was 7.33%, and rejection-class
recalls remained below threshold. That prompt, parser, output evidence, and result are frozen as a
negative baseline; no prompt edit, extra repair, additional seed, classifier training, or 1.7B
rerun is authorized.

M5A.3 authorizes exactly one controlled model-capacity escalation:
`Qwen/Qwen3-4B-Instruct-2507` at immutable model/tokenizer revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, Apache-2.0, BF16, no quantization, and one visible
GPU. The official instruct tokenizer template may differ only as transport formatting; the
semantic prompt, few-shot IDs/labels, corpus, validation order, strict schema/parser, one-repair
policy, generation bounds, metrics, and conjunctive validation/development gates remain identical.
The adapter accepts only the two already recorded exact 1.7B and 4B identities and has no 8B,
moving-revision, quantization, hosted, or automatic fallback path.

M5A.3 adds a project-owned candidate identity, prompt-equivalence contract, immutable staged
evidence, scale-delta report, and independent verifier. Smoke gates validation and validation gates
language development. A failed validation produces no development evidence and cannot trigger a
prompt change or another model. No optimizer, weight update, ACT/controller load, robot
environment, language/control final, M3B test, `m42_final_v0`, or SmolVLA access is permitted.
`passed=true` records correct offline execution; quality and later control authorization remain
separate, and `physical_target_validated=false` throughout.

## D-065 — Quarantine validation and authorize one grammar-constrained neuro-symbolic router

Repeated classifier and Qwen capacity decisions have made the original language validation split
architecture-selection evidence rather than an untouched promotion set. M5A.4 records
`language_validation_quarantined_for_architecture_selection=true`: validation may be replayed only
as a non-selecting diagnostic, while the first promotion decision for the frozen hybrid occurs on
the still-sealed `m5a_language_dev_v0` split. No validation score may change the prompt, lexical
tables, semantic schema, arbiter precedence, agreement rule, or thresholds.

M5A.4 authorizes exactly `NeuroSymbolicRouterV0`. A deterministic project-owned lexical parser and
the exact frozen Qwen3-4B revision produce separate fact frames; only
`DeterministicSafetyArbiterV0` may emit a `RouterDecision`. Routing requires exact object/bin
agreement and a supported pick-and-place action. Malformed, unsupported, ambiguous, conflicting,
or materially disagreeing frames reject without an executable TaskSpec. Direct Qwen3-4B remains a
frozen negative baseline and cannot dispatch.

This decision adds the exact dependency `outlines[transformers]==1.3.1` (Apache-2.0) without
changing Transformers 5.4.0 or Torch. The reviewed public integration wraps the already loaded
Transformers model/tokenizer and compiles one cached Pydantic JSON grammar. The schema permits only
declared keys, enums, finite list lengths, and booleans; downstream Pydantic validation remains
mandatory. If grammar compilation or Qwen generation is incompatible, the experiment stops—there
is no free-form or custom-decoder fallback.

The prompt uses only the frozen 20 train examples. Train smoke and historical validation precede
an fsynced immutable development-access lock; only then may the command open development once and
compare RuleRouter, the frozen classifier, frozen Qwen3-1.7B, frozen direct Qwen3-4B, and the sole
promotable hybrid. Immutable staging, atomic promotion, raw-record metric recomputation, and an
independent verifier remain required. This is offline GPU inference only: no optimizer, ACT,
controller, environment, final split, control schedule, `m42_final_v0`, or SmolVLA is authorized.

A semantic train-smoke failure is itself a terminal, independently verifiable experiment result.
It persists all 20 raw semantic-frame comparisons and the exact failed checks, marks the candidate
rejected, and creates neither historical-validation, lock, nor development evidence. This evidence
lifecycle correction does not change the frozen prompt, schema, parser, arbiter, model, or gate.

## D-066 — Separate execution safety from exact rejection taxonomy before development

The immutable M5A.4 smoke produced all six routeable TaskSpecs correctly and rejected every one of
the fourteen non-routeable commands without a TaskSpec, but its exact four-way status score was
17/20. The three status errors were conservative `reject_ambiguous` to `reject_unsupported`
substitutions caused by Qwen semantic-frame errors. They were not routes and could not dispatch a
controller. Treating those three no-dispatch outcomes as execution failures conflated two distinct
contracts; changing labels, prompt, model, parser, or arbiter after seeing them would be a more
serious protocol violation.

M5A.4.1 therefore versions `SafetyRoutingContractV1` and `RejectionTaxonomyContractV1` separately.
The former alone answers whether a valid TaskSpec may execute; the latter retains exact status,
reason, recall, confusion, and per-family diagnostics. The corpus taxonomy is audited read-only and
remains internally consistent. The observed errors remain model semantic errors, and the separate
exact reason analysis also records the unresolved-correction/conflicting-objects mismatch. No
historical label or score is rewritten.

The amendment is content-bound to the prior smoke and every frozen router input. Its safety gate is
created before the router lock and before validation/development access. Historical validation is
diagnostic only. The untouched development split supplies the first promotion decision under a
strict execution-safety gate that adds explicit zero-route checks for contradictory negation and
unsupported spatial reference. Exact taxonomy failure remains a limitation but does not control a
different physical action: every rejection returns before lookup, reset, or step. This decision
authorizes no training, controller load, environment construction, final source, or robot rollout.

Migrated target machines may retain the same immutable M5A.1 analysis under a different directory
name. M5A.4.1 therefore accepts that root only through the explicit
`--classifier-rejection-evidence` input, validates it with the selected checkpoint before loading a
GPU model, and binds its analysis, checkpoint, and selection fingerprints into the run identity.
Automatic discovery or fallback remains prohibited.

## D-067 — Bind the selected hybrid read-only before one-scene physical control

M5A.4.1 completed its immutable offline protocol and selected the unchanged
`NeuroSymbolicRouterV0` under the execution-safety gate. Exact rejection taxonomy remains a
reported limitation, but every rejection is structurally non-executable. The next separately
authorized work therefore does not reopen language selection: it creates a path-independent
dispatch-source identity over the completed artifact, router lock, exact Qwen files and revisions,
prompt/schema/parser/arbiter fingerprints, and selection flags.

That source may bind only the canonical six-entry frozen PerTask registry through metadata. Model
loading remains exact local-cache-only; there is no download, fallback, prompt edit, controller
reselection, retraining, blending, or shared-ACT path. The first physical command accepts only
`one_scene_control_smoke` and runs Oracle plus the sole selected router. Rejection is tested through
the selected router before controller lookup, reset, or step. Routeable episodes continue to use
the schedule TaskSpec for M1 reset/evaluation and the predicted TaskSpec only for controller
selection.

An independent verifier rehashes the M5A.4.1 source, rebuilt six-controller registry, authoritative
one-scene schedule, rejection probe, all six learned atoms, all six Oracle atoms, active
EpisodeSpecs, and separated action streams. It reports correct stage execution and observed control
success separately. This interface change authorizes no three-scene/full/final stage, M3B test,
historical fresh, `m42_final_v0`, M2 expert, or SmolVLA access and changes no dependency version.

## D-068 — Revalidate the immutable pre-staging control lock instead of rewriting it

The first physical M5A.4.1 dispatch attempt stopped before model, environment, or controller work
because the server's authoritative schedule bundle was created before staged control reduced the
new-build allocation from 12 development plus 30 sealed-final scenes to 10 plus 12. Its corpus,
exclusion, task-order, and language identities were unchanged; only rebuilding with later global
scene-count constants produced a different lock.

Schedule construction remains 10 plus 12. The reader now also recognizes the original 12 plus 30
shape as a legacy immutable contract. It does not trust the recorded shape alone: it recomputes the
ordered unexcluded seed prefix, canonical six-task order, per-scene opaque language IDs, exclusion
digest, both schedule fingerprints, and the sealed-final summary. Only the two declared shapes are
accepted. Staged development reads the first disjoint 1/3/6-scene views from the validated parent;
it neither rewrites that parent nor materializes final text or episodes. No dependency or physical
success criterion changed.

## D-069 — Reuse the semantic extractor's system/user chat transport fingerprint

The second pre-episode attempt verified all 13 pinned Qwen/tokenizer files and the 8.06 GB snapshot,
then stopped because the new dispatch adapter compared the frozen transport fingerprint against the
generic single-user-message helper. The selected M5A.4 and M5A.4.1 router instead used the exact
Outlines transport: the semantic prompt is the system message and `{COMMAND}` is the user message.

The transport renderer is now project-owned alongside `OutlinesQwenSemanticFrameExtractorV0`, and
dispatch compares its hash against the immutable selected evidence before compiling the grammar.
This restores the original prompt transport; it does not change the prompt, tokenizer, model,
generation settings, parser, arbiter, or evidence. File hashes, snapshot size, semantic prompt hash,
and transport hash remain separate hard gates.

## D-070 — Accept the one-scene dispatch gate while reporting controller timeouts separately

The authorized one-scene target run used clean implementation Git
`16911a65540211d84883b354b1d1d9477174daae` on one NVIDIA GeForce RTX 5090 with driver
595.71.05, Python 3.12.13, Torch 2.11.0+cu128, CUDA 12.8, ManiSkill 3.0.1, Transformers 5.4.0,
and Outlines 1.3.1. Headless Vulkan/ManiSkill acceptance used the valid NVIDIA EGL ICD explicitly;
GPU simulation, policy RGB observation, human rendering, stepping, and clean close passed before
control. No dependency version or physical success geometry changed.

The run produced immutable control fingerprint
`sha256:6d6969730b284440094e5ee6281673dc602fd9320d9193fb638c4256b14bea32`, record-set fingerprint
`sha256:589edce7d0417be8947c3cf923d69b8aaa2299605a108e65d5ebde5f780cf1c9`, and completion fingerprint
`sha256:126d8900f949fd0735198d93825e9f6bc3cc917908c8e8d81bab3718e7b2bb5e`. The selected router
emitted all six correct TaskSpecs and dispatched all six matching frozen PerTask controllers. Its
real rejection probe returned `reject_unsupported` and performed zero controller lookup,
environment reset, and environment step. The six learned episodes and six Oracle ceiling episodes
each achieved 5/6 M1 success with one timeout. Both paths had zero wrong-object, wrong-bin,
off-table, invalid/non-finite action, infrastructure, and M2-expert failures. Arm projection was
zero; gripper projection remained explicitly reported (200 learned and 199 Oracle components).

The first independent audit exposed a verifier-only identity defect: the legacy validator required
`RuleRouterV0` even when the selected probe was produced by `NeuroSymbolicRouterV0`. Commit
`cf6fbc07971cc2c507675763f04e6740adce40a4` made the expected rejection-probe router an explicit
read-only verifier input, retained the legacy default, and added a regression test. The fresh
re-audit rehashed existing evidence without rerunning episodes and returned `passed=true`,
`physical_target_validated=true`, 6/6 correct routes, learned success 5/6, and Oracle success 5/6.
Language/control final, M3B test, historical fresh, `m42_final_v0`, M2, and SmolVLA remained
unaccessed. This accepts only one-scene dispatch wiring and safety; it does not authorize the
three-scene screen or reinterpret the two timeouts as success.

## D-071 — Pair every three-scene router rollout with an exact Oracle reset audit

The separately authorized M5A three-scene screen must distinguish language-routing preservation
from the stochastic-looking limitations of the already frozen PerTask controls. The stage therefore
executes Oracle immediately before NeuroSymbolic for each of 18 predeclared scene-task pairs and
captures the three object poses, two bin poses, and Panda qpos immediately after each reset. Exact
snapshot equality, controller checkpoint equality, and a fresh policy reset for all 36 episodes are
hard gate items. Oracle outcome cannot alter, skip, or replace the learned episode.

This is a project-owned orchestration/evidence interface change, not a model, dependency, policy,
prompt, task, success-geometry, or action-runtime change. The existing generic controller dispatcher
retains one optional non-policy audit sidecar so target rollouts can preserve timing, projection,
and reset proof without modifying `EndToEndEpisodeResult` or visual observations. A fixed six-case
selected-router rejection set separately proves zero lookup/reset/step work.

The stage writes a compact fingerprint-owned archive through same-filesystem staging, fsync,
completion-last checks, and atomic promotion. Its independent reader rehashes the underlying 36
generic control atoms before recomputing pairing, action metrics, failure attribution, and the exact
conjunctive gate. `passed=true` means the bounded stage and evidence are valid;
`three_scene_control_screen_passed` alone authorizes, but never starts, full control development.
All final/test/fresh schedules, shared ACT, M2 actions, and SmolVLA remain prohibited.

## D-072 — Accept the paired three-scene screen without hiding shared controller timeouts

The authorized target run used clean implementation Git
`e461c1e20f783e8c5f06e107289842665e08820b`, schedule fingerprint
`sha256:e55ad453f3a8747995594c33bac5a7ff76f6a32399a4bc50d3666ec778e1fa2f`, and development
seeds `2000000001`, `2000000002`, and `2000000003`. Its immutable run fingerprint is
`sha256:d9643f0e9a53ae70b367af87bb80ffad694cc85c582e51d447d9028e210c6480` and its raw
control-record-set fingerprint is
`sha256:14ba3ef901abc0e7a1d33d058523ff0060ce8b34ab7c96df8c03689ce29f351a`.

All 18 NeuroSymbolic routes matched their scheduled TaskSpec and frozen PerTask controller. Exact
three-cube, two-bin, and Panda-qpos reset state matched for all 18 Oracle/learned pairs. Oracle and
NeuroSymbolic each succeeded on 15/18 episodes: 15 pairs both succeeded and the same three pairs
timed out. Thus the success gap is zero and the failures remain attributed to shared frozen control,
not routing. Per-task success was 2/3 red-left, 3/3 red-right, 2/3 green-left, 2/3 green-right, 3/3
blue-left, and 3/3 blue-right on both paths.

The fixed six rejection probes each returned its safe rejection status with zero controller lookup,
policy reset, environment reset, and step. Raw evidence recorded zero wrong-object grasp/bin,
wrong-bin placement, off-table, malformed/non-finite action, arm projection, infrastructure, and M2
calls. Explicit `project` handling affected only the gripper dimension: 647 Oracle and 646
NeuroSymbolic components, with maximum correction `0.060216546058654785`. This remains projection
evidence, not hidden clipping or strict-unprojected success.

Fresh-process independent verification recomputed all gates and returned `passed=true`,
`physical_target_validated=true`, `three_scene_control_screen_passed=true`,
`selected_router_locked=true`, and `full_control_development_authorized=true`. The full six-scene
stage was not run. Language/control final, M3B test, historical fresh, `m42_final_v0`, shared ACT,
M2 actions, and SmolVLA remain unaccessed. This decision accepts the bounded paired screen and its
promotion result; it does not claim 18/18 controller success or authorize any final benchmark.

## D-073 — Keep full control development paired, bounded, and final-sealed

The separately authorized six-scene stage extends the proven three-scene orchestration rather than
introducing another controller or runner. Before model/environment construction it independently
rehashes the parent three-scene compact archive and all 36 parent atoms. The active stage then runs
Oracle immediately before NeuroSymbolic for each of 36 new development scene-task pairs, captures
the same three-cube/two-bin/Panda reset state, resets policy state for all 72 episodes, and preserves
the existing frozen PerTask H=10/`pd_joint_pos`/`project` runtime.

The exact routing gate remains 35/36, representing the previously declared conceptual 95% target at
this discrete sample size. Controller identity must agree on every correct route; it does not
silently change that routing threshold into 36/36. A fixed nine-category rejection set adds
unsupported destination, contradictory negation, and unsupported spatial reference to the earlier
screen cases, while retaining zero lookup/policy-reset/environment-reset/step as hard requirements.

The stage writes one fingerprint-owned compact through fsynced staging, completion-last validation,
and atomic promotion. A fresh-process reader rehashes the underlying 72 atoms and recomputes routing,
paired outcomes, failure attribution, raw/projected/executed action statistics, latency, and every
quality predicate. A positive `M5AFinalAuthorizationV0` is derived only after all predicates pass;
it contains sealed final-lock fingerprints but no final command or seed and never executes final.
This is an orchestration/evidence interface change only: no router, prompt, controller, dependency,
M1 success geometry, action bound, dataset, or model weight changes.

## D-074 — Accept full control development and authorize, but do not execute, sealed final

The authorized full run used clean implementation Git
`f8c583aef35e8e64d758ef5aeddf6f11ea331be9`, schedule fingerprint
`sha256:952029cbd35c4132dbfb32dd7efbf97fd783e233705437825a77c6d915756923`, raw
control fingerprint `sha256:26df7fde9a1d4f3adb8521f2c5c5cd6765643c44873de46ad1ebdc25e8ef597e`,
and immutable compact fingerprint
`sha256:1e01efc61d3dbe45b43287b534745b0f78f5acf2345597c06ed278d4fcb9fd19`.
It ran on one NVIDIA GeForce RTX 5090 with driver 595.71.05, Python 3.12.13, Torch
2.11.0+cu128/CUDA 12.8, and ManiSkill 3.0.1 after the native Vulkan/GPU/RGB gate passed.

All 36 NeuroSymbolic decisions matched the scheduled TaskSpec and frozen controller, and all 36
Oracle/learned initial states matched exactly. Oracle and NeuroSymbolic each succeeded on 25/36:
25 pairs both succeeded and 11 pairs timed out together. Thus language dispatch introduced zero
measured control-success gap. Both paths grasped the target in 35/36 and converted 25/35 grasps to
success. The nine fixed rejection probes performed zero lookup, policy reset, environment reset,
and step.

Independent recomputation found zero wrong-object/wrong-bin route or interaction, false rejection,
off-table, arm projection, malformed/non-finite action, environment/infrastructure failure, and M2
count. Explicit gripper projection remained visible at 2,925 components with maximum correction
`0.05949103832244873`; this is not reported as strict-unprojected success. The fresh verifier
returned `passed=true`, `physical_target_validated=true`, and every quality item true. It created
`M5AFinalAuthorizationV0` fingerprint
`sha256:afb4d59a3549d42ac6e0e2cb010e8261cb39a47792af583fe5dd263645dd19b5` while retaining only sealed
final lock fingerprints. Language/control final, M3B test, historical fresh, `m42_final_v0`, shared
ACT, M2 actions, and SmolVLA remained unaccessed. This decision authorizes only a later separately
requested final command; it does not execute or score final.

## D-075 — Make the separately authorized M5A final single-attempt and independently reproducible

The positive D-074 authorization is consumed only by an explicit sealed-final command. A
project-owned access ledger is opened before final texts or scene seeds are materialized and permits
exactly one attempt. A failure is terminal evidence rather than permission to retry, change a
router, or substitute a schedule. Portable dry-run validates identities and exact counts without
opening this ledger or revealing locked content.

The final language evaluation is fixed at 600 examples and exactly five frozen routers. The physical
evaluation is fixed at 12 scenes x six TaskSpecs with Oracle and NeuroSymbolic paired in reset order,
for 72 state pairs and 144 rollouts. A ten-category rejection set remains outside that rollout
budget and must perform zero lookup, policy reset, environment reset, and step. No controller,
prompt, parser, model, dataset, action runtime, M1 success rule, or dependency changes in this
decision.

The compact archive binds all semantic inputs in a path- and time-independent run fingerprint,
uses fsynced staging and same-filesystem atomic promotion, and writes `complete.json` last. A fresh
process recursively rehashes the archive, recomputes all five language-router metrics, validates all
144 raw control atoms, and recomputes pairing, routing, safety, action, attribution, and quality
gates. Pipeline validity, safety-language quality, exact taxonomy quality, control quality, and the
combined final quality gate remain separate flags. The final command never accesses M3B test,
historical fresh, or `m42_final_v0`, never calls M2, and never starts SmolVLA.

## D-076 — Invalidate the missing-input attempt and require pre-access language-source validation

The first target invocation under D-075 opened its ledger and materialized the locked language-final
split, then stopped before any model inference or robot environment construction because the
migrated server's authoritative M5A.1 artifact was at an explicitly imported root rather than the
default path. Its immutable marker records `RejectionReportError: M5A.1 analysis artifact is
missing`. No language decision, controller lookup, policy reset, environment reset, or `env.step`
occurred. The original attempt remains invalid and is never overwritten.

The final-attempt policy explicitly classifies a missing required input as an infrastructure defect
and permits one fresh-root recovery after a repair commit. The repair moves classifier and both
frozen direct-LLM evidence validation before `open_final_attempt`; it also requires an explicit
`--recovery-from-invalid-attempt` lineage whose terminal marker matches this exact error. The
recovery uses the same router, prompt, parser, models, controllers, schedules, metric gates, and
final examples. Other failures, including routing, model-quality, policy timeout, or controller
quality, cannot be relabelled as recovery authority.

## D-077 — Require an explicit final capability at the shared language evaluator boundary

The first D-076 recovery completed source validation and loaded the frozen Qwen weights, then
stopped before producing its first router record because the reused evaluation helper retained its
historical validation/development-only guard. The invalid marker records
`RouterEvaluationError: language router evaluation allows validation/development only`. No language
decision was scored and no robot environment, controller, reset, or step was reached. The partial
attempt remains immutable.

This is a pre-result orchestration-interface defect, not a model or policy failure. The minimal
repair adds a default-false `authorize_final` capability to `evaluate_language_router` and threads
it only through the sealed-final caller. Every existing validation/development caller is unchanged;
ordinary final calls still fail. A focused regression test proves both the default rejection and
the explicit sealed-final path. A fresh-root recovery must name the exact D-077 invalid marker and
keeps every frozen semantic and physical input unchanged.

## D-078 — Treat safe final false rejections as no-runtime evidence

The first complete sealed-final physical attempt produced all 600-example language results and all
144 control atoms, then failed during compact analysis with `learned audit must be one object`.
The eleven NeuroSymbolic false rejections correctly returned before controller lookup, policy reset,
environment reset, and `env.step`; their atom contract therefore contains a null active EpisodeSpec,
null control, and null paired execution audit. Requiring a physical audit for those records
contradicted the existing rejection no-dispatch boundary and was an evidence-reader defect, not a
model, routing, or controller failure. The invalid attempt and its raw evidence remain immutable.

The analyzer now accepts a null learned audit only for a non-route decision whose executable fields
are null, failure attribution is `routing_false_rejection`, and all dispatch/reset/step indicators
are false or zero. A routed record still requires a complete audit, and any rejected record that
entered the runtime is rejected structurally. A safe false rejection contributes no paired physical
state, so the unchanged 72/72 state-pairing gate fails honestly; the routing and false-rejection
quality gates remain unchanged. This repair changes only evidence interpretation and authorizes no
new model, prompt, schedule, rollout, threshold, or final attempt semantics. Recovery remains
allowlisted to this exact `SealedFinalContractError` type and message; other contract or quality
failures cannot reuse this authority.

## D-079 — Verify the authoritative control lock separately from its staged schedule

The first fresh-process verifier run rejected the completed compact archive with `materialized final
schedules differ from their locks`. The main runner correctly validates the authoritative sealed
control lock against the D-074 authorization and then derives a staged 72-episode schedule whose
fingerprint intentionally binds additional stage material. The verifier incorrectly compared that
derived fingerprint directly with the parent lock fingerprint, so two valid but distinct identities
could never pass the same equality check.

The verifier now compares the authoritative control lock fingerprint with the authorized lock and
continues to compare the complete staged schedule separately with `control_schedule.json`. This is
a read-only verifier correction: it does not change either schedule, any final result, access
ledger, router, model, controller, threshold, or raw episode. The failed verifier report remains
preserved and is not final experimental evidence.

## D-080 — Accept the sealed-final pipeline, reject its quality, and keep SmolVLA sealed

The valid sealed-final recovery used clean implementation Git
`0c5bb7de045e936cd6a3ce850d620bba1d923a3d` and final run fingerprint
`sha256:d4d176d43cd9b6602f0a0e6c352e95a950a2fdbe96913ea0b12401f28712d2e5`. It consumed the exact
D-074 authorization, evaluated five frozen routers on all 600 final-language examples, executed 72
Oracle plus 72 NeuroSymbolic control atoms over 12 unseen scenes, and ran the immutable ten-case
rejection set. The artifact fingerprint is
`sha256:2bfeaefdfb9a4ccb6a6e9a34a57ec053b06c1cb9cfc5ed0e98f5ab67863157c1`.

Fresh-process verifier Git `b2489c73a10546ed1d4a2956649096d6255cd980` recomputed every language
metric, all 144 raw control atoms, 72 pair identities, action statistics, rejection probes, and
quality predicates. Its verification fingerprint is
`sha256:01c7add759f3e6691b5c8aa011c23db56389bb8ee10a8f964edb9aedacc52c08` and it returned
`passed=true`, `final_pipeline_validated=true`, and `physical_target_validated=true`. The earlier
failed verifier report from D-079 remains immutable and is not used as final evidence.

Quality did not pass. NeuroSymbolic achieved 78.33% routeable TaskSpec/object/bin accuracy, zero
unsafe false routes, 21.67% false rejection, and 100% schema validity/repeatability. Its exact
four-way status, rejection-status, and rejection-reason accuracies were 74.17%, 67.92%, and 50.83%.
Oracle succeeded on 55/72 control episodes; NeuroSymbolic succeeded on 46/72 after 61 correct routes,
11 safe false rejections, and 15 routed timeouts. All 61 dispatched correct routes selected the
matching frozen controller. There were zero wrong-object/wrong-bin routes or interactions,
wrong-bin placements, off-table outcomes, arm projections, malformed/non-finite actions,
infrastructure failures, and M2 calls. Explicit projection affected only the gripper dimension:
2,408 Oracle and 2,071 NeuroSymbolic components, with maximum correction
`0.05956220626831055`. Every rejection probe performed zero controller lookup, policy reset,
environment reset, and step.

Three infrastructure attempts remain preserved rather than rewritten: the D-076 missing imported
classifier artifact, the D-077 final-capability guard, and the D-078 no-runtime-audit reader defect.
The pre-access D-078 allowlist launcher failure opened no new final ledger. None changed a model,
prompt, parser, threshold, corpus, schedule, checkpoint, controller, action runtime, or M1 success
rule. The completed final result is therefore terminal experimental evidence, not retry authority:
`final_language_safety_quality_passed=false`, `final_rejection_taxonomy_quality_passed=false`,
`final_control_quality_passed=false`, `final_quality_gate_passed=false`, and `smolvla_go=false`.
M3B test, historical fresh, and `m42_final_v0` remained unaccessed. M5A is closed without starting
SmolVLA; any later milestone requires a new explicit decision and authorization.

## D-081 — Add a zero-action, initial-state-only LatentGuard bridge

M5B adds an optional canonical-JSON boundary for the bounded LatentGuard M6B integration smoke.
The accepted PerTask controller is selected by lexically sorting eligible canonical task IDs; no
rollout result participates. One bridge probe may reset the selected task once, reconstruct its
policy observation, query its accepted ACT checkpoint once, and project the resulting actions. A
counting proxy rejects any `env.step` call.

Projection remains owned by LangMani's existing `BoundedActionEnvPostprocessorV0`. The content-bound
action-space contract can therefore project later candidate JSON without opening a second simulator
or reset. Raw policy actions, raw transformed actions, projected executable actions, and actually
executed actions retain distinct identities. M5B produces no actually executed candidate action.

The future fixed continuation replays a previously recorded projected nominal trajectory after the
ten-action candidate prefix and never requeries the policy after divergence. It is reproducible but
does not provide an intermediate policy snapshot, intermediate exact replay, same-policy
continuation, or closed-loop shielding. The bridge authorizes no training, rollout, task outcome,
performance claim, VLM/LLM, or multi-GPU work.

## D-082 — Freeze v1 as a release-and-portfolio milestone

M6 freezes LangMani v1 at the existing M0–M5A results plus the structural M5B bridge. It adds a
source-controlled results index, report/demo sources, a release-oriented README, an architecture
summary, resume/interview material, and four evidence-bound cases. Generated PDF/video artifacts
remain under `outputs/` and are not committed.

This decision deliberately separates release completeness from model quality. The sealed-final
pipeline and physical verification remain true; the language, control, combined quality gate, and
SmolVLA authorization remain false. M6 cannot retrain, re-evaluate, change thresholds, rewrite
evidence, complete the blocked M5B target probe, or begin a new model. The annotated `v1.0.0` tag
will identify the validated release source and portfolio materials. Future behavioral work requires
a new version or explicit milestone.

## D-083 - Start LangMani 2.0 with a policy-neutral evaluation boundary

LangMani 2.0 Phase 1 preserves the `v1.0.0` source and experiment evidence while adding a new,
project-owned evaluation layer. The six color/bin combinations are one `pick_and_place` skill
family with six task instances. This prevents task parameters from being misreported as six
different robotic skills.

The first adapter wraps the existing selected red-to-left PerTask ACT checkpoint through the
already validated LeRobot preprocessor, policy, postprocessor, and explicit project-mode action
bound processor. The adapter exposes only the generic `ObservationBatch` to `ActionChunk` contract;
the unified evaluator has no language-router dependency. Runtime configuration paths are relative
and content-bound. Absolute paths found in a historical training manifest remain provenance only.

Phase 1 registers no placeholder future models and performs no training. Its three-seed native
smoke is a new v2 runtime check, not a reproduction of historical v1 metrics. Source validation on
a machine without a visible NVIDIA/Vulkan target is explicitly structural and cannot set a
physical-validation claim.

The 2026-07-21 bounded fallback audit did not relax that boundary. Docker Desktop exposed a real
RTX 4090 to PyTorch CUDA but only the llvmpipe CPU Vulkan implementation, and SAPIEN failed before
environment construction. Native Windows loaded the pinned CUDA and Python packages, but SAPIEN
raised access violation `0xC0000005` at the first camera observation after the Unicode asset path
was eliminated. Neither attempt loaded the ACT policy, called `env.step`, or completed an episode.
They are retained only as untracked invalid-infrastructure evidence; the native Linux
NVIDIA/Vulkan three-seed smoke remains the sole runtime acceptance path.

## D-084 - Gate Phase 2 on independently verified real Phase 1 rollouts

The requested LangMani 2.0 Phase 2 explicitly requires an existing real Phase 1 smoke rollout and
instructs the implementation to stop at the minimum recovery when a major Phase 1 acceptance item
is absent. That item is absent: the bounded Docker and Windows attempts both stopped before policy
inference or an environment step, while the available Linux artifact host had no GPU.

Phase 2 therefore adds only `environment/verify_v2_phase1.py` and its project-owned validation
module before stopping. The gate revalidates the frozen v1 release, canonical ACT artifacts, exact
four-file evaluation contract, at least three deterministic seeds, real policy queries, and real
environment steps. It fails closed and emits `phase2_authorized=false` for missing or incompatible
evidence. This is a small recovery of the Phase 1 acceptance boundary, not a redesign or a second
evaluation framework.

No push environment, expert, demonstration archive, multi-skill dataset, SmolVLA dependency,
training, or evaluation is authorized until this gate passes on native Linux NVIDIA/Vulkan
hardware. Historical v1 metrics and invalid infrastructure attempts cannot satisfy it.

## D-085 - Accept the native Phase 1 ACT smoke and open the Phase 2 entry gate

On 2026-07-21 the native target exposed one RTX 5090 with 32 GB VRAM. The accepted runtime used
Python 3.12.13, PyTorch 2.11.0+cu128, ManiSkill 3.0.1, SAPIEN 3.0.3, LeRobot 0.6.0, NVIDIA driver
595.71.05, and the installed EGL NVIDIA Vulkan ICD. The strict installation target passed CUDA,
Vulkan, explicit PhysX GPU simulation, RGB observation/rendering, and one environment step.

An initial GLX launch was invalid because its nested shell lost the EGL ICD environment. The first
correct EGL launch then found a Phase 1 adapter compatibility defect: the historical
`model_config` is a recursively frozen mapping, but the adapter attempted attribute access. Commit
`aa599dd` repaired only that mapping access and added a realistic regression assertion. Neither
invalid attempt reached a policy query, and neither is policy-quality evidence.

The repaired, unchanged three-seed evaluation completed seeds 41001/41002/41003 with 3/3
successes, 40 real ACT queries, 392 real environment steps, zero timeouts, zero invalid actions,
and zero policy/environment failures. The independent verifier rehashed the canonical checkpoint,
recomputed the four-file evidence contract, and returned `phase2_authorized=true`. Phase 2 may now
start with the standard pushing task; this decision does not pre-authorize its expert, data, or
SmolVLA stages.

## D-086 - Define pushing as one parameterized skill family

Phase 2 adds `push_to_region` as one manipulation skill family rather than treating geometry,
target location, or difficulty as separate skills. The first immutable task product is two contact
geometries (`blue_cube`, `orange_cylinder`) by four named target regions by two explicit difficulty
levels, for 16 task instances. A scene seed remains separate from this semantic task identity.

Success is conservative: the complete planar object footprint must fit in the target region, the
object must remain planar and static, an overshoot cannot be accepted, and the condition must hold
for consecutive simulator steps. Workspace exit, lift, topple, grasp, invalid action,
wrong-object contact/displacement, projection, collision, and no-progress stall remain distinct
events so later expert and learned-policy reports do not infer every failure from one final bit.
The pure batched Torch contract is accepted before simulator integration; it is not physical task
evidence.

## D-087 - Use geometry-bounded planar contact for the pushing expert

The Phase 2 privileged expert uses the installed ManiSkill 3.0.1 Panda/mplib integration through
the existing project-owned planner adapter. It closes the gripper before contact, approaches from
behind the selected object, keeps the TCP planar, performs one primary push and at most two bounded
state-dependent corrections, and aborts on lift, grasp, topple, workspace exit, invalid action, or
wrong-object displacement. The expert receives semantic actor handles and the exact
full-containment object-center radius only through its explicit expert context; no new privileged
field is exposed to visual policy observations.

The first real 16-task smoke was valid infrastructure evidence but rejected expert-quality
evidence: direct motion to an 8 cm precontact point toppled every cylinder and displaced two hard
distractors, while two long center-seeking pushes timed out. Bounded single-episode probes showed
that a 12 cm precontact height avoids the target approach collision, while geometry-specific
2.5 cm cube and 1.5 cm cylinder TCP heights prevent grasp and toppling respectively. The expert
therefore uses those fixed geometry-level values and stops at a fixed 3.5 cm margin inside the
exact full-containment center boundary rather than pushing every object to the visual marker
center. Hard distractors remain 8.5 cm lateral to the push path but are placed at 75% of its length
so they do not overlap the Panda's reset-time arm sweep. This changes neither task success geometry nor the 250-step
episode limit. The same immutable all-task smoke and then the fixed 50-standard/30-hard schedule
must pass before Phase 2 data collection may start.

The installed target-renderer helper follows SAPIEN's cylinder convention: its symmetry axis is
local `x`. The orange cylinder intentionally starts with that axis horizontal, so rolling about
local `x` is valid planar motion; cylinder stability therefore measures whether local `x` remains
horizontal instead of incorrectly treating local `z` as an upright axis. The red/white target
helper is rotated 90 degrees about world `y`, matching the installed ManiSkill example so its disk
lies on the table. Native reachability probes also showed that the original forward-region
`x=0.28` extended the Panda TCP to its screw-planner singular boundary. Moving only the two
forward centers to `x=0.22` keeps all four semantic regions distinct and reachable. These are
geometry/orientation corrections: the object sizes, containment radii, success predicates,
planner bounds, and 250-step limit remain unchanged.

## D-088 - Stop Phase 2 before data after the fixed pushing-expert gate

The corrected native all-task smoke at Git `500b09ce9d1cf10f7ac2f6f585a5fb8efed9e686`
completed all 16 episodes with 8/8 standard and 5/8 hard successes. The subsequent predeclared
50-standard/30-hard validation completed without command errors and produced 41/50 standard
successes (82%) and 21/30 hard successes (70%). Hard met its target exactly, but the conjunctive
expert gate failed because standard remained below 90%. This is valid controller-quality evidence,
not an infrastructure failure.

One invalid launcher artifact used the main NumPy 2.2.6 runtime and failed all episodes at planner
initialization before any environment step; the accepted runs use the pinned NumPy 1.26.4/mplib
side runtime. Two later bounded diagnostic probes each reached 10/17 and caused new regressions, so
their candidate changes were rejected. In accordance with the declared stop condition, no pushing
demonstration archive, unified dataset, SmolVLA adapter, model load, training, or learned-policy
evaluation is authorized from this result. Existing Phase 1 and v1 evidence remain unchanged.

## D-089 - Diagnose side pushes at phase boundaries without changing the policy path

Phase 2A begins with a behavior-neutral diagnostic layer because the rejected 41/50 standard
expert report identifies timeouts and verification failures but does not preserve the object and
TCP geometry that preceded those terminal labels. `PushToRegionExpert.diagnostic_trace` therefore
records one immutable privileged snapshot after each existing expert phase. The snapshots include
semantic object poses, the target center, chosen precontact/contact points, intended push
direction, progress and lateral error, a TCP-distance contact proxy, workspace margin, and stable
success state. They are expert-only metadata: no field enters a policy observation, per-step info,
or success calculation, and the expert issues exactly the same actions as before.

`environment/diagnose_push_expert.py` reads the immutable fixed 50/30 report and replays only its
failed `left`/`right` episodes in fresh environments. It produces the requested phase-segment
table and assigns one deterministic preceding root cause rather than treating `timeout` as an
explanation. The contact point is explicitly a TCP-center phase-boundary proxy because the
installed public environment contract does not expose a stable contact-manifold point. This
instrumentation may remain even if all subsequent controller candidates are rejected.

## D-090 - Test one signed 15-degree side-contact compensation candidate

The Phase 2A replay at Git `fab982a` reproduced all 16 previously failed lateral episodes in the
pinned NumPy 1.26.4/mplib 0.1.1 target runtime. Seven failures lost the TCP-distance contact proxy,
two were stable containment boundary cases, two exited the workspace during a correction, and the
remaining five separated into overshoot, primary planning, initial approach, wrong-object sweep,
and joint-action-bound origins. The primary-push traces expose a systematic signed bias: failed
left pushes drift roughly 14--18 degrees toward world `+x`, while failed right pushes drift roughly
14--19 degrees toward the same world `+x` side.

Candidate A therefore changes only `left`/`right` tasks. It rotates the preferred contact normal
15 degrees away from the observed drift and evaluates a fixed four-candidate sequence (preferred,
half, ideal, opposite half). Candidates are ranked by desired-direction alignment, a documented
Panda-base reachability proxy, workspace margin, and distractor swept-segment clearance; unsafe
candidates are not planned, and planning may fall back only to the remaining fixed safe list. The
horizontal cylinder uses a 4 cm lateral tangent-contact offset while forward tasks retain the
original 4.5 cm formula. Stable settling increases from 12 to 16 holds so the two already-contained
static cases can reach the unchanged five-step success predicate. Target positions, region sizes,
success/failure predicates, 250-step limit, forward contact strategy, and low-level action handling
are unchanged. Candidate A remains provisional until the fixed staged experiments accept it.

## D-091 - Reject the shared-geometry candidate and isolate cube compensation

Candidate A recovered eight of the sixteen fixed failed lateral episodes, but its complete
42-episode lateral subset reached only 20/26 standard and 11/16 hard. More importantly, all six
new workspace exits involved `orange_cylinder`: two exited during the primary push and four exited
during a corrective push. The shared 15-degree compensation plus the deeper 4 cm cylinder contact
therefore introduced a systemic geometry-specific regression. Candidate A is rejected before the
16-task smoke and the fixed 50/30 gate.

Candidate B is the one bounded evidence-driven separation of Candidate A. Blue-cube lateral tasks
retain the measured 15-degree compensation, while orange-cylinder lateral tasks restore the
baseline ideal direction and original 4.5 cm contact offset. A zero-degree cylinder strategy emits
only the ideal candidate rather than duplicate planner attempts. The independently motivated
16-step settling change remains because the baseline replay contained two already-contained
static-boundary failures. No target geometry, success predicate, episode limit, action handling,
schedule, or forward strategy changes. Candidate B remains provisional until the unchanged staged
gates accept it.

## D-092 - Bound only lateral-cylinder correction travel before replanning

Candidate B improved the complete fixed lateral subset from 17/26 to 22/26 standard and from 9/16
to 10/16 hard, but it produced three standard workspace-exit terminal statuses where the baseline
reported none. Two correspond to workspace-margin root causes already exposed by the neutral
replay; seed 44044 is a new regression. Candidate B is therefore rejected before smoke under the
predeclared no-increase rule.

The failed-phase traces show that all three exits occur during cylinder correction after a useful
primary push. The object then travels roughly 35--40 cm during a correction even though only
3.5--5 cm of additional center progress is required. Candidate C changes no primary or cube path.
For a lateral cylinder correction only, it commands at most 3 cm from the current contact pose,
uses the measured remaining distance rather than replaying the original endpoint, and further caps
that distance by the object's footprint plus a fixed 1 cm workspace margin. The existing two-push
budget can therefore supply at most 6 cm, covering the observed remaining range without one long
correction. Low-level actions are still rejected rather than clipped. Candidate C remains
provisional until the same bounded replay and complete lateral subset show zero new workspace or
action-bound failures.

## D-093 - Disable the unsafe lateral-cylinder re-contact path

Candidate C reproduced all ten Candidate-B failures with identical terminal statuses and, for the
three workspace exits, identical environment-step counts. The correction endpoint cap was never
reached: the cylinder had already been swept during the high-to-contact re-contact motion.
Candidate C is rejected and its unused distance/margin controls are removed.

Candidate D preserves the primary lateral-cylinder push but makes its later correction phases
explicit no-motion completions. This is a fail-safe response to observed unsafe re-contact, not a
success relaxation: an incomplete primary push still fails the unchanged stable containment
verification. Cube lateral corrections, all forward behavior, environment success/failure
predicates, planner bounds, and action validation remain unchanged. Candidate D must restore the
workspace-exit count before it can proceed beyond the fixed lateral subset.

## D-094 - Reject unsafe plans before execution and use one bounded approach fallback

Candidate D completed the fixed lateral subset at 22/26 standard and 10/16 hard with zero
workspace exits, but the unchanged 16-task smoke remained 8/8 standard and 5/8 hard. Seed 45005
failed while moving to the ideal cylinder precontact because an mplib joint position exceeded the
environment controller bounds at step 51. The failure occurred before object contact, yet the
expert could only discover it by stepping the invalid action and therefore could not try another
already-scored approach.

Candidate E adds an explicit expert-only action-bound accessor and validates every planned joint
position before the first environment step. An out-of-bounds plan remains a classified failure and
is never clipped, projected, or executed. Only for this zero-step precheck failure may the existing
lateral approach loop try the next fixed safe normal. A zero-compensation cylinder now has the
bounded symmetric set 0, plus or minus 7.5, and plus or minus 15 degrees; all candidates retain the
same reachability, workspace, and distractor scoring. This is a general origin-layer action-bound
fix, not a seed exception. Wrong-object and forward paths are otherwise unchanged.

## D-095 - Accept Candidate E and open the pushing-data entry gate

Candidate E at Git `59ca88e9f0514187252a6286ab1b8e06c4318fb4` completed every predeclared
Phase 2A stage in the pinned NumPy 1.26.4/mplib 0.1.1 planner environment on one RTX 5090. The full
fixed lateral subset improved from 17/26 to 22/26 standard and from 9/16 to 11/16 hard. The exact
16-task smoke then reached 8/8 standard and 6/8 hard, satisfying its unchanged joint gate. The
fixed target schedule completed all 80 episodes without command errors at 46/50 standard and 23/30
hard, exceeding the 45/50 and 21/30 thresholds. Standard forward remained 24/24.

The target report contains zero workspace exits, zero action-bound events, zero timeouts, and zero
planner failures. Its remaining failures are four standard cylinder verification failures, three
hard cylinder verification failures, and four hard wrong-object precontact interactions. Independent
recomputation matched every scheduled episode identity in order, found no duplicate identity, and
reproduced the aggregate counts from raw episode records. The accepted report SHA-256 is
`2dd043220a6736bcc8df2f085a6b5322bee76bc96a79b209465cd63d7ecc286c`.

The unchanged expert quality gate therefore passes. Phase 2 may advance to a separately authorized
deterministic demonstration-collection stage under the frozen task, seed, action, and success
contracts. This decision does not claim that a pushing dataset exists, does not authorize bypassing
raw/archive validation, and does not start or validate SmolVLA. SmolVLA remains downstream of an
accepted dataset and a later explicit stage.

## D-096 - Accept the Phase 2B pushing dataset and authorize a separate Phase 2C

Phase 2B ran the frozen Candidate E expert and deterministic schedule on one RTX 5090. The full and
bounded top-up stages attempted 516 episodes, admitted 397 generation successes, and preserved 119
rejected attempts. A separate fresh-environment process action-replayed all 397 generation-accepted
trajectories: categorical outcomes, every transition label, and frame counts matched 397/397; final
position error was zero and maximum quaternion angular error was approximately 5.96e-8 radians. No
replay failure entered the accepted archive.

The derived LeRobot 0.6 dataset contains 397 episodes and 53,297 frames. Its six stable splits have
zero episode-ID, seed, initial-state, trajectory, scene, held-out-language, source-shard, or exported
file overlap. A real readback confirmed 256x256 RGB, 9D Panda state, 8D `pd_joint_pos` actions,
20 Hz control, task strings, and no privileged state. The compatibility audit left the historical
360-episode/64,548-frame pick-and-place root unchanged and created only a content-bound immutable
multi-root index. The push export and multi-root index are therefore accepted as separate immutable
sources rather than a rewritten combined dataset.

The first superseded pilot and an interrupted detached replay remain invalid infrastructure
evidence. The first top-up launcher also failed before collection because it compared the top-up
runtime against the full-stage immutable runtime manifest. Commit
`f081275478a72aa90b7b6529874c974a02e695dc` fixed the contract by assigning full and top-up separate
stage-owned runtime manifests; it did not rename a run, rewrite a manifest, or retrain the expert.
The completed full/top-up collection, replay, export, failure corpus, and source-validation records
remain immutable.

The independent verifier at Git `45009f6b13677413a0262f9521f0e9dcbd74ed5f` passed every collection,
replay, schema, quota, readback, runtime-lineage, leakage, compatibility, source-test, and tracked
metadata gate. Its report SHA-256 is
`b07f97c8ade07e23eac8b710dfee801ebd8f8ef5df44e92c967958122b0310ee` and it records
`pipeline_integrity_validated=true`, `full_collection_validated=true`,
`physical_target_validated=true`, and `smolvla_phase2c_authorized=true`.

This accepts Phase 2B data production and authorizes only a separately invoked Phase 2C consumer.
It does not claim learned-policy quality and does not install, load, train, or evaluate SmolVLA.
Phase 2C must bind the immutable source identities and begin only under a new explicit command.

## D-097 - Bind official SmolVLA without forking its model or projecting learned actions

Phase 2C resolves `lerobot/smolvla_base` to Hugging Face revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205` under installed LeRobot 0.6.0. The published config
uses three cameras and 6D state/action, whereas accepted Phase 2B uses one `base_camera`, 9D
`PandaPolicyStateV0`, and 8D `PandaJointPositionActionV0`. The official implementation pads state
and action to fixed 32D internal maxima, so Phase 2C replaces only the public feature descriptors
before loading the unchanged pretrained tensor file. This is an explicit embodiment adaptation,
not a source fork or randomly initialized substitute.

The generic adapter calls official `predict_action_chunk` and exposes a true-prefix LangMani
`ActionChunk`. Model chunk length remains 50 while validation chooses only among execution horizons
1, 4, and 8. Policy, preprocessor, and postprocessor reset at every episode. Flow-matching noise is
seeded by episode query ordinal, and inference latency, queries, emitted actions, queue resets,
underruns, and invalid outputs are recorded.

Unlike the historical ACT path, learned SmolVLA evaluation fixes the environment action-bound
processor to `reject`. A learned action outside the active bounds is a classified failed episode;
it is never clipped or projected. The only frozen visual-shift domain is a deterministic RGB
photometric transform fixed before any learned outcome. No privileged state enters either mapping.

The currently reachable RTX 5090 host lacks the accepted Phase 2B external roots, while the former
LangMani host is offline. This is an external-artifact availability blocker, not a Phase 2B
verification failure. No optimizer step or simulator competence claim is permitted until the
accepted roots are accessible and the independent verifier passes again.

## D-098 - Replace SmolVLA feature dictionaries completely and launch from audited local bytes

The first target construction probe showed that LeRobot 0.6 recursively merges dictionary CLI
overrides into the published SmolVLA config. Supplying the LangMani camera dictionary therefore
retained `camera1`, `camera2`, and `camera3` and added `base_camera`; that four-camera result is
invalid for Phase 2C and was not accepted as a smoke pass.

The official documented replacement path is now used instead: training supplies
`input_features=null`, after which the official factory derives the complete input and output maps
from the already verified train-only LeRobot metadata. A separate strict construction gate assigns
and asserts exactly `base_camera + 9D state -> 8D action` before loading the unchanged official
weights. The accepted construction loaded 450,046,176 parameters and 99,880,992 trainable
parameters, with no missing or unexpected tensor under strict loading.

The probe also identified three launch-contract defects before any optimizer work: the SmolVLA
extra (including `num2words`) was not installed, optimizer betas were split across two CLI
arguments, and the wrapper pre-created the same output directory that LeRobot refuses to
overwrite. The dependency is now explicit, betas are one JSON argument, and official outputs live
under a fresh `<run>/lerobot/` child. LeRobot 0.6 also resolves Hub config before applying its
revision field and does not expose a SmolVLA policy `dtype` argument, so training launches from the
audited local official snapshot and uses explicit BF16 Accelerate mixed precision instead of an
invented policy flag.

On the exact submitted source, the unmodified official CLI accepted every frozen argument and
reached dataset creation. The deliberate empty-root probe stopped at missing `meta/info.json` and
executed zero optimizer steps. This validates base loading and launch mechanics only. The accepted
Phase 2B roots remain unavailable, so it does not satisfy the real-batch smoke, training,
checkpoint, or closed-loop gates.

## D-099 - Recover Phase 2B by content identity or stop before optimization

Phase 2C.1 treats the lost Phase 2B export as an external immutable asset, not as a reproducible
build target. The accepted evidence retains exact sidecar hashes, split counts, aggregate byte
count, feature/timing contracts, replay evidence, and an independent verifier hash, but it does not
retain a complete per-file digest inventory or the six `meta/info.json` byte hashes. Those missing
identities must not be guessed from a regenerated export.

An `AcceptedDatasetPackage` may therefore be created only after a located candidate re-passes the
retained sidecar identities, all six metadata/count/schema/task gates, complete real LeRobot
readback, and the independent Phase 2B verifier. Package creation additionally computes a complete
path-independent file-tree digest. Recovery verifies the package at its source, copies through an
unaddressed owned staging directory, verifies the staging bytes, atomically activates the expected
root name, and verifies the destination again. A copy interruption cannot materialize the final
dataset path.

If no candidate reaches `EXACT_ACCEPTED_DATASET`, every training gate remains closed and optimizer
steps remain zero. Similar, partial, regenerated, or metadata-repaired data cannot substitute for
the accepted Phase 2B bytes. The honest terminal state is Phase 2C.1 Result B; it does not downgrade
the historical Phase 2B acceptance, claim a learned policy failure, or authorize Phase 2D.

## D-100 - Close Phase 2C.1 as Result B after exhaustive asset recovery

A read-only search covered the current repository outputs, accessible local storage, Git object and
LFS history, GitHub releases, the reachable AutoDL workspace/run roots, its persistent-data mount,
temporary storage, and Hugging Face caches. The old historical endpoint was unreachable. No exact,
potentially matching, or incomplete accepted Phase 2B export was found; the four explicit roots on
the reachable server were all absent.

Because the accepted evidence lacks a complete original file inventory, combined tree digest, and
the six `meta/info.json` byte hashes, no regenerated export can establish byte identity. No package
was created, no copy was attempted, no metadata was repaired, and no real batch was loaded.
Optimizer steps remain zero. Training, checkpoint selection, development evaluation, sealed
evaluation, promotion, and Phase 2D are not run. Phase 2C.1 therefore ends as Result B without
changing the historical acceptance of Phase 2B.

## D-101 - Reconstitute pushing data under a new v2 identity

The missing Phase 2B v1 bytes remain a historical asset loss. Their accepted conclusions are not
withdrawn, but their dataset ID, digests, manifests, counts, and acceptance status cannot authorize
a new runtime dataset. Phase 2B.2 therefore performs a new simulator collection under the distinct
identity `langmani/phase2b-push-v2` with disjoint seeds and newly computed content identities.

The frozen gate first requires at least 95 successes in 100 independently executed expert episodes,
with zero simulator errors, non-finite actions, native action-bound violations, or workspace exits.
Formal collection cannot start if that gate fails. Completed native episodes use an atomic
per-episode commit and verified resume; failed attempts remain evidence but do not enter the default
imitation-learning view. Every generation-accepted episode must independently action-replay in a
fresh environment before LeRobot export.

Acceptance requires at least 400 replay-accepted episodes and 50,000 frames, complete six-split
LeRobot 0.6 readback, zero finite/action/leakage failures, complete raw and export manifests,
content-addressed raw and LeRobot archives, successful restored-copy validation, and one real
SmolVLA batch processed with zero optimizer steps. Two independently restored replicas plus an
explicit pending third copy are a temporary dataset gate; a true third copy is mandatory before any
future 20,000-step training. This milestone cannot start Phase 2C.2 or Phase 2D.

## D-102 - Reject Candidate E at the stricter v2 gate and bound Candidate F

The first Phase 2B.2 gate ran the exact 100-episode schedule at Git
`cd97c4f7cb48be5d45569a5729926fe7ddaa3075` on the pinned NumPy 1.26.4, mplib 0.1.1,
ManiSkill 3.0.1, and LeRobot 0.6.0 runtime. It completed all episodes at 75/100 and is rejected.
The report records seven planning failures, seven timeouts, seven verification failures, and four
wrong-object interactions, with no simulator, finite-action, action-bound, or workspace failure.
Its SHA-256 is `a6a8296a9df7c716698920d782c8e6f0aead6f61f18c228dcadba0a13a1c9489`.

Formal data generation remains prohibited. Candidate F is the bounded response to all four
observed path-level failure families: every precontact uses a fixed high staging pose and the
existing deterministic obstacle-aware candidate set; only intermediate actions on collision-free
free-space plans are sampled at stride two while their exact final joint target and every native
action check remain; the primary push advances in at most twelve 4 cm state-aware segments and may
halve a zero-step failed segment only down to 1 cm. It neither teleports nor mutates object state,
does not change task evaluation, does not extend the 250-step limit, and does not clip or repair an
action. Candidate F is provisional until local validation passes and an untouched 100-episode gate
at a new seed range reaches the unchanged 95% threshold.

Candidate F passed local implementation validation at Git
`6277d77227a51d39d675e328c4df7c233f302484`. The Phase 2B.2 contract therefore binds that exact
commit and freezes seeds 66100 through 66199 for its untouched 100-episode promotion gate. This
does not relabel the v1 Candidate E contract or any historical v1 trajectory. Candidate F remains
unaccepted for formal collection unless the new remote gate reaches at least 95/100 with every
zero-tolerance runtime and action check still at zero.

## D-103 - Reject Candidate F before Phase 2B.2 collection

Candidate F completed its untouched 100-episode gate on seeds 66100 through 66199 from clean Git
`f441b1b035007f0876dafd2a54886b26f6350e40`. It achieved 11/100 and is rejected. The immutable
full report SHA-256 is `94975a838063c4e20891cf478bf775127eaf6a04fbcfcd990caef415fad3087d`.
There were 74 timeouts, ten wrong-object interactions, four planning failures, and one contact
failure. Eighty failures occurred during `primary_push`; all 74 timeouts exhausted the unchanged
250-step limit. The diagnosis is bounded-segment execution cost, not permission to increase the
horizon or relax acceptance. Simulator errors, non-finite actions, action-bound violations, and
workspace violations remained zero. Formal collection, dataset export, and training remain closed.

## D-104 - Freeze Candidate G with a diagnostic probe separated from promotion

Candidate G passed local implementation validation at Git
`24709f1702914e72131245484d4405ac68a6303c`. It executes one direct content-bound primary push by
default and enters the bounded segmented fallback only after a zero-action planning or IK failure.
It also makes full fallback step accounting explicit, raises only free-space staging, restores
controller-executed lateral-cylinder correction, and binds a cylinder-specific containment margin.
It does not extend the 250-step horizon, alter the task predicate, clip or repair actions, or relax
any zero-tolerance gate.

Seeds 66200 through 66231 are reserved for a diagnostic-only 20-standard/12-hard probe. The probe
is marked `formal_gate_eligible=false`, can never set `passed=true`, and exists only to avoid wasting
another full formal gate on an obviously weak path. The untouched Candidate G promotion gate is
frozen at seeds 66300 through 66399 and still requires at least 95/100 with zero simulator errors,
non-finite actions, action-bound violations, and workspace violations. Formal collection remains
closed until that exact gate passes.

## D-105 - Reject Candidate G diagnostically and preserve the formal seed lock

Candidate G completed its 32-episode diagnostic-only probe on seeds 66200 through 66231 from clean
Git `3db2c057a4b9be76a76e3ab3eb9f6874a3d1eb6a`. It achieved 22/32 and is rejected without running
the formal gate. The full report SHA-256 is
`e049ec4b5e0a4ab0a6a187fb938fa402759a0a61e79b8a0f3fd54fafa15a6ba9`. There were four planning
failures, two timeouts, two wrong-object interactions, one correction failure, and one target
workspace exit. Six failures occurred in `move_to_precontact`; four occurred during correction.
The report is bound to `formal_gate_eligible=false`, optimizer steps remained zero, and no
collection began.

Seeds 66300 through 66399 remain untouched. Candidate H addresses only the observed mechanics: it
separates the initial lift from candidate-specific staging and descent, permits a new predeclared
safe candidate after a zero-action planning or bounds precheck failure, restores full-resolution
approach execution, and stops pushing immediately when the unchanged full-containment predicate is
observed so the existing settle phase can establish stable success. It does not alter the 95% gate,
the 250-step horizon, the environment predicate, action bounds, or any zero-tolerance check.

## D-106 - Freeze Candidate H before a larger diagnostic probe

Candidate H passed full local implementation validation at Git
`62db322a6954a362078badcb5f298501cb533bcf`. Its 64-episode diagnostic-only range is 66400 through
66463; these seeds are disjoint from the untouched 66300--66399 formal gate and the collection
ranges. Diagnostic success cannot promote the expert, and a diagnostic failure cancels the formal
gate rather than consuming it. The v2 contract binds `PushToRegionExpert/CandidateH` to that exact
implementation commit. Formal collection remains closed.

## D-107 - Reject Candidate H and bound Candidate I by phase cost

Candidate H completed all 64 diagnostic episodes on seeds 66400 through 66463 from clean Git
`48ab184cb172a5d4533dcff8e2effb16006c2dbd`. It achieved 38/64 and is rejected without touching
the formal gate. The full report SHA-256 is
`a896a8e5645b7e9c63237c94f9cc56359d2b5db9e703eed0cb815c2d3eadbc4c`. Fifteen timeouts dominated
the 26 failures, and successful precontact execution averaged 104.9 of the fixed 250 steps.
Simulator errors, non-finite actions, and action-bound violations remained zero; one target workspace
exit was preserved. No collection or optimization started.

Candidate I keeps the proven candidate-retry structure, restores the frozen free-space stride only
for the initial lift and final descent, and retains full resolution for the high staging sweep where
wrong-object risk was observed. First containment now triggers a bounded controller hold using the
existing settle count, allowing the unchanged stable-success predicate to complete before further
motion. Diagnostic runs may stop early only when a zero-tolerance failure already makes promotion
invalid or the exact arithmetic success ceiling falls below 95%; formal evaluation remains complete
and unchanged.

## D-108 - Freeze Candidate I with cost-bounded diagnostic stopping

Candidate I passed full local validation at Git
`3da6ecf5afccc09eae5fe69ece9170647a3357a0`. Seeds 66500 through 66563 are reserved for its
diagnostic-only probe. The probe may stop and emit a partial failure report only after a
zero-tolerance failure occurs or its exact best-case success count can no longer meet 95%. This
rule never applies to the formal 66300--66399 gate, which remains untouched and complete by
contract. The v2 configuration now binds `PushToRegionExpert/CandidateI` to the implementation
commit. Formal collection remains closed.

## D-109 - Reject Candidate I and bound Candidate J by four observed failures

Candidate I ran from clean Git `c38fcef94504b390f961556221b33ce4ab349f2d` on the disjoint
diagnostic range beginning at seed 66500. After 16 episodes it had 12 successes and four failures,
so even 48 further successes could reach only 60/64, below the unchanged 95% gate. The predeclared
early-stop rule wrote the partial report and stopped. The four failures were two timeouts, one final
verification failure, and one zero-action initial-lift planning failure; all simulator, finite,
action-bound, and workspace checks stayed clean. The report SHA-256 is
`599ff261a1e05c6cd5c5405586fa3e4e0484c7a3bda3154a13bf40a637dd519a`.

Candidate J is limited to those mechanisms. A recoverable zero-step vertical lift may proceed to
the already ranked safe staging poses. Contact establishment and correction re-contact use the
existing content-bound stride while preserving every exact terminal joint target. A contained
target is held only while it remains contained; stable success ends the phase, while drift outside
immediately resumes ordinary correction. The cylinder endpoint restores the existing 3.5 cm
containment margin, and the rare failed direct push uses at most six 8 cm state-aware segments
instead of twelve 4 cm segments. The 250-step horizon, task predicate, action bounds, semantic
schedule, and zero-tolerance gates remain unchanged.

## D-110 - Freeze Candidate J before an isolated diagnostic probe

Candidate J passed full local implementation validation at Git
`d757d91fa7c12f8c341f692118d8d45882623834`. Seeds 66600 through 66663 are reserved for its
diagnostic-only 40-standard/24-hard probe; the unchanged formal range remains 66300--66399 and has
never been materialized. The contract binds `PushToRegionExpert/CandidateJ` to that exact source.
The diagnostic early-stop policy from D-107 remains active, cannot promote the expert, and cannot
alter formal-gate execution. Collection remains closed.

## D-111 - Reject Candidate J on a zero-tolerance workspace exit

Candidate J ran from clean Git `50d28767539237e094657dda1a69798b75392526` and stopped after
14 diagnostic episodes with ten successes. A rightward cylinder correction moved the target outside
the workspace, which triggered the predeclared zero-tolerance early stop. The other failures were a
cube contact event, corrective TCP tracking error, and zero-step contact-planning failure. The full
report SHA-256 is `647e5509490f30211acf2e9aa960cdbc2de94f9b98387230243bf2f2660790ad`.
Formal seeds and collection remain untouched.

Candidate K responds only to those mechanics: all contact paths execute densely; a failed zero-step
direct contact plan may use one fixed midpoint before the same final contact pose; the cylinder
interior margin is fixed at 2 cm; and direct-push planning fallback is bounded to eight 6 cm
segments. The horizon, controller, success predicate, action bounds, formal threshold, and
zero-tolerance checks do not change. Seeds 66700--66763 are diagnostic only.

## D-112 - Bind Candidate K to its verified implementation before isolated probing

Candidate K passed 39 focused runtime-contract tests and Ruff validation at clean implementation
commit `413710deb15a1139b3ac830387c7543d7839c7f7`. The Phase 2B.2 contract binds
`PushToRegionExpert/CandidateK` to that exact source while retaining the untouched formal range
66300--66399. Seeds 66700--66763 are diagnostic only and cannot promote the expert. Collection
remains closed until a later full, untouched 100-episode gate reaches at least 95/100 with every
zero-tolerance count at zero.

## D-113 - Reject Candidate K and bound one final pre-containment braking recovery

Candidate K ran from clean Git `1ec0001f505e33686af01e1d013c1db0aa3881bd` and stopped after
five diagnostic episodes with three successes, one timeout, and one target-workspace exit. The
zero-tolerance workspace event makes the candidate ineligible for promotion. Its immutable report
SHA-256 is `268b925950e7e6e97fe717bb69b7403d579e278910af653fc58429dabc0c9c5f`.

Candidate L is the final bounded recovery inside the existing expert architecture. It adds only an
object-specific pre-containment braking margin, uses the existing privileged target distance inside
the expert, and holds the current joint state before the existing correction phases. It does not
change environment success, planner targets, controller semantics, horizon, action limits, formal
threshold, or zero-tolerance gates. Formal and collection seeds remain sealed.

## D-114 - Bind Candidate L before its isolated diagnostic probe

Candidate L passed 40 focused runtime-contract tests plus full Ruff and format validation at
implementation commit `c7361c60daeaa50740bccee213de9485a4c078ef`. The v2 contract binds
`PushToRegionExpert/CandidateL` to that exact source. Seeds 66800--66863 are diagnostic only;
formal seeds 66300--66399 and all collection ranges remain untouched.

## D-115 - Close Phase 2B.2 as Result B at the expert gate

Candidate L ran from clean Git `fc478d6364e5264b6bceb188f4497da02ad991fe`. Its first four
standard episodes all timed out at the unchanged 250-step horizon. With four failures, its best
possible 64-episode diagnostic result was 60/64 (93.75%), below the unchanged 95% gate, so the
predeclared success-ceiling stop ended the run. Simulator, nonfinite-action, action-bound, and
workspace counts were zero. The full report SHA-256 is
`b2f8c068cdb88a00b4391cdc3c45732eb5a0ee7fc4f6f78eab12771a176a10b6`.

The two complete formal candidate gates remain 75/100 and 11/100. Later G--L probes are diagnostic
only and cannot replace formal evidence. Phase 2B.2 is Result B: the expert gate failed, the formal
66300--66399 range was never consumed, and collection/export/archive/replication/SmolVLA loading,
training, and Phase 2D remain closed. Another expert architecture requires separate authorization.

## D-116 - Continue the already-authorized expert repair with Candidate M

The Phase 2B.2 request explicitly requires repairing an expert below 95% before collection. The
Result B snapshot at `a7e348a` remains preserved, but Candidate M continues that existing scope
without training or formal-seed access. Candidate M retains precontainment braking and replaces the
full high re-contact path for a nearby target with one bounded 8--25 mm controller-executed nudge.
It is bound to implementation commit `1210976e0b926f913504cc8875dc6033d1926224`; seeds
66900--66963 are diagnostic only.

## D-117 - Bind Candidate N with cylinder-specific bounded braking

Candidate N changes only cylinder dynamics: a 5 cm precontainment band and 15 mm maximum in-contact
nudge replace the Candidate M cylinder values while cube behavior stays fixed. It is bound to
implementation commit `f04cd60952428209f660833af0af21910ca4eead`. Diagnostic seeds
68000--68063 avoid the formal and reserved collection ranges and cannot promote the expert.

## D-118 - Reject Candidate N and retarget only drifting cylinder corrections

Candidate N ran from clean Git `48686a57b427ddbf6c99b01b51dc003970e1a5e9` and stopped after
five diagnostic episodes: all four cube tasks succeeded, then the first cylinder/left task crossed
the workspace boundary during settle. The immutable full report SHA-256 is
`9e88c4fcbdfbb60609597c1747e56baacf6d473aab2bbda3fd84f330b8068f30`; formal seeds and collection
remain untouched.

The same already-consumed diagnostic seed was replayed for phase-boundary geometry only. Its
cylinder stayed on the table, but lateral error grew from 3.1 cm after primary push to 11.3 cm after
the second in-contact correction because both corrections reused the initial approach direction.
Candidate O recomputes the current object-to-target-center direction only for lateral-cylinder
in-contact correction. It does not alter cube behavior, primary motion, controller semantics,
success geometry, the 250-step horizon, action bounds, acceptance thresholds, or zero-tolerance
gates. Candidate O requires a new disjoint diagnostic schedule before the formal gate can run.

## D-119 - Bind Candidate O before isolated diagnostic screening

Candidate O passed 41 focused local tests with one expected Windows symlink-capability skip, Ruff,
format, and changed-module mypy validation at implementation commit
`e55198ca44d478df584a85357a697ffbecbc8484`. The v2 contract now binds that exact implementation.
Seeds 68100--68163 are diagnostic only and are disjoint from the untouched 66300--66399 formal
gate, every prior diagnostic range, and the reserved collection/top-up schedule. The diagnostic
probe cannot promote the expert; collection remains closed until a complete formal gate passes.

## D-120 - Reject Candidate O and restore only lost local cylinder contact

Candidate O ran from clean Git `e167852affe047c3e10eb87b8580bc07ba885513` and stopped after
five diagnostic episodes: all four cube tasks succeeded, then the first cylinder/left task crossed
the workspace boundary during settle. Its full report SHA-256 is
`8d738c9e1bd0866a76fbff1d9900b4e1b2947417bac861a9ec108b1eab714a27`; formal and collection
seeds remain untouched.

The same diagnostic seed showed that Candidate O rotated the short correction toward the target,
but the primary push had already left the TCP 9.2 cm from the cylinder. The 15 mm correction
therefore moved in free space while cylinder momentum increased lateral error from 5.0 cm to
12.3 cm. Candidate P adds one low, local contact waypoint only for a near-target lateral cylinder
whose TCP-to-object distance exceeds the existing contact proxy. It then executes the same bounded
nudge. It does not change cube paths, high free-space recovery, task geometry, horizon, action
bounds, thresholds, or zero-tolerance gates.

## D-121 - Bind Candidate P before isolated diagnostic screening

Candidate P passed focused local unit, Ruff, format, and changed-module mypy validation at
implementation commit `de9309355f96065aa62823522a0d0ebeab7b903c`. The v2 contract binds that
exact implementation. Seeds 68200--68263 are diagnostic only, disjoint from every previous probe,
the untouched formal gate, and all collection/top-up ranges. The diagnostic probe remains
ineligible for promotion by itself.

## D-122 - Reject Candidate P and close Phase 2B.2 as Result B

Candidate P ran from clean Git `d54487d1773d2b76c88039943ed45c0d0ccf21a8` on isolated
diagnostic seeds beginning at 68200. It reached 2/5: two successes, one timeout, one precontact
planning failure, and one cylinder workspace exit. The zero-tolerance violation stopped the probe.
Its full report SHA-256 is `6c2eea259b0afeb140dba81b281b05bc8516e3ccffeb24ae9d4071a7db2c21ec`;
the generation-audit SHA-256 is
`d82fdb63f780e8fc1863442a00cafbcd1080e569f87ba1427fa2f4a23f2c97a1`.

The original task requires unstable generation to stop before low-quality export and explicitly
defines Result B for expert-gate failure. With two failed 100-episode formal gates and ten rejected
diagnostic candidates G--P, no further bounded repair is justified inside this milestone. Formal
seeds 66300--66399 and every collection seed remain untouched. Raw collection, LeRobot export,
archives, replication, SmolVLA loading, optimization, training, and Phase 2D did not start.

## D-123 - Close simulator-backed Phase 2B.3 as Result C at clone precondition

Phase 2B.3 began from exact source `8476a72b3ce7504e7fef0a08114c6c567da77ba7` and audited the
installed ManiSkill 3.0.1 public state surface before implementing a simulator-MPC runtime. The
project snapshot covered actor/articulation state, robot joints, empty `pd_joint_pos` controller
state, task counters, elapsed steps, seeds, and RNG state. It intentionally did not claim access to
PhysX private solver/contact caches.

Three native RTX 5090 attempts are retained. Two independently cold-restored environments replayed
the same contact-bearing actions exactly, and all 192 categorical comparisons agreed. However, the
last audit captured three actions before first target contact and still observed 4.394 mm maximum
target-pose divergence between live continuation and a fresh reset-plus-restore. The allowed pose
tolerance was 0.020 mm, and the task's containment clearance is only 5 mm. Configuration and
environment isolation passed, but simulator transition equivalence failed materially.

The architecture is classified `RESULT_C`. Relaxing tolerance, substituting a hand-written model,
or running quality gates anyway would violate the authorized research question. No MPC expert,
development pilot, prediction-agreement evaluation, repair, or formal 100-episode gate ran. Seeds
66300--66399 remain untouched. `phase2b4_collection_authorized=false`,
`phase2c2_training_authorized=false`, and `smolvla_started=false`; no raw archive, dataset,
checkpoint, backward pass, or optimizer step was produced.

## D-124 - Stop reset-replay feasibility at the historical transcript gate

Phase 2B.3.1-RR began from exact MPC Result C commit
`cc31ced97e66220626ef39374b3e3acf84ccd463` in the isolated
`codex/langmani-v2-phase2b3-reset-replay-equivalence` worktree. The independent geometry result
`ae6aae49c61d96c68de434e08dc1167461a29543` remained external: neither result commit is an ancestor
of the other, their merge base is `8476a72b3ce7504e7fef0a08114c6c567da77ba7`, and no geometry
artifact was copied into the MPC package.

Direct SHA-256 checks of immutable Git blob bytes passed all 19 MPC and all ten geometry manifest
entries. The latest no-GPU evidence node was clean at `cc31ced...`; the idle RTX 5090 node held
older source and was not updated or used. Targeted inspection found no complete original
`float32[8]` `pd_joint_pos` transcript for the required diagnostic boundaries. The surviving clone
report records seed/task identity, action counts and indices, and a 12-action content hash, but not
the numeric actions or a complete call-sequence identity bound to them.

The compact Result C and Result D verifiers treat only CRLF-to-LF checkout translation as the
canonical Git text bytes and report which entries needed that normalization. This makes the
immutable Linux-produced hashes portable to a Windows worktree; arbitrary byte changes, including
added whitespace, still fail the manifests.

The predeclared hard stop therefore classifies the phase
`RESULT_D / HISTORICAL_PREFIX_UNAVAILABLE`. This does not overturn the historical Result C and is
not a new physical nondeterminism result. No executable protocol, episode/boundary selection,
environment, reset, sandbox, replay, probe, transition comparison, isolation trial, or runtime
measurement was created. `mpc_pilot_eligible=false`, `mpc_pilot_authorized=false`,
`expert_qualification_authorized=false`, `data_collection_authorized=false`,
`smolvla_training_authorized=false`, `training_started=false`, and
`demonstration_source_validated=false`.

## D-125 - Freeze the bounded Phase 2B.4-F0 PPO teacher feasibility protocol

Phase 2B.4-F0 starts only from RR Result D commit
`3ded856bc0fba7cd749f1fb0caf738fd7fce5751`. The independent geometry commit
`ae6aae49c61d96c68de434e08dc1167461a29543` remains excluded. MPC Result C and RR Result D are
immutable historical evidence: the former invalidated public state-clone continuation for MPC;
the latter found no executable historical action prefix and made no physical nondeterminism claim.

The compatible maintained reference is ManiSkill 3.0.1 PPO at tag commit
`a4a4f9272ad64b1564035874b605ceb687b63ed8`. F0 preserves its separate three-layer 256-unit tanh
actor/critic, rollout, GAE, clipped surrogate, and Adam structure, but rejects the reference
Normal-plus-clamp action path. One Gaussian-latent/tanh/affine distribution, parameterized by the
active environment bounds, now owns stochastic rollout, update likelihood, deterministic
evaluation, and checkpoint reconstruction. No emitted action is clipped, projected, rejected,
replaced, or delegated to an expert.

Before learning, F0 freezes an 87D current privileged teacher observation, the separate
`base_camera + PandaPolicyStateV0[9]` future-student schema, fixed non-fitted state scales, reward
revision 0, five disjoint seed namespaces, 128 vector environments, and exactly 1,048,576 micro
steps. The reward-hacking audit passed without repair. The 48-episode micro gate must pass before
one 60-episode broader development evaluation. A micro failure hard-stops as Result C; an invalid
pipeline/reward stops as Result D.

Formal seeds 66300--66399 and future full-development identities remain sealed. F0 creates no
demonstration, archive, failure corpus, LeRobot dataset, split, student statistic, or accepted
dataset package. Even Result A sets only `ppo_full_training_eligible=true`;
`ppo_full_training_authorized`, `expert_qualification_authorized`,
`data_collection_authorized`, `smolvla_training_authorized`,
`training_started_for_student_policy`, and `demonstration_source_validated` remain false.

## D-126 - Close Phase 2B.4-F0 as Result C at the micro gate

The native RTX 5090 execution used clean source commit
`0740c2fcaf5a34405130f3e3a8a3fac6e9a5d406`. Target installation, prior Result C/Result D
verification, the 128-environment vector audit, reward/action audits, one real PPO update,
checkpoint reconstruction, and the 100,000-sample action legality audit all passed.

The single frozen micro configuration ran exactly 1,048,576 environment steps in 944.172 seconds
(1,110.577 environment steps per second), executed 8,118 optimizer steps and 21,162 resets, and
reported zero simulator exceptions. Its reconstructed checkpoint produced exactly the same
deterministic action as the in-memory policy. Across 21,034 completed training episodes it recorded
zero successes. Mean return improved by only 0.265 between the first and last quartiles, below the
predeclared 0.5 gate; mean target progress improved by only 0.00224 m, below the 0.01 m gate, and
both quartile means remained negative.

The fixed deterministic micro evaluation therefore ran exactly 48 episodes and reached 0
successes: 14 timeouts, 22 wrong-object displacements, and 12 target workspace exits. All 5,462
actions were finite, within the native bounds, and nonconstant; no clip or projection path ran.
The failure is therefore an expert-learning failure under this frozen PPO design, not an action
pipeline or simulator-infrastructure failure.

Reward revision 0 was not repaired. The unsafe evaluation outcomes received their declared large
negative reward terms and did not expose a profitable reward exploit or a success-comparable
unsafe return. The one allowed repair was therefore inapplicable, and parameter tuning or a sweep
was prohibited.

The predeclared hard stop classifies F0 as Result C. The broader 60-episode development evaluation
did not run. Formal seeds 66300--66399, full PPO training, the 95/100 qualification, data
collection, datasets, SmolVLA, and student training remain unaccessed and unauthorized.
`ppo_full_training_eligible=false`, and every authorization flag remains false. This result rejects
the frozen F0 teacher design; it is not a claim that reinforcement learning cannot solve the task.

The compact manifest hashes and sizes use Git-canonical UTF-8 JSON bytes: CRLF is normalized to LF,
while any other carriage return is rejected. This keeps the same immutable evidence independently
verifiable from Windows and Linux checkouts without accepting arbitrary whitespace changes.

## D-127 - Freeze the bounded Phase 2B.4-F1 PPO diagnosis and residual exploration protocol

Phase 2B.4-F1 starts only from immutable F0 Result C commit
`c5892a357da85f753d5d0c3775ccf2b94e0d8910`. The independent geometry commit
`ae6aae49c61d96c68de434e08dc1167461a29543` remains excluded. F1 does not reinterpret F0's valid
pipeline and zero-success result. It reports fields that were not retained in F0, including the
per-episode training termination histogram and per-iteration value explained variance, as
unavailable rather than synthesizing them from new rollouts.

The new action identity is `state_centered_bounded_residual_v1`. Current Panda qpos is normalized
within the live native bounds, mapped through a guarded inverse tanh, incremented by an explicit
per-dimension bounded residual, mapped through tanh, and affinely returned to the unchanged
`float32[8] pd_joint_pos` contract. Arm residual scales are 0.08 and the gripper scale is 0.25.
The complete transform Jacobian is included in PPO log probability using the stable analytic
log-derivative of tanh. The current-coordinate guard is `1e-6` and the zero-residual identity
tolerance is `2e-6`. Emitted actions have no clip,
projection, invalid-action replacement, or fallback path. A 100,000-action legality audit and a
paired 32-by-8-step physical comparison must pass before learning.

F1 Stage 0 uses only the cube, Standard left/forward-right tasks, canonical 250-step stable success,
unchanged physics/geometry, and the unchanged Standard distractor. Probe A keeps reward revision 0,
the F0 PPO architecture/optimizer settings, one fixed seed, and exactly 262,144 environment steps.
Its independent 32-episode gate requires zero wrong-object/workspace/action-integrity events, at
least 25% correct contact, and at least 20% target-directed progress. Only that conjunction permits
one disjoint 262,144-step Probe B. Probe B's 48-episode gate additionally requires at least 20%
success, one success in each included direction, at least 50% correct contact, and at least 40%
progress. No third probe, sweep, resume, or algorithm switch is permitted.

Formal seeds 66300--66399, future full-curriculum identities, Stage 1/2, full PPO training, expert
qualification, collection, LeRobot export, SmolVLA, ACT/VLA-JEPA, and student work remain sealed.
Even a positive F1 result sets eligibility only; all downstream authorization flags remain false.

## D-128 - Close Phase 2B.4-F1 as Result C at the Probe A gate

The target implementation ran from clean commit
`5934a0261f1f0bb6c5bf01df3e0ef0b8f3b50e49` on the migrated RTX 5090 host. The target installation,
F0 compact verifier, 100,000-action legality audit, exact log-probability test, checkpoint
reconstruction, and termination/bootstrap audit passed. The residual audit emitted zero non-finite
or out-of-bound actions, used no clip or projection, and reproduced current targets within
`1.0132789611816406e-6`.

The paired 32-by-8-step physical comparison reduced mean maximum arm target displacement from
0.712 to 0.305 rad and mean maximum TCP step displacement from 0.0902 to 0.0448 m. It improved
locally smooth prefixes from 0/32 to 21/32. Both paths had zero short-prefix wrong-object and
workspace events, so this is a local-kinematics result rather than task competence.

The frozen F0 checkpoint diagnostic on 32 new F1 identities reached the behind-object region in
20 episodes but made zero correct target contacts and produced 14 wrong-object outcomes, nine
workspace exits, and nine timeouts. Correct-contact/containment credit was absent, so no contact
reward revision was justified. Probe A retained reward revision 0.

Probe A ran exactly 262,144 environment steps and observed zero training successes. Its fixed
32-episode evaluation reached zero stable success, zero correct-contact episodes, zero
target-progress episodes, 25 timeouts, seven wrong-object interactions, and zero workspace exits.
All 6,474 evaluation actions remained finite and in bounds with no clip or projection. This fails
the zero-tolerance, contact, and progress gates, so Probe B did not run.

F1 is Result C: the state-centered redesign improved initial locality but did not produce safe
nonzero pushing competence. The custom PPO teacher route is frozen for the current task
formulation; another PPO iteration, Probe C, reward retry, budget increase, or algorithm switch is
not authorized. Formal seeds 66300--66399, full PPO, expert qualification, collection, LeRobot,
SmolVLA, and student training remain unaccessed and unauthorized.

## D-129 - Qualify pinned official demonstrations without reopening the custom route

Phase 2B.5 begins only from immutable F1 Result C commit
`6ca3786702ada4e92be607a5293b9c3c31d2a7d4`. It permanently freezes custom expert, geometry,
simulator-MPC, reset-replay, and PPO development for `LangMani-PushToRegion-v0`. Historical
evidence is rehashed but never rewritten, and no formal custom seed, missing v1 byte, or independent
rejected branch may become new training data.

The official source identity is `haosulab/ManiSkill_Demonstrations` at immutable revision
`d674485bbffdd533914e52d272fdda34c0515608`, used with ManiSkill 3.0.1 and upstream source
revision `a4a4f9272ad64b1564035874b605ceb687b63ed8`. Only required candidate archives are retained
outside the repository, byte-sized, SHA-256 addressed, and safely expanded after member-path and
symlink checks. Source acceptance reads the actual JSON/HDF5 records; a registry entry or sample
video is insufficient.

The preferred common contract is direct Panda `pd_joint_pos`, `float32[8]`, 20 Hz, five simulator
substeps per action. A source using another mode can be classified incompatible or, only with
separate deterministic evidence, convertible; Phase 2B.5 never clips, projects, repairs, or silently
converts it. Official success predicates stay unchanged. Bounded replay uses 20 deterministic,
length-spanning/reset-diverse episodes per candidate. A selected source then uses 100 such episodes.
Both gates require at least 99% categorical agreement, exactly 100% action/frame alignment, zero
invalid actions, and zero simulator exceptions.

The future-student frame allowlist is base-camera `uint8[256,256,3]` RGB without cropping,
`PandaPolicyStateV0 float32[9]`, the recorded action, one task ID, one canonical instruction, and a
20 Hz timestamp. Full environment states and object/goal poses remain separate diagnostics. Only
five replay-passing episodes per selected task may be converted through a separate LeRobot 0.6
environment and read back with the public dataset API. The pilot is not full dataset production.

Result A requires three genuine skill families plus all source, direct-action, replay, observation,
privilege, LeRobot, and split-design gates. Even Result A grants only
`phase2b6_dataset_production_eligible=true`. `act_training_authorized=false`,
`smolvla_training_authorized=false`, `vla_jepa_training_authorized=false`,
`student_policy_training_started=false`, and `custom_push_expert_route_active=false` remain
mandatory, and no policy training follows automatically.

## D-130 - Accept the three-source official demonstration package as Phase 2B.5 Result A

The native qualification finished from clean execution commit
`0b7f4d5e3b6fc909ee370aad5048ea7cc3ab9b17` on the RTX 5090 host. The run used ManiSkill 3.0.1,
SAPIEN 3.0.3, Python 3.12.13, Torch 2.11.0+cu128, driver 595.71.05, and the documented
`/etc/vulkan/icd.d/my_nvidia_icd.json` Vulkan binding. One launch without that required binding
failed before environment construction and is retained only as an infrastructure diagnostic; it
did not contribute to source or replay evidence.

Five Apache-2.0 official archives from immutable Hugging Face revision
`d674485bbffdd533914e52d272fdda34c0515608` were retained outside Git. Their combined size is
138,681,813 bytes and every ZIP member passed traversal/symlink checks. The accepted
motion-planning sources are `PickCube-v1`, `StackCube-v1`, and `PushCube-v1`, representing pick and
place, stacking, and planar pushing. They contain 3,000/3,000 successful trajectories and 254,374
transitions with zero malformed records, duplicate action hashes, non-finite actions, or native-
bound violations. Poke and Pull are rejected from the common contract because the downloaded
official packages expose only delta-action RL sources; no conversion or repair was attempted.

Each selected source passed both 20-episode and 100-episode deterministic length/reset-stratified
replay. Across all six gates, categorical outcome agreement and action/frame alignment were 100%,
with zero invalid actions and zero simulator exceptions. The stronger task-object terminal
translation diagnostics had maxima of 0.397 mm for Pick, 2.589 mm for Stack, and 4.919 mm for Push;
these continuous values do not replace the unchanged official categorical predicates.

The bounded nonprivileged pilot contains 15 episodes and 1,546 aligned policy frames. An isolated
LeRobot 0.6.0 / Torch 2.11.0+cu128 / PyAV 15.1.0 environment wrote it with the public API and read
back every frame with zero image, state, action, task, index, or timestamp failure. The compact
28-file evidence package and fresh no-simulator verifier pass.

Phase 2B.5 is therefore `RESULT_A`: `official_demo_source_validated=true` and
`phase2b6_dataset_production_eligible=true`. The package is a source qualification and pilot, not
a full dataset or policy result. `full_dataset_production_started=false`,
`act_training_authorized=false`, `smolvla_training_authorized=false`,
`vla_jepa_training_authorized=false`, `student_policy_training_started=false`, and
`custom_push_expert_route_active=false`. Phase 2B.6 may start only by a separate instruction.

## D-131 - Freeze Phase 2B.6 official multi-skill dataset production

Phase 2B.6 starts only from immutable Phase 2B.5 Result A commit
`8c81058008bb03c4611c9133adf4112e2a21287f`. Its sole source authority is the accepted official
package for `PickCube-v1`, `StackCube-v1`, and `PushCube-v1` at
`ManiSkill_Demonstrations@d674485bbffdd533914e52d272fdda34c0515608`. Historical custom push,
geometry, simulator-MPC, reset-replay, and PPO branches remain closed and cannot supply source
episodes, task definitions, actions, thresholds, or conclusions.

The immutable production specification is
`configs/langmani_v2/phase2b6_dataset_production.yaml`. It binds all three official ZIP identities,
ManiSkill 3.0.1, the upstream ManiSkill source revision, `physx_cpu` replay, `sapien_cuda`
rendering, the accepted 256-by-256 base camera, `PandaPolicyStateV0`, native `float32[8]
pd_joint_pos`, 20 Hz, five physics substeps, exact episode/transition totals, language banks,
episode identities, splits, post-render visual shift, archive layout, retry policy, and hard
stops. Once full production starts, a material specification change requires a new run identity
and output root while preserving the old partial run.

Every official trajectory is replayed from its official metadata and accepted first-state
mechanism using only its original action bytes. Each policy frame is the observation immediately
before its corresponding action. The post-final-action image is diagnostic only, and no synthetic
terminal action is created. Replay acceptance is conjunctive: all 3,000 official successes must
remain successes, categorical agreement and frame/action alignment must both be 100%, and invalid
actions, non-finite values, and simulator errors must all be zero. Physical failures are never
retried or repaired. At most one identical retry is permitted only for an infrastructure failure
before or outside physical execution.

Primary splits are materialized once per immutable source episode and exposed reset identity:
700/100/100/50/50 per task for train, validation, unseen reset, unseen task language, and visual
shift. Task/split-specific LeRobot 0.6 subroots keep physical media from crossing split
boundaries, while one task container and one unified immutable multi-root index preserve the
requested task-level identities. The 150 visual-shift episodes are disjoint evaluation episodes;
their second rendering applies only the pre-registered post-render exposure and color-temperature
transform. Physics, reset, state, action, timing, geometry, friction, and success remain unchanged.
Cross-skill folds are metadata-only alternate experiment views and do not support an unseen-skill
claim without separately trained future models.

Action horizons 10, 16, and 50 use an explicit boolean `action_is_pad` mask. Future masked losses
must exclude padded timesteps; zero vectors alone carry no padding meaning. Canonical
normalization is computed only from the pooled natural-frame primary training view. Per-task and
equal-task views remain diagnostics, and uniform-task sampling is the recommended default for
future multi-skill baselines so StackCube's longer trajectories do not silently dominate.

Acceptance additionally requires all-frame LeRobot 0.6 structural readback, decoding of every
video frame, exact source-to-derived action equality for all episodes, zero primary-split leakage,
an external deterministic content-addressed archive, and a full file-hash restore plus stratified
LeRobot readback from physically restored bytes. Result A grants only later-phase eligibility.
`act_training_authorized=false`, `smolvla_training_authorized=false`,
`vla_jepa_training_authorized=false`, `student_policy_training_started=false`,
`optimizer_steps=0`, and `backward_passes=0` remain invariant throughout this phase.
