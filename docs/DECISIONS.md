# Environment and dependency decisions

This file records M0 through M3B decisions as of 2026-07-13. “Metadata-compatible” means official package
requirements have a non-empty version intersection; it is not a claim of native Linux GPU or
rendering success.

## D-001 — Platform boundary

The acceptance platform is native Linux with an NVIDIA RTX 4090. Windows-native and WSL-specific
workarounds are intentionally out of scope because ManiSkill documents GPU simulation and rendering
support on native Linux/NVIDIA, while WSL lacks those paths.

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
a candidate until the native Linux gate passes.

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

These are non-physical contract, fake-planner, serialization, command-boundary, and build checks.
mplib is absent because the ManiSkill 3.0.1 dependency marker installs it only on Linux. No real
Panda IK/screw plan, grasp, transport, placement, six-task benchmark, M2 renderer, PhysX GPU run,
or RTX 4090 execution has been physically validated; the full ordered target gate remains pending.

Before the non-Linux boundary guard was added, directly constructing PickCube on this Windows host
terminated the Python process with a SAPIEN access violation in `actor_builder.py`. This is recorded
as out-of-scope evidence, not a passed CPU simulator check and not a reason to add a Windows
workaround.

## Unresolved risks

1. **No native Linux RTX 4090 execution yet.** The selected set is metadata-compatible, but CUDA,
   PhysX GPU, Vulkan rendering, and a saved diagnostic frame remain physically unverified.
2. **ManiSkill does not declare a PyTorch upper bound.** Its metadata cannot prove that PyTorch
   2.11 is runtime-compatible; the strict target run is required.
3. **OpenCV wheel collision.** SAPIEN 3.0.3 requires `opencv-python`, while LeRobot 0.6.0 requires
   `opencv-python-headless`. OpenCV's publishers state that only one wheel sharing the `cv2`
   namespace should be installed. The environment pins both to the same version because both
   upstream metadata requirements must remain satisfied, but this is not an upstream-supported
   resolution. `import cv2`, ManiSkill rendering, and LeRobot import must all pass on the target;
   uninstalling either wheel in-place may damage the other.
4. **System components are not lockable here.** NVIDIA driver, Vulkan ICD, kernel, and distribution
   libraries can still invalidate a correct Python resolution.
5. **No full transitive lockfile yet.** M0 pins the critical direct and renderer packages in the
   environment declaration. Generate and review a platform lock only after the first successful
   target run, so it captures a physically verified rather than merely resolvable environment.
6. **Optional SAPIEN Pinocchio bindings are absent on the review host.** M2 uses the separately
   packaged mplib planner rather than SAPIEN's optional Pinocchio wrapper, so this warning does not
   justify a dependency change. The native target must still prove the selected mplib path works.
7. **M2 physical planning is unverified.** The public mplib 0.1.1 API and ManiSkill 3.0.1 Panda
   examples were inspected, but the current Windows host cannot import Linux-only mplib. Kinematic
   reach, collision behavior, gripper execution, release settling, all six task combinations, and
   diagnostic rendering remain pending until the ordered target gate passes.
8. **The Linux mplib wheel has not been imported beside NumPy 2.2.6 on the target.** Metadata has no
   conflicting bound, but binary-extension compatibility cannot be established from the downloaded
   wheel's Python source. The strict target import and physical benchmark must prove this exact pair.
9. **No authoritative M3A target archive exists yet.** The deterministic schedule, recorder adapter,
   transaction recovery, archive validation, and replay audit passed synthetic and structural tests,
   but no native Linux Panda expert has recorded, inspected, and independently replayed all 360
   accepted episodes. `environment/verify_m3a.py --target-smoke` must first validate the six-task
   physical chain; only `--target-full` can close the full-archive risk after the ordered M0, M1, and
   M2 target gates pass.
10. **No real M3B target export exists yet.** Local generated-array LeRobot/PyAV/Parquet/DataLoader
    integration passes, but the Windows repository path cannot instantiate the Panda renderer and
    no authoritative M3A source exists. `verify_m3b.py --target-smoke` must prove six real
    state-restoration videos and source alignments; only `--target-full` can validate all 60 groups,
    360 episodes, scene-level splits, every decoded video, and the first training-ready derived
    dataset.

## D-027 — Index separate version histories and preserve evidence corrections

On 2026-09-16 (Singapore time), a separately authorized bounded delivery refreshed main to
`6920b52c1f48c278e669cd71b69b8949dd900f3a` and inspected frozen v1
`58434cb17a7234b6d4b2c4fb15aecf8df0621487` and terminal v2
`2dcf2ac68e8627cfed98cbbd93272bf1768b190c`. Neither later commit is an ancestor of that main.
[Versions and evidence](VERSIONS_AND_EVIDENCE.md) now supplies a main entry point to those
histories without importing their runtime or inheriting their target acceptance.

The v1 index's 72/72 physical-pair claim is corrected to the saved independent report's 61 pairs,
with `final_paired_states_validated=false`. Eleven pre-execution false rejections are no-runtime
records. Historical 46/72 end-to-end and 55/72 Oracle success, completed pipeline verification,
and failed final quality gate remain separate. The frozen source and evidence are unchanged.
The local 24-file M5A manifest currently has 16 matching entries and 8 missing entries; that
directory cannot establish complete present-day recovery. Existing portfolio robot footage is
M3B expert footage and cannot demonstrate final learned control.

The authorized attempt found one red-to-left ACT checkpoint with all nine covered files and
completion-marker integrity matching. The designated RTX 5090 target lacked its frozen runtime;
the sole 180-second dependency preflight exited 124 before resolution completed. No dependency
installation, model load, simulation, training, or video followed. This is an environment-preflight
skip, with no new physical validation or model-quality result. The bounded route was stopped
without a replacement candidate. The [receipt](receipts/2026-09-16-single-attempt.md) records
hashes, exact scope, and process/lock closeout. No dependency, runtime interface, quality gate,
or environment declaration changed.
