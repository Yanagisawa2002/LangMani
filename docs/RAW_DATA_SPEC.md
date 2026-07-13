# M3A raw demonstration archive specification

Schema version: `langmani-m3a-raw-v1`.

## Authority and scope

The M3A ManiSkill-native HDF5/JSON archive is the authoritative source. M3B may derive a
LeRobotDataset v3 training representation, but it must not replace this source. M3A contains no
Parquet, policy-camera MP4 export, ACT/SmolVLA training, language paraphrases, augmentation,
multiprocessing, GPU-parallel planning, or policy evaluation.

## Default collection identity

| Field | Default |
| --- | --- |
| environment | `LangMani-PickPlaceByInstruction-v0` |
| control mode | `pd_joint_pos` |
| simulator backend | `physx_cpu` |
| candidate seed start | 0 |
| target complete groups | 60 |
| maximum candidate groups | 120 |
| expert attempts per task | 3 |
| accepted episodes per shard | 60 (10 complete groups) |
| failed raw retention | false |
| replay mode | `action_and_state_audit` |
| final target position tolerance | 0.005 m |
| final target orientation tolerance | 0.05 rad |
| final Panda joint tolerance | 0.05 rad |
| overwrite | error |
| resume | enabled |

The exact planner-side runtime versions of Gymnasium, h5py, ManiSkill, mplib, NumPy, OpenCV,
Pillow, SAPIEN, SciPy, and PyTorch are part of the manifest and must match on resume and independent
replay. Collection and action replay use `LANGMANI_PLANNER_PYTHON`; structural inspection may run
in the main NumPy-2 environment. A missing field or drift in any recorded runtime version rejects
reuse rather than silently resuming under another ABI.

## Deterministic schedule and IDs

Candidate seeds are consecutive and ordered. For each seed the six tasks are scheduled object-major,
bin-minor: red-left, red-right, green-left, green-right, blue-left, blue-right. The same configuration
always produces the same schedule.

Collection-run, candidate-scene, scheduled-episode, expert-attempt, raw-trajectory,
accepted-scene-group, and source-shard IDs use canonical JSON and SHA-256. Environment version,
scene seed/ID, TaskSpec/task ID, expert configuration fingerprint, control mode, and schema version
are included where semantically relevant. Persistent IDs never use Python `hash()` or filesystem
enumeration.

## Counterfactual admission

One `CounterfactualSceneGroup` is one scene seed paired with all six TaskSpecs. It is accepted only
when all six expert attempts succeed, every native trajectory is structurally valid, seeded action
replay succeeds, optional state audit succeeds, final M1 evaluation is success, TaskSpec and all
stable identities agree, and all six recorded initial environment states are identical.

Every bounded attempt persists the exact M2 `ExpertResult`, including expert failures, unexpected
exceptions, and results that claim success but violate collection provenance. A trajectory is
eligible to enter action replay only when the result reports success; its task and scene IDs match
the schedule and reset `EpisodeSpec`; its object, bin, and canonical instruction match the scheduled
`TaskSpec`; its final evaluation exactly agrees with a fresh M1 environment evaluation; neither
wrong-object nor wrong-bin false success is present; and all configured episode, retry, and planning
bounds are respected. A claimed success that violates any of these checks is retained as an
`expert_contract_failure` attempt, with the original `ExpertResult` unchanged, and never reaches
action replay.

Because ManiSkill 3.0.1 intentionally omits static actors from `get_state_dict()`, M3A also captures
an explicit physical snapshot immediately after every reset and before the expert acts. It contains
the seven-value poses of `red_cube`, `green_cube`, `blue_cube`, `left_bin`, and `right_bin`, plus the
nine Panda initial qpos values. Cube poses and Panda qpos are cross-checked against HDF5 state zero;
the complete explicit snapshot is compared across all six tasks, so static bin equality is verified
rather than inferred from a shared seed.

One failed member rejects the whole candidate. Successful siblings become `group_rejected` attempt
records and never enter accepted shards. Every started attempt receives a stable ID and small
diagnostic record. Collection stops with failure if 60 complete groups cannot be obtained within the
120-candidate default bound; it never weakens expert, geometry, replay, or balance criteria.

## Native trajectory pair

M3A uses the installed ManiSkill 3.0.1 `RecordEpisode` wrapper with `save_trajectory=True`,
`save_video=False`, `save_on_reset=False`, `record_reward=False`, `record_env_state=True`, and
`obs_mode="none"`. A valid accepted episode contains:

- float32 Panda actions with shape `(T, 8)`;
- boolean `terminated`, `truncated`, `success`, and `fail` vectors of length `T`;
- an empty `obs` group and no rewards dataset;
- recursive finite numeric environment-state leaves of length `T+1`;
- JSON reset seed and exact TaskSpec;
- exact M2 `ExpertResult`, replay evidence, final M1 evaluation, and stable provenance.
- the explicit initial cube/bin pose and Panda-qpos snapshot captured before the first action.

The transition-time contract is exact: `actions`, `terminated`, `truncated`, and every present
`success`/`fail` label contain exactly `T` entries, while every environment-state leaf contains
exactly `T+1` entries including state zero. Collection and replay never drop the initial state, add
a terminal no-op, duplicate the last transition, normalize an action, translate it to another
control mode, or silently clip it. Replay requires native float32 `(8,)` action rows, validates each
row against the configured environment's explicit action-space bounds, and passes each valid row
unchanged to `env.step` exactly once. An out-of-bounds row is rejected before it reaches the
environment.

Accepted episodes must end successful and terminated, not failed or truncated. `elapsed_steps`, M2
environment steps, HDF5 action count, and replayed action count must agree. Each HDF5/JSON pair is
closed, re-opened, structurally validated, checksummed with SHA-256, and then published. Soft links,
external links, extra trajectory members, non-finite values, invalid dtypes/shapes, and dangling
HDF5/JSON episode references are rejected.

## Manifest and filesystem layout

```text
<dataset-root>/
├── .langmani-m3a-root
├── manifests/
│   ├── collection_manifest.json
│   ├── collection_schedule.json
│   ├── dataset_summary.json
│   ├── attempts.jsonl
│   ├── episodes.jsonl
│   ├── scene_groups.jsonl
│   └── source_shards.jsonl
├── accepted/shards/*.h5 + *.json
├── journal/
│   ├── inflight/candidate-*.json
│   └── shard-*/group-*.h5 + *.json
├── .staging/
└── failed/                 # present only when failed raw retention is enabled
```

`collection_manifest.json` is authoritative and is written last as the generation commit marker.
The JSON/JSONL files are deterministic projections and must match it exactly. Accepted shard paths,
file sizes, checksums, native locations, stable IDs, task semantics, expert/replay evidence, and
group membership are all cross-checked. Orphan HDF5 or JSON files are corruption.

## Resume, interruption, and retention

ManiSkill 3.0.1 opens a recorder pair in write mode, so M3A never appends to an open native file.
It records one candidate in staging and atomically journals an active attempt before execution and a
completed attempt afterward. A resumed collector converts an interrupted candidate into an explicit
rejected group exactly once, preserves its attempt IDs, and advances to the next deterministic seed.

Group bundles remain until an open partial shard is safely reconstructible. Full shards and terminal
partial shards are sealed: checksum or schema failure is a hard error, not silent recollection.
Every accepted group's bundle HDF5 and JSON digests are committed in the manifest; partial-shard
repair first requires the current journal pair to match those trusted digests. It never re-signs a
changed bundle or bypasses action-replay evidence.
Journals and staging are removed only after the authoritative manifest commit. A post-commit resume
idempotently completes any pending failure-retention move before deleting its journal.

Overwrite is allowed only for a root with an exact LangMani marker and a parseable manifest that
claims that same resolved root, or for a narrowly recognized interrupted-initialization residue.
Forged roots are never deleted. With failed-raw retention disabled, only small attempt diagnostics
remain. With it enabled, candidate pairs are moved under `failed/` and never mixed with accepted
shards.

## Replay and audit

Action replay uses one environment and the recorded reset seed plus exact TaskSpec. Before executing
actions, the fresh reset state is compared with recorded state zero across the complete numeric
state tree. Every action is then replayed. Final M1 success, wrong-object/wrong-bin/off-table flags,
target pose, target orientation, Panda qpos, action count, task identity, and configured tolerances
must pass.

Every returned replay failure has one or more stable `ReplayFailureCode` values alongside bounded
human-readable reasons. The codes distinguish malformed recorded data, initialization, task and
initial-state mismatch, out-of-bounds or failed action execution, action-count mismatch, invalid
recorded/final success, wrong-object, wrong-bin, off-table, final-state/tolerance mismatch, failed
state audit, and environment-close failure. A malformed HDF5/JSON time or dtype contract raises a
`ReplayContractError` carrying `recorded_trajectory_invalid` before simulation starts. A passing
result must have no failure codes or reasons; a failing result must have both.

In `action_and_state_audit` mode, every recorded state is restored in order and immediately read
back; every numeric leaf must round-trip within `1e-6`. The final restored state must also satisfy
M1 success. State restoration is an additional audit and never substitutes for seeded action replay.

## Commands and target acceptance

```bash
python environment/collect_raw_demos.py
python environment/inspect_raw_demos.py
python environment/replay_raw_demos.py
python environment/verify_m3a.py
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py --target-smoke
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py \
  --target-full \
  --dataset-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --create-new-run
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py \
  --target-full \
  --dataset-root outputs/datasets/m3a/langmani-pick-place-raw-v1
```

`--target-smoke` first runs `verify_m2.py --target` (which runs M0 and M1), then creates a fresh
diagnostic archive containing one complete counterfactual scene group. All six expert episodes must
close cleanly, pass schema/checksum inspection, preserve identical recorded initial cube/bin/Panda
state, and pass six independent real action replays. Smoke is bounded to 20 ordered candidate scene
seeds and three expert attempts per task. The default smoke root is unique per invocation; an
explicitly supplied smoke root must not already exist.

`--target-full` is the separate authoritative-dataset gate. Without `--create-new-run`, it validates
an existing complete archive directly or resumes a compatible `in_progress` manifest. A missing
archive requires the explicit creation flag. That flag is accepted only when the selected root does
not exist; neither path passes `--overwrite`, and accepted scene groups remain immutable. A complete
archive therefore does not trigger another 360-episode collection. Configuration, schema, control
mode, or expert-fingerprint drift prevents reuse.

Full acceptance requires exactly 60 complete groups, 360 accepted successful episodes, exactly 60
episodes per TaskSpec, zero partial accepted groups, zero accepted expert failures, zero accepted
action-replay failures, zero checksum/schema failures, zero wrong-object or wrong-bin false
successes, and zero unclassified validation failures. The verifier independently inspects every
source pair and independently replays the exact ordered 360 manifest episodes with action plus state
audit. The replay report must contain one strictly parsed passed result per episode.

`verification.json` reports these dimensions independently:

```json
{
  "implementation_validated": true,
  "prior_target_gates_validated": false,
  "expert_collection_smoke_validated": false,
  "action_replay_validated": false,
  "full_dataset_validated": false,
  "physical_target_validated": false
}
```

`physical_acceptance` remains only a compatibility alias for `physical_target_validated`; it does
not replace the independent fields. Target validation is false on non-native-Linux hosts.

No 360-episode physical archive has been produced on the Windows review host. Synthetic HDF5,
transaction, corruption, serialization, and command tests do not constitute native Linux RTX 4090
acceptance.
