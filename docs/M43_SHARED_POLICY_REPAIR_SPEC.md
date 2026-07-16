# M4.3 shared-policy semantic-alignment specification

This document is the normative contract for LangMani milestone M4.3. M4.3 begins with a
zero-training semantic audit of the frozen M4/M4.2 controls. Only after that audit is complete and
immutable may a separate M4.3b change implement exactly one oracle-conditioned
`ACT-Mixed-FactorFiLM` policy. This M4.3a change does not implement, initialize, train, or select a
FactorFiLM model.

M4.3 preserves the M1 task, camera, observation/no-leakage, success, action-space, `num_envs=1`, and
`pd_joint_pos` contracts; the authoritative M3A archive; the immutable M3B export and scene-group
splits; train-only normalization; validation-only checkpoint selection; all historical M4/M4.1/M4.2
artifacts; and the sealed M4.2 final-schedule boundary. OneHot and TaskToken remain oracle discrete
task conditions, not language understanding.

## Starting evidence and interpretation

The completed M4 full experiment and M4.2 target-development evidence are inputs, not mutable
outputs. M4.2 selected execution horizon 10 and the existing explicit `project` runtime. Its
development comparison on the same 72 episodes produced 56/72 PerTask, 35/72 State-OneHot, and
15/72 TaskToken successes. State-OneHot changed its outputs substantially but still grasped a wrong
object 13 times; TaskToken grasped a wrong object 33 times and was rejected. Nonzero action distance
therefore proves only that a condition changes an output. It does not prove that the requested
object/bin semantics were used correctly.

M4.3a uses the immutable selected checkpoints for all six PerTask policies, State-OneHot, and the
rejected TaskToken. It audits two explicitly separated sources:

- all M3B validation scene groups; and
- all `m42_dev_v0` scene groups and their already completed rollout diagnostics.

It must never open M3B test, original M4 fresh seeds, or `m42_final_v0`. Validation and development
identities remain separate inside configuration, source records, summaries, and artifact checksums;
`combined` means an ordered report containing both named sections, never an unlabeled pool.

## Fixed-observation action-chunk audit

For every physical observation and requested canonical `TaskSpec`, the audit holds base-camera RGB
and `PandaPolicyStateV0[9]` fixed. It resets each policy and processor before deterministic
inference, evaluates all six PerTask policies on that one observation, and evaluates State-OneHot
and TaskToken under the requested condition. Policy outputs are compared after saved LeRobot
postprocessing has returned environment-semantic actions but before binary-gripper handling or the
M4.1 runtime projection. Offline comparison never steps the environment.

Every requested observation must have exactly six ordered PerTask reference chunks in canonical
red-left, red-right, green-left, green-right, blue-left, blue-right order. The audit stores stable
source/observation identities rather than images, large tensors, or machine-specific absolute
paths. RGB identity and 9D policy-state identity are independently checked so changing a task
condition cannot silently change the physical observation.

## ActionChunkDistanceV0

`ActionChunkDistanceV0` reports raw L2, action-range-normalized L2,
train-standard-deviation-normalized L2, and finite cosine distance for all action components, arm
components 0 through 6, and gripper component 7. Every family covers the first action, first five
actions, first locked-execution-horizon actions, and full 50-action ACT chunk.

The immutable primary retrieval metric is exactly:

```text
action-range-normalized L2
arm components only (indices 0..6)
first locked M4.2 execution-horizon actions (H=10 for the completed evidence)
```

The gripper is reported but excluded from primary retrieval. Candidate ties are resolved by the
canonical TaskSpec order, never filesystem or dictionary iteration order.

## Semantic retrieval and confusion

Full-task retrieval ranks the candidate chunk against all six PerTask reference chunks. Reports
include top-1, top-2, mean reciprocal rank, the signed margin between the correct reference and
nearest incorrect reference, a six-by-six confusion matrix, and per-TaskSpec results.

Object retrieval forms three centroids: red from red-left/red-right, green from
green-left/green-right, and blue from blue-left/blue-right. Bin retrieval forms left and right
centroids across all three objects. Conditional-bin retrieval compares only the requested object's
left and right PerTask references. Object/bin reports include accuracy, margin, confusion, and
per-label precision/recall. This separation distinguishes target-object confusion from destination
confusion.

When an audit source contains the required exact per-episode interaction fields, the contract can
record requested object, first object grasped/displaced, requested bin, first approached/entered
bin, final object/bin relationship, nearest PerTask chunk, and whether the initial semantic error
predicted the first grasp. It then produces the five declared first-interaction confusion matrices.

The immutable historical M4.2 development evidence does not retain enough exact fields to
reconstruct those records: it has aggregate wrong-object/post-grasp outcomes but not the first
object displaced, first bin approached, first object-entry bin, and final per-object/bin relation
for every episode. M4.3a therefore records first-interaction evidence and its confusions as
explicitly unavailable rather than inferring or fabricating them. It still binds and reports the
real aggregate M4.2 post-grasp failure distribution using the exhaustive taxonomy:

```text
never_grasped_target
wrong_object_grasped
target_grasped_not_lifted
lifted_not_transported
transported_not_descended
descended_not_released
released_outside_success_region
released_but_not_static
target_success_then_lost
timed_out_after_target_grasp
environment_failure
success
```

Conclusions are limited to `condition_not_used`,
`condition_changes_output_but_wrong_semantics`, `target_object_confusion`,
`destination_bin_confusion`, `shared_control_degradation_after_correct_selection`,
`post_grasp_execution_failure`, and `insufficient_evidence`. Output-change magnitude, semantic
retrieval, first interaction, and post-grasp control are reported separately.

## Immutable evidence lifecycle

The audit configuration and all portable records use canonical JSON plus SHA-256. A semantic audit
fingerprint binds the schema, selected audit sources, Git commit, M3B export fingerprint, all six
PerTask checkpoint fingerprints, State-OneHot checkpoint fingerprint, rejected TaskToken checkpoint
fingerprint, locked horizon and gripper runtime, ordered observation/source identities, primary
distance configuration, and the exact policy/runtime provenance required by the implementation.
Paths, hostname, timestamps, temporary directories, and filesystem enumeration order are excluded.

Generation uses a fingerprint-owned sibling staging directory. The producer writes configuration,
source identities, complete machine-readable metrics/confusions/classifications, human-readable
summary, artifact checksums, and a completion marker; then independently validates them before one
atomic promotion. The completion marker is written last. A completed fingerprint directory is
immutable: byte-identical reuse is allowed, while missing, extra, linked, partial, checksum-mismatched,
or semantically incompatible content fails. Staging cleanup is allowed only for a matching owned
incomplete fingerprint.

The portable manifest contains relative artifact references only. A command report may include
resolved local input/output paths for operator diagnostics, but those paths never participate in a
semantic fingerprint or portable manifest.

The on-disk layout is fixed:

```text
<output-root>/
└── evidence/
    └── <config-sha256-hex>/
        ├── owner.json
        ├── config.json
        ├── summary.md
        ├── scopes/
        │   ├── m3b_validation.json                         # when requested
        │   ├── m42_dev_v0.json                             # when requested
        │   └── combined_validation_and_m42_dev_v0.json    # combined only
        ├── manifest.json
        └── complete.json
```

The versioned schemas are `langmani-m43-semantic-audit-config-v0`,
`langmani-m43-semantic-audit-scope-v0`, `langmani-m43-semantic-audit-manifest-v0`, and
`langmani-m43-semantic-audit-complete-v0`. A combined audit must contain the validation,
development, and explicit combined scope files; a combined file never substitutes for either source
artifact. Historical aggregate rollout evidence and exact first-interaction records are separate
fields. For current M4.2 evidence the former is available and the latter is explicitly unavailable,
so the producer cannot fabricate per-observation interactions from summary counts.

## Command boundary

The project-owned audit command is:

```bash
python scripts/audit_act_semantics.py \
  --mode validation \
  --device cpu \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --m4-checkpoint-root outputs/models/act \
  --task-token-checkpoint-root outputs/models/act-task-token \
  --m42-diagnostics-root outputs/diagnostics/m42 \
  --runtime-selection outputs/diagnostics/m42/runtime_ablation/runtime_selection.json \
  --output-root outputs/diagnostics/m43/semantic-audit \
  --report outputs/diagnostics/m43/audit-command.json \
  --dry-run
```

`--mode` is exactly `validation`, `m42_dev_v0`, or `combined`. The first audits M3B validation;
the second audits `m42_dev_v0`; the third emits both independently labeled sections. All input and
checkpoint roots are explicit. `--dry-run` validates paths, identities, schemas, access boundaries,
and the planned source ordering without model inference or evidence promotion. Infrastructure or
contract failure exits nonzero. Output paths must not overlap source, Git, dataset, checkpoint, or
historical diagnostics roots and may not traverse links or junctions.

The restricted input gate always verifies the runtime lock, complete TaskToken validation queue,
all 20 content-bound validation artifacts, and the validation-only ranking that selected the
TaskToken checkpoint. Those files are checkpoint provenance, not development rollout evidence.
In `validation` mode it does not open `development_comparison.json` or
`development_complete.json`; only `m42_dev_v0` and `combined` load those two rollout authorities.
The unused historical M4 diagnostics root is intentionally not a command argument: selected M4
checkpoint provenance comes from each run's allowlisted manifest and validation selection lock.

The non-target implementation verifier is:

```bash
python environment/verify_m43.py
```

It checks imports and M4.3a contracts, distance/retrieval/confusion calculations, portable
serialization, immutable lifecycle behavior, audit-CLI dry-run, forbidden schedule identities, and
truthful physical flags. It does not import or require FactorFiLM and does not claim real checkpoint,
CUDA, simulator, semantic-audit, or physical validation.

## Verification state and M4.3b gate

M4.3a verification distinguishes structural implementation from real inference. A successful local
non-target run may set `semantic_audit_implementation_validated=true` while
`semantic_audit_completed=false`, `factor_film_training_completed=false`,
`final_schedule_accessed=false`, `smolvla_go=false`, and `physical_target_validated=false`.
No implementation fixture may set a semantic quality flag.

FactorFiLM implementation may begin only after all of the following are true:

1. this M4.3a code is formatted, lint-clean, CPU-safe-test clean, built, committed, and pushed as a
   separate clean Git boundary;
2. the real frozen checkpoints and completed M3B/M4/M4.2 inputs pass fingerprint validation;
3. both validation and `m42_dev_v0` semantic audits complete through immutable promotion;
4. the combined evidence validates all source identities, retrieval/confusion outputs, summaries,
   checksums, and completion marker; and
5. the verifier truthfully reports `semantic_audit_completed=true` while final-schedule access and
   SmolVLA remain false.

If the GPU target is unavailable, structural implementation may be completed and pushed, but real
semantic inference and the M4.3b permission gate remain pending. A later target-development command
must use a clean tracked commit and still may not access `m42_final_v0`.

## Explicitly out of scope for M4.3a

M4.3a does not implement or initialize FactorFiLM; train a model; select a checkpoint; alter M3B;
access M3B test, M4 fresh, or `m42_final_v0`; run M2 actions; change M1 success or bounds; add text
encoders, language claims, new data, architecture sweeps, RL, DAgger, or SmolVLA; or authorize a
final benchmark from audit completion alone.

At the current local structural stage no real semantic retrieval metric is recorded in this
document. Results must be copied only from a validated immutable target audit after it exists.

## Local structural validation status (2026-07-16)

The final M4.3a-focused selection reported 147 passed and one Windows symlink-creation permission
skip. The complete CPU-safe suite reported 840 passed, 14 skipped, and 15 hardware/rendering tests
deselected. Ruff formatting and lint checks, `pip check`, sdist/wheel build, and the non-target
`environment/verify_m43.py` command passed. The verifier exercises a combined CLI dry-run and the
immutable promotion/corruption-rejection lifecycle while keeping semantic completion, physical
validation, real checkpoint reload, test/fresh/final access, FactorFiLM, and SmolVLA claims false.

These results validate local structure only. Pinocchio absence, Windows symlink privilege, and the
optional pytest-cache write-permission warning remain nonblocking because no M4.3a structural test
depends on planner kinematics, creating a symlink, or writing that cache. Real frozen-checkpoint
semantic inference and GPU target availability remain pending;
`semantic_audit_completed=false`, `physical_target_validated=false`, and M4.3b remains blocked.
