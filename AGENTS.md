# LangMani contributor instructions

## Purpose

LangMani supports language-conditioned robotic manipulation in ManiSkill. M0 through M3B, M4 full,
M4.1 explicit action-bound target smoke, and M4.2 target-development are complete. M4 full is
experimentally and physically validated, but `baseline_quality_validated=false`; the M4.2 TaskToken
candidate was rejected. M4.3a semantic alignment audit is active. The sealed M4.2 final benchmark,
M4.3b FactorFiLM, and M5 have not started.

M3A remains the sole raw authority and M3B remains the sole derived dataset. M4 must keep
`num_envs=1`, `pd_joint_pos`, the M1 camera/no-leakage and success contracts, exact M3B scene-level
splits, train-only normalization, validation-only checkpoint selection, and a locked test split.
Standard ACT is not language conditioned; task OneHot and TaskToken are oracle command conditions.
M4.2 must not change M1 bounds/success, M3A/M3B data, train-only statistics, historical M4
weights/loss, or existing checkpoint fingerprints. Do not retrain old M4 controls, materialize or
execute `m42_final_v0` during development, start M5, add new data/tasks, invoke M2 during rollout,
or hide projection as clipping. Binary gripper handling is permitted only as the explicit versioned
M4.2 component-7 ablation defined in `docs/M42_ORACLE_CONTROL_SPEC.md`. M4.3a may use only frozen
PerTask, State-OneHot, and rejected TaskToken checkpoints on M3B validation and `m42_dev_v0`.
It must keep those sources separate, must not access M3B test, M4 fresh, or `m42_final_v0`, and must
not implement or initialize FactorFiLM until a real immutable semantic audit is complete.

## Directory ownership

| Path | Ownership |
| --- | --- |
| `src/langmani/` | Project-owned environment and expert runtime packages plus future subsystem interfaces |
| `src/langmani/collection/` | M3A recorder integration, collection orchestration, replay, manifests, and inspection |
| `src/langmani/datasets/` | M3A raw schemas, stable IDs, schedules, and native archive validation |
| `src/langmani/datasets/lerobot_*.py` | M3B source gate, contracts, writer/export, and validation |
| `src/langmani/policies/` | M4 ACT contracts, completed-data views, conditioning, training, checkpoints, rollout, and evaluation |
| `scripts/` | M3B data commands plus M4 train, evaluate, compare, and checkpoint-inspection commands |
| `environment/` | Environment declaration, diagnostics, expert rollout, benchmark, and verification commands |
| `tests/unit/` | Fast tests of project-owned behavior |
| `tests/integration/` | LeRobot and ACT cross-package fixture/target integration |
| `tests/smoke/` | Cross-package, simulator, GPU, and rendering smoke tests |
| `docs/` | Architecture boundaries and append-only decision rationale |
| `outputs/` | Generated diagnostics and experiments; never source-controlled |

Future generated datasets, downloaded simulator assets, videos, caches, and checkpoints belong
outside source-controlled package paths.

## Required commands

Run from the repository root in the `langmani` environment:

```bash
# Format and lint
ruff format .
ruff check .

# CPU-safe tests
pytest -m "not gpu and not rendering"

# Build
python -m build

# CPU-safe installation diagnostics
python environment/verify_install.py

# M1 contract diagnostics (runs simulation as required only on native Linux)
python environment/verify_m1.py

# M2 structural diagnostics everywhere; CPU expert execution on native Linux
python environment/verify_m2.py

# Native Linux planner-side ABI gate; LANGMANI_PLANNER_PYTHON is configured per README.
"$LANGMANI_PLANNER_PYTHON" environment/verify_planner_runtime.py

# M3A structural diagnostics everywhere
python environment/verify_m3a.py

# M3B structural contracts plus a generated-array LeRobot/PyAV fixture
python environment/verify_m3b.py

# M4 contracts plus a real CPU ACT fixture forward/backward and local reload
python environment/verify_m4.py

# One expert rollout and the six-combination benchmark (native Linux target runtime)
python environment/run_expert.py
python environment/benchmark_expert.py

# Resumable raw collection, offline inspection, and independent action replay
python environment/collect_raw_demos.py
python environment/inspect_raw_demos.py
python environment/replay_raw_demos.py

# Derived-data commands
python scripts/export_lerobot_dataset.py --help
python scripts/validate_lerobot_dataset.py --help
python scripts/inspect_lerobot_episode.py --help

# ACT baseline commands
python scripts/train_act.py --help
python scripts/evaluate_act.py --help
python scripts/compare_act_baselines.py --help
python scripts/inspect_act_checkpoint.py --help
python scripts/benchmark_act_evaluation_workers.py --help

# M4.2 oracle-control commands
python scripts/run_m42_runtime_ablation.py --help
python scripts/train_act_task_token.py --help
python scripts/evaluate_m42.py --help
python environment/verify_m42.py

# M4.3a zero-training semantic-audit commands
python scripts/audit_act_semantics.py --help
python environment/verify_m43.py

# Native Linux NVIDIA/Vulkan acceptance gate. The M2 command invokes the
# M0 installation and M1 environment target gates in the main runtime first,
# then delegates planner construction and expert rollouts to the side runtime.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m2.py --target

# Fast native-target chain: ordered M0/M1/M2 gate, one six-task group, and six replays.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py --target-smoke

# First authoritative run: explicitly create, then collect/inspect/replay all 360 episodes.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py --target-full \
  --dataset-root outputs/datasets/m3a/langmani-pick-place-raw-v1 --create-new-run

# Later calls validate an existing complete run or resume a compatible interrupted run.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3a.py --target-full \
  --dataset-root outputs/datasets/m3a/langmani-pick-place-raw-v1

# One real six-task M3B export after the ordered prior target gates.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3b.py --target-smoke

# Full 60-group/360-episode M3B export and independent validation.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3b.py --target-full \
  --source-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1

# M4.1 target smoke: validate prior evidence, reuse existing checkpoints,
# reproduce strict rejection, then execute explicit projected rollouts.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py \
  --target-smoke --action-bound-mode project

# M4 full experiment: completed 360-episode M3B input, all eight ACT runs,
# validation-only selection, locked test, fresh seeds, and comparison report.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m4.py --target-full \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act

# M4.2 development: horizon/gripper ablations, one TaskToken run, validation selection,
# and the 12-scene development benchmark. This must stop before m42_final_v0.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m42.py --target-development

# Separate future authorization only after all development selections are immutable.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m42.py --target-final
```

The exact environment creation commands are maintained in `README.md`.

## Change rules

- Add or update tests for every behavioral change.
- Update `docs/DECISIONS.md` whenever a dependency, public interface, subsystem boundary, or
  environment assumption changes.
- Keep project-owned Python statically annotated and Ruff-clean.
- Do not silently catch dependency, CUDA, Vulkan, rendering, or simulator errors. Diagnostics may
  continue after an error only when the failure is printed and contributes to the final status.
- Do not copy or vendor ManiSkill or LeRobot source.
- Do not commit generated data, downloaded assets, videos, model checkpoints, caches, experiment
  outputs, or fabricated results.
- Record the full Git commit in every M4 training/evaluation run. Full and tiny-overfit evidence
  requires a clean worktree; only an explicitly labeled development run may record a dirty-tree
  override, and that run is never final evidence.
- M4 normalization must be computed from its exact train episode view. Never use M3B whole-dataset
  `meta.stats`, expose test episodes to training/selection, or reopen M3A in normal training.
- M4.2 runtime selection may use only `m42_dev_v0`; TaskToken checkpoint selection may use only the
  M3B validation split. Loading the final lock for audit is allowed, but development must never
  materialize, render, or reset an `m42_final_v0` episode.
- M4.3a must compare environment-semantic policy chunks before runtime projection with
  `ActionChunkDistanceV0`. Its primary retrieval metric is action-range-normalized, arm-only L2 over
  the locked execution horizon. Nonzero distance is not semantic correctness.
- M4.3a completed evidence is fingerprint-owned and immutable. Validation and `m42_dev_v0` records
  must remain explicitly separated, and no test/fresh/final identity may appear in the evidence.
- Historical M4.2 evidence may supply its real aggregate post-grasp distribution, but it lacks the
  exact per-episode fields for M4.3 first-interaction confusion. Keep that evidence explicitly
  unavailable; never reconstruct first interactions from aggregate wrong-object counts.
- Preserve raw, binary-transformed, projected, and executed action evidence as separate contracts.
  OneHot and TaskToken must always be described as oracle conditioning, never language understanding.
- Do not describe a skipped, metadata-only, structural-only, or CPU-only check as physical GPU or
  rendering validation.

## Definition of done

A change is done when its implementation remains within the active milestone, project-owned code
is formatted and lint-clean, relevant CPU-safe tests pass, hardware-only tests are explicitly
marked, documentation and decision records match the implementation, and no generated artifact is
staged. Any change that claims physical GPU/rendering validation or a newly verified dependency
combination must first pass the native Linux target command and record exact observed versions and
hardware results in `docs/DECISIONS.md`. M2 target acceptance additionally requires the M0, M1, and
M2 target commands to pass in the documented order. M3A smoke acceptance requires that ordered gate
before one fresh complete group and six independent action replays. Full-dataset acceptance is a
separate command requiring exactly 60 complete groups (360 accepted episodes, 60 per TaskSpec),
inspection of every shard/checksum, and independent replay of every action sequence. When that
machine is unavailable, any newly requested physical gate remains explicitly pending; previously
accepted immutable M2/M3A/M3B/M4 evidence is not downgraded or re-created. M3B target acceptance
additionally
requires a content-bound M3A target report, real state-restoration rendering, exact scene-level
splits, all videos decoded, source/action/state alignment, a local LeRobot reload, and a DataLoader
batch. M4 implementation or fixture checks are not model-quality evidence. Target smoke requires
the complete M0-through-M3B smoke chain before real CUDA forward/backward, tiny-overfit,
checkpoint/processor reload, and learned-policy M1 rollouts. Full acceptance requires a completed
real 360-episode M3B dataset, six per-task and two mixed runs, validation-only checkpoint selection,
locked test evaluation, the fixed 180-episode fresh-seed benchmark, and provenance-complete reports.
`full_experiment_validated`, `baseline_quality_validated`, and `physical_target_validated` remain
independent flags. M4.2 development additionally requires both source-controlled schedule locks,
the exact 125-seed exclusion audit, immutable horizon and gripper selections from development only,
exactly one TaskToken training run, M3B-validation-only checkpoint selection, and a completed
development benchmark. It must leave `final_benchmark_completed=false` and no SmolVLA go/no-go
claim. M4.2 final is a later separate command requiring immutable selections, clean Git, sealed
schedule authorization, paired PerTask/State-OneHot/TaskToken evaluation, and an explicit go/no-go;
it never starts M5 automatically.
M4.3a local structural completion may validate implementation without claiming a completed semantic
audit or physical execution. M4.3b remains blocked until real validation and `m42_dev_v0` audit
inputs pass immutable promotion with `semantic_audit_completed=true`; audit completion alone does
not authorize a final schedule or SmolVLA.
