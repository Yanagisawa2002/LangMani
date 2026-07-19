# LangMani contributor instructions

## Purpose

LangMani supports language-conditioned robotic manipulation in ManiSkill. M0 through M3B, M4 full,
M4.1 explicit action-bound target smoke, and M4.2 target-development are complete. M4 full is
experimentally and physically validated, but `baseline_quality_validated=false`; the M4.2 TaskToken
candidate was rejected. The real M4.3a semantic-alignment audit and M4.3b FactorFiLM
target-development are complete. FactorFiLM passed experiment/physical verification but failed its
quality gate, so the shared-ACT architecture search is closed and `m42_final_v0` remains sealed.
M5A modular language-to-TaskSpec routing over the six frozen PerTask ACT controllers is the active
milestone. Its implementation now uses separately authorized fixture, tiny-overfit, authoritative
single-seed pilot/resume, language-development, 1-scene smoke, 3-scene screen, and 6-scene full
development stages. The original step-29 classifier pilot and its exact D-061 recovery continuation
completed. Validation selected epoch 4/step 116, but the unchanged full quality gate failed on
rejection generalization. M5A.1 completed the validation-only, no-training post-hoc rejection
analysis and independently verified its immutable evidence. The best safe decoder still failed the
unchanged rejection-reason thresholds, so the classifier is frozen as a rejected offline baseline
and no classifier runtime exists. M5A.2 completed the one authorized Qwen3-1.7B offline comparison;
its train smoke passed, its 300-example validation gate failed, and language development stayed
sealed. M5A.3 completed the bounded direct Qwen3-4B comparison and rejected it as a dispatch
candidate. M5A.4 completed its immutable 20-example smoke with 17/20 exact RouterStatus decisions
and zero unsafe false routes. M5A.4.1 completed its bounded offline taxonomy/safety amendment and
selected the unchanged `NeuroSymbolicRouterV0` on untouched language development. Its separately
authorized one-scene dispatch smoke also completed: all six TaskSpecs routed to the matching frozen
PerTask controller, the selected-router rejection probe did zero lookup/reset/step work, and the
independent physical verifier passed. Learned and Oracle control each succeeded on 5/6 episodes
with one timeout, so dispatch acceptance and observed controller quality remain separate.
Three-scene/full control, the sealed final benchmark, and SmolVLA have not started.
The three-scene paired-screen orchestration and independent verifier are implemented but have not
yet produced target evidence. They require the independently verified one-scene parent, execute
Oracle then NeuroSymbolic for each of 18 scene-task pairs, and atomically promote only the bounded
three-scene evidence. Full development remains a separate authorization.

M3A remains the sole raw authority and M3B remains the sole derived dataset. M4 must keep
`num_envs=1`, `pd_joint_pos`, the M1 camera/no-leakage and success contracts, exact M3B scene-level
splits, train-only normalization, validation-only checkpoint selection, and a locked test split.
Standard ACT is not language conditioned; task OneHot and TaskToken are oracle command conditions.
M4.2 must not change M1 bounds/success, M3A/M3B data, train-only statistics, historical M4
weights/loss, or existing checkpoint fingerprints. Do not retrain old M4 controls, materialize or
execute `m42_final_v0` during development, start SmolVLA, add new robot data/tasks, invoke M2 during
rollout, or hide projection as clipping. Binary gripper handling is permitted only as the explicit versioned
M4.2 component-7 ablation defined in `docs/M42_ORACLE_CONTROL_SPEC.md`. The immutable M4.3a audit
used only frozen PerTask, State-OneHot, and rejected TaskToken checkpoints on M3B validation and
`m42_dev_v0`, with those sources kept separate and M3B test, M4 fresh, and `m42_final_v0` excluded.
M4.3b must keep `PandaPolicyStateV0` at nine dimensions and use only the canonical target-object
visual FiLM path plus destination-bin state/context FiLM path. It must not silently fall back to
State-OneHot/TaskToken, train from semantic-audit observations, select on development data, or
authorize the final schedule or SmolVLA.

M5A must use canonical M1 TaskSpecs and keep natural language out of observations and per-step
info. Corpus template families are not M1 instruction-template IDs. Classifier gradients use train
only; validation alone selects/calibrates; development only measures generalization. One explicitly
pinned local LLM may run through strict JSON validation and at most one repair; no hosted API or
fallback model is permitted. Rejection must contain no executable TaskSpec and return before
controller lookup, policy/environment reset, or `env.step`. The environment reset/evaluation task
remains the oracle schedule task while the predicted TaskSpec selects one frozen controller.
Language/control final locks, M3B test content, historical fresh evaluation, `m42_final_v0`, and
SmolVLA remain inaccessible during M5A target-development.

M5A classifier development uses exactly seed 0, at most five epochs/145 optimization steps, a
step-29 pilot checkpoint, validation-only patience 1, and only `pilot`, `latest`, and
`validation_best` checkpoint roles. A promoted pilot resumes the same run fingerprint and complete
optimizer/scheduler/processor/RNG state; it never restarts from step zero. The sole exception is
the D-061 recovery mode for the exact rejected pilot/run/checkpoint fingerprints. It starts at step
30, preserves the same five-epoch and patience-one bounds, uses the frozen validation-only
seven-key ranking, and does not change ordinary `--target-resume` semantics. Physical candidates are
progressively filtered. The disjoint 1/3/6-scene stages cost at most 18/54/72 episodes and at most
144 before final. The separately authorized 12-scene final costs 144 episodes and is never started
automatically.

## Directory ownership

| Path | Ownership |
| --- | --- |
| `src/langmani/` | Project-owned environment and expert runtime packages plus future subsystem interfaces |
| `src/langmani/collection/` | M3A recorder integration, collection orchestration, replay, manifests, and inspection |
| `src/langmani/datasets/` | M3A raw schemas, stable IDs, schedules, and native archive validation |
| `src/langmani/datasets/lerobot_*.py` | M3B source gate, contracts, writer/export, and validation |
| `src/langmani/policies/` | M4 ACT contracts, completed-data views, conditioning, training, checkpoints, rollout, and evaluation |
| `src/langmani/language/` | M5A corpus, routers, strict schemas, frozen-controller registry, dispatch, evaluation, and attribution |
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

# M4.3 semantic-audit and FactorFiLM structural commands
python scripts/audit_act_semantics.py --help
python scripts/train_act_factor_film.py --help
python scripts/evaluate_act_factor_film.py --help
python scripts/train_act_factor_film.py --dry-run --fixture-contract
python scripts/train_act_factor_film.py --fixture
python environment/verify_m43.py
python environment/verify_m43b.py --help

# M5A modular language-routing commands
python scripts/build_language_corpus.py --help
python scripts/train_text_router.py --help
python scripts/evaluate_language_routers.py --help
python scripts/evaluate_qwen_scale_escalation.py --help
python scripts/evaluate_neuro_symbolic_safety_gate.py --help
python scripts/run_language_control.py --help
python environment/verify_m5a.py
python environment/verify_m5a3.py
python environment/verify_m5a41.py
python environment/verify_m5a_dispatch.py
python environment/verify_m5a_three_scene.py
python scripts/analyze_classifier_rejection.py --help

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

# On the server, fetch and check out the exact clean compatibility producer
# 0088e2937556c123c37c2dbe69f73301b1eebfd0 before this authorized M4.3b target training.
CUDA_VISIBLE_DEVICES=0 python scripts/train_act_factor_film.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act-factor-film \
  --evidence-root outputs/diagnostics/m43/semantic-audit/evidence/930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2 \
  --device cuda \
  --report outputs/diagnostics/m43/factor-film-target-development.json \
  --clean-staging \
  --target-development

# For post-training work, check out the current clean authorization commit containing evaluator/
# verifier implementation 1bacb66d2a6f7c3f2d18d6f65ad7865af9a12cd6. These stages are resumable,
# write outside the immutable training run, and never open test/fresh/final schedules.
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_act_factor_film.py \
  --target-development \
  --training-git-commit 0088e2937556c123c37c2dbe69f73301b1eebfd0
CUDA_VISIBLE_DEVICES=0 python environment/verify_m43b.py \
  --training-git-commit 0088e2937556c123c37c2dbe69f73301b1eebfd0 \
  --training-run-root outputs/models/act-factor-film/<run-fingerprint> \
  --evaluation-evidence-root outputs/diagnostics/m43/<evaluation-evidence-root> \
  --structural-verification outputs/diagnostics/m43/target-development-preflight-verification/verification.json

# M5A begins with one separately authorized fixture. Do not chain later stages in this shell.
CUDA_VISIBLE_DEVICES=0 python scripts/train_text_router.py --fixture \
  --output-root outputs/models/text-router \
  --report outputs/diagnostics/m5a/stages/classifier-fixture.json
python environment/verify_m5a.py --verify-stage classifier_fixture \
  --stage-report outputs/diagnostics/m5a/stages/classifier-fixture.json

# Subsequent separately authorized classifier modes are --tiny-overfit, --target-pilot, then
# --target-resume. The pilot and resume share one authoritative run fingerprint. Physical control
# uses run_language_control.py --stage one_scene_control_smoke, then
# three_scene_control_screen, then full_control_development; later stages consume the preceding
# report. The retired verify_m5a.py --target-development interface must fail and start no work.

# One post-pilot amendment authorizes only the exact rejected run documented by D-061.
CUDA_VISIBLE_DEVICES=0 python scripts/train_text_router.py \
  --target-pilot-recovery-resume \
  --recovery-run-fingerprint sha256:9e3ac659fa2b695c843650df35e3779741d94b3dd70b2aec52a429bc4b2edf49 \
  --recovery-pilot-checkpoint-sha256 sha256:a892c2b73c87884b2b2acf843d22ed9318e6b661640eea6a4964ff8d739b5758 \
  --recovery-maximum-total-epochs 5 \
  --recovery-early-stopping-patience 1
python environment/verify_m5a.py --verify-stage classifier_recovery_training \
  --stage-report outputs/diagnostics/m5a/stages/classifier-recovery-resume.json
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
- M4.3b condition preprocessing must use stable TaskSpec metadata, never instruction parsing. The
  object order is red/green/blue and the bin order is left/right. Object FiLM may modify only the
  ResNet-18 feature map; bin FiLM may modify only the encoded 9D-state token. No combined task token
  or state-appended task feature is permitted.
- FactorFiLM architecture and structural evidence remain bound to
  `8ee0f1babf36b91d1ee2a39701e4a6db6003b660`; target training is bound to the exact clean
  compatibility producer `0088e2937556c123c37c2dbe69f73301b1eebfd0`. Its immutable run and
  checkpoint identities must never be rewritten by a later evaluator commit. Only the exact M3B
  train view may train it, only M3B validation may select a checkpoint, and fixture forward/backward
  or save/reload evidence must never set training, checkpoint, rollout, or physical-validation flags.
- Preserve raw, binary-transformed, projected, and executed action evidence as separate contracts.
  OneHot and TaskToken must always be described as oracle conditioning, never language understanding.
- M4.3b target development is one seed-0 comparison only. Its model/optimization fingerprints,
  H=10/project runtime, exact 20-checkpoint queue, and validation-only selection are immutable.
  Reports, outputs, staging, and resume paths must reject protected or linked content before writes;
  only the latest declared checkpoint or sole next atomically promoted orphan is recoverable.
- Source changes for target-development are authored, tested, committed, and pushed from the local
  repository. The GPU server may only fetch/check out/pull those commits; never hot-patch source on
  the server. Generated checkpoints/evidence remain server outputs and are never committed.
- M5A language examples are split by template family and near-duplicate structural identity.
  Train, validation, development, and final families must remain disjoint; final command texts and
  final control episodes cannot be materialized during target-development.
- M5A routeable decisions use exactly the six canonical `canonical_v0` TaskSpecs. Rejected
  decisions contain no TaskSpec and cannot dispatch or step. Oracle reset TaskSpec, predicted
  TaskSpec, selected controller, and active EpisodeSpec remain separately recorded.
- M5A.4.1 keeps all three rejection statuses physically equivalent no-dispatch outcomes while
  preserving exact status/reason metrics as a separate diagnostic contract. Unsafe false routes,
  false rejections, malformed decisions, or executable rejection fields remain hard failures. It
  cannot relabel corpus examples or modify the frozen M5A.4 model/prompt/schema/parser/arbiter.
- The selected M5A.4.1 dispatch source is checksum-bound to its immutable router lock and remains
  local-cache-only. Its first physical authorization is exactly `one_scene_control_smoke`; it may
  bind metadata for the six frozen PerTask entries but cannot load a controller before a route.
  Independent verification must rehash all Oracle and learned atoms and the real rejection no-op
  probe while reporting controller quality separately from correct stage execution.
- M5A.1 may read only the frozen selected classifier checkpoint plus train/validation corpus
  contracts. It evaluates exactly the four locked decoder candidates and finite threshold grid.
  It cannot construct or step an optimizer, change checkpoint bytes/global step, start another
  seed/run, open development/final/control sources, or publish a runtime unless every original
  quality threshold passes conjunctively.
- M5A uses only the six frozen selected PerTask ACT controllers at H=10 with explicit `project`
  action handling. It cannot reselect/retrain/blend controls, pass confidence into ACT, use a shared
  ACT as the deployed controller, call M2, or collapse raw/binary-transformed/projected/executed
  action evidence. Target development must independently validate a real rejection no-op probe.
- Factorized text-classifier gradients use train only; checkpoint selection, temperature, and
  threshold use validation only. Development uses one seed, no encoder sweep, an explicit pilot
  promotion gate, same-run resume, patience-one early stopping, and bounded checkpoint roles.
  Fixture/tiny/pilot/training completion and promotion remain separate evidence. The single local
  LLM requires explicit pinned model/tokenizer
  revisions, deterministic generation, strict structured validation, and at most one repair. Cloud
  APIs, model/encoder sweeps, prompt edits after development starts, and fabricated confidence are
  prohibited.
- D-061 is a post-pilot amendment and must never be presented as part of the original promotion
  protocol. Recovery is allowlisted to its exact run and pilot SHA-256, performs the 15-item
  read-only audit, resumes at step 30, retains three checkpoint roles, and calibrates only after
  the unchanged full-quality gate passes. A valid quality rejection still has completed/selected
  flags true, calibration false, and no authorization for language development.
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
M4.3a real validation and `m42_dev_v0` inputs passed immutable promotion with
`semantic_audit_completed=true`. M4.3b local structural completion validates only architecture,
processor, fixture optimization, and local checkpoint reload. Target-development acceptance
additionally requires the exact 100,000-step/20-checkpoint run, all 20 x 36 validation rollouts,
immutable seven-key selection, fresh-process `[50,8]` reload equivalence at `atol=rtol=1e-6`, the
paired 3 x 72 `m42_dev_v0` comparison, semantic/first-interaction analysis, and an independent
read-only verifier. Experiment/physical completion and the 16-condition development quality gate
remain separate. Test, historical fresh, `m42_final_v0`, and SmolVLA stay inaccessible throughout.
M5A portable completion requires deterministic exact-count corpus generation, family/near-duplicate
split isolation, all four schedule locks, three router fixtures, strict rejection/JSON behavior, a
portable six-controller registry, zero-dispatch rejection, failure attribution, CPU-safe tests, and
truthful non-target flags. M5A target work is accepted one report at a time: fixture, tiny overfit,
pilot, same-run training, language development, 1-scene smoke, 3-scene screen, and 6-scene full
development. Failed candidates do not consume later budgets. Oracle remains in every physical
stage, RuleRouter remains an offline baseline, and only one screen-selected learned router reaches
the 36-pair full stage. Correct stage execution and next-stage promotion remain separate. Every
development stage must leave language/control final, M3B test content, `m42_final_v0`, and SmolVLA
unaccessed and must not run final automatically.
M5A.1 completion additionally requires a checksum-owned per-example validation diagnostic,
four-way and binary confusion analyses, the predeclared fixed-grid search, a decisive promoted or
frozen conclusion, unchanged checkpoint/weights/step, zero optimizer activity, and independent
stage verification. Correct analysis completion remains separate from classifier quality.
The selected M5A.4.1 one-scene control smoke additionally requires a clean implementation commit,
the exact immutable language evidence, metadata-only six-controller binding, one real rejection
probe with zero lookup/reset/step, six learned routes matching their oracle TaskSpecs and controller
IDs, six Oracle ceiling episodes, preserved raw/projected/executed action evidence, and an
independent read-only verifier. This stage must leave all final/test/`m42_final_v0`/SmolVLA flags
false; correct execution and the observed six-task control success count remain separate claims.
The three-scene screen additionally requires 18 exact initial-state pairs, 36 per-episode policy
resets, the fixed six-category rejection no-op set, paired Oracle/NeuroSymbolic controller identity,
raw action-metric recomputation, atomic compact evidence, and a fresh-process verifier. Its
`passed` flag validates execution/evidence; `three_scene_control_screen_passed` alone controls the
separate full-development authorization, which the screen never executes automatically.
