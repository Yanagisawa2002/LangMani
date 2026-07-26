# LangMani contributor instructions

## Purpose

LangMani supports language-conditioned robotic manipulation in ManiSkill. M0 through M3B, M4 full,
M4.1 explicit action-bound target smoke, and M4.2 target-development are complete. M4 full is
experimentally and physically validated, but `baseline_quality_validated=false`; the M4.2 TaskToken
candidate was rejected. The real M4.3a semantic-alignment audit and M4.3b FactorFiLM
target-development are complete. FactorFiLM passed experiment/physical verification but failed its
quality gate, so the shared-ACT architecture search is closed and `m42_final_v0` remains sealed.
M5A modular language-to-TaskSpec routing over the six frozen PerTask ACT controllers is the latest
completed milestone. Its implementation uses separately authorized fixture, tiny-overfit,
authoritative single-seed pilot/resume, language-development, 1-scene smoke, 3-scene screen, and
6-scene full
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
The separately authorized three-scene paired screen is also complete. Across three unseen
development seeds, Oracle and NeuroSymbolic each succeeded on 15/18 episodes with the same three
paired timeouts. All 18 routes and controller identities matched, all 18 initial physical states
were exactly paired, the fixed six rejection probes did zero lookup/reset/step work, and the
independent physical verifier passed. The separately authorized six-scene full control development
is also complete: all 36 routes and exact reset-state pairs matched; Oracle and NeuroSymbolic each
succeeded on 25/36 with the same 11 timeouts; all nine rejection probes did zero runtime work; and
the independent physical verifier passed. The quality gate created a positive, fingerprint-only
`M5AFinalAuthorizationV0`. The separately authorized sealed final benchmark is now complete and
independently physically verified. Its 600-example language and 72-pair control evidence passed the
pipeline contract, but the final quality gate failed: NeuroSymbolic routed 61/72 tasks, safely
rejected 11 routeable commands, and achieved 46/72 end-to-end successes versus the Oracle ceiling's
55/72. All ten rejection probes performed zero runtime work, and there were no wrong-object or
wrong-bin routes, arm projections, infrastructure failures, M2 calls, or prohibited-source access.
`final_pipeline_validated=true`, `physical_target_validated=true`,
`final_quality_gate_passed=false`, and `smolvla_go=false`; SmolVLA has not started.

M6 is the completed release-and-portfolio freeze. It may index and visualize existing immutable v1
evidence, generate untracked reports/videos, and improve release documentation. It must not modify
v1 runtime behavior, datasets, models, checkpoints, schedules, gates, or experimental evidence.
The release must preserve `final_pipeline_validated=true`, `physical_target_validated=true`,
`final_quality_gate_passed=false`, and `smolvla_go=false` as separate claims.

LangMani 2.0 Phase 1 begins after the immutable `v1.0.0` release. It may validate the v1 release
manifest, audit existing artifacts, wrap one accepted PerTask ACT checkpoint behind a generic
policy/action-chunk contract, define one `pick_and_place` skill family with six task instances, and
run a new language-independent smoke evaluation. It must not rewrite v1 evidence, retrain a model,
introduce a new policy architecture or router, or describe structural/no-GPU checks as real policy
rollouts. Generated v2 evaluation evidence remains under `outputs/` and is not source-controlled.
Phase 1 implementation, release validation, canonical artifact recovery, and native target
acceptance are complete. After the bounded Docker/Windows and one invalid GLX launch attempt were
kept as infrastructure diagnostics, the accepted EGL/Vulkan run at `aa599dd` completed seeds
41001/41002/41003 with 3/3 successes, 40 real ACT queries, and 392 real environment steps. The
independent `environment/verify_v2_phase1.py` gate returned `phase2_authorized=true`. LangMani 2.0
Phase 2 then implemented and physically exercised `LangMani-PushToRegion-v0` plus its deterministic
privileged expert. Phase 2A stabilized lateral pushing without changing forward behavior, success
geometry, schedules, or the 250-step budget. The accepted Candidate E smoke reached 8/8 standard
and 6/8 hard; the fixed 50-standard/30-hard validation reached 46/50 and 23/30, with the protected
standard forward subset still 24/24 and zero workspace-exit or action-bound events. The unchanged
conjunctive expert gate therefore passes. Phase 2B then completed the separately authorized pushing
data pipeline. It preserved all 516 real expert attempts, accepted 397 demonstrations only after
397/397 independent action replays passed, retained 119 rejected attempts as a failure corpus, and
exported 53,297 aligned frames in 397 LeRobot 0.6 episodes. All six stable splits passed scene,
seed, trajectory, language, shard, and file-overlap checks. The frozen 360-episode pick-and-place
dataset remained byte-identical and a content-bound immutable multi-root index passed the real
cross-dataset compatibility audit. The independent final verifier returned
`pipeline_integrity_validated=true`, `full_collection_validated=true`,
`physical_target_validated=true`, and `smolvla_phase2c_authorized=true`. This authorization means
only that a separately invoked Phase 2C may consume the frozen data; no SmolVLA dependency, model,
training run, or learned pushing result exists yet. Rejected probes and invalid infrastructure
attempts remain diagnostics, not quality evidence.

Phase 2B.2 separately reconstitutes the unavailable push bytes under the new identity
`langmani/phase2b-push-v2`. Its two complete formal expert gates reached 75/100 and 11/100. Ten
subsequent G--P candidates ran only on disjoint diagnostic seeds. Candidate P added a bounded local
cylinder re-contact but reached only 2/5 before a zero-tolerance workspace exit stopped its probe;
the other failures were one timeout and one planning failure. Phase 2B.2 is therefore Result B at
the expert gate. Formal seeds 66300--66399 and all collection seeds remain untouched; no v2 raw
episodes, LeRobot export, archive, loader batch, optimizer step, SmolVLA training, or Phase 2D work
exists.

Phase 2B.3 tested a structurally different simulator-backed receding-horizon expert architecture.
Its required clone/restore precondition failed on the native RTX 5090 target: independently
cold-restored sandboxes replayed identically, but a verified target-contact-free snapshot diverged
from the live contact-bearing continuation by 4.394 mm in target-object pose. The error is material
beside the task's 5 mm containment clearance and cannot be repaired by widening tolerance. Phase
2B.3 is therefore Result C, not an expert-quality result. The MPC runtime, development pilot,
prediction gate, bounded repair, formal seeds 66300--66399, collection, training, and SmolVLA were
not run. Phase 2B.4 collection and Phase 2C.2 training remain unauthorized.

Phase 2B.3.1-RR separately audited whether exact reset plus replay of the complete historical
native action prefix could reconstruct contact-bearing transitions. The required original
`float32[8]` `pd_joint_pos` arrays and a complete bound call-sequence identity were unavailable;
only counts, indices, and hashes survived. The phase is therefore
`RESULT_D / HISTORICAL_PREFIX_UNAVAILABLE` and stopped before protocol freeze or simulator
execution. This is not a physical nondeterminism result and does not rewrite Phase 2B.3 Result C.
No reset, replay, fixed probe, runtime measurement, MPC pilot, expert qualification, collection,
or training ran. Every eligibility and authorization flag remains false.

Phase 2B.4-F0 completed as Result C. The 128-environment pipeline, reward/action audits, real PPO
smoke, checkpoint reconstruction, and deterministic execution passed. The single frozen
1,048,576-step micro run had zero training successes and no material return/progress improvement;
its fixed 48-episode evaluation reached 0 successes, 22 wrong-object displacements, and 12 target
workspace exits. The zero-tolerance micro hard stop prevented broader development. Reward revision
0 was not repaired because failures received their declared large negative terms and did not
demonstrate a profitable exploit. Formal seeds 66300--66399, full PPO training, the 95/100
qualification, demonstrations, datasets, collection, SmolVLA, and student training remain sealed.
`ppo_full_training_eligible=false` and every authorization flag remains false.

Phase 2B.4-F1 completed as Result C and is the final bounded PPO-specific diagnosis. The
100,000-action residual audit passed and the real paired short-prefix comparison materially reduced
arm/TCP displacement, but the single 262,144-step Probe A had zero training successes. Its fixed
32-episode evaluation had zero correct contacts, zero target-progress episodes, seven wrong-object
interactions, and zero workspace exits. Probe B did not run. The custom PPO teacher route is frozen
for this task formulation; there is no PPO F2, Probe C, reward retry, sweep, resume, or budget
increase. Formal seeds 66300--66399, Stage 1/2, full PPO training, expert qualification, collection,
LeRobot, SmolVLA, and all student training remain sealed and unauthorized.

Phase 2B.5 completed as Result A without reopening the custom source route. The pinned official
PickCube, StackCube, and PushCube motion-planning sources contributed three skill families, 3,000
successful source trajectories, and 254,374 transitions under one direct Panda
`pd_joint_pos float32[8]` contract. Each task passed 20/20 bounded and 100/100 stronger recorded-
action replays with exact categorical agreement and action/frame alignment, zero invalid actions,
and zero simulator errors. The nonprivileged 15-episode/1,546-frame pilot passed isolated LeRobot
0.6 conversion and all-frame readback. PokeCube and PullCube were not converted because their
downloaded official packages contain only delta-action sources. This sets
`official_demo_source_validated=true` and `phase2b6_dataset_production_eligible=true` only. Full
dataset production, ACT, SmolVLA, VLA-JEPA, student training, and custom-expert reactivation remain
unstarted and unauthorized.

Phase 2B.6 executed full official-source production and closed as Result C. Immutable source
validation still passed for all 3,000 trajectories and 254,374 transitions, and the three-episode
visual-shift pilot passed. Production completed 1,000 PickCube and 938 StackCube replay episodes
before `StackCube-v1` source episode 938 returned a non-retryable physical replay-gate rejection.
The hard stop preserved 1,938 accepted partial replay records and 178,633 frames, one rejection,
and 1,061 unattempted episodes. PushCube production, LeRobot task roots, full readback, full
source-to-derived verification, archive, restore, and canonical normalization did not start. The
partial output is diagnostic only and must not be treated as an accepted dataset or failure
training corpus. All model-training eligibility and authorization flags remain false,
`student_policy_training_started=false`, and `optimizer_steps=0`.

Phase 2B.6.1 is the completed, strictly bounded StackCube episode-938 forensic attempt. Exact
source/action identity and frozen prior evidence passed, but accepted control 936 failed during
SAPIEN Vulkan environment construction before reset or action submission. The hard stop prevented
control 937, target 938, Modes B/C, and all later work. Phase 2B.6.1 is Result D with
`insufficient_evidence` primary and `environment_construction_failure` secondary. It does not
reclassify the frozen Phase 2B.6 Result C or establish any episode-938 physical, determinism,
alignment, or writer claim. Production resume, package acceptance, all training eligibility and
authorization, optimizer creation, backward, and optimizer steps remain false or zero.

Phase 2B.6.1-R is the completed infrastructure-only Vulkan recovery and zero-step environment
gate. The explicit EGL-associated ICD
`/etc/vulkan/icd.d/my_nvidia_icd.json` passed Vulkan, minimal SAPIEN, and three independent
zero-step `StackCube-v1` constructions on the intended RTX 5090; the previous automatic packaged
GLX-associated route again failed before Vulkan instance creation. Every accepted construction
closed with zero explicit resets, steps, actions, and policy frames, and no process or GPU-workload
leak. Phase 2B.6.1-R is Result A and sets only
`phase2b6_1_forensic_restart_eligible=true`. The original forensic replay remains unauthorized,
Phase 2B.6 production remains closed, no accepted multi-skill dataset exists, and every training,
optimizer, backward, and student-policy flag remains false or zero.

Phase 2B.6.1-v2 is the completed bounded StackCube episode-938 replay forensic. The recovered
Vulkan launcher, exact source/action identity, and accepted controls 936/937 passed. Episode 938
then failed `canonical_terminal_success_gate` identically in all three fresh-process Mode A runs:
all 105 actions executed, canonical success began at step 101 for four steps, and final success
was false. The phase is Result B with `deterministic_failure`. The producer has no separate
stable-success acceptance gate; consecutive success remains diagnostic only. Modes B/C were
hard-stopped. `source_episode_exclusion_review_eligible=true` is not an exclusion or
authorization. Clean production, production resume, dataset acceptance, and all training remain
false.

Phase 2B.6-v2 is the completed clean, instrumented official-source production. It restarted from
source episode 0 under the recovered Vulkan launcher and the preregistered
`langmani-phase2b6-v2-exclusion-policy-v1`, without reusing Phase 2B.6 partial records. It
classified all 3,000 sources, accepted 2,998 episodes/254,200 frames, and retained StackCube
episode 938 plus PushCube episode 202 as `EXCLUDED_DETERMINISTIC_PHYSICAL`, each at
`canonical_terminal_success_gate`. All thresholds, source accounting, LeRobot 0.6 readback,
source equality, leakage, padding, normalization, archive, and clean-restore gates passed, giving
Result A. Dataset eligibility for ACT-baseline, SmolVLA, and VLA-JEPA is true; every training
authorization remains false, no policy was loaded, and optimizer/backward/training counts remain
zero.

Phase 2C-A is complete as Result D at its first closed-loop infrastructure smoke. Four seed-0 ACT
training runs and all 16 validation-only offline checkpoint diagnostics completed. The frozen
loss rule selected the final checkpoint for each model, but the selected Pick ACT emitted finite
gripper values above the live native `pd_joint_pos` upper bound on the first policy query. The
evaluator hard-rejected the chunk before `env.step`, with zero executed actions and no simulator
error. The already locked checkpoint was not replaced by an earlier checkpoint after this outcome
was observed, and no clipping, projection, or binary conversion was introduced. The H=1/4/8
development comparison, 30-episode schedules, shared seed 1, unseen-reset/visual-shift final
evaluation, task-ID intervention, and representative videos were not run. This is a
training/consumer-pipeline result, not an ACT quality result:
`closed_loop_development_started=false`, `final_evaluation_started=false`,
`act_baselines_validated=false`, `smolvla_phase_eligible=false`, and both SmolVLA and VLA-JEPA
training remain unauthorized.

Phase 2C-A.1 is the completed single bounded-ACT repair. It replaced only the unbounded action
output with `bounded_action_head_v1`, trained directly against native physical actions with
padding-masked L1, and left the evaluator as a rejecting, non-correcting gate. The 100,000-chunk
audit, padding audit, real GPU smoke, Pick/shared micro-overfits, four seed-0 full runs, and all 16
10,000-query checkpoint screens passed. The selected Pick policy reached 50 real environment
steps, and `H_exec=4` was frozen from the preregistered five-episode comparison. All 180
development and 360 final episodes completed with zero invalid actions and zero simulator errors,
but every policy/task/split group had zero success. Task-ID intervention changed actions without
producing success. The phase is Result B:
`act_baselines_validated=true`, `act_policy_quality_weak=true`, `shared_act_failed=false`,
`act_phase_closed=true`, `further_act_architecture_authorized=false`, and
`smolvla_phase_eligible=true`. The zero shared-minus-per-task success difference is a floor effect,
not evidence that interference is absent. SmolVLA and VLA-JEPA remain untrained and unauthorized;
no further ACT repair, architecture, seed, checkpoint reselection, or data change is authorized.

Phase 2C-B is complete as Result D after the official pretrained SmolVLA comparison. The pinned
base and nested VLM loaded 450,046,176 parameters with no reinitialization; the bounded physical
action transform, 1.6-million-action audit, real GPU smoke, Pick/shared micro-overfits, and the
20,000-step Pick-only run all passed their pipeline gates. The initial locked 20k/H=1 Pick gate
completed 30 episodes with zero success, zero invalid actions, and zero simulator errors. The sole
permitted repair was bound to a demonstrated action-chunk execution-horizon defect and changed
only `H_exec` from 1 to 8. The repeated gate again completed 0/30, with 16 failed grasps, 13
no-initial-motion failures, one object drop, zero invalid actions, and zero simulator errors.
This is a generic SmolVLA competence failure, not an infrastructure or action-contract failure:
`result_d_stop=true`, `smolvla_baselines_validated=false`, and
`other_full_models_authorized=false`. Shared, Stack, and Push full training, final evaluation,
language intervention, VLA-JEPA, LatentGuard, SARM, PPO, and new-data work were not started and
remain unauthorized. No second repair or additional SmolVLA tuning is permitted within Phase
2C-B.

Phase 2C-C is complete as Case B after the final authorized Pick action-formulation experiment.
Its query-state-relative bounded transform reconstructed all 70,239 accepted Pick frames within
`1e-6`; a 100,000-chunk/5,000,000-action audit had zero non-finite values, native-bound
violations, clipping, projection, or replacement. The sole seed-0 relative Pick SmolVLA completed
20,000 optimizer steps, and validation-only offline diagnostics selected step 20,000. The fixed
six-reset screen selected H=8. On the same 30 accepted training resets, the frozen absolute and
relative models each achieved 1/30 success. Relative validation reached 0/30 despite 22 grasps and
four lift events, with zero invalid actions and zero simulator errors. The conditional 50-reset
unseen test remained sealed. Therefore `action_formulation_partially_validated=true`,
`relative_action_formulation_validated=false`, `absolute_action_material_bottleneck=false`,
`langmani_generalization_route_blocked=true`, and `langmani_custom_model_route_eligible=false`.
The recommended project decision is to pivot to a standard benchmark with an established working
policy/data contract. Shared/Stack/Push SmolVLA, VLA-JEPA, another policy family, a second seed,
new data, and Phase 2C-C.1 remain unrun and unauthorized.

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
python environment/verify_m5a_full_control.py
python scripts/analyze_classifier_rejection.py --help

# LangMani 2.0 Phase 1 release and policy-neutral evaluation commands
python scripts/validate_v1_release.py
python scripts/evaluate_policy_v2.py --help
python environment/verify_v2_phase1.py --help

# LangMani 2.0 Phase 2B deterministic pushing-data commands
python environment/collect_push_demos.py --help
python environment/replay_push_demos.py --help
python scripts/export_push_lerobot_dataset.py --help
python scripts/audit_push_pick_compatibility.py --help
python environment/verify_v2_phase2b.py --help

# Phase 2B.3 compact Result C evidence verification (no simulator execution)
python environment/verify_v2_phase2b3.py

# Phase 2B.6-v2 compact Result A/B/C/D artifact verification (no simulator execution)
python environment/verify_v2_phase2b6_v2.py --help

# Phase 2C-A ACT-only consumer, training, and closed-loop evaluation commands
python environment/verify_v2_phase2c_a_identity.py --help
python environment/prepare_v2_phase2c_a.py --help
python scripts/train_v2_phase2c_a_act.py --help
python scripts/evaluate_v2_phase2c_a_act.py --help
python environment/verify_v2_phase2c_a.py --help

# Phase 2C-A.1 bounded-ACT compact Result B verification.
# This reads compact evidence plus externally retained checkpoint components and starts no
# simulator, optimizer, SmolVLA, or VLA-JEPA work.
python environment/verify_v2_phase2c_a1.py --help

# Phase 2C-B official SmolVLA consumer, staged training/evaluation, repair, and verification.
python environment/prepare_v2_phase2c_b.py --help
python environment/audit_v2_phase2c_b_smolvla_base.py --help
python scripts/train_v2_phase2c_b_smolvla.py --help
python scripts/diagnose_v2_phase2c_b_smolvla.py --help
python scripts/select_v2_phase2c_b_checkpoints.py --help
python environment/lock_v2_phase2c_b_evaluation.py --help
python scripts/select_v2_phase2c_b_policy.py --help
python scripts/evaluate_v2_phase2c_b_smolvla.py --help
python scripts/select_v2_phase2c_b_repair.py --help
python environment/verify_v2_phase2c_b.py --help

# Phase 2C-C final Pick action-formulation experiment and compact no-simulator verification.
python environment/prepare_v2_phase2c_c.py --help
python scripts/train_v2_phase2c_c_relative_smolvla.py --help
python scripts/diagnose_v2_phase2c_c_action.py --help
python scripts/select_v2_phase2c_c_policy.py --help
python scripts/evaluate_v2_phase2c_c_pick.py --help
python scripts/finalize_v2_phase2c_c.py --help
python environment/verify_v2_phase2c_c.py \
  --artifact-root artifacts/langmani_v2/phase_2c_c

# Phase 2B.3.1-RR compact Result D evidence verification (no simulator execution)
python environment/verify_v2_phase2b3_rr.py

# Phase 2B.4-F0 deterministic contracts and separately invoked native target stages
python environment/prepare_v2_phase2b4_f0.py
CUDA_VISIBLE_DEVICES=0 python environment/run_v2_phase2b4_ppo.py --smoke
CUDA_VISIBLE_DEVICES=0 python environment/run_v2_phase2b4_ppo.py --train-micro
CUDA_VISIBLE_DEVICES=0 python environment/run_v2_phase2b4_ppo.py \
  --evaluate-micro outputs/diagnostics/v2/phase2b4_f0/micro/micro_checkpoint.pt
python environment/verify_v2_phase2b4_f0.py

# Phase 2B.4-F1 deterministic contracts and separately invoked native target stages.
python environment/prepare_v2_phase2b4_f1.py
CUDA_VISIBLE_DEVICES=0 python environment/run_v2_phase2b4_f1.py \
  --audit-initial-exploration
CUDA_VISIBLE_DEVICES=0 python environment/run_v2_phase2b4_f1.py \
  --diagnose-f0 outputs/diagnostics/v2/phase2b4_f0/micro/micro_checkpoint.pt
CUDA_VISIBLE_DEVICES=0 python environment/run_v2_phase2b4_f1.py --compare-actions
# Probe A is allowed only after the three diagnostics pass. Probe B is allowed only
# when its explicit --probe-a-evaluation report passes every frozen gate.
CUDA_VISIBLE_DEVICES=0 python environment/run_v2_phase2b4_f1.py --train-probe-a

# Phase 2B.5 official-source qualification; stages remain explicitly separate.
python environment/run_v2_phase2b5_official_demos.py --help
python environment/verify_v2_phase2b5.py

# Phase 2B.6 compact Result A or fail-closed Result C evidence verification.
# This command reads only tracked JSON artifacts and starts no simulator or model.
python environment/verify_v2_phase2b6.py

# Phase 2B.6.1 compact Result D evidence verification; no simulator execution.
python environment/verify_v2_phase2b6_1.py

# Phase 2B.6.1-R compact Result A evidence verification; no simulator execution.
python environment/verify_v2_phase2b6_1r.py

# Phase 2B.6.1-v2 compact Result B verification; no simulator execution.
python environment/verify_v2_phase2b6_1_v2.py

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
- LangMani 2.0 Phase 1 runtime configurations must use repository-relative, content-bound artifact
  paths. Historical absolute paths may be retained only as provenance and must not become runtime
  dependencies. The unified evaluator accepts canonical task instances directly and must not import
  or invoke the language router.
- LangMani 2.0 Phase 2B keeps the ManiSkill-native push archive as its raw authority. An expert
  success enters the derived dataset only after fresh-environment action replay reproduces task,
  terminal, transition-label, frame-count, and final-pose contracts. Every failed attempt remains
  in the manifest and failure corpus; collection and replay identities are never inferred from file
  enumeration order.
- Phase 2B full collection and bounded top-up have separate immutable producer/runtime manifests.
  A later stage may combine only explicitly named compatible stages and must not rename, rewrite, or
  silently reuse an incompatible runtime. The historical pick-and-place and new pushing LeRobot
  roots remain immutable and are joined only through the content-bound multi-root index.
- Phase 2B dataset acceptance authorizes only a separately invoked Phase 2C consumer. It does not
  authorize automatic SmolVLA installation, training, evaluation, or a learned-policy quality
  claim. Generated raw archives, failure trajectories, LeRobot data, videos, and verifier reports
  remain under `outputs/`.
- Phase 2B.4-F1 owns only diagnostic logs and at most two small PPO checkpoints under `outputs/`.
  It must never edit F0 evidence, use the geometry branch, touch formal seeds, remove the Standard
  distractor from the environment implementation, change physics/success/250-step limits, run a
  third probe, resume or sweep a failed probe, or create demonstration/dataset/student artifacts.
  Probe A retains F0 reward revision 0. Any Probe B reward revision requires an explicit
  diagnosis-bound decision artifact and cannot be chosen merely because F0 success was zero.
- Phase 2B.5 official source bytes, extracted work files, visual pilots, LeRobot data, videos, and
  runtime logs remain outside source control. Compact evidence may record hashes, schemas,
  replay outcomes, and bounded pilot metadata only. Result A authorizes only a separately invoked
  Phase 2B.6 dataset-production decision; it must not start full production, load a policy, create
  an optimizer, or authorize ACT, SmolVLA, VLA-JEPA, or other training.
- Phase 2B.6.1-R may only establish a process-scoped Vulkan/EGL contract, run minimal SAPIEN
  rendering probes, and construct/inspect/close `StackCube-v1` without reset or step. Its Result A
  creates forensic-restart eligibility, never authorization. Episode 936/937/938 replay, Phase
  2B.6 production, writer/export/archive/restore work, accepted-dataset creation, policy loading,
  optimizer creation, backward, inference, and training require separate later authorization.
- Phase 2B.6.1-v2 is closed at Result B. Its explicit episode-938 failure may open only a separate
  source-exclusion policy review. It must not exclude the source, revise success/stability rules,
  resume Phase 2B.6, reuse the 1,938 partial episodes, run Modes B/C after the Result B hard stop,
  create LeRobot/package/archive outputs, or authorize any model or optimizer activity.
- Phase 2B.6-v2 is that separately authorized versioned source-policy and clean-production
  identity. It must use `configs/langmani_v2/phase2b6_v2_exclusion_policy.yaml`, preserve all
  source identities, start every task at source episode 0, and keep the old partial root read-only.
  It may continue after an explicitly classified physical exclusion only while the frozen per-task
  and total exclusion thresholds remain satisfied. It must not repair, replace, shorten, retry, or
  silently discard a physical failure, and it must never start model training after a dataset pass.

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
The full six-scene stage additionally requires 36 exact Oracle/NeuroSymbolic pairs, 72 fresh policy
resets, the immutable nine-category rejection set, 35/36 correct routes, 100% controller identity
agreement over every correct route, success gap at most two, full action/latency/failure
recomputation, and a fresh-process verifier. It may create `M5AFinalAuthorizationV0` only when every
quality item passes, and must never materialize or execute the sealed final schedules.
The separately authorized sealed final additionally requires exactly 600 final-language examples,
72 Oracle/NeuroSymbolic state pairs, 144 raw control atoms, ten zero-runtime rejection probes, an
access ledger, atomic compact evidence, and fresh-process recomputation. Its completed pipeline and
physical-verification flags do not override a false language, taxonomy, control, or combined quality
gate. A failed final quality gate leaves `smolvla_go=false` and authorizes no automatic next stage.
LangMani 2.0 Phase 1 implementation completion additionally requires an independently validated v1
tag/file manifest, an honest artifact audit, one real ACT adapter, the canonical six-instance task
catalog, policy-neutral evaluator tests, Ruff, CPU-safe regression, and a package build. Runtime
acceptance is separate and requires the recovered checkpoint to produce real actions in the M1
environment for at least one task over three deterministic seeds, with an atomically persisted
runtime manifest, episode JSONL, summary, and completion marker.
LangMani 2.0 Phase 2B completion additionally requires a real corrected pilot, all bounded full and
top-up attempts, immutable accepted and rejected records, fresh-environment action replay of every
generation-accepted episode, exact T/T+1 and transition-label checks, a real LeRobot 0.6 export and
readback, six stable leakage-free splits, preserved failed-attempt diagnostics, and a content-bound
compatibility audit against the unchanged M3B pick-and-place dataset. The final independent
verifier must recompute every admission gate and distinguish
`smolvla_phase2c_authorized=true` from `smolvla_started=false`.
