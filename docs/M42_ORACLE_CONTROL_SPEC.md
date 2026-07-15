# M4.2 oracle-control robustness specification

This document is the normative contract for LangMani milestone M4.2. M4.2 diagnoses the remaining
oracle-conditioned ACT control failures before any language-model policy is introduced. It has
three ordered stages: frozen-checkpoint runtime ablation, exactly one mixed TaskToken training run,
and one separately authorized sealed final benchmark. It does not implement language understanding,
SmolVLA, M5, new demonstrations, reinforcement learning, or a new task.

## Starting evidence and scope

The completed M4 full experiment used the immutable 60-scene/360-episode M3B dataset with
288/36/36 train/validation/test episodes and trained six `ACT-PerTask` models, one
`ACT-Mixed-Unconditioned`, and one `ACT-Mixed-TaskOneHot`. Its accepted aggregate evidence is:

| Control | Locked test | Historical fresh seeds |
| --- | ---: | ---: |
| six PerTask policies | 31/36 (86.1%) | 143/180 (79.4%) |
| Mixed-Unconditioned | 5/36 (13.9%) | 18/180 (10.0%) |
| Mixed-TaskOneHot | 27/36 (75.0%) | 101/180 (56.1%) |

The M4 full experiment is experimentally and physically validated, while
`baseline_quality_validated=false`. The Mixed-TaskOneHot historical fresh benchmark grasped the
target in 141/180 episodes but timed out in 79/180, grasped a wrong object in 25/180, and placed a
wrong object in the target bin in 12/180. Its mean task-pair action-chunk distance was 0.3397,
compared with 0.5186 for the PerTask oracle and zero for Mixed-Unconditioned. These observations
motivate two questions only:

1. Does shorter open-loop execution, especially after grasp, improve frozen ACT control?
2. Can a dedicated Transformer task token close part of the shared-policy gap to PerTask?

M4.2 preserves `LangMani-PickPlaceByInstruction-v0`, `num_envs=1`, `pd_joint_pos`, the M1 camera,
success, observation/no-leakage, and action-space contracts, the M3B scene-level splits, train-only
normalization, validation-only checkpoint selection, and the locked test boundary. It never uses M2
actions in a policy rollout. Existing M3A/M3B artifacts, M4 weights, checkpoint fingerprints,
selection records, test results, and historical fresh-seed reports are immutable.

## Leakage quarantine and committed seed locks

The original M4 test and fresh-seed results are historical evidence, not M4.2 development data.
They cannot select a runtime, architecture, checkpoint, capacity, stopping point, or go/no-go
decision. Before any new rollout, M4.2 jointly generates and commits two disjoint schedules:

| Schedule | Purpose | Scene seeds | Episodes | Fingerprint |
| --- | --- | ---: | ---: | --- |
| `m42_dev_v0` | runtime selection and post-selection development comparison | 12 | 72 per mixed control | `sha256:981547e771b2b5cd3a77e2788bb49d29fc45b3f59c607021a03a4e2ce70b43f1` |
| `m42_final_v0` | one sealed paired benchmark | 30 | 180 per mixed control | `sha256:b2aef313e076201f7a94875c835c2d606d8f255e3f75d53f7ac7fadcdbb267fc` |

The exact development seeds are:

```text
722950248, 1851299512, 713553274, 1707433820, 1405538489, 1953463567,
1994548171, 1805875340, 520095534, 400105183, 1993348904, 584965131
```

The exact sealed final seeds are:

```text
893589536, 982761247, 1647023593, 694109670, 145992996, 544351154,
2073139116, 908051858, 651466276, 2055735584, 1942539159, 978613149,
64981643, 1732076572, 47175200, 451960684, 1196965561, 272982124,
737985780, 1699051323, 138690626, 1098325036, 1544603825, 214610042,
905027018, 568018298, 742414486, 1073839369, 2086293999, 824728764
```

Every scene expands in the canonical object-major/bin-minor order: red-left, red-right,
green-left, green-right, blue-left, blue-right, always with `canonical_v0`. Both locks bind the M3B
export fingerprint
`sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4`,
200 maximum episode steps, `physx_cpu`, the fixed RGB policy camera, 20 Hz control, and evaluation
configuration fingerprint
`sha256:0746ecb022ac67e73bc04d19db39c6e11e76ce264669d5e1dec9474a34cbb6d4`.

The exclusion lock contains exactly 125 unique seeds. It covers M3A accepted seeds and all five
rejected candidate seeds (together 0 through 64), all M3B splits, the 30 observed M4 full fresh
seeds, seed 0 used by M4 smoke/tiny-overfit and M4.1 diagnostics, and the 30 predeclared tiny fresh
seeds. Redundant sources are retained for provenance. Its canonical digest is
`sha256:801e5b9d0595855729a45fa3bab25b85449afab2aec06065dd69233eec0f5610`.

The source locks live at `src/langmani/policies/m42_schedules/exclusion_sources_v0.json`,
`m42_dev_v0.json`, and `m42_final_v0.json`. They are packaged read-only resources and are the only
authoritative seed lists; generated output copies are evidence, not an alternate configuration.

Generation uses `langmani-m42-joint-seed-generation-v0`: for candidate counter values beginning at
zero, canonical JSON containing the generation schema/namespace, M3B fingerprint, exclusion digest,
and counter is SHA-256 hashed; the first 64 digest bits are reduced to a non-negative int31 seed;
excluded and duplicate candidates are skipped. The first 12 accepted candidates become development
and the next 30 become final. The committed stream required 42 candidate attempts. Loader-side
regeneration verifies both lists, their reciprocal references, disjointness, exclusion, and
fingerprints without materializing a final rollout.

Loading the final lock for verification is not final-benchmark access. Materializing final episodes
requires an explicit final authorization, a clean Git tree, the exact final schedule and
implementation fingerprints, and immutable horizon, gripper, and TaskToken-checkpoint selections.
`--target-development` must never supply that authorization, render a final scene, or reset an
environment with a final seed.

## M4.2a execution-horizon ablation

The primary checkpoint is the frozen selected Mixed-TaskOneHot policy. The representative PerTask
checkpoint is the frozen `green_cube` to `left_bin` policy, predeclared because it was the weakest
historical fresh-seed PerTask result. No M4.2a checkpoint is retrained.

The only changed variable is the number of queued actions executed per policy query: exactly 10, 5,
or 1. ACT still predicts a 50-action chunk. All three horizons use the saved model and LeRobot
processors followed by `BoundedActionEnvPostprocessorV0(mode=project)`. The mixed model executes
12 scenes x 6 tasks x 3 horizons = 216 episodes; the representative model executes
12 x 1 x 3 = 36 episodes. `policy.reset()` and the M4.2 execution queue are both cleared at every
episode boundary.

The report records task and safety outcomes, episode length, query count and executed actions per
query, target-grasp/release timing and sign transitions, post-grasp and near-destination re-queries,
inference p50/p95/p99, environment-step latency, raw/projected action audits, and
`strict_unprojected_success`. Raw and executed actions remain separate. Any projected Panda arm
component invalidates runtime selection and target-development acceptance; gripper-only projection
remains explicit evidence rather than being hidden in that gate.

The immutable horizon selection ranks, in order: mixed success; mixed post-grasp timeout; mixed
wrong-object interaction; representative success; successful median steps; lower query/inference
cost; then the larger horizon. Only `m42_dev_v0` may contribute evidence.

## Post-grasp diagnostics

Privileged environment state may be sampled at reset and after executed actions for diagnostics
only. It is never placed in a policy observation or consulted before selecting the next action.
The fixed classifier distinguishes:

```text
never_grasped_target
target_grasped_not_lifted
lifted_not_transported
transported_not_descended
descended_not_released
released_outside_success_region
released_but_not_static
target_success_then_lost
timed_out_after_target_grasp
wrong_object_interaction
environment_failure
success
```

Every failed rollout that grasped the target preserves a compact record with scene/task identity,
checkpoint, horizon, first grasp, maximum height, closest bin distance, first release, final target
position/velocity and M1 evaluation, inferred phase, projection summary, and an optional diagnostic
video path. No trajectory or privileged value becomes a training feature.

## M4.2a gripper-runtime ablation

After the horizon record is locked, M4.2 compares exactly `project` and `binary` at that horizon on
the same development schedule: 144 mixed episodes and 24 representative episodes. `project` is the
existing M4.1 processor. `BinaryGripperEnvPostprocessorV0` is an additional, explicit, versioned
runtime processor for action component 7 only:

```text
raw gripper >= 0 -> +1
raw gripper < 0  -> -1
```

The processor validates the eight-component M1 action schema and rejects malformed, NaN, or Inf
values. It preserves the raw command, every binary change, the binary result, and the final executed
action. Arm components still pass through the existing project processor and are never binarized.
The binary and project configurations have distinct deterministic runtime fingerprints.

The declared ranking is mixed success, post-grasp timeout, wrong-object interaction,
representative success, unnecessary gripper sign transitions, then `project`. For this single-pick
task, at most one open-to-close and one close-to-open transition per episode are necessary; the
reported unnecessary count is the per-episode excess above those two permitted transitions. Binary
is eligible only if wrong-object grasp, wrong-object-in-target-bin, target-in-wrong-bin, off-table,
and invalid-action metrics do not worsen and it gains at least five mixed success percentage
points, reduces mixed post-grasp timeout by at least 25% relative, or gains at least two of the 12
representative episodes. Otherwise `project` remains selected.

## M4.2b ACT-Mixed-TaskToken

M4.2 trains exactly one new oracle-conditioned model, `ACT-Mixed-TaskToken`. It uses
`CanonicalTaskTokenV0`, with the same canonical six-task mapping as `CanonicalTaskOneHotV0`. It is
not a language model, does not tokenize the canonical instruction, and receives no natural-language
text.

Installed LeRobot 0.6.0 exposes a public ACT `FeatureType.ENV` input at
`observation.environment_state`. ACT projects that six-way float32 one-hot through its public
`nn.Linear(6, dim_model)` environment-state projection and inserts the resulting learned
hidden-dimensional vector as the dedicated environment token after latent and Panda-state tokens
and before image tokens. `NormalizationMode.IDENTITY` preserves the canonical command. The six
projection columns plus shared bias are six learned 512D task embeddings. This uses public
configuration/model attributes and does not fork, patch, or copy LeRobot ACT. Contract tests fail
if the ENV projection or three non-image encoder positions change.

`PandaPolicyStateV0` remains exactly 9D; task identity is not concatenated to qpos. Image features,
eight-action output, ResNet-18, transformer depth/width/heads, VAE latent, and loss are unchanged.
Training matches Mixed-TaskOneHot: the same 288 M3B train episodes and 36 validation episodes,
50-action chunks, batch 32, AdamW at `1e-5`, weight decay `1e-4`, no scheduler, KL weight 10,
gradient clipping 10, CUDA bfloat16, 100,000 steps, and checkpoint/validation every 5,000 steps.
Only the exact train view computes normalization. Each training token is derived from immutable M3B
episode provenance, and the same mapping is used at inference.

The effective TaskToken training identity is compared against the frozen Mixed-TaskOneHot run
manifest before training. Apart from replacing the appended task one-hot in policy state with the
dedicated ENV token, model, optimizer, scheduler, seed, data ordering, split, normalization, and
training bounds must match the historical baseline exactly; documentation defaults are not treated
as evidence of fairness.

TaskToken checkpoints retain the existing atomic save/local reload, optimizer/scheduler/RNG state,
saved processors, train-statistics, data/split, Git, and run-identity contracts and additionally bind
the mapping, hidden dimension, ENV injection location, architecture-extension fingerprint, and
selected runtime schema. Resume fails on any semantic mismatch.
Exactly one fingerprint-owned TaskToken run directory may exist under the model root; an extra
complete or incomplete identity is an ambiguity error, while interrupted evidence for the sole
matching identity follows the documented recovery path.

## Validation selection and development comparison

At each configured TaskToken checkpoint, M4.2 computes offline loss and the fixed M3B validation
closed-loop schedule with the already selected horizon and gripper runtime. Selection uses only
validation: highest task success, lowest wrong-object interaction, lowest target-off-table rate,
lowest timeout, lower validation action loss, then earlier step. M3B test, historical M4 results,
`m42_dev_v0`, and `m42_final_v0` are forbidden selection inputs.

After selection is immutable, development compares the frozen Mixed-TaskOneHot, selected TaskToken,
and frozen PerTask oracle where required on `m42_dev_v0`. It reports overall/per-task success,
timeout and post-grasp outcomes, wrong-object/wrong-bin/off-table events, task-pair chunk distances,
task-sensitivity ratio to PerTask, raw and runtime action metrics, and inference latency. Development
results may decide only whether the implementation is ready for final; they cannot change the
checkpoint or any frozen runtime.

## Sealed final benchmark and go/no-go

`m42_final_v0` is a separate explicit command and is run exactly once only after all selections and
the evaluation commit are frozen. It pairs the six PerTask policies, frozen Mixed-TaskOneHot, and
selected TaskToken on the same 180 scene/task episodes and reports episode-paired outcomes and
confidence intervals. An infrastructure failure leaves the partial artifact invalid; a repair may
rerun the unchanged schedule from clean staging but may not change a model or evaluation contract.
The final report includes overall and per-task success, target/post-grasp success, target and wrong-
object grasp, wrong-object/target wrong-bin events, off-table and timeout counts, successful median
steps, inference latency, arm and gripper transformations, maximum raw excess,
strict-unprojected success, task-pair distances and ratio to PerTask, episode-paired outcomes, and
Wilson or bootstrap intervals.

SmolVLA is recommended only if TaskToken passes every final threshold:

- at least 70% and 126/180 overall success;
- no more than a 12-point paired gap to PerTask;
- at least 60% and 18/30 success for every TaskSpec;
- timeout at most 30%, wrong-object grasp at most 8%, and wrong object in target bin at most 3%;
- zero target-in-wrong-bin, target-off-table, arm projection, NaN, Inf, and malformed actions;
- task-sensitivity ratio to PerTask at least 0.75.

An all-pass result writes `go_for_smolvla`; any failure writes
`remain_in_oracle_control_layer`. Neither decision automatically starts M5. The exact next milestone
is chosen only after this immutable report: a GO authorizes planning a separate SmolVLA language
milestone, while NO-GO keeps work in the oracle control layer and follows the failing task/phase
evidence.

Interpretation is also predeclared. A material M4.2a gain identifies execution horizon or gripper
runtime as a control bottleneck. A TaskToken gain that misses the final gate stays in the oracle
layer and reports weak tasks/phases. No task-sensitivity gain triggers an injection/utilization
audit, not added language complexity. Strong PerTask with weak TaskToken points to shared-policy
conditioning/capacity; broadly post-grasp failures point to release/execution control.

## Action evidence and verification

The action path is explicit:

```text
ACT -> saved LeRobot postprocessor -> raw environment action
    -> optional BinaryGripperEnvPostprocessorV0 on component 7
    -> BoundedActionEnvPostprocessorV0(mode=project)
    -> executed environment action
```

Reports keep raw bound excess, binary changes, arm projection, gripper projection, executed-action
validity, and strict-unprojected success as independent fields. Projection is never called clipping;
a projected success does not imply raw-action validity.

One immutable `experiment_manifest.json` is published with the runtime selections. It records the
clean implementation commit, M3B fingerprint, all eight selected M4 checkpoint fingerprints, the
exact prior M4.1 processor fingerprint, both schedule fingerprints, and the horizon/gripper
selection fingerprints. The M4.1 processor fingerprint is derived from the validated `project`
configuration plus the actual action-space contract in all eight historical evaluation-runtime
manifests; it is never a hard-coded checkpoint or processor guess. The runtime selection,
TaskToken run identity/manifest/validation queue, and every development or final evaluation
artifact bind the same experiment-manifest fingerprint. TaskToken architecture and checkpoint
identity remain in their own immutable training and selection manifests, avoiding a circular
fingerprint. Missing or conflicting provenance is an infrastructure failure.
All eight selected checkpoint directories, atomic completion markers, component hashes, and strict
reload contracts are revalidated from current disk before M4.2 consumes the historical evidence,
including the Mixed-Unconditioned control that is retained only for provenance comparison.

The command boundary is:

```bash
python scripts/run_m42_runtime_ablation.py --help
python scripts/train_act_task_token.py --help
python scripts/evaluate_m42.py --help
python environment/verify_m42.py

# This milestone invocation may run development work only.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m42.py --target-development

# Deliberately separate and not authorized by target-development.
CUDA_VISIBLE_DEVICES=0 python environment/verify_m42.py --target-final
```

All commands support dry-run, explicit paths, machine-readable reports, owned staging, fingerprint-
checked resume/reuse, and nonzero infrastructure failure. Completed outputs are immutable. Final
runs require a clean tracked Git state; generated schedules are source, while datasets, videos,
checkpoints, and experiment reports remain ignored outputs.

Verification keeps these flags independent:

```text
implementation_validated
prior_m4_evidence_validated
development_schedule_locked
final_schedule_locked
horizon_ablation_completed
horizon_selection_locked
gripper_ablation_completed
gripper_selection_locked
task_token_training_completed
task_token_checkpoint_selected
validation_only_selection_validated
development_benchmark_completed
final_benchmark_completed
post_grasp_analysis_completed
raw_action_metrics_validated
runtime_action_metrics_validated
go_no_go_decision_completed
smolvla_go
physical_target_validated
passed
```

At implementation time, schedules and contracts may be validated without physical rollout.
Horizon/gripper results, selections, TaskToken checkpoint identity, development metrics, final paired
metrics, and go/no-go remain `pending` until their corresponding target stages actually complete.
In particular, completing `--target-development` leaves `final_benchmark_completed=false`,
`go_no_go_decision_completed=false`, and no SmolVLA decision claim. This repository must not
fabricate or prefill those results.

| Result field | Paused execution value (2026-07-15) |
| --- | --- |
| horizon results / selected horizon | M4.2a evidence completed and reused; `H=10` selected |
| post-grasp development distribution | completed in immutable M4.2a evidence |
| project versus binary results / selected runtime | M4.2a evidence completed and reused; `project` selected |
| TaskToken selected checkpoint / fingerprint | `pending` |
| `m42_dev_v0` policy comparison | `pending` |
| raw and executed development action statistics | M4.2a evidence complete; TaskToken development evaluation `pending` |
| `m42_final_v0` paired benchmark | `not_run` |
| final PerTask / State-OneHot / TaskToken rates | `not_run` |
| final task-sensitivity ratios and safety metrics | `not_run` |
| SmolVLA go/no-go | `not_computed` |

## Paused target-development execution (2026-07-15)

This is an interruption record, not an acceptance claim. The native target-development run used
clean commit `36bd6d81ff5e3b2c74977da1eec0fb0231d2f922`. Its strict immutable-evidence audit accepted the
completed M4.2a runtime ablation without rerunning the 336 physical episodes. The accepted runtime
evidence selected horizon 10 and the existing `project` gripper runtime. It records zero arm
projection, zero target-in-wrong-bin events, and zero target-off-table events. Binary gripper
projection was not selected because it increased wrong-object interaction from 13 to 14 events.

The sole TaskToken run has stable run fingerprint
`sha256:1fec0221b5dbcd28faf2e9b5e465b7a97d7a4b3fc1c141251c19b275af8bf2a8`. Before interruption,
training had reached observed step 82,078 of 100,000 with 16 of 20 atomically complete checkpoints.
The last complete checkpoint was `step-00080000-19bb17efda9f`; the latest observed total loss was
0.10229156166315079, with no NaN or Inf in the latest 1,000 metric records and no incomplete
checkpoint directory. These observations must be rechecked on disk after the instance is restarted;
they do not by themselves prove that the checkpoint survived the platform shutdown.

The AutoDL instance was shut down prematurely while training was still active. Consequently, no
TaskToken training-completion report exists, validation-only selection across all 20 checkpoints has
not run, and `m42_dev_v0` rollout comparison has not run. Any older
`outputs/diagnostics/m42/verification.json` with false flags is stale evidence from an earlier failed
attempt, not the result of this interrupted run. `--target-development` is not validated, and
`--target-final` remains untouched.

When suitable GPU capacity is available again, resume on the exact clean implementation commit,
not on a later documentation-only commit. First verify that commit and the 80,000-step completion
marker, then rerun the same development verifier. Its fingerprint-checked resume path must complete
the remaining checkpoints before validation-only selection and the development benchmark:

```bash
cd /root/autodl-tmp/langmani
git fetch origin
git checkout --detach 36bd6d81ff5e3b2c74977da1eec0fb0231d2f922
test "$(git rev-parse HEAD)" = "36bd6d81ff5e3b2c74977da1eec0fb0231d2f922"
test -z "$(git status --porcelain)"
test -f outputs/models/act-task-token/1fec0221b5dbcd28faf2e9b5e465b7a97d7a4b3fc1c141251c19b275af8bf2a8/checkpoints/step-00080000-19bb17efda9f/complete.json

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/langmani/src
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export XDG_RUNTIME_DIR=/tmp/langmani-xdg
/root/autodl-tmp/conda-envs/langmani/bin/python environment/verify_m42.py --target-development
```

Do not run `--target-final`. After target-development finishes, preserve its exit code and
`outputs/diagnostics/m42/verification.json` before any platform shutdown. A hard failure must remain
stopped with its evidence intact rather than being restarted or hidden.
