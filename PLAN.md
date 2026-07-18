# LangMani roadmap

M0 through M3B, M4 full, M4.1 target smoke, M4.2 target-development, M4.3a, and M4.3b
target-development are complete on native targets. M4 full is experimentally and physically
validated, but its declared quality gate is false. M4.2 rejected TaskToken and M4.3b rejected
FactorFiLM after development; `m42_final_v0` remains sealed and unaccessed. The shared-ACT
architecture search is therefore closed. M5A is the active implementation milestone: it adds
modular language-to-TaskSpec routing over the six frozen PerTask ACT controls. M5A target
classifier/LLM training and language/control development evidence are not yet claimed, its own
final schedules remain sealed, and SmolVLA has not started.

## M0 — Reproducible environment foundation (complete)

Establish the src-layout package, dependency decisions, CPU-safe tests, explicit GPU/rendering
markers, and a strict native Linux diagnostic proving that PyTorch CUDA, ManiSkill, Vulkan
rendering, and base LeRobot imports coexist.

## M1 — Environment and language contracts (complete)

Implement exactly one deterministic, vectorization-compatible ManiSkill environment,
`LangMani-PickPlaceByInstruction-v0`, with three colored cubes, two primitive shallow bins, typed
scene/task metadata, canonical language accessors, privileged-state gating, conservative batched
success metrics, separate policy/diagnostic cameras, and explicit CPU/GPU/rendering acceptance
boundaries. M1 does not include task-solving or data-collection behavior.

## M2 — Privileged motion-planning expert (complete)

Implement one deterministic, privileged-state Panda expert for all six semantic object/bin tasks in
`LangMani-PickPlaceByInstruction-v0`. M2 owns the project-level expert types, a lazy adapter over
the public mplib 0.1.1 planner API, explicit phase execution, verification, structured failure
classification, single-rollout diagnostics, and a six-combination benchmark.

The expert runs only with `num_envs=1` and `pd_joint_pos`. It uses the active `TaskSpec` and narrow
expert-only environment interfaces; it never infers targets from image pixels or actor order. The
stable phase sequence is `initialize`, `move_to_pregrasp`, `approach_target`, `close_gripper`,
`verify_grasp`, `lift_target`, `move_above_destination`, `descend_to_place`, `open_gripper`,
`settle_after_release`, `retreat`, and `verify_task`. Each phase records attempts, completion or a
specific failure, environment steps, planning calls, and elapsed durations in a JSON-serializable
result.

M2 uses a deterministic top grasp and bin-interior-center placement. It preserves M1's visual
no-leakage contract and adds no trajectory recorder, dataset export, training, deployable vision
policy, LLM/VLM call, teleoperation, reinforcement learning, or domain randomization. mplib is not
vectorized and no multiprocessing or RRT fallback is introduced. The native RTX 4090 target gate
passed its six-task smoke and 177/180 balanced benchmark (98.33%) with zero wrong-target successes,
unclassified failures, or crashes.

## M3A — ManiSkill-native raw demonstrations (implementation complete)

Collect a deterministic, resumable authoritative source archive from the M2 expert. Candidate
scene seeds are ordered, and every accepted `CounterfactualSceneGroup` contains one physical layout
paired with all six canonical TaskSpecs. The first target archive is exactly 60 complete groups and
360 successful episodes, with 60 per task combination.

M3A owns typed collection/replay contracts, cryptographic stable IDs, bounded retries, ManiSkill
`RecordEpisode` integration, HDF5/JSON shards with environment states and actions, attempt and
episode manifests, action replay, state audit, corruption/schema checks, resume behavior,
inspection commands, and target verification. It does not create LeRobotDataset, Parquet, policy
videos, training data transformations, or policies. One six-task native target smoke group passed
recording and action replay. Native full acceptance then produced 60 complete groups and 360
accepted/replayed episodes, exactly 60 per TaskSpec, from 65 ordered candidate scenes. Five groups
were rejected and all 404 attempts remain accounted for; no partial group, accepted replay failure,
schema/checksum failure, or unclassified failure was admitted.

## M3B — LeRobotDataset v3 export (implementation complete)

Deterministically convert accepted episodes from a content-bound, validated M3A source into one
local LeRobotDataset v3. M3B restores each recorded pre-action state in a fresh M1 environment,
renders only the fixed 256 x 256 `base_camera`, extracts nine Panda joint positions, preserves the
exact eight-dimensional raw action and canonical task string, and emits exactly T frames for T
actions.

M3B owns typed export contracts, stable export fingerprints, scene-group-level 48/6/6 splits,
source-to-derived mapping, a project lifecycle guard around the public LeRobot 0.6.0 writer,
PyAV H.264/yuv444p video, all-or-nothing staging, independent source/action/state/video/Parquet validation,
DataLoader smoke tests, and structural/smoke/full verification modes. M3A remains authoritative.
M3B adds no training, policy configuration, Hub upload, failure trajectories, additional sensors,
language generation, or parallel export. One six-task target export passed real state-restoration
rendering, H.264 decode, source alignment, finalization, and public reload. Native full acceptance
then exported all 360 episodes and 64,548 frames with exact 288/36/36 episode and 48/6/6 scene-group
splits. Every TaskSpec contributes 48/6/6 episodes; all videos decode, provenance and action/state
alignment pass, and no scene group crosses a split. The accepted export fingerprint is
`sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4`.

## M4 — Reproducible ACT baselines (complete)

Train and compare exactly three controls on one completed M3B dataset: six `per_task` ACT policies
with image plus 9D Panda state; one `mixed_unconditioned` ACT with the same 9D input and deliberate
counterfactual ambiguity; and one `mixed_task_onehot` ACT with image plus a 15D state consisting of
the same Panda state and a project-owned canonical six-way oracle command.

M4.1 owns a versioned environment-action postprocessor after the saved LeRobot postprocessor and
before `env.step`. `reject` preserves strict failure; `project` deterministically bounds finite
actions using the actual M1 action space while retaining raw/executed audits and independent
`task_success` versus `strict_unprojected_success`. A separate runtime fingerprint binds checkpoint
and processor fingerprints, bound configuration, environment/action-space contract, task mapping,
rollout configuration, schema, and current code commit. Existing checkpoint fingerprints remain
unchanged.

M4.1 target smoke
validates the completed M0-through-M3B evidence, reuses the existing PerTask and TaskOneHot
checkpoints without training, reproduces strict bound rejection, executes a projected legal step,
then attempts one PerTask and six same-scene TaskOneHot rollouts.

The clean RTX 4090 smoke passed PerTask 1/1 and TaskOneHot 6/6. All seven task successes required
at least one explicit gripper projection, so strict-unprojected success remains 0/7 and raw-action
bounds validity remains false.

M4 full subsequently trained and sealed all eight 100,000-step runs. PerTask achieved 31/36 on the
locked test and 143/180 on historical fresh seeds; Mixed-Unconditioned achieved 5/36 and 18/180;
Mixed-TaskOneHot achieved 27/36 and 101/180. Validation-only selection, locked test access,
fresh-seed evaluation, provenance, and physical execution passed, so
`full_experiment_validated=true` and `physical_target_validated=true`. The intended conditioning and
fresh-seed quality thresholds did not all pass, so `baseline_quality_validated=false` remains an
equally important result.

## M4.2 — Oracle-control robustness (development complete; final not authorized)

M4.2 first uses frozen Mixed-TaskOneHot and representative green-left PerTask checkpoints to compare
execution horizons 10, 5, and 1 on the committed 12-scene `m42_dev_v0` schedule. After locking one
horizon, it compares the existing explicit `project` action runtime with an explicit component-7
`BinaryGripperEnvPostprocessorV0`. Raw, binary-transformed, projected, and executed actions remain
separate, and privileged post-grasp phases are diagnostics only.

M4.2 then trains exactly one `ACT-Mixed-TaskToken`. Panda policy state remains 9D. The canonical
six-way oracle command uses LeRobot 0.6.0's public `FeatureType.ENV` path, whose linear 6-to-hidden
projection creates a dedicated Transformer environment token. This is discrete oracle conditioning,
not language understanding. Training retains the M4 configuration and train-only statistics; only
M3B validation may select its checkpoint.

Both M4.2 schedules were generated and committed jointly after excluding 125 prior observed or
predeclared seeds. `m42_dev_v0` contains 12 scenes and has fingerprint
`sha256:981547e771b2b5cd3a77e2788bb49d29fc45b3f59c607021a03a4e2ce70b43f1`.
The sealed 30-scene `m42_final_v0` has fingerprint
`sha256:b2aef313e076201f7a94875c835c2d606d8f255e3f75d53f7ac7fadcdbb267fc`.
Development selected horizon 10 and the existing `project` runtime, completed one 100,000-step
TaskToken run, selected its 90,000-step checkpoint from M3B validation only, and completed the fixed
72-episode development comparison. PerTask, State-OneHot, and TaskToken achieved 56/72, 35/72, and
15/72 respectively; TaskToken was rejected. Development stopped without materializing, rendering,
or resetting a final seed. The sealed final result and SmolVLA decision remain absent. The normative
contract is `docs/M42_ORACLE_CONTROL_SPEC.md`.

## M4.3a — Shared-policy semantic alignment audit (complete)

Use the frozen six PerTask, State-OneHot, and rejected TaskToken checkpoints to compare complete
postprocessed ACT chunks on fixed RGB plus `PandaPolicyStateV0[9]` observations. Audit all M3B
validation groups and all `m42_dev_v0` groups while keeping the two sources explicitly separate.
The primary retrieval metric is action-range-normalized, arm-only L2 over the locked execution
horizon. Full-task, object-centroid, global-bin, and object-conditional-bin retrieval, deterministic
confusions, first interaction, and post-grasp failure classes are stored in immutable
fingerprint-owned evidence.

Historical M4.2 development evidence supports aggregate post-grasp failure distributions but lacks
the exact per-episode fields required to reconstruct first-interaction confusion. M4.3a must report
that confusion as unavailable rather than derive it from aggregate wrong-object counts.

M4.3a adds no model or training. It never opened M3B test, M4 fresh, or `m42_final_v0`. The real
clean-Git RTX 5090 audit completed 108 observations and promoted immutable evidence with fingerprint
`sha256:6342bdf4b019df203e6021947cbb39deacac2ea78cc585b5062091ed1d228671`. The combined
State-OneHot result was 37.96% full-task top-1, 76.85% target-object retrieval, and 50.93%
destination-bin retrieval. TaskToken reached 24.07%, 50.00%, and 49.07% respectively and remains
rejected. The audit shows output sensitivity without reliable requested semantics, target-object
confusion, approximately random bin retrieval, and shared-policy post-grasp failures. It sets
`semantic_audit_completed=true` while final/test/historical-fresh access and SmolVLA remain false.
The normative contract is `docs/M43_SHARED_POLICY_REPAIR_SPEC.md`.

## M4.3b — One factorized FiLM repair (target development complete; rejected)

Implement exactly one oracle `ACT-Mixed-FactorFiLM` with separate `TargetObjectConditionV0` and
`DestinationBinConditionV0` mappings. Red/green/blue object embeddings modulate the ResNet-18
layer-4 feature map before ACT image projection; left/right bin embeddings modulate the encoded
nine-dimensional Panda-state token before Transformer processing. Each embedding is 32D. The
residual FiLM projections use standard deviation `1e-5` weights and zero bias, so the initial
transform is close to identity. The policy adds 67,744 trainable parameters and adds neither a
combined task token nor task features to `PandaPolicyStateV0[9]`.

The project-owned LeRobot 0.6.0 adapter subclasses the public `ACTPolicy` and uses instance-local
hooks around the semi-stable backbone and state-projection outputs. Compatibility checks fail closed
on symbol, signature, model-attribute, token-layout, or tensor-shape drift. Training reuses the exact
288/36 M3B train/validation views, train-only statistics, locked seed 0, 100,000-step M4
configuration, a fingerprint-bound 20-checkpoint schedule, validation-only ranking, action chunk
50, locked rollout horizon 10, and `project` runtime.
Stable TaskSpec metadata supplies the two indices; language text is never parsed.

The architecture/structural baseline is frozen at commit
`8ee0f1babf36b91d1ee2a39701e4a6db6003b660`, while the compatibility-only target-training producer
is frozen at `0088e2937556c123c37c2dbe69f73301b1eebfd0`. The authorized target-development sequence is:

1. one exact 100,000-step seed-0 CUDA run and 20 immutable checkpoints at 5,000-step intervals;
2. 36 M3B-validation rollouts for each checkpoint and the predeclared seven-key selection;
3. selected policy plus processors reloaded in a fresh process, with the full postprocessed
   `[50,8]` chunk matching at `atol=rtol=1e-6`;
4. a paired `m42_dev_v0` comparison of PerTask, State-OneHot, and FactorFiLM, 72 episodes each and
   216 episodes total, all at horizon 10 with explicit `project` action handling;
5. validation/development semantic retrieval, new first-interaction evidence, the exact
   16-condition development quality gate, and a separate read-only verifier.

The real run completed 100,000 steps and all 20 checkpoints. M3B-validation-only selection chose
step 70,000; a fresh process reproduced the full postprocessed `[50,8]` chunk with zero absolute
and relative error. The paired physical development result was PerTask 56/72, State-OneHot 36/72,
and FactorFiLM 38/72. FactorFiLM produced six wrong-object grasps, two wrong objects in a target
bin, 34 timeouts, zero target-in-wrong-bin, zero target-off-table, zero arm projections, and no
non-finite or malformed action. The independent verifier passed the experiment and physical
evidence, but the 16-condition quality gate failed. Final authorization remains false; M3B test,
historical M4 fresh seeds, and `m42_final_v0` were not accessed. Further shared-ACT architecture
tuning is prohibited.

## M5A — Modular language-to-TaskSpec routing (implementation in progress)

Use the six immutable selected PerTask ACT checkpoints as the low-level skill library. Build a
deterministic family-split language corpus and compare `RuleRouterV0`, one factorized compact text
classifier, and one explicitly pinned structured local instruct-model router. Every route becomes
one canonical M1 `TaskSpec`; ambiguous, unsupported, or malformed input is rejected before policy
or environment execution.

M5A owns four parent schedule locks, strict structured decisions, classifier validation-only
selection and calibration, one frozen local-LLM prompt whose few-shot examples are train-family
only, a portable six-controller registry, and exact routing/control failure attribution. Classifier
development uses exactly one seed and progresses through fixture, tiny-overfit, a resumable
step-29 authoritative pilot, and at most 145 total steps/five epochs with patience-one validation
early stopping and three fixed checkpoint roles. Physical development progressively promotes
candidates through disjoint 1-scene/3-scene/6-scene schedules, retaining Oracle at every stage and
spending at most 144 routed episodes before final. M1, M3A, M3B, frozen controllers,
H=10/`project` runtime, and prior evidence remain unchanged. Language/control final schedules are
fingerprinted but inaccessible during development.
M3B test sidecar identity may support seed/provenance validation, but its observations, actions,
frames, Parquet content, and videos remain inaccessible. Correct experiment completion and the
learned-router quality gate are separate; authorization never executes final automatically.
The normative contract is `docs/M5A_LANGUAGE_ROUTING_SPEC.md`.

The observed step-29 classifier pilot was valid but failed its original promotion gate. D-061 is
an explicit post-pilot protocol amendment, not part of that predeclared gate: one exact-checkpoint
continuation may resume the same run at step 30, remain within the existing five-epoch budget,
select by a frozen validation-only seven-key ranking, and stop after patience one. It must finish
and verify before any language-development work; calibration is conditional on the unchanged full
quality gate.

The D-061 continuation subsequently completed all 145 steps/five epochs and selected epoch 4,
step 116 from validation only. Routeable TaskSpec/object/bin accuracy reached 100%, while the
unchanged rejection gate still failed (7.5% false-route and 63.64%/72.22%/94.44% rejection-class
recall). M5A.1 is therefore a bounded validation-only analysis of the frozen checkpoint. It records
per-example logits/errors, compares exactly four predeclared decoders over one fixed finite grid,
optionally fits one validation status temperature, and either publishes a new decoder runtime
identity or freezes the classifier as a rejected baseline. It performs no training and cannot open
language development, final, or control sources.

## M5B — Sealed modular final and possible SmolVLA handoff (planned)

Run a separate immutable M5A final benchmark only if at least one learned router passes every
development gate. The sealed final uses 12 unseen scenes x 6 tasks and only Oracle plus the single
locked router (72 + 72 = 144 physical episodes). A future SmolVLA comparison requires a later explicit decision and must preserve
the M5A schedules, frozen-controller ceiling, and failure attribution. M5A never starts it
automatically.

## M6 — Evaluation and release hardening (planned)

Unify policy evaluation, regression thresholds, reproducibility reports, command-line workflows,
and GPU-runner CI when suitable hardware becomes available.
