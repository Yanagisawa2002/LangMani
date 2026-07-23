# LangMani roadmap

## LangMani 2.0 Phase 1 - stable evaluation foundation

Phase 1 starts after the immutable `v1.0.0` release. It validates the frozen release manifest,
audits generated artifacts without claiming reproduction, defines one pick-and-place skill family
with six task instances, recovers one canonical PerTask ACT baseline, and introduces generic
policy/action-chunk plus unified evaluation contracts. It does not add a new architecture, router,
dataset, reward, task, or training run. Native runtime acceptance is a separate real-policy smoke
over one task and three deterministic seeds; no-GPU checks remain structural only.

Implementation, v1 release validation, canonical checkpoint recovery, CPU-safe verification, and
native runtime acceptance are complete. The accepted RTX 5090 EGL/Vulkan run evaluated seeds
41001/41002/41003 with 3/3 successes, 40 real policy queries, and 392 real environment steps. The
independent entry gate returned `phase2_authorized=true`. Earlier Docker, Windows, and GLX failures
remain explicitly invalid infrastructure attempts.

The next phase may compare another real policy architecture only after it implements this same
adapter and evidence contract. It may not modify or relabel the frozen v1 release.

## LangMani 2.0 Phase 2 - multi-skill data foundation (Phase 2B complete)

The required real Phase 1 smoke and its independent verifier pass. Phase 2 implemented the
parameterized `push_to_region` environment, its standard/hard variations, conservative evaluation,
and a deterministic Panda/mplib expert. Phase 2A then stabilized lateral pushing through bounded,
geometry-specific contact selection, fail-safe cylinder correction, and pre-execution action-bound
validation. The unchanged smoke reached 8/8 standard and 6/8 hard; the fixed target validation
reached 46/50 standard successes (92%) and 23/30 hard successes (76.7%). Standard forward remained
24/24, with zero workspace-exit and zero action-bound events. The expert quality gate is true.

Phase 2B completed the separately authorized deterministic pushing-data stage. The real collector
preserved 516 attempts, admitted 397 demonstrations after 397/397 independent action replays, and
retained 119 rejected attempts. The LeRobot 0.6 export contains 397 episodes and 53,297 aligned
frames across six stable, leakage-free splits. Its compatibility audit left the historical
360-episode pick-and-place source immutable and produced only a content-bound multi-root index.
The independent final verifier passed every collection, replay, schema, split, readback,
provenance, and compatibility gate, so `smolvla_phase2c_authorized=true`.

The exact next stage is a separately invoked Phase 2C SmolVLA implementation/training milestone.
That authorization is dataset readiness only: no SmolVLA dependency, adapter, model, training run,
evaluation, or learned-policy Phase 2 result exists yet, and Phase 2B starts none automatically.

The later Phase 2B.2 byte-reconstitution work uses the new `langmani/phase2b-push-v2` identity and
never reuses v1 bytes or hashes. Two complete expert gates reached 75/100 and 11/100; ten later
G--P diagnostic candidates were also rejected. Candidate P reached 2/5 with one timeout, one
planning failure, and one zero-tolerance cylinder workspace exit. No candidate satisfied the
mandatory 95%/100-episode gate, so formal collection never started: v2 episodes=0, frames=0,
optimizer steps=0, and Phase 2C.2 is not authorized. Any further work requires a separately
authorized expert architecture, not another bounded recovery in this milestone.

Phase 2B.3 evaluated that separate simulator-MPC architecture only through its mandatory native
clone/restore precondition. Three audits preserved 100% categorical agreement and exact cold-to-
cold replay, but the final target-contact-free snapshot differed from live continuation by 4.394
mm in object pose. Phase 2B.3 is closed as Result C. Development and formal evaluation did not run,
formal seeds 66300--66399 remain sealed, and Phase 2B.4 collection, Phase 2C.2 training, and SmolVLA
remain unauthorized. A future attempt requires a new state/transition architecture rather than
another configuration adjustment.

Phase 2B.3.1-RR was a separately isolated feasibility audit of reset plus complete historical
prefix replay. It stopped at its transcript prerequisite: the compact diagnostic evidence contains
action counts, indices, and hashes, but no original executable `float32[8]` action arrays from step
zero and no complete call-sequence identity bound to them. It is closed as
`RESULT_D / HISTORICAL_PREFIX_UNAVAILABLE`. No protocol was frozen for execution, no historical
episode or boundary was selected, and no GPU, simulator, reset, replay, probe, runtime measurement,
expert, collection, or training ran. Result D is not a new physical transition result and leaves
the Phase 2B.3 Result C evidence unchanged.

Phase 2B.4-F0 is complete as Result C. Its isolated pipeline froze an 87D current-state teacher
observation, a no-privilege future-student schema, one tanh-affine native action distribution,
reward revision 0, disjoint seed namespaces, 128 GPU environments, and a 1,048,576-step
single-configuration micro budget. Vectorization, reward/action audits, one real PPO update, and
checkpoint reconstruction passed. The micro run had zero training successes and no material
return/progress improvement; deterministic evaluation reached 0/48 with 22 wrong-object
displacements and 12 workspace exits. The hard stop prevented broader development. Formal seeds
66300--66399, full PPO training, the 95/100 qualification, collection, datasets, SmolVLA, and
student training remain sealed; full PPO training is neither eligible nor authorized.

Phase 2B.4-F1 completed as Result C. Its state-centered bounded residual transform passed exact
likelihood, checkpoint, and 100,000-action legality checks and materially reduced local arm/TCP
motion in paired real physics. The only 262,144-step Probe A nevertheless had zero training
successes. Fixed evaluation reached 0/32 success, 0/32 correct contact, 0/32 target progress, seven
wrong-object interactions, and zero workspace exits. Probe B was not authorized or run. The custom
PPO teacher route is frozen for this task formulation. No third probe, reward retry, sweep, resume,
Stage 1/2, full curriculum, formal qualification, collection, dataset export, SmolVLA, or student
training is authorized.

M0 through M3B, M4 full, M4.1 target smoke, M4.2 target-development, M4.3a, and M4.3b
target-development are complete on native targets. M4 full is experimentally and physically
validated, but its declared quality gate is false. M4.2 rejected TaskToken and M4.3b rejected
FactorFiLM after development; `m42_final_v0` remains sealed and unaccessed. The shared-ACT
architecture search is therefore closed. M5A is the latest completed milestone: it adds
modular language-to-TaskSpec routing over the six frozen PerTask ACT controls. Its separately
authorized sealed final has completed with valid physical evidence but failed its conjunctive
language/control quality gate. M5B now implements the optional canonical-JSON LatentGuard
initial-state proposal bridge. Its implementation and CPU fixture gates are complete, but the real
single-seed probe is blocked because the current execution server lacks the accepted controller
registry, runtime-selection record, ACT checkpoint, and processor artifacts. SmolVLA and bounded
M6B execution have not started and remain unauthorized.

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

The real CPU analysis completed at implementation commit `f45115b1d070001e9e82567eed34bb3dcd99149a`
with evidence fingerprint
`sha256:f2401fcb5b054c79c3b7e9674321eefcf9576dc4dcc5407bd96151db9e9b518d`.
The selected conservative decoder used identity temperature, route threshold 0.90, margin 0.00,
object confidence 0.75, and bin confidence 0.85. It achieved zero false routes and 99.44%
routeable full TaskSpec accuracy, but only 71.21%/80.56%/94.44% ambiguous/unsupported/malformed
recall. The full conjunctive gate failed, so no runtime was published, the classifier is frozen,
and additional training or seed authorization remains false. Language development did not start.

M5A.2 adjusted the next stage rather than reopening that result. The classifier remained a frozen
offline negative baseline, RuleRouter remained the deterministic baseline, and the sole learned
candidate was pinned `Qwen/Qwen3-1.7B` through `StructuredLocalLLMRouterV0`. Its 20-example smoke
passed, but the complete 300-example validation reached only 85.56% full TaskSpec accuracy,
14.17% false-route, 92.67% final schema validity, and 59.09%/52.78%/50.00%
ambiguous/unsupported/malformed rejection recall. The conjunctive gate failed, so Qwen3-1.7B is
also frozen as an offline negative baseline and language development did not open.

M5A.3 authorizes one controlled capacity escalation only: `Qwen/Qwen3-4B-Instruct-2507` at exact
revision `cdbee75f17c01a7cc42f958dc650907174af0554`. It keeps the corpus, validation order,
few-shot IDs and labels, semantic prompt, strict parser, one-repair policy, metrics, and gates
unchanged. The stage performs BF16 inference on one GPU with no training, quantization, fallback,
ACT/controller load, or robot environment. Smoke gates validation; validation gates language
development; neither failure authorizes an 8B model or prompt change. Independent immutable
evidence distinguishes correct offline execution from quality promotion, and physical target
validation remains false.

M5A.4.1 completed the bounded offline safety/taxonomy amendment and selected the unchanged frozen
`NeuroSymbolicRouterV0`. Its separately authorized one-scene control smoke is also complete. The
read-only source bound exactly six frozen PerTask ACT controllers, all six learned commands routed
to the matching TaskSpec/controller, and the selected-router rejection probe performed zero
controller lookup, environment reset, or environment step. Six learned and six Oracle episodes ran
physically; each achieved 5/6 task success with one timeout. Independent verification passed and
keeps dispatch correctness separate from observed controller quality. No later development stage,
final split, `m42_final_v0`, or SmolVLA is authorized automatically.

The bounded three-scene paired screen completed on the three disjoint predeclared development
seeds. All 18 Oracle/NeuroSymbolic pairs had exact initial physical state and controller identity,
and every learned route matched its scheduled TaskSpec. Oracle and NeuroSymbolic each achieved
15/18 success; the same three pairs timed out on both paths, with no routing, wrong-object,
wrong-bin, off-table, malformed-action, non-finite-action, infrastructure, or M2 failure. The six
fixed rejection probes performed zero lookup, policy reset, environment reset, and step. The fresh
independent verifier accepted the immutable compact and raw evidence, so the conjunctive gate locks
the selected router and authorizes a separately invoked six-scene development stage. That later
stage has not run.

The full six-scene orchestration is now implemented for the separately authorized target run. It
rehashes the three-scene parent before loading the router, runs Oracle immediately before
NeuroSymbolic for each of 36 new development pairs, requires exact reset-state equality, and uses a
fixed nine-category rejection no-op set. A compact immutable archive and fresh-process verifier
recompute all 72 atoms, the locked 35/36 routing gate, controller/safety/action/latency metrics, and
the bounded `M5AFinalAuthorizationV0`. Implementation completion is not physical evidence; the
target benchmark subsequently completed at implementation commit
`f8c583aef35e8e64d758ef5aeddf6f11ea331be9`. All 36 routes and initial-state pairs matched. Oracle
and NeuroSymbolic each achieved 25/36 success, with 25 pairs both succeeding and 11 pairs sharing a
controller timeout. The nine rejection probes performed zero runtime work, the fresh verifier
passed every item, and the quality gate created the bounded final authorization. Final remains
sealed and was not executed.

## M5A sealed modular final (complete; quality gate failed)

The separately authorized immutable final used 600 language examples, 12 unseen scenes x six tasks,
and only Oracle plus the locked NeuroSymbolic router (72 + 72 = 144 physical atoms). The completed
pipeline and fresh-process verifier passed, but quality did not: NeuroSymbolic routed 61/72 tasks,
safely rejected 11 routeable commands, and succeeded on 46/72 episodes versus Oracle's 55/72. The
final language safety, rejection taxonomy, control, and combined gates are all false. Zero unsafe
wrong-object/wrong-bin route, runtime work after rejection, arm projection, infrastructure failure,
or M2 use was observed.

Implementation Git `0c5bb7de045e936cd6a3ce850d620bba1d923a3d` produced the immutable run;
verifier Git `b2489c73a10546ed1d4a2956649096d6255cd980` independently accepted it with
`final_pipeline_validated=true` and `physical_target_validated=true`. The result is terminal quality
evidence rather than recovery authority: `final_quality_gate_passed=false` and
`smolvla_go=false`.

## M5B — LatentGuard initial-state proposal bridge (implementation complete; target probe blocked)

Implementation commit `9713f7503dae6612309d0ef8a499d93f6c3899a3` adds an optional JSON-only
boundary between the Python 3.12 LangMani runtime and the separate Python 3.11 LatentGuard runtime.
It deterministically selects one accepted PerTask ACT controller, binds its task/checkpoint/
processor/action-space identities, permits exactly one seeded reset and one `[50,8]` policy query,
and reuses `BoundedActionEnvPostprocessorV0` to project candidate actions without opening a second
simulator. The bridge never executes a candidate action and keeps raw, projected, and actually
executed action identities separate.

The implementation, bridge unit suite, clean isolated-Linux validation, build, and installation
diagnostics passed. The real proposal probe did not run because the current execution server does
not contain the accepted controller registry, runtime-selection record, ACT checkpoint, or
processor artifacts required to construct an artifact-validated binding. The bridge failed closed
before environment construction: reset, policy-query, `env.step`, candidate-projection, and outcome
counts are all zero. No canonical real proposal was exported.

M5B target readiness is therefore `blocked`, not `not started`: source implementation is complete,
while the physical reset-only probe remains pending restoration and digest validation of the frozen
controller artifacts. This stage makes no performance or task-outcome claim and does not authorize
bounded M6B execution, SmolVLA, training, rollout, or model selection. A future SmolVLA comparison
would still require a separate explicit research decision and must not reinterpret M5A pipeline
validity as model quality.

## M6 — Release & Portfolio (complete)

Freeze LangMani v1 without changing runtime behavior, model weights, datasets, schedules, quality
thresholds, or historical evidence. M6 owns the `v1.0.0` release tag, a source-controlled frozen
results index, an 8–12 page technical report, a 3–5 minute evidence-labeled demo video, a concise
README homepage and architecture diagram, resume/interview material, and four audited cases:
success, safe false rejection, correct-route control timeout, and safe rejection.

Generated PDF/video outputs remain outside Git. Their source narratives and provenance are
source-controlled. M6 does not authorize M5B target execution, M6B, SmolVLA, controller retraining,
new data, new tasks, or reinterpretation of a failed v1 quality gate. Any new research must begin in
a later version or separately authorized milestone.
