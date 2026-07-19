# M5A modular language routing specification

## Scope and evidence boundary

M5A maps an out-of-band natural-language command to one of the six stable M1
`TaskSpec` values, or rejects the command before controller execution. It then dispatches exactly
one frozen M4 PerTask ACT controller. M5A is a modular language baseline, not an end-to-end VLA and
not a continuation of the shared-ACT architecture search.

The decision follows the completed target-development comparison on `m42_dev_v0`: PerTask reached
56/72, State-OneHot reached 36/72, and FactorFiLM reached 38/72. FactorFiLM training,
validation-only selection, fresh-process reload, and physical development evaluation completed,
but its quality gate failed and final evaluation was not authorized. M5A therefore freezes the six
PerTask controls. It does not retrain them, tune FactorFiLM, open `m42_final_v0`, or start SmolVLA.

The language text remains outside Gym observations and per-step `info`. M1 is reset with the
scheduled oracle `TaskSpec`; a predicted `TaskSpec` selects a controller only. This separation is
required so a wrong route cannot redefine the M1 success oracle and become a false success.

## Supported task space

The only objects are `red_cube`, `green_cube`, and `blue_cube`. The only destinations are
`left_bin` and `right_bin`. All successful routes use M1 `instruction_template_id=canonical_v0`.
M5A template-family IDs are language-corpus metadata and must never be copied into the M1 template
field. Canonical task ordering is object-major then bin-minor.

## Router decision and rejection contract

`RouterStatus` contains exactly:

- `route`
- `reject_ambiguous`
- `reject_unsupported`
- `reject_malformed`

A routed decision contains one valid object, one valid bin, the corresponding stable M1 task ID,
router identity, audit evidence, supported confidence metadata, and a canonical-JSON/SHA-256
decision fingerprint. A rejected decision contains no object, bin, executable `TaskSpec`, or stable
task ID. It contains a machine-readable rejection reason and cannot reach controller lookup,
policy reset, environment reset, or `env.step`.

The rule and classifier routers may expose documented confidence. The local LLM must set
`confidence_available=false` unless a defensible score definition is implemented. Model-written
probabilities are not accepted as confidence.

## Deterministic derived language corpus

M5A owns a derived corpus and does not modify M3A or M3B. Each immutable `LanguageExample` records:

- stable example ID from canonical JSON and SHA-256;
- raw text and audit-only normalized text;
- template-family ID and lexical-variant IDs;
- expected status;
- expected canonical M1 `TaskSpec` only when routeable;
- expected rejection reason only when rejected;
- deterministic generation provenance;
- split and schema version.

Corpus generation is deterministic. Python `hash()` and filesystem enumeration order are not
persistent identities. Routeable classes are exactly balanced. Counts are locked before any model
evaluation:

| Split | Routeable per TaskSpec | Rejected total |
| --- | ---: | ---: |
| train | 100 | 300 |
| validation | 30 | 120 |
| development | 40 | 180 |
| final | 60 | 240 |

The corpus includes direct, polite, object-first, destination-first, telegraphic, contextual,
punctuation/case, verb/determiner, clause-order, indirect-unambiguous, and resolved-correction
families. Rejection families include missing arguments, conflicting objects or bins, unsupported
objects or destinations, multiple tasks, unresolved corrections, contradictory negation, noise or
empty text, safe malformed text, unsupported robot actions, and unsupported spatial references.
Ambiguous or contradictory commands are never converted into guessed routes.

### Family isolation

The atomic split unit is the template family. Lexical substitutions of one structural template
remain in one split. A validator rejects a family ID or near-duplicate structural signature that
appears in train, validation, development, and final more than once. Development and final command
texts are disjoint. Training, checkpoint selection, calibration, threshold selection, and prompt
examples cannot consume development or final examples.

Structural and near-duplicate signatures are derived from the normalized template skeleton and
never include the split name. The isolation report fingerprints those claims, so the check cannot
pass merely because train/validation/development/final was embedded into each signature.

The development-safe corpus builder does not import `_final_language_authority.py`. Its final split
contains only opaque semantic example IDs, exact aggregate counts, and a source-controlled sealed
content digest; raw final commands are absent from the returned corpus and its development archive.
Only a separately authorized future final entry point may pass `authorize_final=true` and lazily
import that authority. At that boundary, a deterministic semantic-slot map resolves each sealed ID
to exactly one content-bound final example before use. Importing or materializing it is forbidden
in target-development.

## Locked language and control schedules

The following identities are generated and locked together before training:

1. `m5a_language_dev_v0`
2. `m5a_language_final_v0`
3. `m5a_control_dev_v0`
4. `m5a_control_final_v0`

The development control authority contains 10 new scene seeds partitioned before execution into
three disjoint, ordered stages: one smoke scene, three screening scenes, and six full-development
scenes. These contain 6, 18, and 36 scene-task pairs respectively. The sealed final authority
contains 12 different new seeds times six tasks, or 72 scene-task pairs. The generator excludes all
available M3A accepted and rejected candidates, all M3B split seeds, M4 fresh/smoke/tiny seeds,
M4.1 diagnostics, both M4.2 locks, and the M4.3 development source. Sidecar test seed identities
may be read only for exclusion; M3B test observations, actions, frames, and videos remain unopened.

Each physical stage has its own content fingerprint and parent-authority fingerprint. Smoke always
runs Oracle for six episodes and one primary promoted learned router for six; a second learned
router runs six more only if language development promoted it, for a maximum of 18. Screening runs
Oracle for 18 and each smoke-promoted learned router for 18, for a maximum of 54. Full development
runs Oracle for 36 and exactly one screen-selected learned router for 36, exactly 72. RuleRouterV0
remains a mandatory offline baseline and does not consume the expanded physical budget. Across all
pre-final stages the maximum routed physical cost is 144 episodes.

Target-development may validate final locks and fingerprints but cannot materialize final language
texts or final control episodes. A future final command requires separate authorization. No M5A
development command automatically grants it.

## Routers

### RuleRouterV0

The rule baseline uses Unicode/case/whitespace normalization, explicit object and bin synonym
tables, and fixed conflict, negation, unsupported-concept, and malformed-input rules. It records
matched spans and the exact fired rule. Normalization preserves negation and cannot erase a
conflict. The router uses no embedding, classifier, or LLM.

### FactorizedTextClassifierV0

The classifier has one shared compact pretrained encoder and three heads:

- status: route, reject ambiguous, reject unsupported, reject malformed;
- object: red, green, blue, none;
- bin: left, right, none.

The object and bin losses are masked for rejected examples. The preferred encoder is
`distilbert-base-uncased`; target execution must pin the exact encoder and tokenizer revisions and
must stop instead of choosing a different model. Train examples are the only gradient source.
Validation alone selects a checkpoint, fits status temperature, and selects a routing threshold.
Development and final cannot affect those choices.

Development uses exactly one training seed (`0`), one encoder/revision, and one optimizer
configuration. It does not claim robustness across initializations. The declared maximum is five
epochs; with the locked 900-example training corpus and batch size 32 this is 29 steps per epoch and
at most 145 optimization steps. The authoritative run pauses after the first completed epoch
(step 29, which is earlier than 20% of the maximum only when those points differ), writes a
resumable checkpoint, and resumes the same immutable run fingerprint. That checkpoint includes
model, optimizer, constant scheduler, processor, Python/NumPy/Torch CPU and CUDA RNG state, best
validation state, validation history, deterministic data-progression metadata, the pinned label
mappings, corpus/split fingerprints, Git/dependency identity, and the preflight/owner references.
The pilot records one finite telemetry row per optimizer step, including component losses,
component/head gradient norms, learning rate, examples processed, epoch progress, throughput,
loader/step latency, and allocated/reserved CUDA memory. Restarting from step zero or changing
seed/encoder is a different run and is rejected.

Before the authoritative run, a fixture must prove gradients, finite masked losses, one optimizer
step, and deterministic reload. A deterministic 48-example tiny-overfit set covers all six
TaskSpecs, all four statuses, and all major rejection classes. Its gate requires at least 98%
status accuracy, at least 98% full TaskSpec accuracy on routeable examples, no executable rejected
decision, finite losses, and exact reload behavior.

The pilot validation gate requires routeable full TaskSpec accuracy at least 85%, object/bin
accuracy at least 90%, rejected-command false-route rate at most 10%, 100% schema validity, finite
values, and nonzero recall for every rejection class. Pilot completion and pilot promotion are
separate facts. Failure preserves the checkpoint and stops the classifier path; it never triggers
another seed or encoder.

At the boundary the command writes exactly three atomic real files: `pilot`, `latest`, and
`validation_best`. It disposes the training instance, reconstructs the pinned model/tokenizer from
local cache, restores model/optimizer/scheduler/RNG/processor/data-progression state, and compares
a fixed validation logit fixture at absolute and relative tolerance `1e-6`. A separately launched
independent verifier repeats that reload in a fresh process, recomputes the complete 300-example
validation evidence (routeable, rejected, per-task, per-family, confusion, and latency fields),
and recomputes the conjunctive promotion decision. The pilot command and verifier never resume a
training step.

### Post-pilot recovery amendment

The original predeclared pilot protocol rejected the seed-0 candidate after step 29: full TaskSpec
accuracy was 85%, object accuracy 85%, bin accuracy 100%, false-route rate 25%, and ambiguous,
unsupported, and malformed rejection recall were 74.24%, 2.78%, and 0%. This result is not
retroactively described as a promoted pilot. After observing validation-only evidence, and before
opening any language-development, final, control, or LLM source, D-061 authorizes one bounded
continuation because the loss remained high and decreasing while the fixture/tiny-overfit and
complete-state reload contracts had passed.

`--target-pilot-recovery-resume` is a versioned exception for exactly the rejected authoritative
run and original `pilot.pt` SHA-256 recorded by D-061. It requires those identities plus explicit
five-total-epoch and patience-one arguments. The amendment binds the current clean implementation
commit, unchanged training configuration, validation schedule, and three-role retention policy.
It is attached to the old run fingerprint and does not create a new seed or run identity. Ordinary
`--target-resume` still requires original pilot promotion and cannot accept a rejected pilot.

Before step 30, the command performs a read-only 15-item audit covering train/validation status and
rejection distributions, object/bin/TaskSpec counts, rejected-loss masking, class indices and
label mapping, template-family isolation, unsupported/malformed labels, pilot status-head
gradient, sampler coverage, and all 900 first-epoch examples. Any defect stops without repairing
the corpus. The exact pilot checkpoint must restore model, optimizer, constant scheduler, RNG,
processor, and data progression with next step 30.

Recovery validates only after complete epochs and ranks checkpoints lexicographically by: higher
full TaskSpec accuracy, lower false-route rate, higher rejection macro recall, higher object
accuracy, higher bin accuracy, lower validation loss, then earlier epoch. One subsequent completed
validation interval without improvement stops training; the total remains at most five epochs.
The original pilot, atomic latest, and atomic validation-best are the only model files. Every epoch
persists the complete routeable/rejected/all validation report and deterministic repeatability.

Bounded completion, checkpoint selection, and full quality are separate flags. The unchanged full
quality gate below is recomputed on a freshly reloaded selected checkpoint. Temperature and
threshold calibration plus a promoted runtime artifact are created only if every quality condition
passes. A valid below-threshold continuation reports `passed=true`,
`classifier_full_quality_gate_passed=false`, and `classifier_calibration_validated=false`, then
stops before language development.

After promotion, validation runs at steps 29, 58, 87, 116, and 145 with patience one completed
interval after the best. Only the fixed `pilot`, `latest`, and `validation_best` checkpoint roles
are retained; role updates are atomic and no periodic checkpoint sweep is created. The final
classifier promotion gate requires routeable full TaskSpec accuracy at least 95%, object/bin
accuracy at least 97%, false-route rate at most 3%, ambiguous rejection recall at least 90%,
unsupported and malformed rejection recall at least 95%, and 100% schema validity. Validation only
selects the best checkpoint, status temperature, and routing threshold.

A command routes only when calibrated status predicts `route`, object and bin heads are valid, and
full-route confidence meets the threshold. The predeclared threshold objective maximizes valid
full-TaskSpec accuracy subject to rejected-command false-route rate at most 3%, then prefers higher
rejection recall and finally the higher threshold.

### StructuredLocalLLMRouterV0

The LLM baseline loads exactly one explicitly configured local instruct model between roughly
0.5B and 3B parameters. Target execution records official model-card/license review, exact model
and tokenizer revisions, dtype, quantization, prompt, train-family few-shot IDs, and generation
configuration. There is no cloud call and no model fallback or sweep.

The target default dtype is `bfloat16`, and the only supported quantization value is `none`. These
values are fingerprinted and reported. Unsupported quantization or an unavailable requested dtype
is a hard failure; M5A does not silently quantize or change precision.

Exactly one versioned prompt artifact is frozen before development. Every few-shot example ID is
from a train family only; validation may be used only to freeze the prompt/configuration and never
as an in-context example source. Language-only evaluation and control evaluation must reuse the
byte-identical prompt template, ordered example IDs, generation configuration, and prompt
fingerprint. Partial development output cannot trigger prompt editing.

The output is a strict JSON object with exactly `status`, `target_object_id`, `target_bin_id`, and
`reason`. The project parser rejects duplicate keys, non-finite constants, Markdown fences,
surrounding prose, missing or extra keys, unknown values, and inconsistent null fields. The
installed generation stack provides no declared JSON-grammar API, so the supported fallback is
deterministic greedy generation (`do_sample=false`), strict parsing, and at most one explicit
format-repair attempt. Temperature zero is recorded as a request; it is not misreported as an
effective sampling temperature when sampling is disabled. Chain of thought is neither requested
nor stored.

## Frozen controller registry and runtime

The immutable registry contains exactly the six selected PerTask checkpoints in canonical order.
It is built through a metadata-only allowlist: the M3B completion marker, completed PerTask run
manifests, validation-only selection records, selected checkpoint artifacts and hashes, selected
validation runtime manifests, and the M4.2 runtime-selection lock plus its referenced experiment
manifest. It rejects comparison/verification files and any test, historical-fresh, or final result
path. In particular, it does not expand the source M4 data contract's evaluation schedules into the
portable registry.

Each entry binds stable task ID, run and selected-checkpoint fingerprints, selected step, selection
fingerprint, model/preprocessor/postprocessor component fingerprints, train-statistics and M3B
fingerprints, producer Git commit, the deployable action/state/image contract projection, an opaque
fingerprint of the full source contract, and the locked runtime. Absolute paths are stored only as
locators and are excluded from the portable registry fingerprint. Before any controller load, the
active M1 environment's action-space bounds must exactly match the frozen selected-validation
contract.

The runtime is the M4.2 selection: `pd_joint_pos`, one environment, H=10 execution, and explicit
`project` action-bound handling. Raw, binary-transformed, projected, and executed actions remain
four separate evidence streams. Because this runtime does not apply binary gripper transformation,
its aligned binary-transformed stream is explicitly absent (`null`) at every step rather than being
silently conflated with projection. Confidence never becomes a continuous controller input.
Controllers are not blended, interpolated, tried in parallel, or selected by post-hoc outcome.

For a routeable control episode, the environment is reset with the scheduled oracle task while the
predicted decision selects the PerTask policy. Policy, preprocessor, postprocessor, and H=10 queue
state reset at every episode. Oracle routing uses the same scene, controller bytes, runtime,
episode limit, renderer, and M1 success definition.

## Evaluation and failure attribution

Language-only evaluation runs RuleRouter, the selected/calibrated classifier when its training gate
passed, and the frozen local LLM on validation and `m5a_language_dev_v0`. It reports routeable TaskSpec/object/bin accuracy,
route recall, false rejection, per-task/family results and confusion matrices; rejected accuracy,
precision, recall, false-route rate and reason recalls; and overall schema validity, malformed
rate, coverage, selective accuracy, supported calibration metrics, latency percentiles, and fixed-
configuration repeatability. Rejected examples never share the ordinary TaskSpec denominator.

Learned candidates are promoted only when development full TaskSpec accuracy is at least 95%,
object/bin accuracy at least 97%, false-route rate at most 3%, ambiguous rejection recall at least
90%, unsupported/malformed rejection recall at least 95%, schema validity at least 99%, and
deterministic repeatability is 100%. The LLM additionally requires malformed output after bounded
repair at most 1%. Promoted routers are ranked by full TaskSpec accuracy, false-route rate,
rejection macro recall, malformed rate, then p95 latency. One primary router is locked for smoke;
the other learned router may remain an offline comparator unless it independently passed.

Physical development progresses through the immutable 1/3/6-scene partitions and excludes every
failed candidate. Each stage records the expected task, decision, dispatched controller, active M1 episode spec,
task outcome, safety/task diagnostics, action evidence, inference latency, and end-to-end latency.
In addition, target execution runs one dedicated rejection probe against the active M1 environment
before the routeable schedule. Its immutable evidence must show zero controller lookup, zero reset,
zero `env.step`, and a safe rejection; the independent validator rechecks that artifact and its
completion reference.
The attachment's failure list contains 14 values, all preserved exactly:

- `routing_correct_control_success`
- `routing_correct_control_failure`
- `routing_wrong_object`
- `routing_wrong_bin`
- `routing_wrong_task`
- `routing_false_rejection`
- `routing_missed_rejection`
- `routing_malformed_output`
- `controller_load_failure`
- `controller_inference_failure`
- `invalid_action`
- `timeout`
- `environment_failure`
- `infrastructure_failure`

Control schedules contain only routeable commands, so each control episode receives exactly one
of these values. A correctly rejected language-only command produces a safe rejection record and
no control episode; it is not mislabeled as control success. Routing-correct controller failure is
not counted as a language error, and wrong routing is not attributed to ACT.

## Progressive physical gates

The one-scene smoke is a wiring/safety gate, not a generalization claim. Any rejection that reaches
controller lookup or `env.step`, controller/TaskSpec disagreement, uncleared policy queue,
infrastructure failure, or M2 expert use stops the candidate.

The three-scene screen permits at most 1/18 routing error, wrong-object route, wrong-bin route, or
false rejection. Controller execution after rejection, target in wrong bin, target off table, arm
projection, non-finite/malformed action, and M2 expert use must be zero. Candidates are ranked by
success gap to Oracle, routing errors, false routes on rejected commands, wrong-object
interactions, then p95 latency. Exactly one router is locked for full development.

The six-scene full-development gate retains the language gate and requires the selected router's
success gap to Oracle at most 2/36; wrong-object and wrong-bin routing at most 2/36 each; false
rejection at most 3/36; and zero target-in-wrong-bin, target-off-table, arm projection,
non-finite/malformed action, post-rejection execution, or M2 expert call. Passing sets
`development_quality_gate_passed=true` and `final_benchmark_authorized=true`; failure leaves both
false. Correct stage execution and quality promotion remain separate.

The observed identity
`P(correct route) * P(control success | correct route)` is reported beside direct counts; it never
replaces them. Passing the development experiment means the declared stages executed correctly,
not that quality passed.

## Lifecycle, outputs, and access flags

Router identities bind corpus/split fingerprints, revisions, architecture or prompt, training and
calibration configuration, selected threshold, Git commit, dependency versions, and M5A schema.
Training and evidence use owned staging, checksums, atomic promotion, and immutable completed
directories. Model weights, corpora, datasets, checkpoints, and videos remain outside Git.
Every public writer rejects symlinked or junction-backed paths and any overlap between its managed
outputs, read-only inputs, source trees, M3A/M3B archives, frozen policy roots, or historical
verification evidence before a stage may run or reuse prior evidence.

The target verifier does not trust a producer summary by itself. It reopens the promoted corpus,
classifier, router-evaluation, and control-evidence roots; recomputes canonical fingerprints,
payload/file checksums, schedule and router identities, metric summaries, controller/action
provenance, and failure-attribution counts; and then cross-checks them against the owner reports and
experiment manifest. A lifecycle flag is set only after its persistent evidence survives that
independent validation.

The stage verifier reports the following independent flags. `passed=true` means only that the
requested stage completed correctly; it does not imply promotion to the next stage:

| Flag or flag group | Required meaning |
| --- | --- |
| `implementation_validated`, `corpus_validated`, `split_isolation_validated`, `rule_router_validated` | Portable implementation, exact counts/isolation, and deterministic RuleRouter contracts passed. |
| `classifier_fixture_validated`, `classifier_tiny_overfit_validated` | The two bounded pre-training gates independently passed. Neither claims full training. |
| `classifier_pilot_completed`, `classifier_pilot_promoted` | The authoritative run reached and saved step 29; the second flag alone records whether validation authorized same-run resume. |
| `classifier_recovery_resume_authorized`, `classifier_recovery_resume_completed`, `classifier_full_quality_gate_passed` | D-061 authorized the exact rejected pilot continuation; bounded execution completed; and, separately, the unchanged full validation gate passed or failed. |
| `classifier_training_completed`, `classifier_checkpoint_selected`, `classifier_calibration_validated` | The promoted single-seed run completed or early-stopped; validation only selected and calibrated it. |
| `llm_router_loaded`, `llm_prompt_locked`, `language_development_completed` | One pinned inference-only LLM and immutable train-example prompt were evaluated with the offline routers. |
| `classifier_candidate_frozen`, `classifier_offline_baseline_validated`, `classifier_dispatch_prohibited` | M5A.2 retained the M5A.1 checkpoint as descriptive read-only evidence and gave it no dispatch authority. |
| `llm_model_identity_validated`, `llm_train_smoke_completed`, `llm_validation_completed`, `llm_validation_gate_passed` | The exact Qwen identity and train-only smoke passed structural/runtime checks; validation completion and quality promotion remain separate. |
| `llm_language_quality_gate_passed`, `learned_router_selected`, `one_scene_control_smoke_authorized` | Development quality alone may lock the local LLM and authorize, but not execute, the next physical stage. |
| `real_gpu_inference_validated`, `physical_target_validated` | M5A.2 sets only the first after real local-model inference; the second stays false because no robot rollout occurs. |
| `one_scene_control_smoke_completed`, `three_scene_control_screen_completed`, `selected_router_locked` | The staged physical gates completed and, separately, screening selected one router. |
| `oracle_control_development_completed`, `predicted_control_development_completed` | Oracle and the single selected router each completed the 36-pair full-development partition. |
| `failure_attribution_validated`, `physical_target_validated` | Independent attribution/action checks passed over the requested real target stage; fixtures never set physical validation. |
| `development_quality_gate_passed`, `final_benchmark_authorized` | The full 6-scene development gate passed and authorizes, but does not execute, final. |
| `language_final_accessed`, `control_final_accessed`, `test_split_accessed`, `smolvla_go` | Must remain false throughout development. |

Non-target verification validates implementation and fixtures only. It must leave classifier
training, development evaluations, authorization, all final access, SmolVLA, and physical target
validation false while still allowing `passed=true`.

Each target command produces one stage report, and `--verify-stage` validates only that report.
The retired monolithic `--target-development` interface fails clearly rather than starting the
chain. A physical stage may set its own completion/physical flags only after its immutable atoms,
failure attribution, action evidence, provenance, and forbidden-access fields pass independent
checks. Pilot completion may pass verification while promotion is false; likewise a completed
screen may pass while no router is selected, and a completed full stage may pass while quality is
false.
It must finish with `language_final_accessed=false`, `control_final_accessed=false`,
`m42_final_accessed=false`, `test_split_accessed=false`, `historical_fresh_accessed=false`,
`final_benchmark_completed=false`, and `smolvla_go=false`. It records but does not execute any final
authorization.

The access flags distinguish identity-only exclusion checks from opening evaluation content:

- `language_final_accessed=false` and `control_final_accessed=false` permit only opaque sealed IDs,
  counts, digests, and lock fingerprints. They prohibit raw final command text, materialized final
  episodes, reset, render, or rollout.
- `test_split_accessed=false` permits finalized M3B sidecar split/scene-seed identities and the
  completion fingerprint solely for exclusion/provenance. It prohibits test Parquet rows,
  observations, actions, frames, videos, tasks, and rollout results.
- `historical_fresh_accessed=false` permits only the committed exclusion summary's prior seed IDs,
  opaque schedule digest, and configuration lock. It prohibits opening historical fresh runtime
  schedules, episode commands, observations/actions, or result reports.
- `m42_final_accessed=false` permits validating the sealed lock identity and exclusion-only seed
  IDs, but prohibits materializing, rendering, resetting, or executing any `m42_final_v0` episode.

## Limitations and future handoff

M5A demonstrates modular command understanding plus dispatch to a fixed skill library. The
classifier and local LLM do not produce continuous control, and oracle PerTask ACT remains the
control ceiling. Only after immutable target-development evidence may a separately authorized
final benchmark compare qualified learned routers. A later SmolVLA study, if authorized, must use
the same language/control schedules, failure attribution, and frozen-controller modular baseline;
M5A never starts it automatically.

## M5A.1 frozen-classifier rejection analysis

The completed authoritative seed-0 classifier run selected epoch 4, step 116 from validation only.
Its routeable TaskSpec, object, and bin accuracies are 100%, but its observed false-route rate is
7.5% and ambiguous/unsupported/malformed rejection recalls are 63.64%, 72.22%, and 94.44%.
M5A.1 diagnoses this frozen result; it is not another training stage.

Inputs are limited to the selected checkpoint, train records for class/provenance auditing, and
the complete validation split. Development/final language, control, M3B test content, historical
fresh evaluation, `m42_final_v0`, and SmolVLA are inaccessible. The analysis reconstructs the
pinned encoder from configuration, loads checkpoint weights directly, runs two deterministic CPU
validation passes, and independently proves unchanged checkpoint bytes, model-state fingerprint,
and global step. It never constructs an optimizer or scheduler and never creates a seed or run.

Before scoring, `DecoderThresholdGridV0` locks route thresholds 0.40 through 0.95 by 0.05, route
margins -0.20 through 0.50 by 0.05, object/bin confidence thresholds 0.40 through 0.95 by 0.05,
and both identity and one validation-fitted status temperature. Exactly four candidates are legal:
`BaselineFourWayArgmaxV0`, `AggregatedRejectThresholdV0`, `HierarchicalStatusDecoderV0`, and
`ConservativeRouteDecoderV0`. No lexical rule, family ID, expected label, or post-inspection grid
refinement may enter decoding.

A configuration is eligible only at false-route <=3%, schema validity 100%, routeable full TaskSpec
>=95%, object >=97%, and bin >=97%. Eligible configurations are ranked, in order, by ambiguous,
unsupported, and malformed recall; rejection macro recall; lower false rejection; higher route
coverage; higher route threshold; then lexicographic configuration fingerprint. Promotion further
requires ambiguous >=90%, unsupported >=95%, and malformed >=95% with all preceding eligibility
conditions. These conditions remain conjunctive and unchanged.

Every validation example records its stable ID/text/family, expected labels, logits/probabilities,
baseline and selected decisions, confidence margins, entropies, and safe TaskSpec-free rejection.
The immutable artifact contains four-way and binary matrices, separated error categories, the ten
required error groups, per-family findings, calibration NLL/ECE, candidate summaries, the grid
lock, checksums, and a no-training audit. If the full gate passes, a separate runtime fingerprint
binds checkpoint, tokenizer, processor, decoder, temperature, Git, corpus, and validation split.
Otherwise no runtime is created, the classifier is frozen as an offline negative baseline, and
both additional-training and additional-seed authorization remain false. Either quality conclusion
can have `passed=true` when the analysis and independent verification are correct.

### Observed M5A.1 result

The real CPU execution at Git `f45115b1d070001e9e82567eed34bb3dcd99149a` produced immutable
analysis fingerprint
`sha256:f2401fcb5b054c79c3b7e9674321eefcf9576dc4dcc5407bd96151db9e9b518d`.
Status temperature fitting changed temperature from 1.0 to 1.4514016224, reduced NLL from
0.4119188027 to 0.3612778088, and reduced ECE from 0.0738781275 to 0.0421380654; nevertheless the
selection objective chose identity temperature. The selected `ConservativeRouteDecoderV0` used
route threshold 0.90, margin 0.00, object confidence 0.75, and bin confidence 0.85. It achieved
zero false routes, 179/180 route coverage, and 99.44% full TaskSpec accuracy, while rejection recall
was 71.21% ambiguous, 80.56% unsupported, and 94.44% malformed. Those three thresholds failed.
Both independent verifiers passed the no-training/checksum/access contracts. The authoritative
conclusion is `classifier_rejected_after_posthoc_calibration`; no runtime exists and additional
training/seed authorization is false.

## M5A.2 offline language-development comparison

M5A.2 changes the language-development eligibility contract without reopening M5A.1. The selected
`ConservativeRouteDecoderV0` remains a mandatory descriptive negative baseline, but it has no
dispatch, promotion, or final-selection authority. `RuleRouterV0` is the mandatory deterministic
offline baseline and is not a learned-router candidate. Exactly one learned candidate exists:
`StructuredLocalLLMRouterV0` backed by `Qwen/Qwen3-1.7B`.

The model and tokenizer both pin Hugging Face revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, use the declared Apache-2.0 license, BF16, no
quantization, and the public Transformers loading APIs. Evidence records the size and SHA-256 of
every required repository file. The official Qwen chat template is invoked with
`enable_thinking=False`; generation is greedy with `do_sample=false`, `num_beams=1`, one tokenizer
EOS policy, and at most 128 new tokens. No chain of thought is requested. Valid schema-only JSON is
retained for audit, while any non-schema text is retained only by hash and character count.

The versioned prompt declares the six object/bin combinations, four statuses, strict four-field
schema, bounded reason-code vocabulary, rejection policy, and no-guess/no-reasoning instructions.
Its fixed few-shot set contains six route examples plus twelve rejection examples, all from train.
The fixed train-only semantic smoke contains one example per canonical TaskSpec and one per each of
the fourteen corpus rejection reasons. A prompt lock is promoted only after real deterministic
smoke inference passes; a fixture cannot create validation/development metrics or authorize
control.

The output parser accepts exactly one JSON object with `status`, `target_object_id`,
`target_bin_id`, and `reason`. Route results require one of the three object IDs and two bin IDs;
all rejection results require both targets to be null. Unknown values, duplicate or extra fields,
multiple JSON objects, surrounding prose, mismatched reason/status pairs, and non-finite constants
are rejected. One schema-only repair is permitted. A failed repair becomes a non-executable
`reject_malformed` decision.

After smoke, the exact prompt/model/parser/generation runtime evaluates the complete 300-example
validation split once. Development opens only when all validation thresholds in this specification
pass conjunctively. It then evaluates all three routers on the complete 420-example
`m5a_language_dev_v0` split without tuning. The local LLM alone may produce a locked
`LanguageRouterSelectionV0`, and only when both validation and development gates pass. That lock
authorizes, but never starts, the later one-scene control smoke.

M5A.2 evidence is runtime-fingerprint-owned, written in staging, flushed, checksum-validated, and
atomically promoted. An independent reader recomputes all metrics and gates from raw records,
checks prompt train-only identity and immutability, verifies the pinned model files, proves the
classifier checkpoint/model remained unchanged with no optimizer, and rejects any development
files after a failed validation gate. A controller-registry boundary artifact contains only the
six canonical task IDs and explicitly records that no registry, ACT checkpoint, controller,
environment, or environment step was opened. Language final, every control schedule, M3B test,
historical fresh evaluation, `m42_final_v0`, and SmolVLA remain inaccessible.

For this language-only stage, `passed=true` means the requested smoke/validation and conditional
development protocol completed and independently verified. It does not imply a quality pass.
`real_gpu_inference_validated=true` requires real pinned-model inference;
`physical_target_validated` remains false because M5A.2 performs no robot rollout.

### Observed M5A.2 result

The fixed Qwen3-1.7B candidate passed all 20 train-smoke examples. On the exact 300-example
validation split it obtained 85.56% full TaskSpec, object, and bin accuracy; 14.17% false-route;
92.67% final schema validity; 7.33% malformed output after its one repair; and
59.09%/52.78%/50.00% ambiguous/unsupported/malformed rejection recall. Deterministic
repeatability was 100%. The validation gate failed, so language development was not accessed and
Qwen3-1.7B is frozen as a non-dispatchable, non-promotable offline negative baseline.

## M5A.3 one-model capacity escalation

M5A.3 asks only whether a larger fixed instruct checkpoint improves the same routing problem. The
sole authorized candidate is `Qwen/Qwen3-4B-Instruct-2507`, with model and tokenizer revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, Apache-2.0, BF16, no quantization, one visible GPU,
and the public Transformers model/tokenizer APIs. It is an inference-only stage: no optimizer,
training, fine-tuning, LoRA, model sweep, 8B fallback, hosted API, ACT controller, robot
environment, or environment step is permitted.

The semantic prompt text, train-only few-shot IDs and labels, 20-example smoke IDs, strict
four-field schema, reason vocabulary, parser, maximum one repair, generation bounds, validation
records/order, metric implementation, and all M5A.2 quality thresholds are unchanged. The 4B
tokenizer's official instruct template is a transport layer only and receives no invented thinking
switch. `prompt_equivalence.json` separately binds the 1.7B semantic prompt fingerprint, the 4B
rendered transport fingerprint, semantic-content fingerprint, few-shot IDs, schema, parser, and
repair-policy fingerprints. Validation cannot begin unless `semantic_prompt_content_equal=true`.

The exact train-only smoke must complete 20 examples, cover all six TaskSpecs and four statuses,
keep rejections non-executable, reach at least 99% final schema validity, and repeat
deterministically. Smoke failure stops before validation. A passing smoke opens the same 300
validation examples exactly once. The unchanged validation gate is conjunctive: TaskSpec >=95%,
object/bin >=97%, false-route <=3%, ambiguous >=90%, unsupported/malformed >=95%, final schema
validity >=99%, malformed after repair <=1%, deterministic repeatability 100%, and no prohibited
source access. Failure creates no development directory and records `stopped_at_validation=true`.

Only a validation pass may open complete `m5a_language_dev_v0`. Development evaluates RuleRouter,
the frozen classifier, frozen Qwen3-1.7B, and the locked Qwen3-4B runtime. It retains the M5A.2
gate plus every-TaskSpec >=90%, rejection-family false-route <=10%, and explicit non-routing of
empty/meaningless, conflicting-object, conflicting-bin, and unsupported-action inputs. A passing
development gate may set `learned_router_selected=true` and authorize a later separately executed
one-scene control smoke, but M5A.3 never runs robot control itself.

Evidence lives below
`outputs/diagnostics/m5a/qwen4b-escalation/<runtime-fingerprint>/`. Staging, fsync, checksum
validation, atomic promotion, immutable reuse, and an independent raw-record verifier are
mandatory. The scale report uses identical ordered validation IDs and reports 4B-minus-1.7B deltas
for semantic accuracy, false routes, rejection recall, schema/repair behavior, latency, generated
tokens, and GPU memory. `passed=true` means the authorized offline protocol and evidence completed;
it does not imply either quality gate passed. `physical_target_validated=false`, and language final,
control development/final, M3B test, `m42_final_v0`, and SmolVLA remain inaccessible.

## M5A.4 neuro-symbolic safety arbitration

The repeatedly observed validation split is quarantined as historical architecture-selection
evidence. `NeuroSymbolicRouterV0` is frozen from train-only examples plus already recorded
diagnostics, then promoted or rejected exactly once on untouched `m5a_language_dev_v0` after an
immutable access lock. Historical validation replay is explicitly
`post_selection_diagnostic_only=true` and cannot alter behavior.

`SymbolicLexicalFrameV0` records normalized lexical facts and every matched span without labels,
split identity, or dispatch. `QwenSemanticFrameV0` is a strict fact-only Pydantic object produced by
the fixed Qwen3-4B model under one cached Outlines 1.3.1 JSON grammar. It has no RouterStatus,
TaskSpec, final reason, confidence, or reasoning field. `DeterministicSafetyArbiterV0` is the sole
decision authority and applies malformed, unsupported, ambiguous, then route precedence. A route
requires one supported object and bin, exact symbolic/semantic slot agreement, supported action,
and no blocker. Any material disagreement rejects; no threshold or architecture sweep exists.

The target sequence is train smoke, non-selecting historical replay, immutable prompt/runtime lock,
one complete 420-example development pass, and independent verification. Development compares five
fixed routers, but only the hybrid is promotable. The original conjunctive quality thresholds and
safety-specific zero-false-route checks are unchanged. Correct offline execution may report
`passed=true` while the quality gate is false. Physical validation stays false and no control stage
starts automatically.

If the frozen train-only smoke fails, the command stops there and persists an immutable terminal
rejection with all 20 raw comparisons. The independent verifier requires historical replay, the
development lock, and development artifacts to be absent in that case. No semantic policy may be
changed from the smoke scores.

Evidence is fingerprint-owned under
`outputs/diagnostics/m5a/neuro-symbolic-router/<runtime-fingerprint>/`. It includes the symbolic,
schema, decoder, model, prompt, arbiter, runtime, smoke, historical, five-router development,
selection, manifest, and completion records. Completion is written last after fsync/checksum
validation and same-filesystem atomic promotion.

## M5A.4.1 safety-routing and rejection-taxonomy separation

M5A.4.1 is a pre-development protocol amendment over the immutable M5A.4 smoke evidence. It does
not alter `NeuroSymbolicRouterV0`, the Qwen3-4B weights or revision, prompt text, few-shot IDs,
Outlines grammar, semantic-frame schema, symbolic lexical parser, arbiter precedence, agreement
policy, corpus text, or labels. The original exact four-way smoke result remains 17/20. The three
status errors are all `reject_ambiguous` to `reject_unsupported`; every output is structurally
non-executable and no controller can be dispatched. Exact reason-code analysis also retains the
separate unresolved-correction/conflicting-objects diagnostic error.

`SafetyRoutingContractV1` distinguishes executable route, safe rejection, unsafe false route,
false rejection, and malformed decision. Any rejection status is safe only when object, bin,
TaskSpec, and task ID are all null. `RejectionTaxonomyContractV1` separately preserves exact
RouterStatus/reason accuracy, status recalls, macro recall, confusion matrices, and per-family
metrics. Because all three rejection statuses have the same no-dispatch behavior in M5A, taxonomy
quality is a required limitation and diagnostic, not an independent physical-execution gate.

The immutable amendment binds implementation Git, the source M5A.4 runtime and smoke artifact,
corpus/split identities, model/prompt/schema/parser/arbiter identities, and both new contract
fingerprints. The existing 20 records are reassessed without model inference. Safety smoke requires
6/6 correct complete TaskSpecs, 14/14 safe rejections, zero unsafe routes, zero false rejections,
zero malformed decisions, 100% schema validity and determinism, and no executable rejection.
Only after this gate passes is the unchanged router lock written. The lock precedes both the
300-example post-selection validation diagnostic and the first access to all 420 untouched
`m5a_language_dev_v0` examples.

Development compares RuleRouterV0 and the three frozen negative baselines with the sole promotable
NeuroSymbolicRouterV0. Promotion requires the predeclared routeable accuracy thresholds plus safe
rejection recall >= 97%, unsafe false-route rate <= 3%, 100% schema validity/determinism, zero route
on each named safety family, no executable rejection, no rejection family above 10% unsafe routes,
and no prohibited access. Exact taxonomy metrics never disappear and cannot be described as
calibrated or reliable when they fail. `passed=true` means the offline protocol and independent
verification completed; it does not imply taxonomy quality, safety promotion, or physical robot
validation. Control, language final, M3B test, `m42_final_v0`, and SmolVLA remain sealed.

Evidence is written under
`outputs/diagnostics/m5a/neuro-symbolic-safety-gate/<runtime-fingerprint>/` through fsynced staging,
checksum validation, same-filesystem atomic promotion, and a completion marker written last.

## Selected-router dispatch integration and one-scene control smoke

The completed M5A.4.1 artifact is the sole eligible learned-router source. A path-independent
dispatch identity binds its runtime/artifact fingerprints, router-lock fingerprint, exact model and
tokenizer revisions, model-file hashes, prompt and rendered-transport fingerprints, semantic
schema, lexical parser, safety arbiter, and the diagnostic exact-taxonomy limitation. Revalidation
must pass before a model, controller, or environment runtime is entered. The model cache is local
only; download fallback, another router, prompt change, and later physical stages are prohibited.

The read-only controller binding covers exactly the canonical six TaskSpec IDs in registry order.
It may inspect finalized metadata but sets controller-loaded/environment-created/reset/step fields
to false. A routed decision then selects exactly one PerTask checkpoint while the schedule TaskSpec
remains the M1 reset/evaluation oracle. A rejection probe uses the selected router on the active M1
environment boundary and must return with zero controller lookup, zero reset, and zero `env.step`.

The one-scene smoke executes six learned episodes and the six corresponding Oracle ceiling
episodes. Each learned decision must match its scheduled TaskSpec, controller ID, and active
EpisodeSpec; wrong-object/wrong-bin routes, false rejection, malformed/non-finite actions,
infrastructure failures, dispatch after rejection, and M2 calls are zero. Raw, binary-transformed,
projected, and executed action evidence remains separate. Independent verification rehashes every
atom and reports the learned/Oracle success counts separately from correct experiment execution.
The smoke cannot authorize or access the three-scene screen, full development, language/control
final, M3B test, historical fresh, `m42_final_v0`, or SmolVLA.

The authorized native target execution completed this gate at implementation Git
`16911a65540211d84883b354b1d1d9477174daae`. All six learned routes matched their scheduled
TaskSpecs and frozen controller IDs; the real selected-router rejection probe performed zero
lookup/reset/step work. Learned and Oracle execution each succeeded on 5/6 episodes and each had one
timeout, with zero wrong-object, wrong-bin, off-table, invalid-action, infrastructure, or M2-expert
failures. Independent verifier Git `cf6fbc07971cc2c507675763f04e6740adce40a4` accepted the immutable
evidence with `physical_target_validated=true`. The timeout counts remain observed controller-quality
evidence and do not weaken or redefine the dispatch gate.

## Three-scene paired control screen

This separately authorized stage consumes the completed one-scene report and its independent
verification as immutable parent authority. It materializes only the next three predeclared
development seeds and their canonical six-task order. For each of the 18 scene-task atoms it runs
Oracle first, then the unchanged `NeuroSymbolicRouterV0`, without using the Oracle outcome to alter
the learned rollout. Both paths reset policy state and use the same frozen PerTask controller,
H=10, `pd_joint_pos`, and explicit `project` action runtime.

Immediately after each reset and before any action, the stage captures the privileged diagnostic
state of all three cube poses, both bin poses, and the nine-dimensional Panda qpos. The independent
reader requires exact equality inside every Oracle/learned pair. The reset snapshot is evidence
only; it is not added to a visual observation, task input, or per-step info. A fixed six-command
rejection set covers conflicting objects, conflicting bins, unsupported action, unsupported
object, empty/noise, and unresolved correction. Every probe must return before controller lookup,
policy reset, environment reset, and `env.step`.

The compact screen archive is run-fingerprint-owned and contains the parent authority, exact
three-scene schedule and commands, frozen router/registry/runtime identities, 18 Oracle and 18
NeuroSymbolic records, pair analysis, rejection probes, failure attribution, gate result, manifest,
and completion marker. It is written through same-filesystem staging, fsync, checksum validation,
completion-last ordering, and atomic promotion. A fresh-process verifier rehashes both the compact
archive and the underlying 36 raw control atoms, reconstructs action streams and failure
attribution, recomputes the paired analysis and conjunctive gate, and checks all prohibited-access
flags.

Correct stage execution has `passed=true` and `physical_target_validated=true` even when controller
quality makes the screen gate false. Only a true `three_scene_control_screen_passed` may set
`selected_router_locked=true` and `full_control_development_authorized=true`. The full stage is
never executed automatically, and language/control final, M3B test, historical fresh,
`m42_final_v0`, M2 expert actions, and SmolVLA remain unaccessed.

The authorized target screen used clean implementation Git
`e461c1e20f783e8c5f06e107289842665e08820b` and the three locked seeds `2000000001` through
`2000000003`. All 18 routes, controller identities, and exact initial-state pairs passed. Oracle and
NeuroSymbolic each achieved 15/18 M1 success, with the same three paired timeouts and zero success
gap. The fixed six-category rejection set performed zero lookup, policy reset, environment reset,
or step. Raw recomputation found zero arm projection, wrong-object, wrong-bin, off-table,
malformed/non-finite action, infrastructure, and M2 counts. Gripper-only projection remained
explicitly reported at 647 Oracle and 646 NeuroSymbolic components. The independent verifier
accepted both compact and underlying atom evidence with `physical_target_validated=true`; the
screen quality gate authorized but did not execute full control development.

## Full six-scene paired control development

The separately authorized full stage first rehashes both the compact three-scene archive and all
36 underlying parent atoms. It then materializes only the six remaining development seeds and their
canonical 36 commands. Each pair runs Oracle immediately before the locked NeuroSymbolic router,
captures the three cube poses, two bin poses, and Panda qpos after both resets, and requires exact
state equality. Both branches reset the matching frozen PerTask policy and retain H=10,
`pd_joint_pos`, and explicit `project` handling.

The immutable rejection set contains nine predeclared categories: conflicting objects, conflicting
bins, unsupported action, unsupported object, unsupported destination, empty/noise, unresolved
correction, contradictory negation, and unsupported spatial reference. Each must terminate with no
TaskSpec, controller lookup, policy reset, environment reset, or `env.step`. Probe outcomes do not
consume the 72-rollout budget.

The exact conjunctive gate requires at least 35/36 correct TaskSpecs, at most two wrong-object and
two wrong-bin routes, at most three false rejections, no malformed executable decision, and
controller identity agreement for every correct route. NeuroSymbolic success may differ from Oracle
by at most two. All 36 initial states must match; arm projection, wrong-bin placement, off-table,
malformed/non-finite action, environment/infrastructure failure, and M2 counts must be zero. Raw,
projected, and executed action statistics, per-task/per-family results, paired outcomes, failure
attribution, grasp/release timing, and router/policy/environment latencies remain separate.

The fingerprint-owned archive is written through fsynced staging, completion-last validation, and
same-filesystem atomic promotion. A fresh process rehashes the archive and all 72 raw atoms and
recomputes every gate. `passed=true` means the authorized experiment and evidence completed
correctly. Only a true quality gate creates a positive `M5AFinalAuthorizationV0`; that record
contains final lock fingerprints but no command text or scene seed and never executes final.

The authorized native target run completed at implementation Git
`f8c583aef35e8e64d758ef5aeddf6f11ea331be9`. All 36 routed TaskSpecs, frozen controller identities,
and paired reset states matched. Oracle and NeuroSymbolic each achieved 25/36 M1 success, with 25
pairs both succeeding and 11 pairs sharing a timeout; no asymmetric outcome occurred. Per-task
success for both paths was red-left 2/6, red-right 3/6, green-left 6/6, green-right 6/6, blue-left
4/6, and blue-right 4/6. Target grasp occurred in 35/36 episodes on each path and post-grasp success
was 25/35.

All nine rejection probes performed zero lookup, policy reset, environment reset, and step. Raw
recomputation found zero wrong-object/wrong-bin interaction, off-table, arm projection,
malformed/non-finite action, environment/infrastructure failure, and M2 count. Explicit gripper
projection remained visible at 2,925 components with maximum correction
`0.05949103832244873`. The fresh verifier accepted the raw and compact evidence with
`physical_target_validated=true`, all conjunctive quality items true, and positive final
authorization fingerprint
`sha256:afb4d59a3549d42ac6e0e2cb010e8261cb39a47792af583fe5dd263645dd19b5`.
Final command text, scene seeds, and rollouts remained unmaterialized.
