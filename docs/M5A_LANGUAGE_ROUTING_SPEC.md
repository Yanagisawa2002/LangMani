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
| `classifier_training_completed`, `classifier_checkpoint_selected`, `classifier_calibration_validated` | The promoted single-seed run completed or early-stopped; validation only selected and calibrated it. |
| `llm_router_loaded`, `llm_prompt_locked`, `language_development_completed` | One pinned inference-only LLM and immutable train-example prompt were evaluated with the offline routers. |
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
