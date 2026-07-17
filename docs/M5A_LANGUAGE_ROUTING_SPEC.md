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

Development control contains 12 new scene seeds times six canonical tasks, 72 episodes. Final
control contains 30 different new seeds times six tasks, 180 episodes. The generator excludes all
available M3A accepted and rejected candidates, all M3B split seeds, M4 fresh/smoke/tiny seeds,
M4.1 diagnostics, both M4.2 locks, and the M4.3 development source. Sidecar test seed identities
may be read only for exclusion; M3B test observations, actions, frames, and videos remain unopened.

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

Language-only evaluation runs RuleRouter, the selected/calibrated classifier, and the frozen local
LLM on validation and `m5a_language_dev_v0`. It reports routeable TaskSpec/object/bin accuracy,
route recall, false rejection, per-task/family results and confusion matrices; rejected accuracy,
precision, recall, false-route rate and reason recalls; and overall schema validity, malformed
rate, coverage, selective accuracy, supported calibration metrics, latency percentiles, and fixed-
configuration repeatability. Rejected examples never share the ordinary TaskSpec denominator.

Control development evaluates oracle routing and all three routers on the same 72 routeable
episodes. It records the expected task, decision, dispatched controller, active M1 episode spec,
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

## Development quality gate

At least one learned router (classifier or local LLM) must satisfy every applicable language gate:
routeable full TaskSpec accuracy at least 95%, object/bin accuracy at least 97%, false-route rate at
most 3%, ambiguous rejection recall at least 90%, unsupported and malformed rejection recall at
least 95%, schema validity at least 99%, and deterministic repeatability 100%. The LLM additionally
requires malformed structured output after bounded repair at most 1%.

Its predicted control result must be within five percentage points of oracle, with at most 2/72
wrong-object routes, 2/72 wrong-bin routes, and 3/72 false rejections. Target-in-wrong-bin,
target-off-table, arm projection, non-finite action, malformed action, post-rejection dispatch, and
M2 expert calls must all be zero. PerTask controller timeouts do not by themselves invalidate the
router.

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

The principal target-development flags have these meanings:

| Flag or flag group | Required meaning |
| --- | --- |
| `implementation_validated` | The declared target-development chain is internally complete and has no failed check; this is not a quality claim. |
| `prior_m43b_target_validated` | The supplied immutable M4.3b independent verification passed and retained all forbidden-access flags as false. |
| `corpus_validated`, `split_isolation_validated`, `final_schedules_locked` | Exact corpus counts, visible-family isolation, sealed-final attestation, and all four schedule identities were independently checked. |
| `rule_router_validated`, `controller_registry_validated` | The deterministic router contracts and exactly six metadata-only frozen controller entries/checkpoint artifacts passed validation. |
| `classifier_training_completed`, `classifier_checkpoint_selected`, `classifier_calibration_validated` | One classifier trained from train only; validation only selected its checkpoint, temperature, and threshold. |
| `llm_router_loaded`, `llm_prompt_locked` | Exactly one pinned local model loaded and the train-example-only prompt/generation identity was frozen before development. |
| `llm_license_reviewed`, `llm_model_card_reviewed` | The operator explicitly attested review of the declared license and official model card; these flags do not infer legal approval. |
| `language_validation_completed`, `language_development_completed` | All three routers completed the declared language-only validation and development schedules with immutable evidence. |
| `oracle_control_development_completed`, `predicted_control_development_completed`, `control_development_completed` | The paired oracle plus three-router 72-episode control development completed under the same frozen controller/runtime identities. |
| `rejection_noop_probe_validated` | A real active-M1 rejection probe proved zero controller lookup, policy call, reset, and environment step, and the independent validator accepted its immutable evidence. |
| `failure_attribution_validated`, `physical_target_validated` | Independent attribution/action checks passed over real target rollouts; the latter is never set by portable fixtures. |
| `experiment_manifest_validated` | The final target-development manifest matches independently rederived corpus, router, controller, schedule, and evidence fingerprints. |
| `development_quality_gate_passed`, `final_benchmark_authorized` | These are equal to the OR of the learned-router gates. Authorization records permission only and is independent of correct experiment execution. |
| `final_benchmark_completed` | Always false in target-development. |

Non-target verification validates implementation and fixtures only. It must leave classifier
training, development evaluations, authorization, all final access, SmolVLA, and physical target
validation false while still allowing `passed=true`.

Target-development may set experiment/physical flags only after real classifier training,
validation-only selection/calibration, one pinned local LLM, language development, oracle and
predicted 72-episode control development, failure attribution, and independent evidence checks.
It must finish with `language_final_accessed=false`, `control_final_accessed=false`,
`m42_final_accessed=false`, `test_split_accessed=false`, `historical_fresh_accessed=false`,
`final_benchmark_completed=false`, and `smolvla_go=false`. It records but does not execute any final
authorization. `passed=true` means the experiment and evidence chain completed correctly; it does
not imply `development_quality_gate_passed=true`.

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
