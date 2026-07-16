# M4.3 shared-policy semantic-alignment specification

This document is the normative contract for LangMani milestone M4.3. M4.3 begins with a
zero-training semantic audit of the frozen M4/M4.2 controls. That real M4.3a audit is complete and
immutable. A separate M4.3b change now implements exactly one oracle-conditioned
`ACT-Mixed-FactorFiLM` policy plus its training/checkpoint contracts and local non-target fixture.
M4.3b does not execute the 100,000-step target run, select a real checkpoint, run `m42_dev_v0`,
access `m42_final_v0`, or start SmolVLA.

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
truthful physical flags. M4.3b extends the same non-target verifier with FactorFiLM mappings,
architecture/API checks, fixture forward/backward, optimizer, processor/checkpoint reload,
deterministic inference, dry-run command, validation-only selection, and final-access prohibition.
Without an explicit valid real-evidence root it cannot claim semantic-audit completion, and no
non-target invocation can claim CUDA training, simulator rollout, model quality, or physical
validation.

## Verification state and M4.3b gate

M4.3a verification distinguishes structural implementation from real inference. A successful local
non-target run may set `semantic_audit_implementation_validated=true` while
`semantic_audit_completed=false`, `factor_film_training_completed=false`,
`final_schedule_accessed=false`, `smolvla_go=false`, and `physical_target_validated=false`.
No implementation fixture may set a semantic quality flag.

FactorFiLM implementation was permitted only after all of the following became true:

1. this M4.3a code is formatted, lint-clean, CPU-safe-test clean, built, committed, and pushed as a
   separate clean Git boundary;
2. the real frozen checkpoints and completed M3B/M4/M4.2 inputs pass fingerprint validation;
3. both validation and `m42_dev_v0` semantic audits complete through immutable promotion;
4. the combined evidence validates all source identities, retrieval/confusion outputs, summaries,
   checksums, and completion marker; and
5. the verifier truthfully reports `semantic_audit_completed=true` while final-schedule access and
   SmolVLA remain false.

The real RTX 5090 combined audit satisfied this gate at Git
`dfea8b3d7d28274909ff178cb9087a9a90e17ee7`. It processed 36 M3B-validation and 72
`m42_dev_v0` observations, retained both named source sections, and promoted evidence fingerprint
`sha256:6342bdf4b019df203e6021947cbb39deacac2ea78cc585b5062091ed1d228671`. It bound M3B
fingerprint `sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4`, split
fingerprint `sha256:d86d29374ef7956a4ad8a1d9a111ed0924387b5b456d86e8c9ad5a31752f65e5`, and
`m42_dev_v0` fingerprint `sha256:981547e771b2b5cd3a77e2788bb49d29fc45b3f59c607021a03a4e2ce70b43f1`.
It reported `semantic_audit_completed=true`, `final_benchmark_authorized=false`,
`final_schedule_accessed=false`, `test_split_accessed=false`, `fresh_seed_accessed=false`, and
`smolvla_go=false`.

The combined primary-metric results were:

| Control | Full-task top-1 | Top-2 | MRR | Correct margin | Object | Bin | Conditional bin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| State-OneHot | 37.96% | 70.37% | 0.6289 | -0.002428 | 76.85% | 50.93% | 50.93% |
| TaskToken | 24.07% | 50.00% | 0.4887 | -0.006511 | 50.00% | 49.07% | 50.93% |

Both conditions changed outputs but did not reliably align them with requested semantics. The
object confusion matrices were `[[34,2,0],[11,21,4],[2,6,28]]` for State-OneHot and
`[[30,4,2],[18,10,8],[15,7,14]]` for TaskToken. Their bin matrices were
`[[22,32],[21,33]]` and `[[27,27],[28,26]]`. Target-object confusion, approximately random bin
retrieval, and shared-policy post-grasp failures therefore justify one factorized repair. TaskToken
remains rejected. These findings do not validate FactorFiLM quality or authorize a final schedule.

## Explicitly out of scope for M4.3a

M4.3a does not implement or initialize FactorFiLM; train a model; select a checkpoint; alter M3B;
access M3B test, M4 fresh, or `m42_final_v0`; run M2 actions; change M1 success or bounds; add text
encoders, language claims, new data, architecture sweeps, RL, DAgger, or SmolVLA; or authorize a
final benchmark from audit completion alone.

At the original local structural stage no real semantic retrieval metric was recorded. The table
above was copied only after the immutable target audit existed and independently validated.

## Local structural validation status (2026-07-16)

The standalone M4.3a implementation baseline is Git commit
`ddead2c65134dd23c9188d35e8fe995143c79a88` (`feat: complete M4.3 semantic alignment audit`).
This hash identifies the code and documentation that passed the structural gate below; the
subsequent documentation-only binding commit does not change the implementation.

The final M4.3a-focused selection reported 147 passed and one Windows symlink-creation permission
skip. The complete CPU-safe suite reported 840 passed, 14 skipped, and 15 hardware/rendering tests
deselected. Ruff formatting and lint checks, `pip check`, sdist/wheel build, and the non-target
`environment/verify_m43.py` command passed. The verifier exercises a combined CLI dry-run and the
immutable promotion/corruption-rejection lifecycle while keeping semantic completion, physical
validation, real checkpoint reload, test/fresh/final access, FactorFiLM, and SmolVLA claims false.

These results validated local structure only. Pinocchio absence, Windows symlink privilege, and the
optional pytest-cache write-permission warning were nonblocking because no M4.3a structural test
depended on planner kinematics, creating a symlink, or writing that cache. The later real audit is
the separate immutable evidence recorded above; it completes semantic inference without changing
the historical meaning of this local gate.

## M4.3b FactorFiLM architecture contract

M4.3b owns exactly one new oracle-conditioned architecture:

```text
ACT-Mixed-FactorFiLM
```

It consumes the unchanged `observation.images.base_camera` plus `PandaPolicyStateV0[9]`. Stable
TaskSpec metadata, never instruction text, is factorized by the same project-owned component at
training and inference into:

| Mapping | Canonical index order |
| --- | --- |
| `TargetObjectConditionV0` | `red_cube=0`, `green_cube=1`, `blue_cube=2` |
| `DestinationBinConditionV0` | `left_bin=0`, `right_bin=1` |

Unknown TaskSpec IDs, unknown object/bin IDs, malformed combinations, wrong batch size, wrong dtype,
wrong device, and out-of-range indices fail clearly. `PandaPolicyStateV0` remains exactly nine
dimensions. M4.3b adds no object/bin/task one-hot to state, no ENV or combined task token, no
language embedding, and no M3B feature or dataset rewrite.

### Target-object visual path

`TargetObjectConditionV0` uses one learned embedding table `[3,32]` and a learned projection
`Linear(32,1024)`. The projection splits into gamma and beta `[B,512]`, explicitly broadcasts them
to `[B,512,1,1]`, and modulates the ResNet-18 layer-4 feature map `[B,512,8,8]`:

```text
visual_conditioned = visual_features * (1 + gamma_object) + beta_object
```

This occurs after backbone extraction and before `encoder_img_feat_input_proj` consumes the feature
map. The object path never receives destination identity and never modifies the encoded state token.

### Destination-bin state/context path

`DestinationBinConditionV0` uses a separate learned embedding table `[2,32]` and projection
`Linear(32,1024)`. Its gamma and beta are `[B,512]` and modulate the encoded Panda-state token after
the public-model `Linear(9,512)` projection but before ACT Transformer sequence processing:

```text
state_conditioned = state_features * (1 + gamma_bin) + beta_bin
```

The bin path never receives target-object identity and never modifies the visual feature map.

Both final FiLM projections use weight initialization `normal(mean=0,std=1e-5)` and zero biases.
The resulting residual transform is close to identity while preserving nonzero first-step
embedding gradients. For the full locked ACT configuration, the object path adds 33,888 parameters
and the bin path adds 33,856. The 51,576,712-parameter base ACT therefore becomes 51,644,456
parameters, an exact 67,744 increase. No other capacity, backbone, action head, or loss changes.

## LeRobot 0.6.0 integration and risk boundary

The isolated adapter subclasses these public LeRobot interfaces:

```text
lerobot.configs.PreTrainedConfig
lerobot.policies.act.ACTConfig
lerobot.policies.act.ACTPolicy
lerobot.policies.act.make_act_pre_post_processors
PreTrainedConfig.from_pretrained(pretrained_name_or_path, *, force_download=False,
                                 resume_download=None, proxies=None, token=None,
                                 cache_dir=None, local_files_only=False, revision=None,
                                 **policy_kwargs)
ACTPolicy(config: ACTConfig, **kwargs)
ACTPolicy.forward(batch)
ACTPolicy.predict_action_chunk(batch)
ACTPolicy.select_action(batch)
ACTPolicy.save_pretrained(save_directory, *, state_dict=None, repo_id=None,
                          push_to_hub=False, card_kwargs=None, **push_to_hub_kwargs)
ACTPolicy.from_pretrained(pretrained_name_or_path, *, config=None, force_download=False,
                          resume_download=None, proxies=None, token=None, cache_dir=None,
                          local_files_only=False, revision=None, strict=False, **kwargs)
make_act_pre_post_processors(config, dataset_stats=None)
```

LeRobot 0.6.0 exposes no public callback at the two required intermediate representations. The
adapter therefore registers instance-local Torch forward hooks on the semi-stable internal
attributes `policy.model.backbone` and `policy.model.encoder_robot_state_input_proj`. It does not
import or copy the private `lerobot.policies.act.modeling_act.ACT` implementation and does not
monkey-patch a class or global ACT behavior.

The upstream compatibility identity records public import paths/signatures, LeRobot version,
underlying model module/name, backbone output key, image/state projection structures, position
embeddings, decoder/action head, hidden dimension, and expected shapes:

```text
image                          [B,3,256,256]
ResNet-18 layer-4 feature map  [B,512,8,8]
encoded state token            [B,dim_model]
encoder sequence               latent, state, 64 image tokens
action chunk                   [B,chunk_size,8]
```

Exactly one visual and one state hook must fire per conditioned policy call. Hook state is
instance-local and cleared even on exceptions. Any LeRobot version, signature, attribute, token-
layout, key, shape, or call-count drift is a hard compatibility failure. There is no State-OneHot,
TaskToken, or unconditioned fallback.

## Training, identity, and checkpoint contract

The intended target-development run is a fair architectural comparison with State-OneHot. It uses:

- the same 288 M3B train and 36 validation episodes;
- dataset fingerprint `sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4`;
- split fingerprint `sha256:d86d29374ef7956a4ad8a1d9a111ed0924387b5b456d86e8c9ad5a31752f65e5`;
- ResNet-18 without pretrained weights, action chunk 50, one observation step, 512D ACT, and the
  unchanged VAE/Transformer/action-head configuration;
- batch size 32, AdamW, learning rates `1e-5`, weight decay `1e-4`, KL weight 10, no scheduler,
  global gradient norm 10, locked seed 0, deterministic data order, 100,000 steps, and bfloat16 CUDA;
- checkpoint interval 5,000 and exactly 20 expected checkpoints;
- train-only image/state/action normalization from the exact M3B train view;
- locked execution horizon 10 and explicit `project` runtime for later development rollout.

Task conditions are attached only after the installed LeRobot preprocessor has produced the
standard image/state/action tensors. They are stripped before ACT receives its ordinary feature
dictionary, then consumed only by the two scoped hooks. This preserves the saved 9D state processor
and one common train/inference mapping.

Architecture, mapping, run, checkpoint, manifest, validation-queue, selection, and dry-run records
are immutable JSON-serializable project-owned contracts with canonical-JSON/SHA-256 fingerprints.
A run identity binds the audit evidence, M3B/split/statistics fingerprints, exact ordered episode
views, base ACT configuration, FactorFiLM configuration and upstream identity, optimizer/data
configuration, seed, Git commit, and package versions. Hostname, timestamps, absolute paths,
temporary directories, and filesystem enumeration order do not affect it.
Target-development parsing independently requires strictly increasing, unique 288-train and
36-validation lists and binds each list's count and canonical fingerprint into the data contract;
the manifest rechecks the base ACT and H=10/project runtime against that identity.

The existing atomic M4 checkpoint lifecycle is extended, not replaced. Each published checkpoint
contains or references base ACT weights; both embeddings and projections; architecture identity;
both mappings; ACT config; processors; train-only statistics; optimizer, scheduler, RNG, and global
training state; dataset/split/Git/version/schema identities; checksums; and a last completion marker.
Reload is local-only and strict. Direct policy saves require a new or empty real directory before
any upstream file is written. Resume rejects mismatched dataset, split, statistics, mapping,
embedding dimension, injection location, base ACT config, optimizer state, schema, run identity, or
completed immutable run. It accepts only the latest manifest checkpoint or one sole, complete,
exactly-next 5,000-step promoted orphan; multiple or out-of-order orphans fail. `--clean-staging`
may remove only an owned, matching, unlinked incomplete staging directory; it never mutates a
published checkpoint. Command reports and outputs are rejected before writes when they overlap
M3B/evidence/source/Git/historical artifacts or traverse a symlink/junction.

## Validation selection and access locks

The future selection queue contains exactly the 20 expected checkpoint steps, has its own canonical
fingerprint, and evaluates only M3B validation. Each item includes the offline validation loss read
from the checkpoint's fingerprint-bound training metric. A selection record embeds that queue and
must match all 20 `(checkpoint fingerprint, step, loss)` tuples and its validation schedule. Its
deterministic ranking is:

1. highest validation task-success rate;
2. lowest wrong-object interaction rate;
3. lowest wrong object in target bin rate;
4. lowest target-off-table rate;
5. lowest timeout rate;
6. lower validation action loss;
7. earlier checkpoint step.

M3B test, historical M4 fresh seeds, `m42_dev_v0`, `m42_final_v0`, and semantic-audit observations
are prohibited training/selection inputs. `m42_dev_v0` becomes available only after validation
selection is immutable, as a later closed-loop development benchmark. The sealed final lock may be
validated as a lock artifact only; no M4.3b implementation, dry-run, fixture, training, resume, or
selection command may enumerate its seeds, construct observations, reset an environment, run a
policy, or authorize final evaluation.

## Structural command and fixture boundary

The project-owned command supports `--dry-run`, `--fixture`, and a future explicit
`--target-development` contract:

```bash
python scripts/train_act_factor_film.py \
  --dataset-root outputs/fixtures/m43/factor-film-dataset-contract \
  --output-root outputs/fixtures/m43/factor-film-dry-run \
  --device cpu \
  --report outputs/diagnostics/m43/factor-film-fixture-contract.json \
  --dry-run \
  --fixture-contract

python scripts/train_act_factor_film.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act-factor-film \
  --evidence-root outputs/diagnostics/m43/remote-audit-dfea8b3/930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2 \
  --device cpu \
  --report outputs/diagnostics/m43/factor-film-dry-run.json \
  --dry-run

python scripts/train_act_factor_film.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act-factor-film \
  --evidence-root outputs/diagnostics/m43/remote-audit-dfea8b3/930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2 \
  --device cpu \
  --report outputs/diagnostics/m43/factor-film-fixture.json \
  --fixture
```

The fixture constructs a deterministic reduced ACT batch, applies both conditions, verifies finite
loss/backward and gradients in base ACT, object embedding, bin embedding, object projection, and bin
projection, takes one optimizer step, saves model/processors/checkpoint state, loads a fresh policy
instance, and compares deterministic inference with `atol=1e-6` and `rtol=1e-6`. It separately proves
object-only visual changes, bin-only state changes, untouched opposite pre-FiLM paths, explicit
broadcasting, and near-identity initialization. It is never checkpoint-quality or physical evidence.

The later GPU target-development command is exactly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_act_factor_film.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --output-root outputs/models/act-factor-film \
  --evidence-root outputs/diagnostics/m43/remote-audit-dfea8b3/930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2 \
  --device cuda \
  --report outputs/diagnostics/m43/factor-film-target-development.json \
  --clean-staging \
  --target-development
```

An interruption may resume only by adding the explicit `--resume-checkpoint` for a compatible
incomplete run. This structural milestone does not execute either target path or create a real
training directory.

## M4.3b non-target verification truth

A passing local implementation stage may report:

```text
implementation_validated=true
semantic_audit_completed=true        # only when the real evidence root above validates
factor_film_implementation_validated=true
factor_film_fixture_training_validated=true
factor_film_training_completed=false
factor_film_checkpoints_complete=false
factor_film_checkpoints_validated=false
factor_film_checkpoint_selected=false
factor_film_reload_validated=false
development_benchmark_completed=false
development_quality_gate_passed=false
factor_film_development_evaluation_completed=false
factor_film_development_quality_validated=false
final_benchmark_authorized=false
final_schedule_accessed=false
smolvla_go=false
physical_target_validated=false
passed=true
```

Fixture optimization must never set `factor_film_training_completed`. No full CUDA training,
real checkpoint selection, `m42_dev_v0` rollout, final benchmark, go/no-go, or SmolVLA code belongs
to this structural milestone.
