# ACT baseline specification

This document is the normative M4 contract for reproducible ACT behavioral-cloning baselines and
closed-loop evaluation on `LangMani-PickPlaceByInstruction-v0`. M4 consumes a completed M3B
LeRobotDataset v3; it does not reopen M3A during normal training and does not change any M1 task,
camera, success, action, or observation contract.

M4 evaluates control learning and task conditioning. Standard ACT does not consume natural
language. The canonical instruction may remain in LeRobot task metadata, but no ACT tensor input is
constructed from that text. The six-way task one-hot is an oracle command condition, not language
understanding. SmolVLA, text encoders, language embeddings, paraphrases, RL, DAgger, Hub upload,
distributed training, data augmentation, new demonstrations, and M2 expert use during policy
rollout are outside M4.

M4 full is now complete: `full_experiment_validated=true`,
`physical_target_validated=true`, and `baseline_quality_validated=false`. This document remains the
immutable contract for those eight historical runs. M4.2 does not add a fourth `ActVariant` or
change an existing run/checkpoint identity; its independent extension contract is
[`M42_ORACLE_CONTROL_SPEC.md`](M42_ORACLE_CONTROL_SPEC.md).

## Baselines and policy features

`ActVariant` contains exactly these values:

| Variant | Policies | Training episodes | Policy state | Scientific role |
| --- | ---: | ---: | --- | --- |
| `per_task` | six, one per stable TaskSpec | 48 per policy | `PandaPolicyStateV0[9]` | primary control and data-quality baseline |
| `mixed_unconditioned` | one | all 288 train episodes | `PandaPolicyStateV0[9]` | deliberately ambiguous counterfactual control |
| `mixed_task_onehot` | one | all 288 train episodes | `PandaPolicyStateV0[9] + CanonicalTaskOneHotV0[6]` | oracle discrete-task-conditioned upper bound |

All variants also receive `observation.images.base_camera` and predict the exact eight-dimensional
`pd_joint_pos` action. `PandaPolicyStateV0` is float32 in this order:

```text
panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5,
panda_joint6, panda_joint7, panda_finger_joint1, panda_finger_joint2
```

`CanonicalTaskOneHotV0` is float32, has exactly one active component, and follows the existing M3A
object-major/bin-minor order:

```text
red_cube/left_bin, red_cube/right_bin,
green_cube/left_bin, green_cube/right_bin,
blue_cube/left_bin, blue_cube/right_bin
```

The implementation derives each component from the corresponding stable `TaskSpec` ID; it never
parses instruction text. At training time the episode-to-task mapping comes from the M3B sidecar.
At inference time the active M1 command `TaskSpec` is encoded through the same project-owned
component. The one-hot is appended in memory and the M3B dataset is never rewritten or duplicated.

No variant may consume object or bin pose, target indices from privileged state, full simulator
state, source seed, scene ID, expert phase, success oracle, split label, task text, task ID as a
scalar tensor, or any M2 planner value. Only the task-one-hot variant receives a task condition.

## Completed-M3B input gate

Normal M4 work accepts one immutable completed M3B root only. The gate validates the LangMani
completion marker, typed export manifest, summary, independent M3B validation report, export and
source fingerprints, `v3.0` LeRobot codebase version, full mode, finalization state, exact sidecar
cross-references, every stored source alignment and video result, local Parquet/video presence and
frame totals, and the policy feature allowlist.

The full gate requires:

- one `observation.images.base_camera` RGB video feature at 256 x 256;
- one float32 `observation.state[9]` using `PandaPolicyStateV0`;
- one float32 `action[8]` using the unchanged `pd_joint_pos` contract;
- 20 FPS;
- 60 complete counterfactual scene groups and 360 episodes, 60 per TaskSpec;
- scene-level 48/6/6 train/validation/test groups, hence 288/36/36 episodes;
- no scene leakage and no privileged feature;
- local PyAV reads with downloads and Hub access disabled.

The gate trusts the immutable, content-bound M3B acceptance evidence and verifies its local
cross-references. It does not rerun the M3A inspector, recalculate M3A/M3B admission, or accept a
fixture as full data. A fixture can exercise M4 code paths only and remains ineligible for real
training or physical-validation flags.

## Dataset views and train-only statistics

M4 derives explicit ordered episode views from the validated M3B manifest: global train,
validation, and test views, plus all three views for each of the six stable TaskSpec IDs. It checks
the episode set returned by LeRobot's public episode filter instead of assuming that indices are
contiguous or that request order controls row order. Training loaders receive train views only;
validation and test views never enter checkpoint optimization. The test view is not opened for a
full experiment until checkpoint selection is locked.

Normalization statistics are recomputed from current frames of the selected train view. M4 never
uses the M3B dataset-wide `meta.stats`, because LeRobot keeps those full-dataset statistics even
when an episode subset is loaded. Per-task runs use exactly their 48 train episodes; mixed runs use
all 288 train episodes.

The algorithm is deterministic batched Welford accumulation in float64 with population standard
deviation. State and action counts are frame counts. Image values are converted to float32 [0, 1]
and accumulated per RGB channel over all train pixels. The saved record includes min, max, mean,
standard deviation, component order, source episode indices, calculation algorithm, LeRobot and
PyTorch versions, and a canonical SHA-256 fingerprint. A separate set audit proves that no
validation or test episode is present and that every selected train frame appears once.

For the 15D variant, the first nine state statistics are the train-only Panda statistics. The final
six synthetic components use mean 0 and standard deviation 1, with min 0 and max 1. Therefore the
installed mean/std normalizer leaves the appended binary one-hot unchanged. Conditioning occurs
before the ACT preprocessor at both training and inference; no full-dataset or text-derived fallback
exists.

## Inspected LeRobot 0.6.0 boundary

M4 uses installed public interfaces and does not copy ACT source:

```python
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
from lerobot.policies import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy, make_act_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
```

The inspected constructors/functions used by the project are below. Type names use the imports
above rather than repeating fully qualified module paths; parameter names, ordering, and defaults
match `inspect.signature()` from the installed 0.6.0 package.

```text
ACTPolicy(config: ACTConfig, **kwargs)
ACTConfig(n_obs_steps: int = 1, input_features: dict[str, PolicyFeature] | None = <factory>,
          output_features: dict[str, PolicyFeature] | None = <factory>, device: str | None = None,
          use_amp: bool = False, use_peft: bool = False, push_to_hub: bool = True,
          repo_id: str | None = None, private: bool | None = None, tags: list[str] | None = None,
          license: str | None = None, pretrained_path: Path | None = None,
          pretrained_revision: str | None = None, chunk_size: int = 100,
          n_action_steps: int = 100, normalization_mapping: dict[str, NormalizationMode] = <factory>,
          vision_backbone: str = 'resnet18',
          pretrained_backbone_weights: str | None = 'ResNet18_Weights.IMAGENET1K_V1',
          replace_final_stride_with_dilation: int = False, pre_norm: bool = False,
          dim_model: int = 512, n_heads: int = 8, dim_feedforward: int = 3200,
          feedforward_activation: str = 'relu', n_encoder_layers: int = 4,
          n_decoder_layers: int = 1, use_vae: bool = True, latent_dim: int = 32,
          n_vae_encoder_layers: int = 4, temporal_ensemble_coeff: float | None = None,
          dropout: float = 0.1, kl_weight: float = 10.0, optimizer_lr: float = 1e-05,
          optimizer_weight_decay: float = 0.0001, optimizer_lr_backbone: float = 1e-05) -> None
make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig
make_policy(cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata | None = None,
            env_cfg: EnvConfig | None = None,
            rename_map: dict[str, str] | None = None) -> PreTrainedPolicy
make_pre_post_processors(policy_cfg: PreTrainedConfig, pretrained_path: str | None = None,
                         pretrained_revision: str | None = None,
                         **kwargs: Unpack[ProcessorConfigKwargs]) -> tuple[PolicyProcessorPipeline,
                         PolicyProcessorPipeline]
resolve_delta_timestamps(cfg: PreTrainedConfig | RewardModelConfig,
                         ds_meta: LeRobotDatasetMetadata) -> dict[str, list] | None
make_act_pre_post_processors(config: ACTConfig,
                             dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None)
                             -> tuple[DataProcessorPipeline, DataProcessorPipeline]
LeRobotDataset(repo_id: str, root: str | Path | None = None,
                episodes: list[int] | None = None,
                episode_filter: Callable[[dict], bool] | None = None,
                image_transforms: Callable | None = None,
                delta_timestamps: dict[str, list[float]] | None = None,
                tolerance_s: float = 0.0001, revision: str | None = None,
                force_cache_sync: bool = False, download_videos: bool = True,
                video_backend: str | None = None, return_uint8: bool = False,
                depth_output_unit: str = 'mm', batch_encoding_size: int = 1,
                rgb_encoder: RGBEncoderConfig | None = None,
                depth_encoder: DepthEncoderConfig | None = None,
                encoder_threads: int | None = None, streaming_encoding: bool = False,
                encoder_queue_maxsize: int = 30)
LeRobotDatasetMetadata(repo_id: str, root: str | Path | None = None,
                       revision: str | None = None, force_cache_sync: bool = False,
                       metadata_buffer_size: int = 10)
PolicyProcessorPipeline.save_pretrained(self, save_directory: str | Path | None = None, *,
                                        repo_id: str | None = None, push_to_hub: bool = False,
                                        card_kwargs: dict[str, Any] | None = None,
                                        config_filename: str | None = None, **push_to_hub_kwargs)
PolicyProcessorPipeline.from_pretrained(pretrained_model_name_or_path: str | Path,
                                        config_filename: str, *, force_download: bool = False,
                                        resume_download: bool | None = None,
                                        proxies: dict[str, str] | None = None,
                                        token: str | bool | None = None,
                                        cache_dir: str | Path | None = None,
                                        local_files_only: bool = False,
                                        revision: str | None = None,
                                        overrides: dict[str, Any] | None = None,
                                        to_transition: Callable | None = None,
                                        to_output: Callable | None = None, **kwargs)
                                        -> DataProcessorPipeline
```

`ACTPolicy.forward(batch)` returns `(loss, loss_dict)`, where installed ACT exposes `l1_loss` and,
when VAE is active, `kld_loss`. `ACTPolicy.select_action(batch)` returns one action and internally
uses its action queue. `ACTPolicy.reset()` clears that queue. Policy and processors expose public
`save_pretrained`; the policy exposes `from_pretrained`, and `PolicyProcessorPipeline` exposes
`from_pretrained`, `save_pretrained`, `reset`, and `__call__`. All M4 reloads use local paths and
never contact or authenticate to the Hub.

Direct `ACTPolicy(config)` construction leaves the module on CPU even when `config.device` is CUDA;
the inspected public LeRobot factory explicitly calls `policy.to(cfg.device)`. LangMani mirrors
that required public step and rejects construction unless every policy parameter and buffer reaches
the configured device. The ACT preprocessor performs renaming, batch insertion, device transfer,
and normalization; the
postprocessor unnormalizes actions and returns them to CPU. Installed ACT preprocessing does not
divide uint8 images by 255. The official LeRobot trainer performs that conversion before the
preprocessor, so LangMani explicitly converts decoded uint8 CHW images to float32 [0, 1] and also
validates an already decoded float32 [0, 1] input. Task strings and every non-allowlisted metadata
field are removed before processor invocation.

Installed `ACTConfig` defaults to 100-step chunks and 100 executed actions, ImageNet-pretrained
ResNet-18, model dimension 512, eight heads, feedforward dimension 3200, four encoder layers, one
decoder layer, VAE latent dimension 32/four VAE encoder layers, KL weight 10, AdamW learning rate
`1e-5`, weight decay `1e-4`, and no scheduler. Its default `push_to_hub=True` is unsafe for M4 and is
always overridden to false. `use_amp` is configuration only; LangMani owns the actual autocast and
gradient handling in its training loop.

## Effective primary configuration

All three variants use one fixed primary configuration except the required 9D versus 15D state
width:

| Parameter | M4 primary value |
| --- | ---: |
| Image/state/action shapes | `(3,256,256)`, 9 or 15, 8 |
| Backbone | ResNet-18, no pretrained weights |
| Observation steps | 1 |
| Action chunk / actions executed per query | 50 / 10 |
| Model / feedforward dimensions | 512 / 3200 |
| Attention heads | 8 |
| Transformer encoder / decoder layers | 4 / 1 |
| VAE / latent dimension / VAE encoder layers | true / 32 / 4 |
| Dropout / KL weight / pre-norm | 0.1 / 10 / false |
| Temporal ensembling | disabled |
| Optimizer | AdamW |
| Learning rate / backbone learning rate | `1e-5` / `1e-5` |
| Weight decay | `1e-4` |
| Normalization mapping | visual/state/action `MEAN_STD` |
| Final-stride dilation / feedforward activation | false / `relu` |
| Observation / action delta indices | none / `range(50)` |
| PEFT / Hub and pretrained references | disabled / all null |
| Scheduler / warmup | none / 0 |
| Gradient clipping | global norm 10 |
| Primary mixed precision | CUDA bfloat16 |
| Batch size / loader workers | 32 / 4 |
| Training / checkpoint / validation intervals | 100000 / 5000 / 5000 steps |
| Model/source dtype | float32 |
| Hub publication | disabled |

Pretrained backbone weights are disabled so an offline run cannot depend on an unstored network
download. Batch size, chunk length, and executed actions are the only allowed bounded development
search dimensions, and their search space must be declared before test performance is available.
The table records the selected primary values; fixtures use a much smaller model solely to validate
the API path and are never quality evidence. Peak CUDA memory and training throughput are pending a
real target run and must be copied from structured metrics, never estimated or fabricated.

## Temporal sampling and action chunks

The public `resolve_delta_timestamps(ACTConfig, LeRobotDatasetMetadata)` helper is authoritative.
At 20 FPS, current observation state and image use time zero; no future observation is requested.
The 50 actions use indices `0..49`, hence timestamps `0.00, 0.05, ..., 2.45` seconds. LeRobot clamps
queries at an episode boundary and returns `action_is_pad` for padded positions. Contract tests
cover the first and final frame and require an action tensor shaped `[50, 8]` plus a padding mask
shaped `[50]`. LangMani does not manually reproduce temporal indexing or allow a chunk to read an
action from another episode.

During closed-loop inference, installed `select_action` owns the action queue: a new chunk is
queried only when the queue is empty and the first ten actions are emitted across successive calls.
Policy and both processor pipelines are reset before every episode so queued actions cannot cross
episode boundaries.

## Run identity, Git, and artifacts

The tracked baseline through M3B is Git commit
`6920b52c1f48c278e669cd71b69b8949dd900f3a`, tagged `m3b-implementation`. Every run records its
actual full `HEAD`, not merely that baseline tag. Full and tiny-overfit evidence requires a clean
tree. A dirty tree is accepted only in an explicitly labeled development run with the
development-only override recorded; it can never set a final validation flag.

The run fingerprint is canonical JSON plus SHA-256 over M3B export fingerprint, split-manifest
digest, ordered train and validation episode indices, variant and per-task stable ID when present,
task-one-hot mapping version, effective ACT/data/optimization contracts, train-statistics
fingerprint, seed, LeRobot/PyTorch/CUDA versions, Git commit, and M4 schema. Paths, hostname,
timestamps, temporary directories, enumeration order, and Python `hash()` are excluded. A completed
fingerprint directory is immutable.

Generated runs remain under ignored output paths:

```text
outputs/models/act/<run-fingerprint>/
├── run_manifest.json
├── config.json
├── dataset_contract.json
├── train_stats.json
├── metrics.jsonl
├── checkpoints/
├── validation/
├── test/
├── fresh_seed/
├── reports/
└── complete.json
```

## Training and checkpoints

The project-owned loop seeds Python, NumPy, Torch CPU/CUDA, and DataLoader workers; enables strict
Torch deterministic algorithms; sets deterministic cuDNN behavior; disables cuDNN benchmarking and
TF32; and records all settings. Unsupported nondeterministic CUDA behavior is an explicit failure,
not a silent fallback.

Each bounded step obtains a train-only batch, projects the feature allowlist, performs explicit
image conversion, runs the saved preprocessor and ACT forward pass, verifies finite total and
reported component losses, backpropagates under explicit CUDA bfloat16 autocast, verifies gradients,
clips global norm, steps AdamW, clears gradients, and logs JSONL. Logged fields include total,
action, and available KL loss; learning rate; gradient norm; examples and step; DataLoader and step
time; throughput; validation loss when scheduled; checkpoint path when one is saved; and CUDA
allocated/reserved memory. Missing loss components remain null rather than being invented. The
primary path deliberately avoids float16,
so it does not require a `GradScaler`; a target GPU without bfloat16 support fails instead of
silently changing precision. Nonfinite input, loss, or gradient stops the run.

Checkpoints are saved to a same-filesystem staging directory and atomically promoted. A completion
marker is written and flushed in staging before the directory is promoted. They contain the public
ACT pretrained directory,
preprocessor and postprocessor configs, optimizer and optional scheduler state, RNG state, global
training state, run identity/manifest, content checksums, M3B and split fingerprints, statistics
fingerprint, seed, Git commit, and runtime versions. Reload validates every checksum and semantic
identity before loading the model and processors locally. A deterministic batch must produce the
same inference output within the test tolerance after a fresh reload.

Resume rejects dataset, split, ACT, variant, task mapping, train-statistics, optimizer, or run
fingerprint mismatch, missing/corrupt artifacts, and a completed immutable run. Checkpoint or
processor load never contacts Hugging Face Hub and M4 never pushes artifacts. If interruption lands
between checkpoint promotion and run-manifest update, resume may adopt only that single next-step
content-valid orphan, including the first scheduled checkpoint, and recovers its checkpoint-bound
metric record. A run interrupted before any checkpoint is atomically moved into a preserved
`_abandoned_precheckpoint` diagnostic area and restarted from the same deterministic identity; it is
never mistaken for completed evidence.

Run finalization updates the validated manifest and then publishes `complete.json`. The comparison
command may repair only the narrow missing-marker crash window after it has revalidated the typed
manifest, selection lock, selected-checkpoint evaluation artifacts, sensitivity evidence, and
training summary; an inconsistent existing marker is never replaced.

The training summary records full wall duration, including validation and checkpoint serialization,
for uninterrupted runs. After a resumed run, that field is explicitly null because elapsed wall
time before the process boundary is not reconstructable; the separately named DataLoader/optimizer
duration remains a measured lower bound rather than being mislabeled as total training time.

Dry-run validates every gate and reports fingerprints, variant/task, shapes, episode counts,
train-statistics identity, effective ACT configuration, run directory, and schedules without
starting a long optimization. Tiny-overfit is a sanity gate, not full evidence. Its fixed-view loss
is explicitly recorded as a training-view overfit diagnostic, not held-out validation; the run
identity therefore keeps formal validation indices empty while binding the diagnostic role and
episode indices in the data contract. Per-task tiny overfit requires final fixed-view offline loss
no greater than 50% of the pre-training loss on that same deterministic tiny view and records its
same-scene rollout outcome. Task-one-hot tiny overfit uses one complete six-task train group, varies
only the oracle one-hot, and attempts all six rollouts;
its target quality gate is 6/6. A miss is reported and investigated without changing labels,
success geometry, test data, or temporal/action contracts.

## Closed-loop policy evaluation

The adapter creates the M1 environment with `num_envs=1`, RGB policy observations,
`pd_joint_pos`, and 20 Hz control. It does not import or invoke M2. Each query reads only
`base_camera` RGB and `PandaPolicyStateV0`; the task-one-hot variant additionally encodes the active
nonprivileged command `TaskSpec`. It applies the saved preprocessor, calls installed
`ACTPolicy.select_action`, applies the saved postprocessor, then applies the explicit M4.1
environment-action processor to float32 `[1,8]`. Single-action bounds are resolved through
Gymnasium's wrapper attribute contract or
the unwrapped ManiSkill environment because ManiSkill 3.0.1's `TimeLimitWrapper` does not expose
`single_action_space` as a direct wrapper attribute.

In `reject` mode, out-of-bounds actions are recorded as `invalid_action` and terminate before a
step. In `project` mode, finite violations are recorded and explicitly projected; malformed and
nonfinite actions still terminate. Inference failures, environment failures, off-table outcomes,
truncation, and timeouts remain distinct. Inference, environment-step, and whole-episode timing are
measured separately. Raw records retain conservative M1 evaluation fields, grasp/wrong-grasp
events, raw/executed action audits and compact projection metrics, per-joint ranges, event times,
latency samples, and failure reasons.

Validation and test use the exact six M3B scene groups: mixed variants run 36 episodes per split,
while each per-task policy runs its matching six episodes. The fresh benchmark fixes 30 unseen
seeds before results are observed, excludes every available accepted and rejected source seed, and
pairs every seed with all six tasks. Mixed variants therefore run 180 episodes and each per-task
policy runs 30. The schedule and exclusion set are saved with a SHA-256 digest.

Reports include overall, per-task, and per-scene rates, the full failure taxonomy, raw episode
records, Wilson confidence intervals, target/wrong-bin/wrong-object/off-table signals, timing and
latency percentiles, action statistics, and the exact clean evaluation Git state. Optional videos do
not replace failed episode records. Each evaluation is first written beneath an identity-owned
same-filesystem staging directory. Raw episodes and all aggregates are re-derived and checked before
atomic promotion. Selection locking, sensitivity-report publication, and idempotent run finalization
occur only after promotion, so a process interruption can be resumed without accepting partial
evidence or overwriting an accepted result.

## Checkpoint selection and test lock

Every candidate checkpoint is evaluated on one immutable validation schedule. Ranking is declared
before test access:

1. highest validation task-success rate;
2. lowest wrong-object interaction rate;
3. lowest target-off-table rate;
4. lowest offline validation action loss;
5. earliest checkpoint step.

The offline validation loss used here is checkpoint-bound evidence. The legacy field name
`offline_validation_action_loss` stores the established total ACT validation objective: action
reconstruction plus the configured weighted KL contribution when VAE is active. Historical M4,
M4.2, and FactorFiLM comparisons preserve that value and schema; they do not reinterpret it as an
arm-only loss. Evaluation requires the external JSONL row to match the metric stored under the
checkpoint fingerprint, so editing `metrics.jsonl` cannot change ranking.

The atomic, immutable selection record stores every candidate metric, deterministic ranking,
selected fingerprint, timestamp, and the validation schedule digest. Test evaluation in full mode
requires that exact selected checkpoint and exact predeclared test schedule. Test results are
written separately and cannot replace selection or resume/tune training. A development override is
explicitly nonfinal and cannot set `full_experiment_validated`.

## Counterfactual audits and interpretation

Before interpreting the unconditioned baseline, the offline audit checks every train scene group:
the six episodes, decoded initial RGB, Panda state, and physical observation equivalence; expert
first actions/chunks; and all task-pair action distances. Exact initial RGB digest equality remains
a reported diagnostic, but it is not the admission rule because independently encoded H.264
episodes can decode the same source frame to slightly different pixels. Admission instead requires
all six decoded first frames to be codec-equivalent under the M3B export's already frozen MAE and
PSNR thresholds, all six Panda states to match, and at least one task-dependent expert chunk in
every train scene group. The report records the exact and codec-equivalent fractions, worst
pairwise first-frame MAE, worst pairwise first-frame PSNR, thresholds, and one-to-many action
statistics. M4 does not reopen M3A during normal training; it relies on the completed M3B source-
reconstruction and video-quality gate for the source-to-decoded-frame authority.

For prediction sensitivity, RGB and Panda state are held fixed. Mixed-unconditioned inputs are
identical and deterministic predictions must be identical. Mixed-task-onehot varies only the six
one-hot basis vectors. Per-task compares six independently trained policies on the same fixed
physical observation. First actions and chunks receive all 15 pairwise distances and may be
compared with the six task-matched expert target chunks. Reports separately aggregate different
target/same-bin and same-target/different-bin distances. This measures task sensitivity, not
semantic language understanding.

The final comparison binds dataset/code/model fingerprints, parameter counts, train time, actual
peak GPU memory and throughput, losses, validation and locked-test rollouts, fresh-seed metrics,
per-task failures, counterfactual sensitivity, inference latency, and checkpoint-selection
evidence. It distinguishes control learning, ambiguity without a condition, oracle task
conditioning, and language understanding (not tested).

## M4.1 environment-action boundary

ACT is a continuous regression model. Even when every source action lies inside the finite M1
space, denormalized predictions can overshoot a bound; the target smoke observed raw gripper
outputs around `1.0508`--`1.0684` for Mixed-TaskOneHot and `1.1280` for PerTask while the active
gripper upper bound was `1.0`. The LeRobot policy postprocessor owns normalization inversion. It
does not own simulator admissibility. LangMani therefore applies the separately versioned
`BoundedActionEnvPostprocessorV0` after LeRobot postprocessing and immediately before `env.step`:

```text
model -> LeRobot policy postprocessor -> raw environment action
      -> BoundedActionEnvPostprocessorV0 -> executed action -> env.step
```

The configured mode is always explicit. `reject` preserves the strict audit: malformed,
nonfinite, unsupported-space, and finite bound violations fail before `env.step`. `project`
retains malformed/nonfinite actions as hard failures and deterministically computes each finite
component as `executed = min(max(raw, low), high)`, using the active environment action space rather
than hard-coded `[-1, 1]`. In-range values are returned unchanged. Projection is not described as
raw validity and is never hidden as implicit clipping. Binary gripper thresholding is not part of
M4.1; a future `BinaryGripperEnvPostprocessorV0` would be a separate ablation.

Each policy action has a machine-readable record containing the raw and executed arrays, actual
bounds, violation mask, projection count, lower/upper/maximum excess, L1/L2/L-infinity correction,
dtype, shape, step, and mode. Long per-step evidence is stored in a checksummed JSONL sidecar;
episode and benchmark JSON keep compact totals, rates, per-dimension counts, first projection,
maximum/mean excess and correction, and malformed/nonfinite flags. `task_success` means M1 success
under the executed actions. `strict_unprojected_success` additionally requires zero projected
actions. Projected task success is eligible for the M4.1 tiny-overfit task gate but never for the
strict metric.

Checkpoint identity remains unchanged. Every evaluation writes a separate runtime manifest whose
canonical SHA-256 identity binds the checkpoint fingerprint, saved preprocessor and postprocessor
fingerprints, serialized action-bound configuration, environment ID and actual action-space
contract, task-one-hot mapping version, rollout configuration, current code commit, and M4.1
runtime schema. Paths, hostname, wall time, and output location are nonsemantic. Reload validation
loads the model and both saved LeRobot processors locally, reconstructs the bound processor from
serialized configuration, checks deterministic raw output, then records the runtime fingerprint.
Interrupted evaluation staging is preserved under the owning run as failure evidence before a
new runtime is attempted.

## Commands and verification

The M4 command boundary is:

```bash
python scripts/train_act.py --help
python scripts/evaluate_act.py --help
python scripts/compare_act_baselines.py --help
python scripts/inspect_act_checkpoint.py --help
python environment/verify_m4.py
```

A representative full training invocation is:

```bash
python scripts/train_act.py \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --variant mixed_task_onehot --seed 0 --output-root outputs/models/act --full
```

For `per_task`, pass the exact stable TaskSpec ID through `--task-id`; the CLI also accepts a
human-facing alias such as `red_cube__left_bin` and immediately resolves it to that stable ID.
Dry-run, test-only fixture probes, tiny-overfit, development, full, resume, device, batch-size, and step controls remain
explicit and every failure returns nonzero.

`scripts/train_act.py --dry-run --planned-mode full` is a no-training semantic preflight: it uses
the exact `ExperimentMode.FULL` identity, fixed optimization, completed-data requirement, clean Git
gate, split views, train-only statistics, and runtime versions that the future full run will use.
It returns before creating the fingerprint-owned model directory. The outer
`environment/verify_m4.py --target-full --dry-run --action-bound-mode project` executes this for the
canonical six per-task runs and the two mixed runs, requires eight unique identities, and writes one
ordered plan. It validates existing M3B full evidence instead of rebuilding M3A/M3B and does not
train, select a checkpoint, access test rollouts, or claim physical M4 acceptance.

Non-target verification covers installed imports, immutable contracts, completed-data/feature/split
gates, train-only statistics and leakage audit, one-hot equality, shapes, public delta timestamps,
fingerprints, fixture forward/backward/optimizer work, checkpoint/processor local reload, selection,
test lock, rollout serialization, and truthful physical flags. Fixture success is not real dataset,
CUDA, model-quality, or physical evidence.

M4.1 target smoke requires explicit `--action-bound-mode project`. It validates the already
completed M0--M3B smoke report and its six-episode M3B dataset as the current-machine prerequisite,
then reuses the existing 5000-step PerTask and 10000-step Mixed-TaskOneHot checkpoints without
training. Reused checkpoints are evaluated only with the exact six-episode dataset root and export
fingerprint stored in both run manifests; this checkpoint-bound archive may have a different path
from a newly generated current-machine smoke export, but it must pass full storage validation.
Each runtime-specific evaluation retains its own `analysis.json`. If a clean canonical analysis for
the same run, checkpoint, and schema already exists from another runtime, revalidation preserves it
instead of overwriting it; a different run/checkpoint/schema or dirty canonical artifact still
fails. The current runtime analysis remains addressable through the evaluation output path.
Before full rollouts it runs
one strict probe: reproduce the known raw violation, prove reject blocks the step, reconstruct and
validate project mode, and execute exactly one projected legal M1 action. It then attempts one
PerTask rollout and all six one-hot counterfactual tasks from the shared scene group, retains raw
and executed action evidence, proves the rollout source has no M2 expert dependency, and uses 6/6
one-hot `task_success` as the primary quality gate. Full target mode requires the
completed 360-episode M3B dataset, offline counterfactual audit, all six per-task policies and both
mixed policies, validation-only selection, locked test, the 180-episode fresh benchmark, and the
final comparison/provenance audit. Every target-full evaluation command receives explicit
`--action-bound-mode project`; the evaluator's strict default is never allowed to silently select
the full experiment runtime. Both target modes reject a dirty current worktree before reusing
historical artifacts. Full mode resumes the latest identity-compatible checkpoint, recovers the sole
valid promoted orphan, revalidates already promoted evaluation directories, and reuses only a fully
completed immutable run instead of unconditionally retraining it.

An evaluation-only compatibility repair must not manufacture a new training identity. The explicit
`environment/verify_m4.py --target-full --reuse-completed-evidence --action-bound-mode project`
path therefore loads only the eight fingerprints in the existing passed full dry-run plan. It
requires the plan's clean training Git, exact dataset/split identity, canonical ordering, fixed
configuration and schedules, 20 sealed checkpoints and validations per run, immutable selection,
locked test, fresh-seed, analysis, and completion evidence. It invokes only
`compare_act_baselines.py`; missing or incompatible evidence is an error and can never fall back to
`train_act.py` or `evaluate_act.py`. Normal target-full also refuses implicit retraining when it
finds matching completed historical evidence; a genuinely new experiment uses a new output root.

Training and evaluation provenance are separate contracts. The run and checkpoint fingerprints
retain their original training Git. Every validation, test, and fresh-seed benchmark must instead
match the code Git and runtime fingerprint in its sibling `EvaluationRuntimeManifest`, including
`project` action handling, `physx_cpu`, the M1 environment/action space, task mapping, exact split,
fixed evaluation configuration, deterministic reload checks, and all per-episode runtime IDs. The
comparison report records one `training_git_commit` plus the audited `evaluation_git_commits` set.

The structured report keeps these flags independent:

```text
implementation_validated, git_baseline_validated, source_dataset_validated,
fixture_training_validated, cuda_training_validated, tiny_overfit_validated,
checkpoint_reload_validated, checkpoint_model_reload_validated,
policy_processor_reload_validated, action_bound_processor_reload_validated,
raw_action_bounds_validated, projected_action_bounds_validated,
closed_loop_inference_validated, strict_unprojected_rollout_validated,
tiny_overfit_task_success_validated,
train_stats_leakage_validated, validation_selection_validated, test_lock_validated,
per_task_experiment_completed, mixed_unconditioned_experiment_completed,
mixed_task_onehot_experiment_completed, fresh_seed_benchmark_completed,
completed_evidence_reuse_validated, reused_training_git_commit,
reused_evaluation_git_commits,
full_experiment_validated, baseline_quality_validated, physical_target_validated
```

The backward-compatible `checkpoint_reload_validated` aggregate is true only when the model,
LeRobot processors, and action-bound processor reload checks all pass (and its pre-M4.1 structural
checkpoint check remains true). `tiny_overfit_validated` mirrors the M4.1 task-success gate.
`raw_action_bounds_validated` can remain false while `projected_action_bounds_validated` and
`closed_loop_inference_validated` are true; that combination means the declared runtime corrected
audited finite regression overshoot before physical execution. `strict_unprojected_rollout_validated`
is intentionally independent and remains false if any successful episode required projection.

`full_experiment_validated` means the declared experiment completed correctly; it does not promise
good model quality. `baseline_quality_validated` remains separate and may be false with truthful
results. The RTX 4090 smoke chain has produced one real six-episode M3B dataset and the two CUDA
tiny-overfit checkpoints; their losses converged and one-hot changes predictions. The original
strict rollout correctly stopped on raw bound overshoot before its first step. The clean M4.1
projected smoke passed: PerTask was 1/1, Mixed-TaskOneHot was 6/6, combined task success was 7/7,
and strict-unprojected success was 0/7. The report correctly keeps raw-bound validity false while
projected-bound validity and physical closed-loop validation are true.

The later M4 full run completed all eight 100,000-step policies with validation-only selection,
locked test, 180-episode historical fresh-seed evaluation, and provenance-complete physical
reports. PerTask aggregate reached 31/36 locked-test and 143/180 fresh-seed success;
Mixed-Unconditioned reached 5/36 and 18/180; Mixed-TaskOneHot reached 27/36 and 101/180. Experiment
validity and physical acceptance are true, while baseline quality is false. These observed test and
fresh results are quarantined as historical evidence and cannot tune M4.2.

## M4.2 compatibility boundary

M4.2 preserves this document's M3B gate, exact split views, train-only normalization, model/loss,
atomic checkpoint, validation selection, test lock, and raw-versus-executed action contracts. The
existing `ActVariant` enum and all eight M4 identities remain unchanged.

Runtime ablation wraps the installed action-chunk prediction interface with an independent queue
that executes exactly 10, 5, or 1 actions from the unchanged 50-action chunk. After the horizon is
selected, `BinaryGripperEnvPostprocessorV0` may explicitly map component 7 by sign before the
existing project processor. It does not redefine M4.1 `reject` or `project`, and every raw,
binary-transformed, projected, and executed value remains independently auditable.

The one new `ACT-Mixed-TaskToken` is an independent M4.2 run type. It keeps Panda state at 9D and
uses LeRobot 0.6.0's public `FeatureType.ENV` input plus `nn.Linear(6, dim_model)` projection as a
dedicated Transformer token. It neither concatenates the command to qpos nor changes the historical
15D State-OneHot checkpoint. `CanonicalTaskTokenV0` is the same six-way oracle mapping and is not
natural-language understanding. Only the M3B validation split may select its checkpoint.

M4.2 locks a 12-scene development schedule and a disjoint, sealed 30-scene final schedule after
excluding 125 prior seeds. Development runtime selection never uses the old M4 test/fresh results
and never materializes or executes the new final schedule. A separate final authorization is
required after all runtime, architecture, and checkpoint choices are immutable.

M4.2 target-development completed with horizon 10 and the existing `project` runtime. Its
validation-selected 90,000-step TaskToken checkpoint achieved 15/72 development successes versus
35/72 for State-OneHot and 56/72 for PerTask, with 33 TaskToken wrong-object grasps. TaskToken is
rejected and its final benchmark remains unrun.

## M4.3a semantic-audit compatibility boundary

M4.3a consumes the immutable selected six PerTask, State-OneHot, and rejected TaskToken checkpoints
without changing any training or checkpoint identity. It uses all M3B validation and all
`m42_dev_v0` observations as separately labeled audit sources. M3B test, historical M4 fresh, and
`m42_final_v0` are forbidden.

The audit holds base-camera RGB and `PandaPolicyStateV0[9]` fixed while generating all six PerTask
reference chunks and each requested candidate chunk. It compares saved-postprocessor outputs before
runtime transforms. `ActionChunkDistanceV0` reports multiple scale/window/component families, while
its primary retrieval metric is fixed to action-range-normalized, arm-only L2 over the first locked
execution-horizon actions. Full-task, object/bin centroid, conditional-bin, first-interaction, and
post-grasp evidence remain distinct; nonzero action distance is not semantic correctness.

Evidence uses canonical SHA-256 identity, owned staging, independent checksum/schema validation,
atomic promotion, and a last completion marker. A completed fingerprint root is immutable. Local
fixtures may validate the implementation but cannot claim real checkpoint inference, semantic
completion, or physical target validation.

The real combined M4.3a audit completed at Git
`dfea8b3d7d28274909ff178cb9087a9a90e17ee7` over 108 observations and promoted evidence
fingerprint `sha256:6342bdf4b019df203e6021947cbb39deacac2ea78cc585b5062091ed1d228671`.
State-OneHot reached 37.96% full-task top-1, 76.85% target-object retrieval, and 50.93%
destination-bin retrieval. TaskToken reached 24.07%, 50.00%, and 49.07% respectively and remains
rejected. These results authorize one factorized repair because requested-object confusion and
approximately random bin retrieval persist; they do not validate any repair quality or authorize
the sealed final schedule.

## M4.3b FactorFiLM compatibility boundary

M4.3b introduces exactly one independent architecture, `ACT-Mixed-FactorFiLM`. It does not add a
historical `ActVariant`, change any M4/M4.2 checkpoint, or alter the M3B feature schema. One shared
training/inference conditioning component maps stable TaskSpec metadata into two canonical indices:

- `TargetObjectConditionV0`: `red_cube=0`, `green_cube=1`, `blue_cube=2`;
- `DestinationBinConditionV0`: `left_bin=0`, `right_bin=1`.

Instruction text is never parsed. The policy still consumes only `observation.images.base_camera`
and `PandaPolicyStateV0[9]`; there is no state-appended one-hot, object/bin one-hot, combined task
token, language embedding, or LeRobot ENV feature.

The object path uses a learned 32D embedding followed by a projection to 1,024 values. The first
512 are gamma and the remaining 512 beta for the ResNet-18 layer-4 feature map
`[B,512,8,8]`. Explicit `[B,512,1,1]` broadcasting applies
`visual * (1 + gamma_object) + beta_object` after backbone extraction and before ACT's image
projection. Destination identity cannot enter this path.

The bin path uses a separate learned 32D embedding and projection to 1,024 values. Its gamma/beta
each have shape `[B,512]` and modulate the encoded nine-dimensional Panda-state token after
`nn.Linear(9,512)` and before Transformer processing. Target-object identity cannot enter this
path. Projection weights use `normal(mean=0,std=1e-5)` and projection biases are zero, making both
initial transforms close to identity without eliminating the first-step embedding gradient.

For the locked primary ACT configuration, the unmodified ACT has 51,576,712 parameters. The object
embedding/projection adds 33,888 and the bin embedding/projection adds 33,856, for an exact 67,744
increase and 51,644,456 total. No other model capacity changes.

The adapter uses these public LeRobot 0.6.0 symbols:

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

`ACTPolicy` exposes no public hook at the required intermediate stages. A single project-owned
adapter therefore subclasses it and registers instance-local Torch forward hooks on the semi-stable
`policy.model.backbone` and `policy.model.encoder_robot_state_input_proj` outputs. It neither imports
the private `lerobot.policies.act.modeling_act.ACT` class directly nor monkey-patches global ACT
behavior. A compatibility contract binds public signatures, LeRobot version, underlying model type,
backbone dictionary key, `[B,512,8,8]` feature map, `[B,dim_model]` state projection, latent/state/64-
image-token encoder structure, decoder position count, and `[B,chunk_size,8]` action head. Drift is
a hard failure and never falls back to State-OneHot or TaskToken.

FactorFiLM training extends the existing lifecycle after the installed LeRobot preprocessor. Its
architecture/structural baseline is fixed at clean Git commit
`8ee0f1babf36b91d1ee2a39701e4a6db6003b660`; D-054 explicitly reauthorizes clean compatibility
commit `0088e2937556c123c37c2dbe69f73301b1eebfd0` as the target-training producer. Later evaluation
and verifier commits cannot alter that run or checkpoint identity. Post-training evaluator/verifier
implementation is fixed at `1bacb66d2a6f7c3f2d18d6f65ad7865af9a12cd6`. It uses the same 288 M3B
train episodes, 36 validation episodes, train-only statistics, primary model and optimizer
configuration, locked seed 0/data order, action chunk 50, 100,000 steps, 5,000-step checkpoint
interval, expected 20 checkpoints, batch size 32, bfloat16 CUDA behavior, horizon 10, and `project`
runtime as State-OneHot. Only the separated conditioning modules increase parameters.

FactorFiLM run, architecture, mapping, processor, checkpoint, and selection identities are
independent canonical-JSON/SHA-256 contracts. Checkpoints extend the M4 atomic format with both
embeddings/projections and the architecture sidecar, while retaining base ACT, processors,
train-only statistics, optimizer/scheduler/RNG state, versions, Git, dataset/split fingerprints,
and global step. Direct saves reject nonempty destinations before upstream files are written.
Resume rejects every dataset, split, mapping, embedding, injection, ACT, statistics, optimizer,
schema, or completed-run mismatch, and accepts only the latest declared checkpoint or sole next
atomically promoted orphan.

Only M3B validation can select among the expected 20 checkpoints. The queue has a canonical
fingerprint; selection embeds it and must match its run, schedule, checkpoint fingerprints, and
steps. The ranking is success,
wrong-object interaction, wrong object in target bin, target off table, timeout, action loss, then
earlier step. M3B test, historical M4 fresh, `m42_dev_v0`, `m42_final_v0`, and semantic-audit
observations are not selection inputs. The queue field `offline_validation_action_loss` retains the
historical total ACT objective described above. Development rollout begins only after immutable
validation selection and a fresh-process reload proof.

The local fixture may construct, optimize, save, and reload a reduced ACT model and processors. It
must prove finite forward/backward, gradients in the base model and all four new parameter groups,
separated injection, near-identity initialization, and deterministic fresh-instance inference. That
evidence can set only FactorFiLM implementation/fixture flags. It cannot set training, checkpoint,
selection, development, physical, final, or SmolVLA flags.

## M4.3b target-development evaluation

The authorized experiment consists of one producer run and post-training evidence stages. The
producer performs exactly 100,000 seed-0 CUDA steps and publishes checkpoints at steps
`5000,10000,...,100000`. All 20 are evaluated on exactly six M3B validation scene groups times six
tasks (36 episodes per checkpoint). The immutable seven-key ranking is highest success count,
lowest wrong-object interaction count, lowest wrong-object-in-target-bin count, lowest off-table
count, lowest timeout count, lower checkpoint-bound validation objective, then earlier step.
Development observations cannot affect selection.

After selection, a fresh operating-system process reconstructs the policy, both FactorFiLM
mappings, preprocessor, policy postprocessor, and explicit H=10/`project` runtime. The complete
postprocessed environment-semantic action chunk must have shape `[50,8]` and match the producer
reference at `atol=1e-6`, `rtol=1e-6`. Any fingerprint, shape, value, processor, or runtime mismatch
stops before a development reset.

The fixed development matrix uses the existing `m42_dev_v0` schedule only after that lock. It runs
the frozen six PerTask controls, frozen State-OneHot, and selected FactorFiLM on identical scene/
task identities: 72 episodes per control and exactly 216 total. All use `num_envs=1`,
`pd_joint_pos`, the unchanged M1 success contract, horizon 10, explicit project action handling,
identical reset semantics, and no M2 expert. Validation and development semantic retrieval stay
separately identified and use the M4.3a action-range-normalized arm-only H=10 metric before runtime
projection. Newly executed development episodes retain exact first-interaction and post-grasp
diagnostics; unavailable values are never reconstructed from aggregates.

Final evaluation is authorized only when all 16 conditions pass conjunctively: at least 50/72
successes and 69.44%, gap to PerTask at most eight, at least 7/12 per TaskSpec, at most six
wrong-object grasps, at most two wrong objects in the target bin, at most 22 timeouts, zero target
in wrong bin, zero target off table, zero arm projections, zero NaN, zero Inf, zero malformed
action, at least 70% primary full-task top-1, at least 80% target-object retrieval, and task-
sensitivity ratio to PerTask at least 0.75. No rounding, proxy, or threshold relaxation is allowed.
Correct execution may therefore have `passed=true` and `physical_target_validated=true` while
`development_quality_gate_passed=false` and `final_benchmark_authorized=false`.

The post-training evaluator is `scripts/evaluate_act_factor_film.py`; its implementation lineage is
`1bacb66d2a6f7c3f2d18d6f65ad7865af9a12cd6`, its resumable evidence root is outside the immutable
training run, and it binds both producer and evaluator Git identities. The
read-only `environment/verify_m43b.py` independently validates all 20 checkpoints and validation
records, ranking, reload, 216 paired identities, semantic/first-interaction/post-grasp reports,
quality calculation, checksums, provenance, and access flags. It never trains, changes selection,
or executes a rollout.

## Handoff

The immediate handoff is completion and independent verification of the authorized FactorFiLM
target-development sequence, not M4.2-final or SmolVLA. Until real evidence passes, training,
selection, reload, development, and physical flags remain unclaimed. M3B test, historical M4 fresh,
`m42_final_v0`, automatic retraining, SmolVLA, and M5 stay inaccessible, and no command starts a
final or language-policy milestone automatically.
