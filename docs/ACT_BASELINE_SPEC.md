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

The ACT preprocessor performs renaming, batch insertion, device transfer, and normalization; the
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
starting a long optimization. Tiny-overfit is a sanity gate, not full evidence. Per-task tiny
overfit requires final fixed-view offline loss no greater than 50% of the pre-training loss on that
same deterministic tiny view and records its same-scene rollout outcome. Task-one-hot tiny overfit uses
one complete six-task train group, varies only the oracle one-hot, and attempts all six rollouts;
its target quality gate is 6/6. A miss is reported and investigated without changing labels,
success geometry, test data, or temporal/action contracts.

## Closed-loop policy evaluation

The adapter creates the M1 environment with `num_envs=1`, RGB policy observations,
`pd_joint_pos`, and 20 Hz control. It does not import or invoke M2. Each query reads only
`base_camera` RGB and `PandaPolicyStateV0`; the task-one-hot variant additionally encodes the active
nonprivileged command `TaskSpec`. It applies the saved preprocessor, calls installed
`ACTPolicy.select_action`, applies the saved postprocessor, and checks float32 `[1,8]`, finiteness,
and M1 bounds.

Out-of-bounds actions are recorded as `invalid_action` and terminate the episode. They are never
silently clipped. Inference failures, environment failures, off-table outcomes, truncation, and
timeouts remain distinct. Inference, environment-step, and whole-episode timing are measured
separately. Raw records retain conservative M1 evaluation fields, grasp/wrong-grasp events, action
magnitudes and per-joint ranges, event times, latency samples, and failure reasons.

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

The offline validation loss used here is checkpoint-bound evidence. Evaluation requires the
external JSONL row to match the metric stored under the checkpoint fingerprint; editing
`metrics.jsonl` cannot change ranking.

The atomic, immutable selection record stores every candidate metric, deterministic ranking,
selected fingerprint, timestamp, and the validation schedule digest. Test evaluation in full mode
requires that exact selected checkpoint and exact predeclared test schedule. Test results are
written separately and cannot replace selection or resume/tune training. A development override is
explicitly nonfinal and cannot set `full_experiment_validated`.

## Counterfactual audits and interpretation

Before interpreting the unconditioned baseline, the offline audit checks every train scene group:
the six episodes, initial RGB digest, Panda state, and physical observation equivalence; expert
first actions/chunks; and all task-pair action distances. It reports the fraction of identical
initial observations and the one-to-many target variance.

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

Non-target verification covers installed imports, immutable contracts, completed-data/feature/split
gates, train-only statistics and leakage audit, one-hot equality, shapes, public delta timestamps,
fingerprints, fixture forward/backward/optimizer work, checkpoint/processor local reload, selection,
test lock, rollout serialization, and truthful physical flags. Fixture success is not real dataset,
CUDA, model-quality, or physical evidence.

Target smoke first requires the M0/M1/M2 target gate, M3A target smoke, M3B target smoke, and one
real six-episode M3B group. It then requires CUDA forward/backward, both declared tiny-overfit
paths, checkpoint/processor reload, and real closed-loop M1 steps. Full target mode requires the
completed 360-episode M3B dataset, offline counterfactual audit, all six per-task policies and both
mixed policies, validation-only selection, locked test, the 180-episode fresh benchmark, and the
final comparison/provenance audit. Both target modes reject a dirty current worktree before reusing
historical artifacts. Full mode resumes the latest identity-compatible checkpoint, recovers the sole
valid promoted orphan, revalidates already promoted evaluation directories, and reuses only a fully
completed immutable run instead of unconditionally retraining it.

The structured report keeps these flags independent:

```text
implementation_validated, git_baseline_validated, source_dataset_validated,
fixture_training_validated, cuda_training_validated, tiny_overfit_validated,
checkpoint_reload_validated, closed_loop_inference_validated,
train_stats_leakage_validated, validation_selection_validated, test_lock_validated,
per_task_experiment_completed, mixed_unconditioned_experiment_completed,
mixed_task_onehot_experiment_completed, fresh_seed_benchmark_completed,
full_experiment_validated, baseline_quality_validated, physical_target_validated
```

`full_experiment_validated` means the declared experiment completed correctly; it does not promise
good model quality. `baseline_quality_validated` remains separate and may be false with truthful
results. As of this implementation review, no authoritative M3B target dataset, CUDA training,
tiny-overfit 6/6 result, real learned-policy rollout, GPU-memory/throughput measurement, locked
test/fresh benchmark, or physical target acceptance has been produced. Those items remain pending
the native Linux RTX 4090 target.

## Handoff

The next language-conditioned milestone should add the declared SmolVLA/text path only after M4's
real target dataset and ACT controls are complete. It must reuse M4's scene splits, train-only
statistics discipline, validation selection, test lock, fresh-seed schedule, and result provenance,
and compare language conditioning against both the ambiguous unconditioned model and the oracle
task-one-hot upper bound.
