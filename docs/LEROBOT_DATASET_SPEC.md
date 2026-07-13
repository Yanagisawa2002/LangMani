# LeRobot Dataset Specification

## Status and boundary

M3B deterministically derives one local LeRobotDataset v3 from a validated M3A archive for
`LangMani-PickPlaceByInstruction-v0`. The M3A HDF5/JSON archive remains the sole authority. The
derived dataset may be deleted and regenerated; M3B never changes M3A files or acceptance
decisions.

M3B does not train ACT or SmolVLA, publish to the Hub, add paraphrases, collect trajectories, or
export failed attempts. It uses one process, one environment, `pd_joint_pos`, RGB only, and the
existing canonical six-task language contract.

## Source acceptance gate

No HDF5 trajectory can be read until `validate_m3a_source_for_export()` succeeds. Its order is
fixed:

1. run the complete `inspect_raw_dataset()` corruption, schema, checksum, projection, provenance,
   action-replay, state-audit, and initial-layout validator;
2. load the authoritative collection manifest;
3. consume a content-bound `langmani-m3a-verification-v3` target report;
4. match source root, collection run ID, configuration fingerprint, archive digest, mode, and all
   independent physical validation flags;
5. derive accepted groups from `manifest.scene_groups` in candidate order and episodes from each
   group's canonical six-task order;
6. prove that this ordered index is an exact one-to-one set with `manifest.raw_episodes`.

Smoke input is exactly one complete group and six episodes. Full input is exactly 60 complete
groups and 360 episodes, with 60 episodes per TaskSpec. Expert failures, replay failures, partial
groups, wrong-object/wrong-bin false successes, unclassified failures, retained failure
trajectories, and orphan HDF5 entries are rejected.

The source archive digest is a canonical SHA-256 over collection/run identity, ordered accepted
group and episode IDs, group bundle checksums, and ordered source-shard checksums. It excludes
paths and timestamps. Each shard is checksummed again immediately before first use, and the whole
source is inspected and re-digested before promotion.

## Installed LeRobot 0.6.0 API

M3B uses the installed public imports:

```python
from lerobot.configs import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
```

The verified writer lifecycle is:

```python
dataset = LeRobotDataset.create(
    repo_id=...,
    fps=20,
    features=...,
    root=nonexistent_staging_child,
    robot_type="panda",
    use_videos=True,
    video_backend="pyav",
    rgb_encoder=...,
    image_writer_processes=0,
    image_writer_threads=0,
    batch_encoding_size=1,
    streaming_encoding=False,
    encoder_threads=1,
    data_files_size_in_mb=100,
    video_files_size_in_mb=200,
)
dataset.add_frame(fresh_frame_dict)
dataset.save_episode(parallel_encoding=False)
dataset.finalize()
```

Installed signatures were checked for `LeRobotDataset`, `create`, `add_frame`, `save_episode`, and
`finalize`. `create` supports data/video file-size settings but not a public `chunks_size` argument,
so M3B does not expose one. `add_frame` mutates its input by removing the special `task` field;
M3B passes a new frame dictionary every time. `save_episode` writes Parquet before video encoding
and is not transactional. The project adapter therefore owns an `OPEN -> FINALIZED | FAILED`
state machine and permits exactly one successful `finalize()` call.

The inspected 0.6.0 signatures are:

```text
LeRobotDataset(repo_id, root=None, episodes=None, episode_filter=None,
               image_transforms=None, delta_timestamps=None, tolerance_s=0.0001,
               revision=None, force_cache_sync=False, download_videos=True,
               video_backend=None, return_uint8=False, depth_output_unit="mm",
               batch_encoding_size=1, rgb_encoder=None, depth_encoder=None,
               encoder_threads=None, streaming_encoding=False,
               encoder_queue_maxsize=30)
LeRobotDataset.create(repo_id, fps, features, root=None, robot_type=None,
                      use_videos=True, tolerance_s=0.0001,
                      image_writer_processes=0, image_writer_threads=0,
                      video_backend=None, batch_encoding_size=1,
                      rgb_encoder=None, depth_encoder=None,
                      metadata_buffer_size=10, streaming_encoding=False,
                      encoder_queue_maxsize=30, encoder_threads=None,
                      video_files_size_in_mb=None, data_files_size_in_mb=None)
add_frame(self, frame)
save_episode(self, episode_data=None, parallel_encoding=True)
finalize(self)
```

Local loading always uses:

```python
LeRobotDataset(repo_id=..., root=..., video_backend="pyav", return_uint8=True)
```

The explicit PyAV backend is required because LeRobot 0.6.0 selects TorchCodec by package
presence, while this Windows installation's TorchCodec DLL chain is not loadable.

## Policy feature schema

Only these user-defined features are created:

| Key | LeRobot dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `observation.images.base_camera` | `video` | `(256, 256, 3)` | uint8 RGB, HWC at writer input |
| `observation.state` | `float32` | `(9,)` | `PandaPolicyStateV0` qpos |
| `action` | `float32` | `(8,)` | exact recorded `pd_joint_pos` action |

LeRobot legitimately adds `timestamp`, `frame_index`, `episode_index`, `index`, and `task_index`.
`task` is supplied as the accepted canonical string and stored through LeRobot task metadata; it
is not a numeric policy observation.

No full environment state, object/bin pose, target index, color ID, task ID, scene ID, expert
phase/status, success oracle, planner state, TCP target, collision geometry, depth, segmentation,
human camera, or wrist camera is exported.

## PandaPolicyStateV0

The state is unnormalized joint position in this semantic order:

1. `panda_joint1`
2. `panda_joint2`
3. `panda_joint3`
4. `panda_joint4`
5. `panda_joint5`
6. `panda_joint6`
7. `panda_joint7`
8. `panda_finger_joint1`
9. `panda_finger_joint2`

The exporter obtains active-joint names and qpos through the public robot API, requires exactly
these nine unique names, and reorders by name. It does not flatten the M1 observation dictionary,
which would also expose qvel, and it does not use hard-coded articulation-state offsets.

## Action schema

The eight float32 components retain their M2/M3A `pd_joint_pos` values and semantics:

1. `panda_joint1_position_target`
2. `panda_joint2_position_target`
3. `panda_joint3_position_target`
4. `panda_joint4_position_target`
5. `panda_joint5_position_target`
6. `panda_joint6_position_target`
7. `panda_joint7_position_target`
8. `panda_gripper_mimic_command`

M3B does not normalize, clip, reinterpret, interpolate, convert control modes, drop the first
action, or append a terminal action. `action_tolerance` is fixed to `0.0`, and independent
validation uses exact array equality rather than approximate numeric matching.

## Camera and reconstruction

The source is the M1 `base_camera`:

| Property | Value |
| --- | --- |
| Eye / target | `(0.65, -0.75, 0.70)` / `(-0.04, 0.0, 0.08)` |
| Resolution | 256 x 256 RGB |
| FOV | 1.05 rad |
| Near / far | 0.01 m / 10.0 m |
| Shader | `minimal` |
| Target render backend | `sapien_cuda` |

For every episode a fresh M1 environment uses `num_envs=1`, `obs_mode="rgb"`,
`reward_mode="none"`, source `sim_backend`, and `pd_joint_pos`. It first resets with the recorded
scene seed and exact TaskSpec, then verifies `get_episode_specs()`. For every training frame it
calls public `set_state_dict(state[t])`, then public `get_obs()`. ManiSkill 3.0.1 synchronizes the
scene, captures sensors, reads camera data, and synchronizes CUDA inside `get_obs()`; M3B never
uses an artificial `env.step`, human render camera, direct camera capture, or private GPU/render
method.

## Temporal contract

An M3A trajectory contains T actions and T+1 states. M3B emits exactly T frames:

```text
frame[t] = {
  observation.images.base_camera: render(state[t]),
  observation.state: Panda qpos at state[t],
  action: raw_action[t],
  task: accepted canonical instruction,
}
timestamp[t] = t / 20
```

`state[T]` is retained only in M3A for audit. It is never emitted because no `action[T]` exists.

## Language metadata

All frames in an episode use the canonical instruction already stored in the accepted M3A record.
M3B neither regenerates language from IDs nor adds paraphrases. The six strings remain the red,
green, and blue cube paired with left and right bin instructions defined by M1.

## Stable export identity and source mapping

The export fingerprint is `sha256:<64 hex>` over canonical JSON containing at least:

- M3A collection/config/archive identity and ordered accepted episode IDs;
- source/output schema versions and environment version;
- LeRobot, PyAV, and libavcodec versions;
- feature, policy-state, action, camera, codec, control-mode, FPS, and split contracts.

It excludes source/output paths, staging paths, wall-clock time, filesystem enumeration, and
Python `hash()`. Every derived episode record maps its LeRobot index to collection run ID, source
fingerprint and archive digest, raw trajectory ID, accepted scene-group ID, scene seed/ID, task
ID/specification, canonical instruction, shard ID and relative path, HDF5 group, HDF5/JSON
checksums, aggregate source checksum, source T, split, raw RGB digest, output frame count, and
status.

The raw RGB digest hashes a version tag and, for every frame, length-delimited canonical frame
index/shape/dtype metadata plus contiguous raw RGB bytes. It identifies deterministic rendering;
it is not compared directly with lossy decoded video bytes.

## Scene-level split

Groups are ranked by SHA-256 of canonical `{split_seed, scene_group_id}`; equal digests use stable
scene-group ID as a deterministic tie-break. Full assignment is:

| Split | Scene groups | Episodes | Episodes per TaskSpec |
| --- | ---: | ---: | ---: |
| train | 48 | 288 | 48 |
| validation | 6 | 36 | 6 |
| test | 6 | 36 | 6 |

Smoke uses train/validation/test group counts 1/0/0. Split membership is stored as episode-index
lists for one physical dataset; video and Parquet data are not duplicated.

## Video encoding and measured quality gate

The fixed RGB configuration is software H.264 (PyAV resolves `h264` to `libx264`), `yuv444p`,
CRF 18, GOP 2, preset `medium`, `fast_decode=0`, one encoder thread, 20 FPS, and explicit PyAV.
Local inspection recorded LeRobot 0.6.0, PyAV 15.1.0, libavcodec 61.19.101, and libx264 core 165.
The version/configuration values become provenance rather than being silently substituted.

Initial AV1/yuv420p calibration produced good results on a smooth fixture but a second high-chroma
fixture fell to 29.70 dB; lowering AV1 CRF from 30 to 25 did not remove the 4:2:0 chroma ceiling.
The selected H.264/yuv444p configuration measured mean absolute pixel error 0.773-1.069 and PSNR
44.13-47.63 dB on that seven-frame fixture. Before target runs, M3B retained acceptance thresholds
at mean absolute error no greater than 5.0 and minimum PSNR at least 30 dB. Validation decodes every video frame,
structurally validates every frame, and compares all frames for short episodes or at least five
deterministic first/interior/final samples per episode to directly reconstructed raw RGB.

## Staging, finalization, and output layout

`LeRobotDataset.create()` requires a nonexistent dataset root. M3B creates a fingerprint-owned
container and passes its nonexistent `dataset/` child to LeRobot. A prior incomplete container is
never resumed; `--clean-staging` must explicitly remove an owned staging directory. A complete
destination is immutable.

On Windows, LeRobot/PyAV ffconcat fails when multiple episode videos are joined under this
repository's non-ASCII path. M3B therefore uses a documented ASCII staging directory at the root
of the same volume, then performs a same-filesystem atomic directory rename. Final datasets load
normally from Unicode paths. Linux uses the destination-adjacent staging directory.

Successful order is:

1. write every episode once;
2. call LeRobot `finalize()` once;
3. write namespaced sidecars;
4. independently load and validate the staged dataset;
5. revalidate unchanged M3A source;
6. flush staged files;
7. atomically rename into the final destination;
8. remove the empty staging owner;
9. write `langmani/complete.json` last.

The output is:

```text
data/.../*.parquet
meta/info.json
meta/tasks.parquet
meta/episodes/.../*.parquet
meta/stats.json
videos/observation.images.base_camera/.../*.mp4
langmani/export_config.json
langmani/export_manifest.json
langmani/source_episode_mapping.json
langmani/split_manifest.json
langmani/validation_report.json
langmani/dataset_summary.json
langmani/dataset_card.md
langmani/complete.json
```

Sidecars store semantic IDs and relative source paths only. They contain no Windows drive path or
machine-specific absolute source path. A failure leaves structured diagnostics in staging and no
completion marker; partially finalized LeRobot data is never promoted or repaired in place. The
narrow crash window after atomic promotion but before `complete.json` remains explicitly
incomplete: cleanup requires `--clean-staging`, and deletion is permitted only when the promoted
export manifest proves the same fingerprint. A completion marker always makes the destination
immutable.

## Independent validation

The validator checks local files before constructing `LeRobotDataset`, preventing a corrupt root
from falling back to a Hub download. It then independently verifies:

- public local load, exact LeRobot `v3.0` metadata, feature allowlist, episode/task/frame counts,
  and canonical order;
- full 60/360/60-per-task counts or smoke 1/6 counts;
- 48/6/6 split balance and no physical-scene leakage;
- one-to-one source/derived mapping, episode boundaries, timestamps, and task strings;
- strict raw action equality and reconstructed qpos alignment;
- raw render digests and sampled lossy-video quality;
- all expected video files, every decoded frame, and frame dimensions/content;
- every Parquet file through PyArrow;
- absence of privileged features and incomplete markers;
- completion marker/fingerprint agreement when validating a final destination;
- first and final dataset item, one deterministic item per nonempty split, and a PyTorch
  `DataLoader(..., num_workers=0)` batch.

Metadata-only validation is a fast diagnostic and cannot set full/source/video acceptance flags.

## Commands and acceptance

Dry-run (no Parquet or MP4 creation):

```bash
python scripts/export_lerobot_dataset.py \
  --source-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --output-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --repo-id langmani/pick-place-by-instruction-v0 \
  --full --dry-run
```

Export, independent validation, and inspection:

```bash
python scripts/export_lerobot_dataset.py --source-root ... --output-root ... --full
python scripts/validate_lerobot_dataset.py --dataset-root ... --source-root ... --full
python scripts/inspect_lerobot_episode.py --dataset-root ... --episode-index 0
```

Verification modes:

```bash
python environment/verify_m3b.py
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3b.py --target-smoke
CUDA_VISIBLE_DEVICES=0 python environment/verify_m3b.py --target-full \
  --source-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1
```

Structural mode runs contract tests and a real generated-array LeRobot/PyAV fixture. It never sets
physical validation. Target smoke first runs the ordered M0/M1/M2 plus M3A target-smoke gate, then
exports and validates one real six-task group. Target full requires a content-bound M3A full report
and validates all 60 groups/360 episodes. The verification report keeps implementation, source,
smoke/full export, load, Parquet, video, source alignment, split, leakage, and physical flags
independent.

## Known limitations and M4 handoff

Native Linux RTX 4090 full acceptance passed for all 60 groups, 360 episodes, and 64,548 frames.
The exact 288/36/36 episode split, 48/6/6 scene-group split, per-task balance, video decode,
source/action/state alignment, public reload, and DataLoader gate all passed; the export fingerprint
is `sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4`.
Windows fixture video writing/reading also passes, but SAPIEN cannot instantiate the Panda from the
current non-ASCII repository path, so no local fixture is described as real simulation validation.
Both OpenCV wheels still share the `cv2` namespace; M3B does not import OpenCV and uses LeRobot/PyAV
instead.

The proposed M4 ACT baseline consumes only a completed local M3B root and project split manifest:

```text
ActBaselineConfig(
  dataset_root,
  expected_export_fingerprint,
  train_episode_indices,
  validation_episode_indices,
  image_key="observation.images.base_camera",
  state_key="observation.state",
  action_key="action",
  fps=20,
)
```

M4 may read LeRobot task strings/task indices for canonical language conditioning, but it must not
open the M3A archive, recompute acceptance, change the feature schema, or use the test split for
model selection.
