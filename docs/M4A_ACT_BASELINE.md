# M4A: one language-independent ACT baseline

This implementation starts from main `6920b52c1f48c278e669cd71b69b8949dd900f3a` (M3B), on
`codex/m4a-act-baseline`. Later historical experiment branches are not dependencies. Unrecoverable
historical data must be regenerated on a new AutoDL native Linux instance. Generated-array tests
are explicitly fixtures, never substitutes for that archive, formal training, or physical evaluation.

## Architecture inspected before implementation

| Existing component | Contract retained |
| --- | --- |
| `src/langmani/environments/pick_place_by_instruction.py` | `LangMani-PickPlaceByInstruction-v0`, one Panda, 20 Hz, `pd_joint_pos`, 200-step horizon, existing geometric containment/release/static success predicate |
| `src/langmani/experts/pick_place.py` and `planner.py` | Deterministic twelve-phase privileged grasp/transport/place expert and upstream mplib planning |
| `collection/recording.py`, `datasets/archive.py`, `datasets/types.py` | M3A native HDF5/JSON authority: T unchanged actions and T transition labels, T+1 states; stable scene/task/episode identity, checksums and independent replay |
| `datasets/lerobot_export.py`, `lerobot_writer.py` | Reconstruct public state[t]/RGB[t], pair with original action[t]; one staged, finalized LeRobot v3 dataset plus `langmani/` provenance sidecars |
| `datasets/policy_state.py` | `PandaPolicyStateV0`: seven arm joints and two measured finger positions; semantic name-based ordering |
| `datasets/lerobot_types.py` | uint8 HWC writer images; reader CHW; state float32[9]; action float32[8], seven absolute arm targets plus normalized gripper mimic command |
| M1 camera configuration | Policy `base_camera` 256×256 RGB, look-at (0.65,-0.75,0.70)→(-0.04,0,0.08); separate 512×512 human camera; no wrist/depth/segmentation inputs |
| `environments/specs.py` | Exactly six canonical sentences: red/green/blue cube × left/right bin; deterministic TaskSpec metadata |
| `datasets/splits.py` | Hash-ranked scene groups, saved split seed, 48/6/6 full scene split; six task counterparts stay together |
| Existing tests/CLIs | pytest CPU-safe markers, Ruff, package build, read-only M3B validator and independent Linux target verification |

M4A adds three focused modules and one CLI. `m4a_data` fully validates decoded data and selects
red-cube/left-bin episodes; `m4a_training` configures stock ACT and invokes the official trainer;
`m4a_evaluation` resets the environment, uses upstream `select_action` queue semantics, and
evaluates the existing expert on the same explicit seeds. The public camera extractor is a wrapper
over the unchanged M3B extraction function, preserving reconstruction behavior.

## Policy and expert boundary

ACT receives only normalized RGB and qpos. Its 9D state never contains task IDs, object/bin poses,
grasp flags, planner output or language. Feature keys, component names, shapes and finite values are
allowlisted in validation and checkpoint loading. Evaluation builds a new two-key dictionary each
step. The expert's privileged context is instantiated only in the separate expert rollout.

The evaluator hashes `get_state_dict()` immediately after reset for pairing; these bytes do not
enter the policy. Static M1 bins are fixed by environment code, while the hash checks dynamic
objects/robot. Seed lists exclude every source scene. Both controllers receive the same TaskSpec,
backend and 200-step limit, and use the environment's success predicate. Expert planner status is
also recorded separately. Infrastructure errors invalidate the comparison, preserve error rows,
produce null gaps, and return a nonzero command exit. A valid zero-success evaluation stays zero.

ACT predicts 50 future actions; `select_action` queues ten, then queries again using the latest
observation. `policy.reset()` and both processors reset on every episode. Returned actions are
unnormalized before `env.step`. Invalid/nonfinite actions and out-of-bounds absolute arm joint
commands end the episode explicitly. The normalized gripper command saturates to `[-1, 1]`,
matching ManiSkill's existing mimic-controller preprocessing exactly; arm commands are not clipped.
Per-episode and aggregate saturation counts and maximum overshoots are recorded. This is continuous
native gripper saturation, with no binary replacement or expert rescue. An evaluation with zero ACT
environment steps fails and cannot report a valid comparison. See D-031 for the retained original
strict-rejection failure and the native-controller equivalence diagnostic.

## Upstream compatibility and reproducibility

- Pinned public implementation: `lerobot==0.6.0`, `ACTConfig`, `ACTPolicy`, processors and
  `lerobot.scripts.lerobot_train.train`. No copied LeRobot source or custom optimizer/trainer loop.
- ResNet-18 ImageNet initialization, stock 512D ACT transformer/VAE, 50-step chunks, ten executed
  actions; **51,576,712 parameters** observed in the real CPU fixture. Smoke and full use the same
  architecture; only the optimization budget differs. First use needs the cached torchvision weights
  or permission/network access to download those public weights.
- Upstream 0.6's episode filtering does not recompute `meta.stats`. A scoped factory adapter uses
  only selected training rows for state/action statistics and fixed ImageNet RGB statistics. It
  restores the factory on exit, including exceptions, and edits no installed package or dataset.
- Dataset rows are validated in full before training. The split records source/export identity,
  a content digest over all files, the M3B random seed, raw/derived episode IDs, scene seeds/groups,
  partitions, exact frame counts and excluded test IDs. Data changes invalidate resume/evaluation.
- The last planned training checkpoint is evaluated; M4A performs no sweep or checkpoint selection.
  The six held-out validation episodes remain available for later explicit diagnostics. No test
  episode enters training, normalization or this fresh-seed benchmark.
- Resume uses upstream optimizer/scheduler/RNG restoration. The source Git commit, dataset split,
  mode, seed, device, batch size and recomputed normalization must match. `--steps` is the desired
  **total** global step, not additional steps. Full training requires a clean checkout.
- Windows CPU review writes `last_checkpoint.json` instead of the optional upstream `last` symlink,
  whose creation failed here with WinError 1314. Actual model/config/processor/training-state saving
  is upstream and unchanged. Linux retains upstream symlink behavior. All commands use explicit
  checkpoint paths, so neither mechanism is needed for loading.

Upstream references: [ACT documentation](https://huggingface.co/docs/lerobot/act) and
[pinned trainer](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/scripts/lerobot_train.py).
Installed 0.6 source and real fixture execution, rather than an assumption about a newer API, drive
this adapter. Only dataset/training extras are activated; no SmolVLA dependency is added.

## Formal Linux execution order

Set up `environment/environment.yml`, the NumPy-1 planner overlay, and the checkout as in README.
Export `LANGMANI_PLANNER_PYTHON` for the entire chain; M2/M3A use it for the expert, and paired M4A
evaluation retains a separate expert worker result/log/runtime record for each seed. The worker
uses the same environment configuration as ACT, and exact initial-state hashes must match.
Preserve the output
paths across the following commands. They reuse the existing target gates and never replace their
results with generated arrays:

```bash
python environment/verify_m3a.py --target-smoke
python environment/verify_m3a.py --target-full \
  --dataset-root outputs/datasets/m3a/langmani-pick-place-raw-v1 --create-new-run
python environment/verify_m3b.py --target-full \
  --source-root outputs/datasets/m3a/langmani-pick-place-raw-v1 \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1
```

Stop if a gate fails; repair the actual infrastructure/expert issue before collection or training.
The original Linux mplib/NumPy ABI and Vulkan conditions remain to be physically checked on the new
instance. No runtime is provisioned or claimed verified by this local implementation.

The exact validation, split, smoke, full training and paired evaluation commands are in README's
**M4A — ACT Baseline** section. To resume an interrupted full run, use its last *completed* checkpoint:

```bash
python scripts/act_baseline.py train \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --split results/act_baseline/split.json --output results/act_baseline/full-seed0 \
  --mode full --device cuda --batch-size 8 --steps 100000 --seed 0 \
  --resume results/act_baseline/full-seed0/training/checkpoints/050000/pretrained_model
```

The example resumes checkpoint 50,000; replace that step with the actual completed step. Never
resume from a partially written checkpoint. Use `--wandb` at run creation for optional logging.
The local path is mandatory; optional `--repo-id` on validation/splitting must match M3B metadata.
There is no Hub publication or remote-data fallback.

## Local fixture and regression commands

```bash
python -m pytest tests/unit/test_m4a_baseline.py tests/integration/test_m4a_upstream.py -q
python -m pytest -m "not gpu and not rendering" -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The integration fixture writes twelve generated three-frame episodes (36 frames), validates every
frame, selects one task, performs a real stock ACT forward/backward/optimizer step, writes and
reloads its model and processors, then restores optimizer/RNG state and advances from step 1 to 2.
It is a CPU compatibility test, **not** a real M3B export, expert trajectory or closed-loop success.

## Result artifacts and stage interpretation

```text
results/act_baseline/<run-id>/
  config.json                 # run identity, Git commit/dirty flag, arguments, source location
  split.json                  # source hashes, seed, episode IDs, frame counts
  validation.json             # full decoded-row audit and statistics
  normalization.json          # train-only state/action stats + fixed ImageNet RGB stats
  checkpoint_metadata.json    # final step, model/processor hashes, reload, workload/timing
  status.json                 # training completion flags
  training/checkpoints/...    # upstream models, processors, optimizer and RNG state
  evaluation-seed42000/
    config.json               # explicit reset schedule, runtime and evaluator provenance
    split.json
    episodes.csv              # one row per controller/seed, outcomes, reasons, pairing hashes
    metrics.json               # both rates, absolute/signed gap, lengths, counts, validity
```

`implementation complete` requires reviewed code and passing automated checks. `smoke tested`
must name its scope (generated CPU fixture versus real-data/Linux smoke). `full dataset validated`
requires a real finalized M3B source that passes the source target gate and exhaustive consumer
validation. `full ACT trained` requires the declared full run and a reloaded saved checkpoint.
`closed-loop evaluated` requires all scheduled real simulator episodes, matching reset hashes and
no infrastructure errors. These last three are **pending execution** for the new formal dataset.
No fabricated `metrics.json` or success rates are committed. All result trees are Git-ignored.

## Workload and GPU planning

The new formal dataset does not exist yet, so its actual frame count, measured GPU memory and
wall-clock training time are **unavailable**. Historical frame counts must not be reused. After
regeneration, `split.json` records actual `train_frames`, `held_out_frames`, `excluded_test_frames`,
selected-task `total_frames` and `dataset_total_frames`. The completed checkpoint receipt reports:

- Sample presentations = `steps × batch_size` (default 100,000 × 8 = **800,000** anchors).
- Equivalent train passes = `800,000 / actual_train_frames`.
- At most 40 million action target positions before episode-end padding; upstream masks pads.
- Actual elapsed training invocation time and last-step/reload peak CUDA allocated bytes (null
  for CPU). Upstream resets its CUDA peak counter each optimization step, so this is not a run-wide peak.

For planning, budget **12–16 GB VRAM** for batch 8, one 256×256 view, FP32 ACT; this is an estimate,
not a measured requirement. A 24 GB instance leaves additional space for simulation and decoding.
Choose after the real-data smoke measures allocation and throughput. Dataset size primarily affects
storage, decoding and reuse frequency; do not infer VRAM or hours simply from frame count. A fresh
non-resumed smoke measures setup-inclusive timing only; use steady-state trainer logs to estimate
`100000 / measured_steps_per_second`, and retain rendering/evaluation time separately.
