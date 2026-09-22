# M4B delivery and execution record

Status: implementation under validation; native pairing, real-data smoke, full training and language
evaluation are pending. Synthetic contract tests do not establish any benchmark result.
M4A remains frozen at `717a07d`, including its 10/20 ACT and 20/20 expert result.

## A. Design

See [M4B_PROTOCOL.md](M4B_PROTOCOL.md). Two semantic goals select the existing red cube's destination.
The same RGB/qpos/state under different text is paired with the same flow noise. There are 20 fresh
scenes, 100 physical policy rollouts, 160 scored condition/goal rows, and 40 expert reference rollouts.
Goal IDs are independent of strings. Existing green/blue cubes remain distractors; environment and
success logic are unchanged. Expert performance is reported without introducing a new success-rate
threshold; infrastructure, exact input pairing and nonzero execution are hard validity gates.

## B. Implementation

- `src/langmani/policies/m4b_protocol.py`: semantic/text protocol, schedule, scored-row reuse, metrics.
- `src/langmani/policies/m4b_data.py`: immutable two-goal split and exhaustive/content-bound validation.
- `src/langmani/policies/m4b_training.py`: pinned upstream model/processors/trainer, resume and memory.
- `src/langmani/policies/m4b_evaluation.py`: exact reset gate, separate scorer, shared noise, video.
- `scripts/smolvla_baseline.py`: validate/prepare/gate/train/evaluate commands and identity checks.
- `scripts/prepare_smolvla_assets.py`: pinned public snapshot preparation and hashes.
- `tests/unit/test_m4b_baseline.py`: synthetic leakage, pairing, scoring, tamper and artifact tests.
- `environment/m4b-runtime.txt`, `pyproject.toml`: isolated optional SmolVLA dependencies.
- `AGENTS.md`, README, protocol/delivery/decision documents: scope, commands and evidence status.

M4A modules/tests/data/checkpoints are unchanged. Full decoded validation is reused only after every
source file has been hashed again and the exact inventory matches; any change requires revalidation.
Training statistics cover only 96 selected training episodes. Fresh official SmolVLA processors use
this robot's RGB/qpos/action schema and train statistics; resume loads the saved processors. No model
source is copied or changed. State/action projection padding stays upstream. Text enters its tokenizer.

## C. Data

The frozen full M3B archive contains 360 episodes/64,548 frames across six semantic tasks. M4B selects
120 red-cube demonstrations: 96 train, 12 validation, 12 excluded test. Exact selected frame counts,
paired-video differences, source content hash and split hash remain pending exhaustive M4B validation.
The original M3A archive remains the raw authority and is reused read-only.

## D–E. Exact command contract

Run inside the M4B checkout, with its `src` on `PYTHONPATH`; a new isolated native overlay installs
`environment/m4b-runtime.txt`. `LANGMANI_PLANNER_PYTHON` selects the unchanged NumPy-1 planner launcher.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=0`.
The following variables identify the read-only source and generated M4B paths:

```bash
DATA=/root/autodl-tmp/langmani-m4a/outputs/datasets/m3b/langmani-pick-place-lerobot-v1
PRIOR=/root/autodl-tmp/langmani-m4a/results/act_baseline/full-seed0/evaluation-seed42000-native-gripper-v2/config.json
ASSETS=/root/autodl-tmp/langmani-m4b-assets/assets.json
python scripts/prepare_smolvla_assets.py --output /root/autodl-tmp/langmani-m4b-assets
python scripts/smolvla_baseline.py validate --dataset-root "$DATA" --output results/smolvla_baseline/validation.json
python scripts/smolvla_baseline.py prepare --dataset-root "$DATA" --validation results/smolvla_baseline/validation.json --m4a-evaluation-config "$PRIOR" --output results/smolvla_baseline/protocol --seed 43000 --scenes 20
COMMON=(--dataset-root "$DATA" --validation results/smolvla_baseline/validation.json --split results/smolvla_baseline/protocol/split.json --schedule results/smolvla_baseline/protocol/schedule.json)
python scripts/smolvla_baseline.py gate "${COMMON[@]}" --output results/smolvla_baseline/native-gate --sim-backend physx_cpu
TRAIN=("${COMMON[@]}" --assets "$ASSETS" --gate results/smolvla_baseline/native-gate --device cuda)
python scripts/smolvla_baseline.py train "${TRAIN[@]}" --output results/smolvla_baseline/smoke --mode smoke --steps 3 --batch-size 8 --learning-rate 0.0001 --seed 0
python scripts/smolvla_baseline.py train "${TRAIN[@]}" --output results/smolvla_baseline/smoke --mode smoke --steps 4 --batch-size 8 --learning-rate 0.0001 --seed 0 --resume results/smolvla_baseline/smoke/training/checkpoints/000003
python scripts/smolvla_baseline.py train "${TRAIN[@]}" --output results/smolvla_baseline/full-seed0 --mode full --steps 20000 --batch-size 8 --learning-rate 0.0001 --seed 0
python scripts/smolvla_baseline.py evaluate "${TRAIN[@]}" --output results/smolvla_baseline/full-seed0/evaluation-seed43000 --checkpoint results/smolvla_baseline/full-seed0/training/checkpoints/020000 --sim-backend physx_cpu
```

Check actual upstream checkpoint directory padding in the smoke receipt before invoking resume.
Full-run resume uses the same command with increased total `--steps` and `--resume` within that run.
Optional `--wandb` uses existing upstream support and defaults off. No Hub publication.
Fixed model revisions: SmolVLA `d9f33c94a60fb382c90dea2164c96845bd955e28`; backbone/tokenizer
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Snapshot file hashes are recorded in `assets.json`.

## F–I. Results, failures, measurements and limits

Pending execution: correct/swapped/blank/paraphrase results, per-goal results, canonical paired goal
switching, paraphrase pair following, achieved-goal/action differences, timing, GPU samples, peak
allocated memory and final checkpoint hash. Optimizer-step peaks are accumulated across every
upstream update, since upstream resets its PyTorch peak counter each step. NVIDIA samples remain
sampled maxima, distinct from the allocator peak and total device memory.

Failure taxonomy distinguishes wrong goal, timeout, invalid action, arm bounds, off-table and
infrastructure. Videos and exact final grasp/containment/static flags support inspection; they do not
automatically identify a physical cause. Invalid comparisons suppress causal sensitivity metrics.
The correct-minus-swapped score alone is insufficient: swapped prompt-goal achievement and paired
goal switching must also be inspected. Constant-destination behavior is explicitly tested.
No direct architecture superiority over single-task ACT, open-ended language understanding or
pretraining-unseen paraphrase claim is justified. Final results will replace the pending entries.
