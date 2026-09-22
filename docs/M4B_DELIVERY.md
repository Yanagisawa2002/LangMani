# M4B delivery and execution record

Status: **implementation complete; smoke tested; full dataset validated; full SmolVLA trained;
closed-loop evaluated**. Native pairing, final checkpoint reload, local archive hashes and independent
evaluation/media audits passed. Execution uses clean commit
`1b2d76cd994197b064bbd364e44a215cb7b68bbf`. Synthetic contract tests establish no benchmark result.
M4A remains frozen at `717a07d`, including its 10/20 ACT and 20/20 expert result.

The measured answer is limited: canonical instructions selected the corresponding different goal
in **6/20 identical-scene pairs**, but correct-instruction success was only **15/40** and held-out
paraphrase success was **0/40**. This demonstrates some template-dependent goal selection, with no
demonstrated robustness to the tested linguistic reformulations.

## A. Design

See [M4B_PROTOCOL.md](M4B_PROTOCOL.md). Two semantic goals select the existing red cube's destination.
The same RGB/qpos/state under different text is paired with the same flow noise. There are 20 fresh
scenes, 100 physical policy rollouts, 160 scored condition/goal rows, and 40 expert reference rollouts.
Goal IDs are independent of strings. Existing green/blue cubes remain distractors; environment and
success logic are unchanged. Expert performance is reported without introducing a new success-rate
threshold; infrastructure, exact input pairing and nonzero execution are hard validity gates.

## B. Changed files

The change from frozen M4A comprises these 14 files:

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

Exhaustive native validation decoded the frozen full M3B archive: 360 episodes/64,548 frames across
six semantic tasks. M4B selects 120 red-cube demonstrations/21,459 frames, preserving source splits:

| Split | Episodes | Frames | Left episodes/frames | Right episodes/frames |
| --- | ---: | ---: | ---: | ---: |
| Train | 96 | 17,149 | 48 / 8,588 | 48 / 8,561 |
| Validation, unused for checkpoint selection | 12 | 2,154 | 6 / 1,073 | 6 / 1,081 |
| Excluded source test | 12 | 2,156 | 6 / 1,077 | 6 / 1,079 |

All 60 left/right source pairs have exactly equal initial qpos and decoded first RGB frames
(uint8 absolute difference min/mean/max: 0/0/0). The original M3A archive remains the raw authority
and is reused read-only. Source byte inventory and exact split/schedule identities are:

- Dataset content: `cd97abe24a5dde623046fdf67da4810c5d7c8aaa2d1e59ec6e66b2531b212b2a`.
- M4B split: `7bbd3e9a68d72cc5681c75884f3b0e1072e0d62ea8afe2626f1a4dee307bb68f`.
- M4B evaluation schedule: `4cadb2fcab49f120485fd0f449a505500ac6ba7b06d4eca1ffb1d6a5b0a0d1d8`.

The native M0 → M1 → M2 gate passed in the isolated M4B overlay. CPU-safe native tests passed
433 in the main interpreter and 2 in the separate planner interpreter; GPU/rendering acceptance was
measured separately by the native target gate. The local suite passed 430 with 5 native-only skips
and 5 GPU/rendering deselections; lint, format and wheel/sdist build passed.

The pre-training paired expert reference completed 39/40 successful episodes. Seed `659916340`,
`red_cube:right_bin`, failed planning after 92 steps and stays in the reference. All 40 episodes had
nonzero execution, and all 20 pairs had exact native state/RGB/qpos identity across goals and across
the planner/main runtimes. This establishes pairing validity, not a claim of perfect expert success.
The old M2 balanced expert gate is a prerequisite, not a language-ablation comparator.

## D. Exact training commands

Run inside the M4B checkout, with its `src` on `PYTHONPATH`; a new isolated native overlay installs
`environment/m4b-runtime.txt`. `LANGMANI_PLANNER_PYTHON` selects the unchanged NumPy-1 planner launcher.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=0`.
The following variables identify the read-only source and generated M4B paths:

```bash
cd /root/autodl-tmp/langmani-m4b
source /root/autodl-tmp/langmani-m4b-ops/runtime.env
# runtime.env selects the isolated M4B interpreter and src, the unchanged planner,
# CUDA device 0, two OMP/MKL threads, and deterministic cuBLAS workspace.
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
```

The real-data smoke saved and reloaded step `000003`; resume restored optimizer/RNG/data order and
completed step `000004`. A repeated real-frame inference probe with common noise was bit-identical.
These checks validate the interface and persistence, not manipulation success.
Full-run resume uses the same command with `--resume` within that run and a total `--steps` greater
than the saved step; an interrupted run can retain its original 20,000-step total budget.
Resume requires the original execution commit, data/assets/split identity and saved normalization.
Restore into the recorded absolute paths, or prepare a separate new run; silently rewriting a saved
run's paths/hashes or replacing its source commit is rejected. A documentation-only delivery commit
does not change the execution identity of the recorded checkpoint.
Optional `--wandb` uses existing upstream support and defaults off. No Hub publication.
Fixed model revisions: SmolVLA `d9f33c94a60fb382c90dea2164c96845bd955e28`; backbone/tokenizer
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Snapshot file hashes are recorded in `assets.json`.

## E. Exact evaluation command

With the same checkout, runtime and `TRAIN` argument array above, the executed command was:

```bash
python scripts/smolvla_baseline.py evaluate "${TRAIN[@]}" \
  --output results/smolvla_baseline/full-seed0/evaluation-seed43000 \
  --checkpoint results/smolvla_baseline/full-seed0/training/checkpoints/020000/pretrained_model \
  --sim-backend physx_cpu
```

The receipts retain expanded argument lists, runtime variables and the complete upstream config.
These are the completed run's commands. Outputs are immutable: a new evaluation uses a new output
directory, while compatible training continuation uses the saved run's resume contract.

## F. Language-ablation results

All 100 physical policy rollouts completed without infrastructure error. The four conditions are
scored against each original requested goal; canonical and blank rollouts are explicitly reused.

| Condition | Requested-goal success | Left goal | Right goal | Timeout scoring rows | Mean successful length |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correct canonical text | 15/40 (37.5%) | 8/20 (40%) | 7/20 (35%) | 25 | 138.67 steps |
| Swapped canonical text | 0/40 (0%) | 0/20 | 0/20 | 25 | N/A, no requested-goal successes |
| Blank text | 0/40 (0%) | 0/20 | 0/20 | 40 | N/A |
| Held-out paraphrase | 0/40 (0%) | 0/20 | 0/20 | 40 | N/A |

Correct-text successful means are 134.50 steps for left and 143.43 for right. Left/right denote the
environment's fixed semantic bin IDs; the camera views the scene obliquely.

The swapped condition reached the **goal specified by its supplied text in 15/40 rows (37.5%)**:
7/20 when the original requested goal was left and 8/20 when it was right. These are the same 15
physical canonical successes scored against the opposite request. They are intentional wrong-request
outcomes, not evidence that the policy ignored the swapped instruction. Correct-minus-swapped and
correct-minus-blank requested-goal success differences are both 37.5 percentage points.

| Direct language-use measurement | Observed result |
| --- | ---: |
| Canonical-left reaches left AND canonical-right reaches right in one paired scene | 6/20 (30%) |
| Both canonical rollouts reach different goals | 6/20 (30%) |
| Canonical action trajectories differ after changing text | 20/20 (100%); insufficient alone to establish goal selection |
| Both paraphrase prompts reach their corresponding goals | 0/20 (0%) |
| Canonical/paraphrase non-null achieved-goal agreement | 0/40 (0%); two failures do not count as agreement |

The six successful goal-switch seeds are `375419956`, `2053748918`, `737375675`, `2062869193`,
`468995337` and `1417035984`. Six scenes succeeded for both canonical destinations, three for only
one, and eleven for neither. All 20 groups have equal initial physical-state/RGB/qpos and inference
noise hashes. The schedule, checkpoint and 200-step cap stayed fixed, with no selection or retuning
after final outcomes. Independent CSV/JSON recomputation confirmed the denominators and switch count.

The reference expert succeeded in 39/40 separate paired episodes (left 20/20; right 19/20), retaining
the planning failure documented above. This does not change the validity denominator of 20 scenes.

## G. Failure taxonomy

| Physical termination | Count across 100 physical policy rollouts |
| --- | ---: |
| Goal reached | 15 |
| 200-step timeout | 85: 25 canonical, 20 blank, 40 paraphrase |
| Invalid action | 0 |
| Arm action out of bounds | 0 |
| Target off table | 0 |
| Other native termination | 0 |
| Infrastructure error | 0 |

At scoring-row level, swapped text has 15 `wrong_goal` outcomes plus 25 timeouts. Its termination
distribution is still 15 goal-reached/25 timeout because the alternate instructed destination was
actually reached. Correct text has the same termination counts, and blank/paraphrase each have 40
timeout rows. The blank count represents 20 physical runs scored twice; the 160 rows are not 160
independent trials.

All 85 timeout final states marked the red cube static and failed both destination success tests.
Representative initial/middle/final video frames show a canonical timeout with the arm at a bin
while the red cube remains on the table; blank/paraphrase examples also leave the cube on the table
in the inspected middle/final frames. These observations do not establish a complete physical or
model-internal cause for every timeout. No failure was repaired, reseeded or excluded.

The inherited native gripper saturation was applied to 1,058/19,080 executed actions; maximum
overshoot was 0.1300433 before clipping to [-1, 1]. Independent action-file inspection confirmed
every saturation and unchanged arm values. No arm clipping was introduced.

All 100 MP4s decoded locally: **19,180 frames = 19,080 actions + 100 initial frames**, each 256×256.
The saved representative index includes successes, timeouts and swapped wrong-request outcomes.
For a matched success pair, inspect `seed-375419956-canonical_left.mp4` (141 steps) and
`seed-375419956-canonical_right.mp4` (131 steps). Seed `1362180809` supplies the inspected canonical,
blank and paraphrase timeout examples. All videos remain in the evidence archive.

## H. Training measurements

The planned full run completed with exit code 0 on 2026-09-22 at 05:08:11 UTC. No training restart or
checkpoint selection occurred. The final model and its saved processors reloaded successfully.

| Measurement | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB reported device capacity |
| Training budget | 20,000 optimizer steps, batch 8, seed 0 |
| Learning rate | AdamW peak 1e-4; final logged 2.5e-6 |
| Effective schedule | 666 warmup steps, 20,000 decay steps |
| Train view | 17,149 frames in 96 episodes |
| Sample presentations | 160,000; 9.32999 equivalent train passes |
| Training invocation | 6,768.605 seconds (112.81 minutes), including upstream initialization/checkpoint saves |
| Whole training stage | 03:14:44–05:08:11 UTC (113.45 minutes), including wrapper work/reload/hash checks |
| Final logged training loss | 0.016; training objective only, not task or language success |
| Optimizer-step peak allocated VRAM | 2,845,703,680 bytes (2.650 GiB) |
| Sampled device memory maximum | 3,376 MiB; 1,361 samples at 5-second intervals |
| Sampled GPU utilization | Mean 22.81%, maximum 60%; includes initialization and checkpoint work |
| Parameters | 450,046,176 total; 99,880,992 trainable in the verified model configuration |
| Final stored tensor elements by dtype | 148,811,856 F32; 301,234,320 BF16 |

Upstream logged automatic scaling from the saved scheduler defaults of 1,000 warmup/30,000 decay
steps to the effective values above; the persisted config retains those defaults. Optimizer settings
are AdamW betas (0.9, 0.95), epsilon 1e-8, weight decay 1e-10 and gradient clip norm 10.0.
The run used LeRobot 0.6.0, Torch 2.11.0+cu128, Transformers 5.5.4, Tokenizers 0.22.2,
Accelerate 1.15.0 and NumPy 2.2.6, in the isolated native overlay. W&B was disabled.

Optimizer-step peaks are accumulated across every upstream update, since upstream resets its
PyTorch peak counter each step. This is an allocator measurement during updates. NVIDIA samples are
sampled whole-device values, distinct from an exact process peak or device capacity.

Final checkpoint: `training/checkpoints/020000/pretrained_model`. Model SHA256:
`4aabc676a85bc1b521ed4098678723c21d238d79bcb4454cbe489a8121a7243e`.
The 12-file model/processors/optimizer/RNG/scheduler checkpoint tree digest is
`5426493f0949d94b0701af8d2400e25bb16ce062b289387b4b1f19c99a1695de`.
Language evaluation ran from 05:08:11 to 05:25:47 UTC: 1,056 seconds for the whole command, with
1,022.165 seconds (17.04 minutes) inside the rollout/evidence loop. It executed 19,080 physical
environment steps. Its 212 GPU samples had a maximum memory use of **1,929 MiB**, mean GPU
utilization 10.37% and maximum 24%. An exact evaluation allocator peak was not collected.

## I. Justified claims and limitations

Changing only canonical language caused the corresponding successful destination change in 6/20
paired scenes, under identical starting observations and common inference noise. This is direct,
bounded evidence of language-dependent goal selection. The overall 37.5% canonical success and
0% paraphrase success show that reliable manipulation and linguistic robustness remain unsolved.
The result does not demonstrate open-ended understanding, object-selection generalization, or
foundation-pretraining-unseen language. The paraphrases were excluded only from this fine-tuning
dataset. Two goals, four fixed paraphrase strings, 20 evaluation scenes, one training seed and one
fixed final checkpoint limit broader conclusions. No numerical acceptance threshold was invented.

M4A's single-task non-language ACT result is frozen and not a fair direct architecture comparison:
the task conditioning, data subset, training budget and evaluation schedule differ. No categorical
goal-ID ACT control or additional training experiment was added.

### Evidence and recovery

Generated evidence is stored locally under
`D:/LangMani-worktrees/m4b-smolvla-language/outputs/m4b-review/`, outside Git. Final archives were
verified by outer SHA256 and by every member's name, size and SHA256; extracted evidence was checked
again, followed by independent scoring and full video/action audits.

| Verified archive | Compressed bytes | Files |
| --- | ---: | ---: |
| `m4b-preflight-evidence.tar.gz` | 1,901,950 | 234 |
| `m4b-pretrained-assets.tar.gz` | 721,606,113 | 18 |
| `m4b-final-checkpoint.tar.gz` | 1,074,583,993 | 12 |
| `m4b-final-evidence.tar.gz` | 16,143,161 | 566 |

Final checkpoint archive SHA256:
`6bb6488b4b194c53a39e1f980373db2bd0a3b6b32c237e5597e7bd5f56e9ccaf`.
Final evidence archive SHA256:
`732c9db325dc126690899b42e23d7aa551b2bb9584ea00809e682ed0742a1dc5`.
Their `*-manifest.json` and `*-verification.json` receipts retain complete inventories. The
checkpoint archive contains model/processors plus optimizer, RNG, scheduler and step state; restore
its two top-level directories under `training/checkpoints/020000`. Pinned backbone/tokenizer assets
must also be restored to their recorded paths.

The read-only source recovery archives remain in the M4A review directory, verified again during
M4B and referenced by `source-recovery-reference.json`: raw M3A SHA256
`4c315f829797556250a395ae1d865f562fb1a8ecf1a322b8d54912ed10a45333`;
derived M3B SHA256 `358080f1e937b2a3cec7650548e8abe84a85ba82653150d8192e155a77aa8dad`.

Extracted results are under
`final-evidence/repo/results/smolvla_baseline/full-seed0/evaluation-seed43000/`: `metrics.json`,
`episodes.csv`, `rollouts.csv`, `expert_episodes.json`, `config.json`, `representative_videos.json`
and all per-rollout JSON/NPZ/MP4 files. Local `independent-evaluation-audit.json` and
`independent-media-audit.json` record the independent checks. `media-audit/` contains inspected
frame sheets. The local closeout supplement preserves verifier scripts, receipts, frame sheets and
final documentation; its manifest records the documentation delivery commit separately from the
unchanged experiment commit. Large data/model/evidence files are deliberately not committed to Git.
