# M4A delivery — 2026-09-22 Asia/Singapore

| Stage | Status and evidence boundary |
| --- | --- |
| implementation complete | Implemented and tested on the isolated M3B-based branch |
| smoke tested | **Passed real-data Linux CUDA smoke**: three optimizer steps, save/reload and three paired closed-loop scenes; ACT 0/3 timeout with 600 executed steps, expert 3/3. Generated-array fixtures are separate compatibility tests |
| full dataset validated | **Passed**: real 60-group/360-episode M3A collection and independent replay; complete M3B export/validation; exhaustive M4A validation and inherited split |
| full ACT trained | **Passed**: 100,000 steps, batch 8, seed 0; final checkpoint saved, hashed and reloaded |
| closed-loop evaluated | **Passed**: final checkpoint on 20 fresh paired scenes; ACT 10/20, expert 20/20, 3,311 ACT steps, exact initial-state pairing and no infrastructure errors |

SmolVLA was not started. No generated benchmark, dataset or checkpoint is committed.

## A. Architecture

Existing M1 environment → privileged M2 expert → authoritative M3A native archive → existing M3B
renderer/export → exhaustive M4A consumer validator → inherited red-cube/left-bin train/held-out
episode view → official LeRobot 0.6 ACT trainer → upstream checkpoint → paired ACT/expert evaluator.
ACT sees one 256×256 RGB image and 9D qpos, predicts 8D actions, and ignores all language metadata.
The [technical guide](M4A_ACT_BASELINE.md) records the inspected source map and policy/expert boundary.

## B. Every changed file

| File | Purpose |
| --- | --- |
| `.gitignore` | Ignore all generated `results/` artifacts |
| `AGENTS.md` | Define the authorized M4A scope and five separate completion states |
| `README.md` | Add the M4A section, exact commands, measured workload and explicit benchmark status |
| `environment/environment.yml` | Activate LeRobot's official training dependencies on Linux |
| `environment/planner-runtime.txt` | Pin the isolated NumPy-1/SciPy/OpenCV overlay required by native mplib |
| `environment/verify_m1.py` | Run CPU and GPU PhysX target checks in separate processes |
| `environment/verify_m2.py` | Verify and select the isolated planner interpreter for expert target gates |
| `environment/verify_m3a.py` | Use the selected planner runtime for real collection and independent action replay |
| `environment/verify_planner_runtime.py` | Record imported runtime pins and exercise native Panda planner construction |
| `pyproject.toml` | Keep package dependency extras aligned with the environment declaration |
| `src/langmani/datasets/observation_reconstruction.py` | Expose the existing RGB extraction contract through a public wrapper |
| `src/langmani/datasets/types.py` | Include effective planner dependency versions in authoritative collection provenance |
| `src/langmani/experts/planner.py` | Reject an incompatible NumPy ABI before constructing mplib |
| `src/langmani/experts/pick_place.py` | Remove redundant waypoint holds and accommodate measured native tracking residuals within the unchanged task/horizon contract |
| `src/langmani/experts/runtime.py` | Select the virtualenv launcher without dereferencing symlinks and probe imported versions |
| `src/langmani/policies/__init__.py` | Declare the small policy package |
| `src/langmani/policies/m4a_data.py` | Exhaustive read-only row/video validation, no-download guard, hashes, statistics, semantic/scene checks and single-task split manifest |
| `src/langmani/policies/m4a_training.py` | Stock ACT configuration, train-only normalization, official trainer integration, resume and checkpoint/processor reload |
| `src/langmani/policies/m4a_evaluation.py` | Closed-loop queue, strict arm bounds, audited native gripper saturation, paired reset audit, expert reference and JSON/CSV metrics |
| `scripts/act_baseline.py` | `validate`, `split`, `train`, `evaluate` CLI with provenance and safe output/checkpoint paths |
| `tests/unit/test_m4a_baseline.py` | Malformed/valid datasets, no leakage, deterministic splits, source identity, metrics/pairing, path safety, arm rejection, gripper saturation and zero-step rejection |
| `tests/integration/test_m4a_upstream.py` | Real generated-video LeRobot readback and upstream ACT optimizer/save/reload/resume compatibility |
| `tests/unit/test_m1_commands.py` | Cover isolated CPU/GPU PhysX verification commands |
| `tests/unit/test_m2_commands.py` | Cover planner-runtime checks and interpreter selection |
| `tests/unit/test_m3a_commands.py` | Cover planner interpreter propagation through collection and replay gates |
| `tests/unit/test_m3a_types_schedule.py` | Cover the expanded authoritative runtime fingerprint |
| `tests/unit/test_planner_adapter.py` | Cover the native NumPy ABI guard |
| `tests/unit/test_pick_place_expert.py` | Cover tracking-residual rejection, preserved final task authority and absence of redundant holds |
| `tests/unit/test_planner_runtime.py` | Cover effective version probes and virtualenv launcher preservation |
| `docs/DECISIONS.md` | D-027 through D-032 rationale, dependency boundaries, failures and observed evidence |
| `docs/M4A_ACT_BASELINE.md` | Architecture, formal regeneration order, resume, result schema and workload guidance |
| `docs/M4A_DELIVERY.md` | This delivery inventory and execution status |

## C. Smoke commands

The command actually tested locally, after installing this checkout in a CPU review environment:

```bash
python -m pytest tests/unit/test_m4a_baseline.py tests/integration/test_m4a_upstream.py -q
```

The real-data Linux smoke **passed**, after the existing full M3A/M3B target gates:

```bash
DATA=outputs/datasets/m3b/langmani-pick-place-lerobot-v1
python scripts/act_baseline.py validate --dataset-root "$DATA" \
  --report results/act_baseline/validation.json
python scripts/act_baseline.py split --dataset-root "$DATA" \
  --output results/act_baseline/split.json
python scripts/act_baseline.py train --dataset-root "$DATA" \
  --split results/act_baseline/split.json --output results/act_baseline/native-smoke-seed0 \
  --mode smoke --device cuda --batch-size 8 --steps 3 --seed 0
python scripts/act_baseline.py evaluate --dataset-root "$DATA" \
  --split results/act_baseline/split.json \
  --checkpoint results/act_baseline/native-smoke-seed0/training/checkpoints/000003/pretrained_model \
  --output results/act_baseline/native-smoke-seed0/evaluation-seed42000 \
  --device cuda --episodes 3 --seed 42000 --sim-backend physx_cpu
```

This short trained policy is a pipeline smoke, not evidence of task proficiency.

## D. Full training command

```bash
python scripts/act_baseline.py train \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --split results/act_baseline/split.json --output results/act_baseline/full-seed0 \
  --mode full --device cuda --batch-size 8 --steps 100000 --seed 0
```

Optional W&B: add `--wandb`. Resume the same run using `--resume <completed-step>/pretrained_model`
with `--steps` equal to the intended total global step. No checkpoint selection or language input.

## E. Full evaluation command

```bash
python scripts/act_baseline.py evaluate \
  --dataset-root outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --split results/act_baseline/split.json \
  --checkpoint results/act_baseline/full-seed0/training/checkpoints/100000/pretrained_model \
  --output results/act_baseline/full-seed0/evaluation-seed42000-native-gripper-v2 \
  --device cuda --episodes 20 --seed 42000 --sim-backend physx_cpu
```

Produces 20 expert + 20 ACT episode rows on identical reset seeds, plus `metrics.json` with both
success rates, failures, episode length mean/std, termination counts, reset pairing validity and
absolute/signed success-rate gaps. Arm violations are rejected; normalized gripper saturation
matches the native controller and records its count and maximum overshoot. Nonfinite actions are
rejected. The original `evaluation-seed42000` directory remains unchanged as failure evidence.
Commands describe recorded runs; choose new output paths for a future independent replication.

Accepted results, completed **2026-09-21 15:56:06 UTC** (23:56:06 Asia/Singapore):

| Metric | ACT | Privileged expert |
| --- | --- | --- |
| Episodes | 20 | 20 |
| Successes / failures | 10 / 10 | 20 / 0 |
| Success rate | 50% | 100% |
| Mean episode length (steps) | 165.55 | 178.00 |
| Episode length population standard deviation | 34.544862 | 5.830952 |
| Termination reasons | 10 success, 10 timeout | 20 success |
| Target off table | 0 | 0 |

Absolute and expert-minus-ACT success-rate gaps are **0.5 (50 percentage points)**. ACT executed
3,311 steps; all ten failures reached the unchanged 200-step limit. All 20 exact initial-state
hash pairs matched, all source scene seeds were excluded, and no infrastructure errors occurred.
The schedule RNG seed is 42000; the exact generated scene seeds are recorded in `config.json`.
Its schedule SHA-256 is `d12a28d002aca9cca203a78dde12b6fd761272ba9d4f9f0528b576c63d074ae9`.
Normalized gripper saturation occurred 464 times, with maximum overshoot 0.0211760998;
this is the documented native-controller behavior, with no arm clipping or expert rescue.

## F. Test evidence

- CPU-safe regression after the native-runtime/expert compatibility changes: **406 passed, 5 skipped,
  5 deselected**. The skips require native Linux; deselections are GPU/rendering tests.
- After the D-031 evaluator repair at `0dafc31`: Windows **412 passed, 5 skipped, 5 deselected**;
  Linux main runtime **415 passed, 5 deselected**, plus **2 native planner tests passed**
  (417 total). Ruff format/lint and wheel/sdist build passed.
- M3B standalone verifier: **114 passed**. Real generated video/Parquet/DataLoader fixture included.
- Installation, M1, M2 and M3A non-target diagnostics: passed, with physical work explicitly skipped.
- Ruff lint/format and wheel/sdist build: passed; logs are generated under
  `outputs/m4a-review/`.
- Runtime used: Windows Python 3.12.10, torch 2.11.0+cpu, LeRobot 0.6.0, Accelerate 1.14.0.
  A visible RTX 4090 does not make this CPU-only PyTorch environment GPU-validated.
- New native Linux execution at `915d823`: **409 main-runtime tests + 2 planner-runtime tests
  passed**. Five GPU/rendering pytest cases were deselected; hardware acceptance was instead
  exercised by the separate ordered M0/M1/M2 target commands, all passing. M2 scored **177/180**,
  with three classified planning failures and no crashes. Real M3A smoke collected and independently
  replayed all six episodes. The M2 score is a prerequisite, not the ACT comparison.
- Native runtime: Ubuntu 22.04, Python 3.12.13, torch 2.11.0+cu128, NVIDIA RTX 4090
  (24,564 MiB), driver 595.71.05, SAPIEN 3.0.3, ManiSkill 3.0.1 and LeRobot 0.6.0.
  Main NumPy is 2.2.6; planner NumPy is 1.26.4. Exact main/planner freezes and diagnostic
  reports are retained in `outputs/m4a-review/native-accepted-preflight.tar.gz`.

## G. Blockers, limitations and recovery evidence

Formal data and training ran at `915d8233c2b8897b63c3c46121794021caa31d25`. M3A accepted 60 scene
groups (360 episodes), rejected five candidate groups, and independently replayed every accepted
episode. M3B produced 64,548 frames across six tasks. The selected task has 60 episodes/10,738 frames,
split into 48/8,588 train, 6/1,073 held-out and 6/1,077 excluded test episodes/frames.

Full training completed at **2026-09-21 15:05:30 UTC**, with final model SHA-256
`542d66bfbe6f0d1431bc639281c0f8ac905bafd6e616b61bd488867aa45410cf`.
The first final evaluation then rejected all 20 initial ACT gripper predictions (1.0011–1.0105)
before any environment step. The native gripper controller would have saturated them to 1; its
physical targets were verified identical before/after saturation. This was an interface failure,
not an executed ACT benchmark result. The original chain exited 1 and remains preserved.

D-031's narrow repair is committed as `0dafc315c90d68751027621dd396f71ecd5f0187`; it changes only
evaluation behavior and associated tests/docs. `eval-resume-v2.sh` verified all original checkpoint
file hashes, reran native planner tests, and evaluated the same final checkpoint and exact 20-seed
schedule in a new directory. It did not repeat data generation or training. The continuation and
evaluation command both exited zero. Independent local verification recomputed the metrics from
all 40 CSV rows and checked every pair, source-seed exclusion, original schedule and checkpoint.

There are **no unresolved M4A execution blockers**. The result is limited to one task, one training
seed, 48 training demonstrations and 20 fresh evaluation scenes. Ten ACT episodes still timed out;
their exact physical failure mechanism was not diagnosed in this milestone. No checkpoint sweep,
post-result model tuning, language-generalization claim or SmolVLA work was performed. Optional
online W&B was not exercised. Pipeline acceptance does not imply expert-level policy performance.

Recovery evidence is stored outside Git in local `outputs/m4a-review/`:

| Artifact | Local verification |
| --- | --- |
| `m3a-full-validated-backup.tar.gz` | Archive SHA-256 and all 20 raw file hashes/sizes passed |
| `m3b-full-validated-backup.tar.gz` | Archive SHA-256 and all 14 derived file hashes/sizes passed |
| `native-act-smoke-evidence.tar.gz` | Hash, three exact reset pairs and source-seed exclusion passed |
| `act-full-strict-bounds-failure.tar.gz` | Remote/local archive SHA-256 matched; original zero-step failure retained |
| `action-bounds-diagnostic.json` | Remote/local SHA-256 matched; all 20 first predictions and native equivalence retained |
| `act-full-100000-backup.tar.gz` | Archive SHA-256 and all 11 checkpoint file hashes/sizes passed; includes model/processors, optimizer/RNG/step, configs/logs/runtime/GPU evidence |
| `act-final-evaluation-backup.tar.gz` | Archive SHA-256 and all 80 evidence file hashes/sizes passed; includes metrics, CSV, config/split, expert workers, GPU samples, logs, exits and evaluation commit |

The final checkpoint archive is 553,733,411 bytes; its SHA-256 is
`29d6cd6d3c9cf9b5d6daf5a6800f7becbcc194dba646b4922e9880478e4ffd90`.
Local verification at 2026-09-21 15:44:57 UTC checked all 619,200,513 uncompressed checkpoint bytes.
`act-full-backup-verification.json` records each file hash. The raw/derived backup receipts,
validation/split reports and exact run configuration are alongside it. The final evaluation archive
is 25,247 bytes, SHA-256 `b801d17da24b86ba7bdeed61b71c6aae55712a2b224718e36a06e9c38d1290f4`.
Its 80 evidence files passed local verification at 2026-09-21 16:05:31 UTC.
Directly inspect `outputs/m4a-review/native-act-final-evaluation/metrics.json`, `episodes.csv`,
`config.json` and `split.json`; `act-final-evaluation-backup-verification.json` contains all 20
paired state hashes and the recomputed acceptance receipt. The metrics SHA-256 is
`4373be2f3d3922405b42e47f6851a162adbe11a5b9e2aa69b851e7e65623e780`; the CSV SHA-256 is
`563fec209fa60a06a96a40674e822174bb1ead4a72990853bf986b04fd57a6f2`.

The recurring continuation is paused at delivery after evidence verification and the final push.
The server remains running; no shutdown or reboot is authorized. Large datasets, model files and
generated evidence remain outside Git; Git contains implementation, tests and documentation.

## H. GPU and workload

Measured full training: **100,000 steps × batch 8 = 800,000 sample presentations**, equivalent to
**93.153237 passes** over the actual 8,588 training frames. ACT has 51,576,712 parameters and predicts
50 action targets per sampled anchor; at most 40 million target positions precede episode-end padding.
The upstream training invocation took **17,200.8279 s** (4 h 46 min 41 s), excluding the preceding
dataset preflight and final evaluation. The final logged loss was 0.021; loss is not a success rate.

The full stage has **3,533 five-second NVIDIA samples**, including preflight, with a maximum
**1,634 MiB** and sampled maximum GPU utilization **22%**. These are sampled maxima, not exact peaks
or minimum hardware requirements. PyTorch reports **1,036,664,320 bytes** peak allocated during the
last step/reload only, because upstream resets the counter each optimization step. The accepted
evaluation command took **972 seconds**, including exhaustive data preflight. Its **191** five-second
samples showed maxima of **1,460 MiB** and **11%** GPU utilization. These actual measurements
replace the earlier 12–16 GB planning estimate for this exact batch-8, single-256×256-view configuration.

For reproducing the entire validated generation/training/simulation chain, use the tested
**24 GB RTX 4090 class** configuration. Training alone used much less memory in this run; an
**8 GB training-only budget** would leave substantial margin over the observed allocation, but
it is a planning estimate and has not been validated on an 8 GB device. No smaller-device or
throughput claim follows from these measurements.
