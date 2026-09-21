# M4A delivery — 2026-09-21

| Stage | Status and evidence boundary |
| --- | --- |
| implementation complete | Implemented and tested on the isolated M3B-based branch |
| smoke tested | **Generated-array CPU fixture only**: official ACT forward/backward, optimizer, save, processor/model reload and step-1 → step-2 resume |
| full dataset validated | **Pending execution**: regenerate real M3A/M3B on the new AutoDL Linux instance |
| full ACT trained | **Pending execution** |
| closed-loop evaluated | **Pending execution**: no real environment rollout in this delivery |

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
| `README.md` | Add the conservative M4A section, exact commands and pending benchmark table |
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
| `src/langmani/policies/m4a_evaluation.py` | Closed-loop action queue, strict action bounds, paired seeds/reset audit, expert reference and JSON/CSV metrics |
| `scripts/act_baseline.py` | `validate`, `split`, `train`, `evaluate` CLI with provenance and safe output/checkpoint paths |
| `tests/unit/test_m4a_baseline.py` | Malformed/valid datasets, no leakage, deterministic splits, source identity, metrics/pairing, path safety and action rejection |
| `tests/integration/test_m4a_upstream.py` | Real generated-video LeRobot readback and upstream ACT optimizer/save/reload/resume compatibility |
| `tests/unit/test_m1_commands.py` | Cover isolated CPU/GPU PhysX verification commands |
| `tests/unit/test_m2_commands.py` | Cover planner-runtime checks and interpreter selection |
| `tests/unit/test_m3a_commands.py` | Cover planner interpreter propagation through collection and replay gates |
| `tests/unit/test_m3a_types_schedule.py` | Cover the expanded authoritative runtime fingerprint |
| `tests/unit/test_planner_adapter.py` | Cover the native NumPy ABI guard |
| `tests/unit/test_pick_place_expert.py` | Cover tracking-residual rejection, preserved final task authority and absence of redundant holds |
| `tests/unit/test_planner_runtime.py` | Cover effective version probes and virtualenv launcher preservation |
| `docs/DECISIONS.md` | D-027/D-028/D-029 rationale, dependency boundaries and observed evidence |
| `docs/M4A_ACT_BASELINE.md` | Architecture, formal regeneration order, resume, result schema and workload guidance |
| `docs/M4A_DELIVERY.md` | This delivery inventory and execution status |

## C. Smoke commands

The command actually tested locally, after installing this checkout in a CPU review environment:

```bash
python -m pytest tests/unit/test_m4a_baseline.py tests/integration/test_m4a_upstream.py -q
```

The real-data Linux smoke is **pending**, and requires the existing full M3A/M3B target gates first:

```bash
DATA=outputs/datasets/m3b/langmani-pick-place-lerobot-v1
python scripts/act_baseline.py validate --dataset-root "$DATA" \
  --report results/act_baseline/validation.json
python scripts/act_baseline.py split --dataset-root "$DATA" \
  --output results/act_baseline/split.json
python scripts/act_baseline.py train --dataset-root "$DATA" \
  --split results/act_baseline/split.json --output results/act_baseline/smoke-seed0 \
  --mode smoke --device cuda --batch-size 8 --steps 3 --seed 0
python scripts/act_baseline.py evaluate --dataset-root "$DATA" \
  --split results/act_baseline/split.json \
  --checkpoint results/act_baseline/smoke-seed0/training/checkpoints/000003/pretrained_model \
  --output results/act_baseline/smoke-seed0/evaluation-seed42000 \
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
  --output results/act_baseline/full-seed0/evaluation-seed42000 \
  --device cuda --episodes 20 --seed 42000 --sim-backend physx_cpu
```

Produces 20 expert + 20 ACT episode rows on identical reset seeds, plus `metrics.json` with both
success rates, failures, episode length mean/std, termination counts, reset pairing validity and
absolute/signed success-rate gaps. The explicit bounds policy is rejection, never silent clipping.

## F. Test evidence

- CPU-safe regression after the native-runtime/expert compatibility changes: **406 passed, 5 skipped,
  5 deselected**. The skips require native Linux; deselections are GPU/rendering tests.
- M3B standalone verifier: **114 passed**. Real generated video/Parquet/DataLoader fixture included.
- Installation, M1, M2 and M3A non-target diagnostics: passed, with physical work explicitly skipped.
- Ruff lint/format and wheel/sdist build: passed; logs are generated under
  `outputs/m4a-review/`.
- Runtime used: Windows Python 3.12.10, torch 2.11.0+cpu, LeRobot 0.6.0, Accelerate 1.14.0.
  A visible RTX 4090 does not make this CPU-only PyTorch environment GPU-validated.

## G. Remaining blockers

The formal data cannot be recovered. The new native Linux AutoDL runtime has passed M0/M1 target
checks, including CUDA, Vulkan and real RGB simulation. Original M2 expert acceptance failed;
D-029 records its bounded diagnosis and minimal repair. The repaired M2 target gate remains
pending. Pass that gate, regenerate and independently replay real M3A trajectories, and
export/validate full M3B before actual-data smoke/full training/evaluation. No full frame count,
trained benchmark or real closed-loop success rate is available. Optional online W&B is wired but
was not exercised. This delivery makes no physical acceptance or model-quality claim.

## H. GPU and workload

Planning estimate: **12–16 GB VRAM**, with 24 GB allowing more room for simulation; target usage is
unmeasured. Default training is **100,000 steps × batch 8 = 800,000 sample presentations**.
Equivalent passes are `800000 / split.json:train_frames`. The new actual frame count remains
unavailable until regeneration; the split and checkpoint receipts calculate and preserve it from
real data. Estimate hours only after measuring target throughput. See the guide for padding,
timing and memory-counter limits.
