# M4B.1 and M4C delivery record

M4B.1 is complete and separately delivered at `3104578386242bcf62a2b09ea97a86f1b2a92dc0`.
M4C implementation, full source-data/language validation and real-data smoke passed.
Complete L1/L5/L10 training, closed-loop evaluations and local final recovery are pending. No M4C outcome
is inferred from M4B, fixtures or smoke. M4A `717a07d` and M4B `99139de` remain frozen.

The first full-chain attempt started at2026-09-22T08:00:12Z (16:00 Singapore), executing `b9308c6`.
L1 stopped after2,143 updates on the first epoch's5-frame tail because the sample audit incorrectly
required every batch to contain8 frames. The failed invocation took786.099s and saved no checkpoint;
its files/logs are retained. Recovery uses fresh `results/m4c-v2` paths after a native boundary check.
L5/L10 and all full evaluations have not yet run. The same-task
`langmani-m4c` continuation is active until verified recovery and A–Q delivery are finished.

## A–D. Completed diagnostic decomposition

[M4B1_DELIVERY.md](M4B1_DELIVERY.md) supplies the architecture, exact behavioral thresholds,
taxonomy precedence and reanalysis. [M4B1_PROTOCOL.md](M4B1_PROTOCOL.md) fixes objective definitions.
Canonical destination approach40/40 leads to red contact27/40, grasp20/40 and full success15/40.
Paraphrase destination approach32/40 leads to red contact/grasp/success0/40. Paired destination
switching is20/20 canonical and12/20 paraphrase, distinct from6/20 and0/20 full-success pairs.
The dominant timeout category is correct-side approach without red contact45/85. Empty-hand
destination approach does not establish comprehension of pickup or of the full instruction.

## E–G. Language design, statistics and leakage

[M4C_PROTOCOL.md](M4C_PROTOCOL.md) fixes the comparison before training or evaluation.
The same96 training trajectories contain17,149 frames (left48/8,588; right48/8,561).
L1/L5/L10 enumerate96/480/960 permitted trajectory-label rows, respectively; these are not
independent robot demonstrations. Each run presents159,973 actual robot-frame samples over20,000
optimizer updates: configured batch8, nine5-frame epoch tails, and160,000 nominal `steps*8`.
The frozen upstream metadata's top-level sample count is nominal; M4C actual counts and optimizer
step/row logs are reported separately. Original validation12/2,154 and test12/2,156 red-goal episodes/frames stay out
of training; the authoritative six-task source remains360 episodes/64,548 frames.

Nested catalogs have1/5/10 training expressions per goal and six held-out expressions per goal,
two each lexical/syntactic/natural. All models receive identical evaluation wording. Common
canonical wording is the primary seen condition. Every expression, source episode, semantic
goal, template ID, family and hash is persisted. Native source-byte and language validation passed:
20 training plus12 held-out expressions, zero normalized-text/template-ID overlap, no original
M4B held-out text in training, maximum20 tokens including the upstream newline (limit48).
Manifest SHA256: `0ebb548845f7f3531e4d3d24347c1f7019110f2f0f4176e58d1c7c698db396ca`.
Full L10 adds natural phrasing,
so this intervention changes expression count and family coverage together.

## H–I. Fixed commands

Run from the separate Linux M4C checkout with the frozen M4B runtime environment and M4C `src`
on PYTHONPATH. The CLI records exact argv/Git/dependency provenance and rejects dirty Git runs.

```bash
python scripts/m4c_baseline.py prepare \
  --dataset-root /root/autodl-tmp/langmani-m4a/outputs/datasets/m3b/langmani-pick-place-lerobot-v1 \
  --validation /root/autodl-tmp/langmani-m4b/results/smolvla_baseline/validation.json \
  --split /root/autodl-tmp/langmani-m4b/results/smolvla_baseline/protocol/split.json \
  --m4b-schedule /root/autodl-tmp/langmani-m4b/results/smolvla_baseline/protocol/schedule.json \
  --assets /root/autodl-tmp/langmani-m4b-assets/assets.json \
  --gate /root/autodl-tmp/langmani-m4b/results/smolvla_baseline/native-gate \
  --output results/m4c/protocol
python scripts/m4c_baseline.py control-smoke --protocol results/m4c/protocol \
  --output results/m4c/control-smoke
for level in L1 L5 L10; do
  python scripts/m4c_baseline.py train --protocol results/m4c/protocol --smoke \
    --level "$level" --output "results/m4c/smoke-$level"
done
python scripts/m4c_baseline.py compare --protocol results/m4c/protocol --smoke \
  --runs results/m4c/smoke-L1 results/m4c/smoke-L5 results/m4c/smoke-L10 \
  --control results/m4c/control-smoke --output results/m4c/smoke-comparison.json
python scripts/m4c_baseline.py evaluate --protocol results/m4c/protocol --smoke \
  --run results/m4c/smoke-L10 --output results/m4c/evaluation-smoke
# Preserve the failed attempt. Copy the protocol byte-for-byte to results/m4c-v2/protocol;
# run the native short-batch boundary check before recovery (see D038).
for level in L1 L5 L10; do
  python scripts/m4c_baseline.py train --protocol results/m4c-v2/protocol \
    --level "$level" --smoke-validation results/m4c/smoke-comparison.json \
    --output "results/m4c-v2/full-$level-seed0"
done
python scripts/m4c_baseline.py compare --protocol results/m4c-v2/protocol \
  --runs results/m4c-v2/full-L1-seed0 results/m4c-v2/full-L5-seed0 results/m4c-v2/full-L10-seed0 \
  --output results/m4c-v2/full-comparison.json
for level in L1 L5 L10; do
  python scripts/m4c_baseline.py evaluate --protocol results/m4c-v2/protocol \
    --run "results/m4c-v2/full-$level-seed0" --output "results/m4c-v2/evaluation-$level-seed0"
done
```

The orchestration must stop on any failed stage. Each group retains its final model, optimizer,
RNG and official processors. Compatible interrupted training uses `--resume` and preserves
previous logs; completed training cannot be repeated. Full evaluations total540 physical rollouts
and720 scoring rows; swapped reuses seen, and blank is scored against both requests.

## J–L. Results, resource measurements and videos

All M4C full-result cells, runtimes and VRAM measurements are **pending**, not zero. Final delivery
will report separate full-success and destination-approach tables, contact/grasp/switch counts,
timeout and failure categories for six conditions, plus per-goal counts. Training runtime comes
from actual logs; optimizer-step allocator peaks and sampled GPU memory are separate measures.
All action arrays, T+1 telemetry/video frames and paired state/image/qpos/noise hashes are retained.
Representative videos are selected by numeric scene seed/prompt within each observed category.
The completed M4B.1 video index and inspected contact sheets are linked in its delivery document.

## M–N. Changes and validation

M4C adds `m4c_language.py` (catalogs/manifests/sampling), `m4c_training.py` (text-only adapter and
sample/model audit), `m4c_evaluation.py` (common interventions/diagnostics), `m4c_baseline.py` (CLI)
and `test_m4c_baseline.py`. This protocol/delivery, README and DECISIONS document scope/results.
The full cumulative file audit and measured test receipts will be recorded after native validation.
No frozen environment, dataset, expert, M4A or M4B implementation changes are planned.

Initial local implementation validation:450 passed,5 native skips,5 GPU/rendering deselections
(31.85s); Ruff format/check passed; wheel and sdist built. These include12 M4C fixture tests and
the438 previously passing tests. At implementation `3bfdfd9`, native453 main plus2 planner tests
and the ordered M0/M1/M2 target gates passed (M2 remains177/180, a prerequisite).

Preparation initially stopped on a serialized string/Path interface error before training. Fix
`b9308c6` passed13 targeted tests locally and natively. The original failure logs/exit1 remain,
and `preflight-v2` resumes from preparation without repeating native expert gates. L1 and unwrapped
20-step smoke checkpoints already match SHA256
`93cab1ab1fa3e53cb6abe59e1f31cf6e945849241225c7867c8f08574c1eba88`.
All three20-step smokes passed and share initial-state hash
`216bc01fa8255d933a2c30c51dd6e5816d34a52d965d306aed2aa0de3da473e8` and robot-sample hash
`c8501ce479c8b6b36e9f07e07cc442810c90a09ea019a8c167a5e8b683e8b112` (160 presentations each).
Nine inference-smoke rollouts completed:1,800 actions/1,809 frames,12 scoring rows,0 infrastructure
errors. All9 timed out after20 training updates; these are pipeline checks, not full results.
Local independent audits recomputed telemetry events, pairing, metrics/CSV, frame and language
sampling identities. Three deterministic representative frame sheets were inspected without
inferring failure causes. The190-file preflight recovery archive passed outer and per-file hashes:
`52e82a498886b3a01aebeba012777090382bce2613725b37e951c31c562a356f` (2,622,497 bytes).
All five immutable data/model/assets recovery-reference archives were rehashed locally.

## O–Q. Claims, limitations and next action

M4B.1 supports observed destination sensitivity with incomplete object interaction and execution.
No M4C improvement claim is available before actual full evaluations. Even a positive result is
restricted to one training seed,20 paired scenes,two known goals and these held-out expressions.
Do not claim general language understanding, open vocabulary or transfer. Language variants and
reused scoring rows are not independent demonstrations/trials. After the prescribed experiment,
recommend packaging, additional seeds or a specific failure investigation from measured evidence;
do not automatically launch another milestone or seed sweep.
