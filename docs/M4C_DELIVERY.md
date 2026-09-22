# M4B.1 and M4C delivery record

M4B.1 is complete and separately delivered at `3104578386242bcf62a2b09ea97a86f1b2a92dc0`.
M4C implementation is being validated. M4C real-data smoke, language/data validation, complete
L1/L5/L10 training, closed-loop evaluations and local final recovery are pending. No M4C outcome
is inferred from M4B, fixtures or smoke. M4A `717a07d` and M4B `99139de` remain frozen.

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
independent robot demonstrations. Each run presents160,000 robot-frame samples over20,000
optimizer updates. Original validation12/2,154 and test12/2,156 red-goal episodes/frames stay out
of training; the authoritative six-task source remains360 episodes/64,548 frames.

Nested catalogs have1/5/10 training expressions per goal and six held-out expressions per goal,
two each lexical/syntactic/natural. All models receive identical evaluation wording. Common
canonical wording is the primary seen condition. Every expression, source episode, semantic
goal, template ID, family and hash is persisted. Runtime leakage/token-length validation remains
pending; fixture checks do not validate the physical dataset. Full L10 adds natural phrasing,
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
# Only after all preceding stages passed:
for level in L1 L5 L10; do
  python scripts/m4c_baseline.py train --protocol results/m4c/protocol \
    --level "$level" --smoke-validation results/m4c/smoke-comparison.json \
    --output "results/m4c/full-$level-seed0"
done
python scripts/m4c_baseline.py compare --protocol results/m4c/protocol \
  --runs results/m4c/full-L1-seed0 results/m4c/full-L5-seed0 results/m4c/full-L10-seed0 \
  --output results/m4c/full-comparison.json
for level in L1 L5 L10; do
  python scripts/m4c_baseline.py evaluate --protocol results/m4c/protocol \
    --run "results/m4c/full-$level-seed0" --output "results/m4c/evaluation-$level-seed0"
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
the438 previously passing tests. Native gates and real-data wrapper equality remain pending.

## O–Q. Claims, limitations and next action

M4B.1 supports observed destination sensitivity with incomplete object interaction and execution.
No M4C improvement claim is available before actual full evaluations. Even a positive result is
restricted to one training seed,20 paired scenes,two known goals and these held-out expressions.
Do not claim general language understanding, open vocabulary or transfer. Language variants and
reused scoring rows are not independent demonstrations/trials. After the prescribed experiment,
recommend packaging, additional seeds or a specific failure investigation from measured evidence;
do not automatically launch another milestone or seed sweep.
