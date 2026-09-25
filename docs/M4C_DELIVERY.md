# M4B.1 and M4C final delivery (A–Q)

**M4C seed 0 training and closed-loop evaluation are complete and locally audited.**
Implementation complete, real-data smoke tested, full dataset validated, full SmolVLA trained,
and closed-loop evaluated are five separately satisfied gates. Each of L1/L5/L10 completed
20,000 optimizer updates on the same 96 robot trajectories; all three models, optimizer/RNG
states and full evidence have verified local recovery copies. The evaluation totals are
540 physical rollouts, 720 scoring rows, 99,853 actions and 100,393 decoded video frames,
with zero infrastructure failures. Complete success and destination approach are reported separately below.

Execution identity is `f8781c2710ed3bcd3e4f70cf9afe1764ee9b5b22`, on native Linux / RTX 4090.
The versioned chain started 2026-09-22T08:42:32Z. Evaluations finished at 16:03:39Z (L1),
16:33:47Z (L5), and 17:03:09Z (L10); `full-chain-v2.exit` and `final-backup.exit` are both 0.
No completed long training or evaluation was repeated. M4B.1 was separately delivered at
`3104578386242bcf62a2b09ea97a86f1b2a92dc0`; M4A `717a07d` and M4B `99139de` remain frozen.

The preserved first L1 attempt (`b9308c6`) stopped after 2,143 updates, 786.099 seconds,
with no checkpoint, because its audit rejected the official five-frame epoch tail. Fix `f8781c2`
kept upstream `drop_last=False` and aligned resumed loader epochs. Formal 8/5/8/8 boundary
checks and independent 174-row acceptance passed before fresh `results/m4c-v2` output began.
The old `results/m4c` logs and exit 1 are retained as historical failure evidence.

Operational shutdown is separate from scientific completion. The user explicitly authorized power-off
after delivery and an idle-job check. At closeout on 2026-09-25 the old helper no longer existed and
the SSH endpoint refused TCP connections, preventing a fresh process/screen/GPU inventory or
submission of a shutdown command. **Power-off and billing status are unverified.** Keep the existing
continuation active for this remaining operation; never restart the server or infer shutdown from
connection refusal. Local `shutdown-status.json` records subsequent connection and command evidence.

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


### Catalog coverage and actual text use

The complete catalog is persisted in the language manifest and [m4c_language.py](../src/langmani/policies/m4c_language.py).
Training uses deterministic sample-ordinal hashing; it consumes no model/sampler RNG.
For each group, all actual left/right presentations total 80,059/79,914. Counts by training template follow.

| Template | L1 left/right | L5 left/right | L10 left/right |
| --- | ---: | ---: | ---: |
| train_00 | 80059/79914 | 15935/15956 | 8042/8038 |
| train_01 | 0/0 | 16011/15963 | 8029/7982 |
| train_02 | 0/0 | 15925/16165 | 7933/8088 |
| train_03 | 0/0 | 15920/15976 | 7965/7986 |
| train_04 | 0/0 | 16268/15854 | 8130/7987 |
| train_05 | 0/0 | 0/0 | 7893/7918 |
| train_06 | 0/0 | 0/0 | 7982/7981 |
| train_07 | 0/0 | 0/0 | 7992/8077 |
| train_08 | 0/0 | 0/0 | 7955/7990 |
| train_09 | 0/0 | 0/0 | 8138/7867 |

Zero counts indicate a template outside that group’s permitted catalog, not missing robot data.
## H–I. Exact training and common evaluation

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


The pinned model is `lerobot/smolvla_base` revision `d9f33c94a60fb382c90dea2164c96845bd955e28`;
backbone `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
LeRobot 0.6, Torch 2.11.0+cu128, transformers 5.5.4, tokenizers 0.22.2, NumPy 2.2.6 and
Accelerate 1.15.0 are recorded in native evidence. Input is one base-camera RGB 256×256 frame,
9-value robot state and text; output is 8-value absolute `pd_joint_pos`. Upstream pads/resizes
images to 512×512; chunk 50, execute 10, 10 denoising steps, token limit 48, train-only mean/std
state/action normalization, no image augmentation. There are 450,046,176 total and 99,880,992 trainable parameters.
Vision is frozen; `train_expert_only=true`, `train_state_proj=true`, `use_amp=false`.
All use seed 0, zero loader workers, AdamW LR 1e-4, betas (0.9,0.95), eps 1e-8,
weight decay 1e-10 and gradient clipping 10. Saved preset warmup/decay are 1,000/30,000;
the actual native scheduler log auto-scales them to **666/20,000**, ending at LR 2.5e-6.
Checkpoints are saved every 5,000 steps; the final 020000 state, not a best-eval selection, is evaluated.

Evaluation uses all 20 original paired M4B scenes, nine physical prompts per scene, and frozen
200-step termination/success predicates. Each scene has 20 shared CPU-generated flow-noise chunks
of shape [1,50,32]. All three models match initial state/RGB/qpos and noise hashes for all 180 prompt IDs.
The same seen/lexical/syntactic/natural wording is used across models. Swapped reuses 40 seen
trajectories and blank reuses 20 trajectories for two requested-goal scores; blank supplied-goal
and instruction-switch metrics are undefined. Frozen expert reference is 39/40, including its
retained right-goal planning failure at seed 659916340. M2 177/180 is only a prior gate.

## J. Full results and paired decomposition

### Full manipulation success

| Diversity | Seen | Lexical | Syntactic | Natural | Swapped requested | Swapped supplied | Blank requested |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | 15/40 (37.5%) | 2/40 (5.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 0/40 (0.0%) | 15/40 (37.5%) | 0/40 (0.0%) |
| L5 | 20/40 (50.0%) | 15/40 (37.5%) | 0/40 (0.0%) | 4/40 (10.0%) | 0/40 (0.0%) | 20/40 (50.0%) | 0/40 (0.0%) |
| L10 | 24/40 (60.0%) | 22/40 (55.0%) | 9/40 (22.5%) | 12/40 (30.0%) | 0/40 (0.0%) | 24/40 (60.0%) | 0/40 (0.0%) |

### First destination approach (includes empty hands)

| Diversity | Seen | Lexical | Syntactic | Natural | Swapped requested | Swapped supplied | Blank requested |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | 40/40 (100.0%) | 40/40 (100.0%) | 36/40 (90.0%) | 39/40 (97.5%) | 0/40 (0.0%) | 40/40 (100.0%) | 12/40 (30.0%) |
| L5 | 40/40 (100.0%) | 40/40 (100.0%) | 36/40 (90.0%) | 40/40 (100.0%) | 0/40 (0.0%) | 40/40 (100.0%) | 15/40 (37.5%) |
| L10 | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) | 0/40 (0.0%) | 40/40 (100.0%) | 19/40 (47.5%) |

### Object interaction, instruction switching and timeouts

| Diversity | Condition | First red contact | First red grasp | Destination switch pairs | Both full successes | Timeout scoring rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L1 | blank | 0/40 (0.0%) | 0/40 (0.0%) | undefined | undefined | 40/40 (100.0%) |
| L1 | lexical | 6/40 (15.0%) | 4/40 (10.0%) | 20/20 (100.0%) | 1/20 (5.0%) | 38/40 (95.0%) |
| L1 | natural | 0/40 (0.0%) | 0/40 (0.0%) | 19/20 (95.0%) | 0/20 (0.0%) | 40/40 (100.0%) |
| L1 | seen | 27/40 (67.5%) | 20/40 (50.0%) | 20/20 (100.0%) | 6/20 (30.0%) | 25/40 (62.5%) |
| L1 | swapped | 27/40 (67.5%) | 20/40 (50.0%) | 20/20 (100.0%) | 6/20 (30.0%) | 25/40 (62.5%) |
| L1 | syntactic | 0/40 (0.0%) | 0/40 (0.0%) | 16/20 (80.0%) | 0/20 (0.0%) | 40/40 (100.0%) |
| L5 | blank | 0/40 (0.0%) | 0/40 (0.0%) | undefined | undefined | 40/40 (100.0%) |
| L5 | lexical | 24/40 (60.0%) | 22/40 (55.0%) | 20/20 (100.0%) | 6/20 (30.0%) | 25/40 (62.5%) |
| L5 | natural | 8/40 (20.0%) | 7/40 (17.5%) | 20/20 (100.0%) | 2/20 (10.0%) | 36/40 (90.0%) |
| L5 | seen | 34/40 (85.0%) | 26/40 (65.0%) | 20/20 (100.0%) | 9/20 (45.0%) | 20/40 (50.0%) |
| L5 | swapped | 34/40 (85.0%) | 26/40 (65.0%) | 20/20 (100.0%) | 9/20 (45.0%) | 20/40 (50.0%) |
| L5 | syntactic | 0/40 (0.0%) | 0/40 (0.0%) | 16/20 (80.0%) | 0/20 (0.0%) | 40/40 (100.0%) |
| L10 | blank | 2/40 (5.0%) | 0/40 (0.0%) | undefined | undefined | 40/40 (100.0%) |
| L10 | lexical | 36/40 (90.0%) | 27/40 (67.5%) | 20/20 (100.0%) | 10/20 (50.0%) | 18/40 (45.0%) |
| L10 | natural | 23/40 (57.5%) | 18/40 (45.0%) | 20/20 (100.0%) | 4/20 (20.0%) | 28/40 (70.0%) |
| L10 | seen | 36/40 (90.0%) | 35/40 (87.5%) | 20/20 (100.0%) | 11/20 (55.0%) | 16/40 (40.0%) |
| L10 | swapped | 36/40 (90.0%) | 35/40 (87.5%) | 20/20 (100.0%) | 11/20 (55.0%) | 16/40 (40.0%) |
| L10 | syntactic | 21/40 (52.5%) | 17/40 (42.5%) | 20/20 (100.0%) | 3/20 (15.0%) | 31/40 (77.5%) |

Destination switch is scored against supplied goals, including swapped reuse. Red interaction does not distinguish left/right.
No result here supports broad language understanding, open-vocabulary performance or transfer to other manipulation tasks.

### Pair-category partition

Each category denominator is 20 scene pairs. Raw one-correct is overlapping and is not another partition category.
Blank has no instruction pair; swapped categories reuse seen.

| Model | Condition | Both corresponding | Same target | Only one corresponding | Indeterminate | Both wrong opposite | Raw one-correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | seen | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L1 | lexical | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L1 | syntactic | 16/20 | 2/20 | 2/20 | 0/20 | 0/20 | 4/20 |
| L1 | natural | 19/20 | 0/20 | 1/20 | 0/20 | 0/20 | 1/20 |
| L1 | swapped | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L5 | seen | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L5 | lexical | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L5 | syntactic | 16/20 | 0/20 | 4/20 | 0/20 | 0/20 | 4/20 |
| L5 | natural | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L5 | swapped | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L10 | seen | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L10 | lexical | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L10 | syntactic | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L10 | natural | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L10 | swapped | 20/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |

### Per requested goal

Every cell is a count out of 20. Left/right are semantic bins, not image coordinates.
Swapped supplied scores equal seen; blank rows reuse the same 20 physical runs.

| Model | Condition | Side | Full success | Destination approach | Bin first contact | Red first contact | Red first grasp | Placement attempt |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 | seen | left_bin | 8/20 | 20/20 | 0/20 | 15/20 | 11/20 | 8/20 |
| L1 | seen | right_bin | 7/20 | 20/20 | 4/20 | 12/20 | 9/20 | 7/20 |
| L1 | lexical | left_bin | 1/20 | 20/20 | 0/20 | 3/20 | 2/20 | 1/20 |
| L1 | lexical | right_bin | 1/20 | 20/20 | 6/20 | 3/20 | 2/20 | 1/20 |
| L1 | syntactic | left_bin | 0/20 | 16/20 | 6/20 | 0/20 | 0/20 | 0/20 |
| L1 | syntactic | right_bin | 0/20 | 20/20 | 9/20 | 0/20 | 0/20 | 0/20 |
| L1 | natural | left_bin | 0/20 | 19/20 | 1/20 | 0/20 | 0/20 | 0/20 |
| L1 | natural | right_bin | 0/20 | 20/20 | 14/20 | 0/20 | 0/20 | 0/20 |
| L1 | swapped | left_bin | 0/20 | 0/20 | 0/20 | 12/20 | 9/20 | 0/20 |
| L1 | swapped | right_bin | 0/20 | 0/20 | 0/20 | 15/20 | 11/20 | 0/20 |
| L1 | blank | left_bin | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L1 | blank | right_bin | 0/20 | 12/20 | 7/20 | 0/20 | 0/20 | 0/20 |
| L5 | seen | left_bin | 10/20 | 20/20 | 4/20 | 18/20 | 14/20 | 10/20 |
| L5 | seen | right_bin | 10/20 | 20/20 | 1/20 | 16/20 | 12/20 | 10/20 |
| L5 | lexical | left_bin | 7/20 | 20/20 | 4/20 | 12/20 | 11/20 | 7/20 |
| L5 | lexical | right_bin | 8/20 | 20/20 | 8/20 | 12/20 | 11/20 | 8/20 |
| L5 | syntactic | left_bin | 0/20 | 16/20 | 6/20 | 0/20 | 0/20 | 0/20 |
| L5 | syntactic | right_bin | 0/20 | 20/20 | 13/20 | 0/20 | 0/20 | 0/20 |
| L5 | natural | left_bin | 2/20 | 20/20 | 8/20 | 5/20 | 4/20 | 2/20 |
| L5 | natural | right_bin | 2/20 | 20/20 | 3/20 | 3/20 | 3/20 | 2/20 |
| L5 | swapped | left_bin | 0/20 | 0/20 | 0/20 | 16/20 | 12/20 | 0/20 |
| L5 | swapped | right_bin | 0/20 | 0/20 | 0/20 | 18/20 | 14/20 | 0/20 |
| L5 | blank | left_bin | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| L5 | blank | right_bin | 0/20 | 15/20 | 17/20 | 0/20 | 0/20 | 0/20 |
| L10 | seen | left_bin | 12/20 | 20/20 | 2/20 | 19/20 | 18/20 | 12/20 |
| L10 | seen | right_bin | 12/20 | 20/20 | 1/20 | 17/20 | 17/20 | 12/20 |
| L10 | lexical | left_bin | 10/20 | 20/20 | 2/20 | 19/20 | 12/20 | 10/20 |
| L10 | lexical | right_bin | 12/20 | 20/20 | 4/20 | 17/20 | 15/20 | 12/20 |
| L10 | syntactic | left_bin | 5/20 | 20/20 | 13/20 | 10/20 | 9/20 | 5/20 |
| L10 | syntactic | right_bin | 4/20 | 20/20 | 10/20 | 11/20 | 8/20 | 4/20 |
| L10 | natural | left_bin | 6/20 | 20/20 | 9/20 | 12/20 | 9/20 | 6/20 |
| L10 | natural | right_bin | 6/20 | 20/20 | 5/20 | 11/20 | 9/20 | 6/20 |
| L10 | swapped | left_bin | 0/20 | 0/20 | 0/20 | 17/20 | 17/20 | 0/20 |
| L10 | swapped | right_bin | 0/20 | 0/20 | 0/20 | 19/20 | 18/20 | 0/20 |
| L10 | blank | left_bin | 0/20 | 0/20 | 0/20 | 1/20 | 0/20 | 0/20 |
| L10 | blank | right_bin | 0/20 | 19/20 | 17/20 | 1/20 | 0/20 | 0/20 |

### Placement, bin contact and cumulative task funnel

Non-cumulative bin/placement counts and the six-stage cumulative funnel retain separate meanings.
Funnel stages: red approach / red contact / red grasp / correct destination / correct placement / full success.
Each funnel cell uses all rows of the indicated reference as its denominator; stages are joint achievements, not temporal ordering.

| Model | Condition / reference | Bin first contact | Placement attempt | Cumulative counts (denominator) | Success after correct first destination |
| --- | --- | ---: | ---: | --- | ---: |
| L1 | seen / requested_goal | 4/40 | 15/40 | 40 / 27 / 20 / 20 / 15 / 15 (each /40) | 15/40 |
| L1 | lexical / requested_goal | 6/40 | 2/40 | 6 / 6 / 4 / 4 / 2 / 2 (each /40) | 2/40 |
| L1 | syntactic / requested_goal | 15/40 | 0/40 | 0 / 0 / 0 / 0 / 0 / 0 (each /40) | 0/36 |
| L1 | natural / requested_goal | 15/40 | 0/40 | 3 / 0 / 0 / 0 / 0 / 0 (each /40) | 0/39 |
| L1 | swapped / requested_goal | 0/40 | 0/40 | 40 / 27 / 20 / 0 / 0 / 0 (each /40) | 0/0 |
| L1 | swapped / supplied_goal | 4/40 | 15/40 | 40 / 27 / 20 / 20 / 15 / 15 (each /40) | 15/40 |
| L1 | blank / requested_goal | 7/40 | 0/40 | 0 / 0 / 0 / 0 / 0 / 0 (each /40) | 0/12 |
| L5 | seen / requested_goal | 5/40 | 20/40 | 39 / 34 / 26 / 26 / 20 / 20 (each /40) | 20/40 |
| L5 | lexical / requested_goal | 12/40 | 15/40 | 34 / 28 / 22 / 22 / 15 / 15 (each /40) | 15/40 |
| L5 | syntactic / requested_goal | 19/40 | 0/40 | 4 / 0 / 0 / 0 / 0 / 0 (each /40) | 0/36 |
| L5 | natural / requested_goal | 11/40 | 4/40 | 22 / 8 / 7 / 7 / 4 / 4 (each /40) | 4/40 |
| L5 | swapped / requested_goal | 0/40 | 0/40 | 39 / 34 / 26 / 0 / 0 / 0 (each /40) | 0/0 |
| L5 | swapped / supplied_goal | 5/40 | 20/40 | 39 / 34 / 26 / 26 / 20 / 20 (each /40) | 20/40 |
| L5 | blank / requested_goal | 17/40 | 0/40 | 0 / 0 / 0 / 0 / 0 / 0 (each /40) | 0/15 |
| L10 | seen / requested_goal | 3/40 | 24/40 | 40 / 36 / 35 / 35 / 24 / 24 (each /40) | 24/40 |
| L10 | lexical / requested_goal | 6/40 | 22/40 | 40 / 36 / 27 / 27 / 22 / 22 (each /40) | 22/40 |
| L10 | syntactic / requested_goal | 23/40 | 9/40 | 30 / 21 / 17 / 17 / 9 / 9 (each /40) | 9/40 |
| L10 | natural / requested_goal | 14/40 | 12/40 | 39 / 23 / 18 / 18 / 12 / 12 (each /40) | 12/40 |
| L10 | swapped / requested_goal | 0/40 | 0/40 | 40 / 36 / 35 / 0 / 0 / 0 (each /40) | 0/0 |
| L10 | swapped / supplied_goal | 3/40 | 24/40 | 40 / 36 / 35 / 35 / 24 / 24 (each /40) | 24/40 |
| L10 | blank / requested_goal | 17/40 | 0/40 | 6 / 2 / 0 / 0 / 0 / 0 (each /40) | 0/19 |

### Deterministic failure categories

Physical supplied-goal categories partition each condition, including success. Blank uses its neutral physical taxonomy (20 runs);
all other rows have 40 runs. Swapped is shown for interpretation but reuses seen; exclude it when totaling physical runs.

| Model | Condition | Counts |
| --- | --- | --- |
| L1 | seen | correct_goal_contact_no_grasp: 7; correct_goal_execution_timeout: 4; correct_goal_grasp_then_drop: 1; correct_goal_no_contact: 13; success: 15 |
| L1 | lexical | correct_goal_contact_no_grasp: 2; correct_goal_grasp_then_drop: 2; correct_goal_no_contact: 34; success: 2 |
| L1 | syntactic | correct_goal_no_contact: 36; no_meaningful_interaction: 2; wrong_goal_selected: 2 |
| L1 | natural | correct_goal_no_contact: 39; no_meaningful_interaction: 1 |
| L1 | swapped | correct_goal_contact_no_grasp: 7; correct_goal_execution_timeout: 4; correct_goal_grasp_then_drop: 1; correct_goal_no_contact: 13; success: 15 |
| L1 | blank | no_meaningful_interaction: 6; oscillation_or_stall: 2; unprompted_destination_failure: 12 |
| L5 | seen | correct_goal_contact_no_grasp: 8; correct_goal_execution_timeout: 2; correct_goal_grasp_then_drop: 4; correct_goal_no_contact: 6; success: 20 |
| L5 | lexical | correct_goal_contact_no_grasp: 6; correct_goal_execution_timeout: 5; correct_goal_grasp_then_drop: 2; correct_goal_no_contact: 12; success: 15 |
| L5 | syntactic | correct_goal_no_contact: 36; no_meaningful_interaction: 2; oscillation_or_stall: 2 |
| L5 | natural | correct_goal_contact_no_grasp: 1; correct_goal_execution_timeout: 1; correct_goal_grasp_then_drop: 2; correct_goal_no_contact: 32; success: 4 |
| L5 | swapped | correct_goal_contact_no_grasp: 8; correct_goal_execution_timeout: 2; correct_goal_grasp_then_drop: 4; correct_goal_no_contact: 6; success: 20 |
| L5 | blank | no_meaningful_interaction: 2; oscillation_or_stall: 3; unprompted_destination_failure: 15 |
| L10 | seen | correct_goal_contact_no_grasp: 1; correct_goal_execution_timeout: 5; correct_goal_grasp_then_drop: 6; correct_goal_no_contact: 4; success: 24 |
| L10 | lexical | correct_goal_contact_no_grasp: 9; correct_goal_execution_timeout: 2; correct_goal_grasp_then_drop: 3; correct_goal_no_contact: 4; success: 22 |
| L10 | syntactic | correct_goal_contact_no_grasp: 4; correct_goal_execution_timeout: 2; correct_goal_grasp_then_drop: 6; correct_goal_no_contact: 19; success: 9 |
| L10 | natural | correct_goal_contact_no_grasp: 5; correct_goal_execution_timeout: 3; correct_goal_grasp_then_drop: 3; correct_goal_no_contact: 17; success: 12 |
| L10 | swapped | correct_goal_contact_no_grasp: 1; correct_goal_execution_timeout: 5; correct_goal_grasp_then_drop: 6; correct_goal_no_contact: 4; success: 24 |
| L10 | blank | oscillation_or_stall: 1; unprompted_destination_failure: 19 |

The independent closeout audit recomputes every per-goal metric, cumulative funnel, conditional denominator, physical taxonomy and all five paired categories. Full raw audits also recompute sampled telemetry events, action hashes/clipping, frame counts, final flags and CSV consistency. Blank red contact 2/40 for L10 represents one physical contact scored twice. No selected-goal conditional denominator of zero has a defined rate.

## K. Actual runtime and memory

| Model | Optimizer updates | Actual samples | Training seconds | Optimizer peak bytes (MiB) | Training GPU samples / peak MiB / max utilization | Evaluation seconds | Evaluation GPU samples / peak MiB / max utilization |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| L1 | 20,000 | 159,973 | 7152.985 | 2,852,236,288 (2720.104) | 1439 / 3380 / 60% | 1959.008 | 399 / 1929 / 24% |
| L5 | 20,000 | 159,973 | 7226.463 | 2,853,337,088 (2721.154) | 1453 / 3380 / 57% | 1772.162 | 362 / 1929 / 25% |
| L10 | 20,000 | 159,973 | 7243.786 | 2,852,236,288 (2720.104) | 1457 / 3380 / 57% | 1726.341 | 353 / 1929 / 25% |

Training invocation total is 21,623.234 seconds (6.006 hours); evaluation metric total is 5,457.511 seconds (90.959 minutes).
The abandoned 786.099-second invocation is additional. Evaluation wrapper stage durations were 1,995/1,808/1,762 seconds and include startup overhead.
Device memory/utilization are approximately five-second samples, not continuous peaks. Optimizer allocator peaks are separately measured inside optimizer steps.
GPU logs use server UTC+8; sample endpoints and CSV hashes are retained in `L*-training-resources.json` and `independent-closeout-audit.json`.
Physical success/timeout terminations are 17/163, 39/141 and 67/113 out of 180 for L1/L5/L10 respectively.

## L. Representative media and actual inspection

Native representative indexes select the first numeric scene seed/prompt within each observed category.
The local renderer produced 17/21/22 sheets for L1/L5/L10. **Visual review covered the following nine sheets plus three switching-pair sheets: 12 sheets, 15 physical trajectories.**
This was keyframe inspection, not full playback of all 540 videos or visual acceptance of every generated sheet.
Blank stage slots mean the event was unavailable. Exact selected frame numbers are in `manual-media-review.json` and the media indexes.
All 100,393 video frames were decoded automatically in the separate raw audit. Semantic left/right must not be read as image left/right.

Links below resolve in the local checkout; ignored evidence/media are not shipped through Git.

| Model | Rollout / video | Inspected sheet | Observation and limit |
| --- | --- | --- | --- |
| L1 | [seed-1327019584-syntactic_left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L1-seed0/rollouts/seed-1327019584-syntactic_left.mp4) | [frames](../outputs/m4c-review/media-L1/seed-1327019584-syntactic_left.png) | Wrong-goal proxy: red remains on the table; the first-destination snapshot is close to the central/bin region. Side classification is telemetry-based, not inferred from the camera projection. |
| L1 | [seed-234742277-seen_right](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L1-seed0/rollouts/seed-234742277-seen_right.mp4) | [frames](../outputs/m4c-review/media-L1/seed-234742277-seen_right.png) | Red moves from its source position to the near-image bin; final frame shows it inside that bin, consistent with recorded success. |
| L5 | [seed-468995337-natural_left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L5-seed0/rollouts/seed-468995337-natural_left.mp4) | [frames](../outputs/m4c-review/media-L5/seed-468995337-natural_left.png) | Red is raised and then lowered into the far-image bin; final frame is consistent with recorded success. |
| L5 | [seed-1327019584-syntactic_left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L5-seed0/rollouts/seed-1327019584-syntactic_left.mp4) | [frames](../outputs/m4c-review/media-L5/seed-1327019584-syntactic_left.png) | Red remains at its source location; final hand pose differs from reset. Two endpoint images cannot independently establish a temporal stall. |
| L10 | [seed-170897538-natural_left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L10-seed0/rollouts/seed-170897538-natural_left.mp4) | [frames](../outputs/m4c-review/media-L10/seed-170897538-natural_left.png) | Contact/grasp event frames place the hand by the red cube, but red is still outside both bins at the destination and final frames. No placement event is recorded. |
| L10 | [seed-375419956-syntactic_left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L10-seed0/rollouts/seed-375419956-syntactic_left.mp4) | [frames](../outputs/m4c-review/media-L10/seed-375419956-syntactic_left.png) | The sequence shows the red cube lifted and lowered into the far-image bin; final state is consistent with recorded success. |
| L1 | [seed-234742277-seen_left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L1-seed0/rollouts/seed-234742277-seen_left.mp4) | [frames](../outputs/m4c-review/media-L1/seed-234742277-seen_left.png) | Red is elevated near the hand at the destination frame, but remains outside the far-image bin at the final frame; the drop label comes from the full telemetry. |
| L10 | [seed-170897538-lexical_left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L10-seed0/rollouts/seed-170897538-lexical_left.mp4) | [frames](../outputs/m4c-review/media-L10/seed-170897538-lexical_left.png) | The hand contacts near red, then reaches the far-image bin while red remains on the table. No grasp frame is available. |
| L10 | [seed-836627643-blank](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L10-seed0/rollouts/seed-836627643-blank.mp4) | [frames](../outputs/m4c-review/media-L10/seed-836627643-blank.png) | Red remains outside the bins while the hand ends near the near-image bin; no qualifying destination/grasp/placement event. Temporal stall label requires telemetry. |
| L1 | Seen pair seed 375419956: [left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L1-seed0/rollouts/seed-375419956-seen_left.mp4) / [right](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L1-seed0/rollouts/seed-375419956-seen_right.mp4) | [pair frames](../outputs/m4c-review/media-switch-pairs/L1-seen-switch.png) | Both rows share the same initial scene. Red is transferred to different bins in the two final images, consistent with two full successes and opposite supplied goals. |
| L5 | Seen pair seed 234742277: [left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L5-seed0/rollouts/seed-234742277-seen_left.mp4) / [right](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L5-seed0/rollouts/seed-234742277-seen_right.mp4) | [pair frames](../outputs/m4c-review/media-switch-pairs/L5-seen-switch.png) | Both rows share the same initial scene. Red is transferred to different bins in the two final images, consistent with two full successes and opposite supplied goals. |
| L10 | Seen pair seed 206132263: [left](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L10-seed0/rollouts/seed-206132263-seen_left.mp4) / [right](../outputs/m4c-review/final-evidence/repo/results/m4c-v2/evaluation-L10-seed0/rollouts/seed-206132263-seen_right.mp4) | [pair frames](../outputs/m4c-review/media-switch-pairs/L10-seen-switch.png) | Both rows share the same initial scene. Red is transferred to different bins in the two final images, consistent with two full successes and opposite supplied goals. |

## M. Changed files and frozen boundaries

The cumulative change against frozen M4B `99139de` is limited to the following 16 files:

| Files | Purpose |
| --- | --- |
| `AGENTS.md` | Authorized M4B.1/M4C scope added before implementation; latest direct shutdown authorization supersedes its older no-shutdown sentence. |
| `README.md`, `docs/DECISIONS.md` | Evidence-backed milestone status and decision rationale. |
| `docs/M4B1_PROTOCOL.md`, `docs/M4B1_DELIVERY.md` | Frozen diagnostic definitions and separately delivered original-M4B reanalysis. |
| `docs/M4C_PROTOCOL.md`, `docs/M4C_DELIVERY.md` | Precommitted language experiment and this A–Q delivery. |
| `scripts/diagnose_m4b.py` | Separate diagnostic CLI. |
| `src/langmani/policies/m4b1_diagnostics.py`, `src/langmani/policies/m4b1_runtime.py` | Observational diagnostic decomposition and replay integration. |
| `tests/unit/test_m4b1_diagnostics.py` | Diagnostic threshold, pairing and taxonomy tests. |
| `scripts/m4c_baseline.py` | Separate M4C prepare/train/compare/evaluate CLI. |
| `src/langmani/policies/m4c_language.py` | Fixed catalogs, trajectory-label manifests, deterministic text assignment and leakage checks. |
| `src/langmani/policies/m4c_training.py` | Text-only dataset adapter, consumed-sample audit, compatible recovery and model comparisons. |
| `src/langmani/policies/m4c_evaluation.py` | Identical cross-model interventions, diagnostics and result provenance. |
| `tests/unit/test_m4c_baseline.py` | M4C contracts, short-batch and restored-loader regression tests. |

Frozen pre-M4C code/tests/environments remain byte-identical; M4B.1 implementation and its two
protocol/delivery documents remain unchanged from `3104578`. The final closeout edits only README,
this delivery and DECISIONS. Previous uncommitted progress documents are preserved locally under
`outputs/m4c-review/pre-closeout-documents`. Models, videos, data and generated audits stay outside Git.

## N. Tests, independent acceptance and local recovery

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

Partial-batch correction `f8781c2`: local454 passed/5 native skips/5 GPU deselections (28.47s),
Ruff/build passed; native16 targeted tests passed (6.68s). Real-data loading around steps2,142 and
4,286 yielded8/5/8/8 batches in all language groups, with every non-text tensor equal to the
unwrapped reference. Local independent checks verified174 CSV rows, language choices, optimizer
step indices and sequence hashes. These are boundary checks with zero optimizer updates, not
new benchmark trials. Failed-attempt16-file backup SHA256:
`d8f018739f7be52cafc3169c772f58556c060db6780ec2e0fce738b5236be7f6`;
boundary12-file backup: `f78ba8ed07345699c839793ec23d183a060228c2d561d604a3c753701f33020b`.
Both archives passed outer and per-file hashes and safe extraction. The local boundary acceptance
receipt gates the versioned recovery chain. Original source, language manifest and native expert
gates remain unchanged; no completed long training was repeated.

### Recovery inventory and identities

Local root: `D:/LangMani-worktrees/m4b1-diagnostics-m4c-paraphrase/outputs/m4c-review`.
Every archive listed below passed outer SHA-256, exact member inventory, per-file byte/hash checks and safe extraction.
Each model archive has 22 files including all 12 final checkpoint files: model, official processors, optimizer, scheduler, RNG and saved step 20,000/batch 8/processes 1.
Recovery acceptance verifies saved bytes/state; it does not claim a new optimizer update or a newly executed full resume.
The final evidence archive contains 3,062 files including all evaluation telemetry/actions/videos/CSV, metrics, input schedules and native logs.

| Archive | Compressed bytes | Verified files | SHA-256 |
| --- | ---: | ---: | --- |
| m4c-L1.tar.gz | 1,076,785,454 | 22 | `6403e38ca0043f0a60176e38103364bfbfe7c9de4e4fff602d6839504bccd0dc` |
| m4c-L5.tar.gz | 1,077,020,223 | 22 | `db0acf056eb5bd095e7f888ecaa8f990c80e45242e2000fe538c3f1ca18a594f` |
| m4c-L10.tar.gz | 1,077,073,814 | 22 | `62b2d72e8b451e61ec2c6d8eec17792885a6d463fa952c23d145e56f5cccceaf` |
| m4c-final.tar.gz | 96,930,792 | 3062 | `7322ced8ea5e3cc33032f7526223f1ff105bb8c9338e74330e5e6be9b32a4529` |

| Model | Final model SHA-256 |
| --- | --- |
| L1 | `4aabc676a85bc1b521ed4098678723c21d238d79bcb4454cbe489a8121a7243e` |
| L5 | `c0a79bb6ff356e81fdec78354c2238c013b878d7fb4fa494251c855a52515ac1` |
| L10 | `bdb10da9ac5968a00aa7a8dd6facbe1be0b91ce9bb89a778bbe8b598154d93fe` |

L1 final model bytes exactly match frozen M4B; this is a control check, not a substitute for the new shared held-out evaluation.
| Shared identity | SHA-256 |
| --- | --- |
| Initialization | `216bc01fa8255d933a2c30c51dd6e5816d34a52d965d306aed2aa0de3da473e8` |
| All 159,973 ordered robot samples | `1be02b926e83194cf019e719f26c8493c4e6a12efa9c04a8e09dcaaf996fa34f` |
| Canonical train-only normalization | `f08b62673e6d21609242b4cfeef3a35aa7fa4bfe3c010459d2cc0571c65e5848` |
| Fixed training configuration excluding only language level | `806dddf25a93170102ab8d0ba4182af0709ac3c179d8ebe81acceaf41bbd9316` |
| Source dataset content | `cd97abe24a5dde623046fdf67da4810c5d7c8aaa2d1e59ec6e66b2531b212b2a` |
| Frozen split | `7bbd3e9a68d72cc5681c75884f3b0e1072e0d62ea8afe2626f1a4dee307bb68f` |
| Shared evaluation schedule | `19ec9d813853ca546b45aafc52423ec211a6403687b6fe9008999b63c31cceda` |
| 180 paired input records, canonical JSON | `78c16b5643c17a028fa5640739a362976e581414d1cd134747303c2f12b9b65a` |

The canonical input hash covers seed/prompt/text/template and initial state/RGB/qpos/noise hashes; `paired-inputs.json` retains every record.
The three immutable training-audit bundles retain the full 479,919-row sample evidence and 27 short batches.
Five frozen source/model/asset recovery references remain in `source-recovery-reference.json`, bound to prior per-file receipts and current archive hashes.
The source M3A/M3B archives and pretrained assets are required for restoration; the final M4C archive alone is not self-contained.
See local `RECOVERY.md`, `manual-media-review.json`, `independent-closeout-audit.json`, all `*-verification.json` / `*-recovery-audit.json` receipts, and `m4c-local-closeout-manifest.json`.
These local records include exact paths, verification commands and final delivery commit; they are excluded from Git.

## O. Measured conclusion

At equal robot-data and optimizer budgets, L10 improves full task success over L1 from
15/40 to 24/40 seen (+22.5 percentage points), 2/40 to 22/40 lexical (+50.0 points),
0/40 to 9/40 syntactic (+22.5 points), and 0/40 to 12/40 natural (+30.0 points).
L5 is intermediate except syntax remains 0/40. Red-contact/grasp rates also improve.
Destination approach was already high under L1 (90–100% on prompted conditions), so the
measured change extends beyond selecting a destination toward performing pickup/placement.
Blank remains 0/40 full success for every model, and swapped supplied-goal successes match
seen while opposite-original-request successes remain zero. Those are reused scoring controls.

## P. Limits

One training seed, 20 paired scenes, two familiar visual goals, and 12 held-out expressions
do not establish broad language understanding, open vocabulary or transfer. No across-training-seed
uncertainty was measured. Paired scenes, reused swapped trajectories and doubled blank scoring
must not be treated as independent trials. Catalog size and phrasing-family coverage change
together; no per-family causal attribution is identified. M4C's held-out catalog differs from
original M4B, so original paraphrase 0/40 is not substituted into this controlled comparison.
Destination approach can be empty-handed; red contact/grasp cannot identify left versus right.
Diagnostics describe observable trajectories, not hidden language representations or failure causes.
L10 still fails 18/40 lexical, 31/40 syntactic and 28/40 natural requests. Success is far from robust.

## Q. Disposition

Package this seed-0 milestone and stop experiment work. No extra seeds or next milestone are launched.
If separately authorized later, an L1/L10 multi-seed replication would test reproducibility;
a focused pickup/placement failure investigation could test mechanisms without revising these frozen results.
The remaining operational action is the already-authorized idle-server shutdown. Verify current
processes, screen jobs and GPU compute jobs before any shutdown, wait for other jobs without
terminating them, and retain a local command/time/result receipt. SSH refusal is insufficient
evidence of power-off. Keep the continuation active while this operation cannot be verified.
