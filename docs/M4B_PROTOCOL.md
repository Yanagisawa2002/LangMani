# M4B: does changing the instruction change the destination?

M4A is frozen at `717a07d`. Its ACT result is a non-language, single-task reference, not a fair
architecture comparison against this two-goal benchmark. M4B adds no environment or raw-data change.

## Minimal benchmark

Use the existing scene, red cube, left bin and right bin. The green/blue cubes remain distractors.
Two semantic IDs, `red_cube:left_bin` and `red_cube:right_bin`, are independent of text strings.
The existing six canonical sentences denote six object/destination combinations, not six paraphrases.
Only the two red-cube canonical commands enter M4B fine-tuning. All six remain in immutable metadata.
Both destinations have expert demonstrations in the accepted 360-episode M3A/M3B archive. M4B selects
their 120 episodes and inherits scene-level train/validation/test assignments; it never repartitions
frames or writes to that source. Actual frame counts and hashes come from exhaustive validation.

At reset, the task metadata changes the success query but neither physical layout nor RGB/qpos.
Native acceptance must verify identical state, image and qpos hashes across goals. Thus one initial
policy-visible observation corresponds to two different valid action sequences, selected by text.
Dataset first-frame images can differ slightly because separate lossy video streams were encoded;
report their differences separately, without claiming compressed bytes are identical. The decisive
causal test uses identical native reset observations and common flow-matching inference noise.

## Frozen run and language conditions

Use upstream `lerobot/smolvla_base`, with immutable model and backbone/tokenizer revisions and file
hashes, LeRobot 0.6.0, one RGB view plus 9D qpos and instruction text. Output remains 8D absolute
`pd_joint_pos`; 50-action chunks, 10 executed actions before re-observation, 10 flow steps. Keep the
upstream pretrained architecture and fine-tuning defaults. Initial budget: 20,000 optimizer steps,
batch 8, learning rate 1e-4, seed 0. Real-data smoke/save/reload/resume must pass before full training.
Measure memory and throughput on the actual GPU; no numerical language-success threshold is assumed.

Use 20 fresh scene seeds from schedule seed 43000, excluding every source seed and M4A's 20 final
evaluation scenes. Freeze the schedule before inspecting policy outcomes. Evaluate the final planned
checkpoint, without checkpoint selection. Per scene run these five distinct policy inputs:

1. Canonical red → left.
2. Canonical red → right.
3. Blank string (processed by the upstream tokenizer, with no fallback to the canonical command).
4. Held-out red → left paraphrase.
5. Held-out red → right paraphrase.

Paraphrases alternate between two fixed sentence pairs by schedule index. They are unseen in this
fine-tuning dataset; absence from the foundation model's pretraining corpus is not claimed.
Reset the same scene and policy queue for each input. Use identical per-scene/per-chunk noise across
conditions, and hash it. Keep the 200-step cap and existing success predicate. Evaluate the existing
predicate for both semantic goals for scoring only; no simulator state reaches action selection.
Stop a policy rollout when either red-cube destination satisfies that unchanged predicate, on native
failure/truncation, or on invalid actions. Apply the already verified native gripper saturation and
strict absolute-arm bounds. Record every saturation, executed action sequence, flags and video.

Each scene yields eight scored rows: two requested goals × correct/swapped/blank/paraphrase.
Canonical rollouts serve both their correct-goal score and the opposite requested-goal swapped score;
the blank rollout is likewise scored against both goals. Every row identifies its physical rollout.
This is 100 physical policy rollouts, not 160 independent trials. Expert reference uses 40 separate
scene/goal rollouts. Reused scoring avoids pretending identical inputs are independent evidence.

## Metrics and claims

Report original-request success under each condition, per-goal success, timeout count, successful-only
mean episode length, termination distribution, and achieved-goal distribution. For swapped prompts,
also report success at the goal actually specified by the swapped text. Wrong-request failure alone
does not establish language use: the policy should actually reach the alternate instructed goal.

Primary sensitivity evidence is the fraction of scene pairs in which canonical-left reaches left
AND canonical-right reaches right. Also report achieved-goal changes, action trajectory differences,
the analogous paraphrase pair result, and correct-minus-swapped/blank rate differences. These are
measured values with explicit denominators, not invented acceptance thresholds. A constant destination
policy can score 50% across balanced goals but scores zero on the two-goal pair-following metric.
Infrastructure, hash-pairing, missing artifacts or zero-step failures invalidate comparison.

Retain exact configs/splits, data/model/revision hashes, noise/observation/state/action hashes,
rollout JSON/CSV, scored episode CSV, aggregate metrics, all policy videos and representative links.
Classify wrong goal, invalid action, out-of-bounds arm, off-table failure, timeout and infrastructure
separately. For timeouts retain existing grasp/containment/static flags; do not infer unobserved causes.
High task success without paired language sensitivity supports no language-use claim. This benchmark
does not establish open-ended understanding or generalization beyond its two goals and fixed prompts.
