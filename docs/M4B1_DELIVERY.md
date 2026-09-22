# M4B.1 diagnostic decomposition

**Complete:** implementation, native diagnostic smoke, all 100 frozen-action replays, independent
telemetry/media audits and local backup hashes passed. M4A `717a07d` and M4B delivery `99139de` remain
frozen. Diagnostic execution commit: `86e6d8eae210e5ce3d57b4b4331125c6e090d959`.

## Architecture and measurement boundary

`m4b1_runtime.DiagnosticObserver` wraps the unchanged environment's `step`, then records TCP,
cube and bin poses, pairwise finger contact forces and native Panda grasp flags. Each sample
asserts public state hashes match before/after reading. No diagnostic is passed to the model.
`m4b1_diagnostics` reduces saved arrays offline; the M1 environment and M4B policy/scoring
modules are unchanged. Thresholds, persistence, ambiguity handling, taxonomy precedence and
metric denominators are defined in [M4B1_PROTOCOL.md](M4B1_PROTOCOL.md).

The primary goal-selection proxy is first sustained TCP approach to a destination. It includes
empty-handed movement and does not establish internal intent. Both instructions operate on the
same red cube: object first contact/grasp cannot identify the left/right destination. Separate
bin first-contact accuracy is available but bin contact is not required for task success.
No destination-grasp accuracy is fabricated. Forces are sampled at control boundaries and can
miss brief contacts between samples. Native `is_grasping` is a force/angle predicate, not a
guarantee of secure transport. Full task success retains the exact original predicate.

## Commands and evidence validation

Use the accepted M4B runtime overlay and set `PYTHONPATH` to this checkout's `src`:

```bash
source /root/autodl-tmp/langmani-m4b1-ops/runtime.env
cd /root/autodl-tmp/langmani-m4b1
python scripts/diagnose_m4b.py \
  --source /root/autodl-tmp/langmani-m4b/results/smolvla_baseline/full-seed0/evaluation-seed43000 \
  --output results/m4b1/smoke-v1 \
  --checkpoint /root/autodl-tmp/langmani-m4b/results/smolvla_baseline/full-seed0/training/checkpoints/020000/pretrained_model \
  --smoke
python scripts/diagnose_m4b.py \
  --source /root/autodl-tmp/langmani-m4b/results/smolvla_baseline/full-seed0/evaluation-seed43000 \
  --output results/m4b1/full-v1
```

Output paths must be new. The first command uses the numerically first original successful
scene, all five physical prompt IDs, independent uninstrumented replay controls, and actual
frozen-checkpoint inference on its first successful prompt. Smoke took 73.951 seconds: five
recorded trajectories (949 actions), five controls (949 actions), and one online run (149
actions). All action/video equality and per-step public state invariance checks passed.

The full command replays all original 100 physical trajectories; it does not create 100 new
independent model trials. Canonical and blank scoring reuse yields the same 160 episode rows.
`per_episode_diagnostics.csv`, `failure_taxonomy.json`, `metrics.json`, detailed JSON/NPZ traces
and deterministic representative video references are written separately from frozen evidence.

## Tests and frozen boundary

Local CPU suite: 438 passed, 5 native-only skipped, 5 GPU/rendering deselected. Eight new rule
tests cover threshold persistence, ties, object/side separation, grasp/lift/placement/drop,
invalid telemetry, precedence, swapped/null references and paired partitions. Ruff format/check
and wheel/sdist build passed. Native main suite: 441 passed, 5 GPU/rendering deselected; planner
suite: 2 passed. Ordered M0/M1/M2 native target gate passed, retaining M2 177/180 and its three
planning failures. M2 results are environment prerequisites, never language experiment scores.
All 100 replays (19,080 actions) matched original reset state/RGB/qpos, final flags, termination
and byte-identical rendered videos; full replay took 426.100 seconds. Independent local audit
recomputed approach/contact/grasp/placement/drop/final geometry and metric counts from raw arrays;
1,375 grasp-positive object samples and 3,464 contact-positive target samples were observed.
All 100 videos decoded to 19,180 frames. Six deterministic representative frame sheets were
inspected against event times; visual inspection did not assign or alter any failure category.

Changed files: new diagnostic runtime, pure reducer, CLI, rule tests and this delivery/protocol;
AGENTS.md records authorized milestone boundaries; DECISIONS.md records D035; README will link
the completed diagnostic result. All earlier environment/dataset/expert/M4A/M4B source and test
files remain byte-identical to `99139de`. No datasets, models or generated results enter Git.

Local review root: `D:/LangMani-worktrees/m4b1-diagnostics-m4c-paraphrase/outputs/m4b1-review`.
Five frozen recovery references (raw M3A, derived M3B, M4B checkpoint, M4B evidence, pinned
pretrained assets) passed fresh outer SHA-256 checks; original per-file verification receipts
remain preserved. `m4b1-evidence.tar.gz` is 17,726,154 bytes, contains 395 verified files
(21,654,663 uncompressed bytes), and has SHA-256
`d8c3e739889ed398920851926c913e42405f250cb569be45d74a6bd54053c263`.
The archive was safely extracted under `outputs/m4b1-review/evidence`; every member's size/hash
matched its manifest. No diagnostic GPU-memory peak was sampled, so that value is unavailable.
This milestone performs no new training and does not replace any M4B checkpoint.

## Diagnostic results and the five requested answers

All selection numbers below mean the fixed, geometric **destination-approach proxy**, including
empty-handed movement. They are not full semantic/task understanding. First object contact and
grasp refer to the shared red cube; first bin contact measures the side but is not a success prerequisite.

| Condition / scoring reference | Correct destination approach | Red first contact | Red first grasp | Correct placement attempt | Full success | Success after correct approach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct canonical / supplied = requested | 40/40 (100%) | 27/40 (67.5%) | 20/40 (50%) | 15/40 (37.5%) | 15/40 (37.5%) | 15/40 (37.5%) |
| Swapped / supplied | 40/40 (100%) | 27/40 (67.5%) | 20/40 (50%) | 15/40 (37.5%) | 15/40 (37.5%) | 15/40 (37.5%) |
| Swapped / original requested | 0/40 | 27/40 (67.5%) | 20/40 (50%) | 0/40 | 0/40 | 0/0 (unavailable) |
| Unseen paraphrase / supplied = requested | 32/40 (80%) | 0/40 | 0/40 | 0/40 | 0/40 | 0/32 |
| Blank / original requested | 12/40 (30%) | 0/40 | 0/40 | 0/40 | 0/40 | 0/12 |

Blank has no supplied instruction goal, so supplied-goal accuracy is undefined, not 0%.
Its 20 physical rollouts are scored twice against left/right. Swapped reuses canonical physical
rollouts and is not another 40 trials. Side-discriminating first bin contact was 4/40 correct
canonical, 4/40 supplied-swapped, 0/40 original-request-swapped, 28/40 paraphrase, and 7/40
requested-blank. High bin contact under paraphrase coexists with zero red contact and zero success.

1. **How often is the instructed destination selected?** Canonical 40/40; tested held-out
   paraphrases 32/40. Paraphrase has one opposite-side approach and seven unobserved side choices.
   These thresholds were committed before telemetry; no rule was fitted to these results.
2. **How often does correct selection lead to completion?** Canonical 15/40; paraphrase 0/32.
   The cumulative canonical funnel is red approach 40/40 → contact 27/40 → grasp 20/40 →
   correct destination approach with those preceding achievements 20/40 → placement 15/40 →
   full success 15/40. The independent destination-stage count is still 40/40. Paraphrase red
   approach is only 1/40, followed by 0/40 contact/grasp/placement/success. Funnel stages are
   cumulative achievements, not a claim that their timestamps always follow this order.
3. **Grounding or execution bottleneck?** Wrong destination choice does not explain canonical
   failures: all 25 canonical timeouts approached the correct side. Thirteen never contacted
   red, seven contacted but did not achieve a sustained native grasp, four satisfied that predicate but
   never lifted red by the defined 2 cm, and one lifted/grasped then released outside a bin.
   Thus observed canonical losses are in the object-interaction/execution stages. Paraphrases
   often direct the hand to the correct bin while failing to interact with red. These data
   cannot decide whether that omission is a pickup/subtask semantic error or a control failure;
   destination-direction evidence must not be promoted to understanding of the full instruction.
4. **Does swapped behavior remain consistent before completion?** Yes: supplied-side approach
   40/40, original opposite-side approach 0/40. Canonical paired_instruction_switch_accuracy is
   20/20 from approach, compared with only 6/20 pairs succeeding at both complete tasks. Paraphrase
   approach switching is 12/20; seven pairs have only one corresponding choice and one pair
   approaches the same side under both texts. Both-full-success paraphrase pairs remain 0/20.
   Same-target and one-correct categories overlap in raw counts; the documented partition gives
   same-target precedence and separately records eight paraphrase pairs with one correct side.
5. **What dominates the original 85 timeouts?** See the exhaustive physical counts below. The
   largest category is correct destination approach without red contact: 45/85 (13 canonical,
   32 paraphrase). Blank has neutral categories because no semantic instruction was supplied.

| First-match physical timeout category | Count / 85 | Percent |
| --- | ---: | ---: |
| correct_goal_no_contact | 45/85 | 52.94% |
| unprompted_destination_failure | 12/85 | 14.12% |
| no_meaningful_interaction | 10/85 | 11.76% |
| correct_goal_contact_no_grasp | 7/85 | 8.24% |
| oscillation_or_stall | 5/85 | 5.88% |
| correct_goal_execution_timeout | 4/85 | 4.71% |
| correct_goal_grasp_then_drop | 1/85 | 1.18% |
| wrong_goal_selected | 1/85 | 1.18% |

Zero new infrastructure errors. Requested-goal and supplied-goal taxonomies are separate CSV
columns. The one wrong-side approach first qualifies at step 200 and has no object contact;
it is a late geometric proxy event, not evidence of a completed wrong-object/goal manipulation.
No threshold was adjusted to make this case look more decisive.

## Deterministic video index and limitations

Videos are under extracted `repo/results/m4b1/full-v1/rollouts/`; category selection sorts by
numeric scene seed then prompt ID. The index is `representative_videos.json`:

| Category | Video stem |
| --- | --- |
| Successful correct instruction | seed-234742277-canonical_right |
| Successful instruction switching | seed-375419956-canonical_left and seed-375419956-canonical_right |
| Wrong destination approach | seed-1327019584-paraphrase_left |
| Correct destination with execution failure | seed-170897538-canonical_left |
| Timeout/stall | seed-170897538-blank |

The representative execution failure shows sampled grasp forces but no qualifying red lift,
followed by empty-handed destination motion. Successful examples show red transported and
released in the requested bin. Videos corroborate event traces; they do not reveal internal
intent or explain all failures. All MP4s, not just representatives, are retained.

These one-training-seed, 20-scene results refine the frozen interpretation: 0/40 full paraphrase
success does **not** imply 0/40 destination-direction sensitivity. They still establish no robust
paraphrase-conditioned pick-and-place, broad language comprehension, open vocabulary behavior
or transfer. The next authorized experiment is M4C's fixed-budget language-diversity comparison,
using both this destination proxy and red-interaction/full-success metrics. This completed
diagnostic stage must be separately committed before its implementation begins.
