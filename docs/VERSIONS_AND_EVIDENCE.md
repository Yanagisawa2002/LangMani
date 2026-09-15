# Versions, evidence, and demonstration provenance

Snapshot: **2026-09-16 Singapore time** (2026-09-15 UTC). Repository visibility was checked as
public. These links pin the source versions used for each claim.

## Choose an entry point

| Version | Source and entry | What is implemented | Acceptance boundary |
| --- | --- | --- | --- |
| Current main | [`6920b52c1f48c278e669cd71b69b8949dd900f3a`](https://github.com/Yanagisawa2002/LangMani/tree/6920b52c1f48c278e669cd71b69b8949dd900f3a), [README](../README.md) | M0–M3B: environment, privileged expert, raw collection/replay, LeRobot export and validation | This SHA's decision record leaves native target M3A/M3B acceptance pending. Later branch results do not validate this SHA. |
| Frozen v1 | [`v1.0.0` source, `58434cb17a7234b6d4b2c4fb15aecf8df0621487`](https://github.com/Yanagisawa2002/LangMani/tree/58434cb17a7234b6d4b2c4fb15aecf8df0621487), [technical report](https://github.com/Yanagisawa2002/LangMani/blob/58434cb17a7234b6d4b2c4fb15aecf8df0621487/docs/portfolio/TECHNICAL_REPORT.md) | M0–M5A, eight ACT baselines, language routing to six frozen task controllers; M5B bridge structure; portfolio materials | Historical pipeline and physical verification passed; final model quality gate failed. M5B target probe remains blocked. |
| Stopped v2 research | [`2dcf2ac68e8627cfed98cbbd93272bf1768b190c`](https://github.com/Yanagisawa2002/LangMani/tree/2dcf2ac68e8627cfed98cbbd93272bf1768b190c), [terminal Phase 2C-C result](https://github.com/Yanagisawa2002/LangMani/blob/2dcf2ac68e8627cfed98cbbd93272bf1768b190c/docs/langmani_v2/phase_2c_c_result.md) | Multi-skill data derivation and bounded ACT/SmolVLA experiments | Data pipeline results and failed learning results are separate. No new training or final evaluation is authorized by this guide. |

Neither pinned v1 nor v2 commit is an ancestor of the main snapshot (`git merge-base
--is-ancestor` returned 1 for both after fetching origin). The `v1.0.0` Git tag identifies a source
freeze; this change does not publish a GitHub Release or merge either branch into main.

## v1: preserve the quality failure and correct the pairing claim

The [frozen results index](https://github.com/Yanagisawa2002/LangMani/blob/58434cb17a7234b6d4b2c4fb15aecf8df0621487/docs/RESULTS_INDEX.md)
contains an overstatement in its “Exactly paired initial states: 72 / 72” row. The saved independent
verification report and [decision D-078](https://github.com/Yanagisawa2002/LangMani/blob/58434cb17a7234b6d4b2c4fb15aecf8df0621487/docs/DECISIONS.md#d-078--treat-safe-final-false-rejections-as-no-runtime-evidence)
support the following correction:

| Historical measure | Correct interpretation |
| --- | --- |
| Control records | 72 Oracle records and 72 NeuroSymbolic records |
| Physical initial-state pairs | **61**, with `final_paired_states_validated=false` |
| Routeable commands rejected before execution | **11**; no controller lookup, policy/environment reset, or step for those learned records |
| NeuroSymbolic end-to-end success | **46/72** |
| Oracle task-controller ceiling | **55/72** |
| Correctly routed control timeouts | **15** |
| Final pipeline / historical physical verification | `true` / `true` |
| Final quality gate / SmolVLA go flag at the v1 freeze | `false` / `false` |

“144 control records” must not be described as 144 physical executions or 72 exact physical pairs.
The rejected commands remain in the 72-command success denominator. Historical physical pipeline
verification does not override the failed pairing and quality gates. Later v2 work is a separate
authorization history; it does not change the v1 flags.

The original tag and evidence remain immutable. This guide is the correction layer. The
[attempt receipt](receipts/2026-09-16-single-attempt.md) records the exact report hash and current
recovery inventory. No sealed benchmark was executed for this update.

## Current recovery inventory

The local M5A evidence manifest lists 24 files: **16 matched their recorded bytes and SHA-256,
8 are missing, and 0 present files mismatched**. The missing prediction/dispatch/episode records
prevent a claim that the complete final evidence can currently be independently reconstructed
from that directory. The existing aggregate reports remain historical evidence.

One red-to-left PerTask ACT checkpoint is present: its **9 manifest-covered files**, manifest,
and completion marker passed byte-integrity checks. This establishes availability of one
checkpoint, not restoration of all six deployed controllers or a successful policy reload.

The [receipt](receipts/2026-09-16-single-attempt.md) gives source-relative recovery paths,
the missing-file list, checkpoint/model hashes, and the dependency blocker. Large models,
datasets, and raw experiment outputs remain outside Git.

## Which video shows which controller?

The existing portfolio film `langmani_v1_demo_4min.mp4` uses
`m3b_target_smoke_six_tasks.mp4`, which is **M3B expert-source footage**. Its documented SHA-256
and source are in the [frozen artifact record](https://github.com/Yanagisawa2002/LangMani/blob/58434cb17a7234b6d4b2c4fb15aecf8df0621487/docs/portfolio/ARTIFACTS.md).
It illustrates the environment and data pipeline; it is not footage of the final learned ACT
controller or M5A language-controlled execution.

The 2026-09-16 attempt produced **no new learned-policy video**. The dependency preflight stopped
before model loading or simulator creation. There is consequently no learned-video viewing link
or episode success rate for this attempt. A future demonstration needs separate authorization,
an available frozen runtime, and logs binding code, model, fixed development inputs, raw/projected/
executed policy actions, all episode outcomes, and all resulting videos. A small demonstration
cannot promote a historical quality gate.

## v2: data production and policy quality

| Historical result | Meaning and source |
| --- | --- |
| 2,998 accepted episodes / 254,200 frames from 3,000 official source identities | Replay, filtering, conversion and provenance work over upstream demonstrations; not 3,000 newly self-collected expert trajectories. [Dataset result](https://github.com/Yanagisawa2002/LangMani/blob/2dcf2ac68e8627cfed98cbbd93272bf1768b190c/docs/langmani_v2/phase_2b6_v2_result.md) |
| Bounded ACT: 0/360 final episodes successful; all timed out | Completed experiment and valid physical path with failed learned control. [ACT result](https://github.com/Yanagisawa2002/LangMani/blob/2dcf2ac68e8627cfed98cbbd93272bf1768b190c/docs/langmani_v2/phase_2c_a1_result.md) |
| Relative-action Pick SmolVLA: 1/30 training resets, 0/30 validation | Terminal Case B; final unseen-reset result unavailable. [Phase 2C-C](https://github.com/Yanagisawa2002/LangMani/blob/2dcf2ac68e8627cfed98cbbd93272bf1768b190c/docs/langmani_v2/phase_2c_c_result.md) |

These are pinned historical reports, not GPU experiments repeated by this documentation change.
