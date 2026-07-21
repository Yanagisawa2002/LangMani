# LangMani 2.0 Phase 2C: push-only SmolVLA

Phase 2C asks whether a genuinely pretrained SmolVLA policy can learn the accepted planar-pushing
task and retain competence under held-out scenes, instructions, hard conditions, and a frozen
photometric shift. Training loss and offline action reconstruction are diagnostics; only real
closed-loop ManiSkill rollouts establish control competence.

## Immutable inputs

The consumer starts from Phase 2B commit
`7394faba1ed2e29ab26b70dbc4058f46e5164041`. Before any optimizer step,
`environment/verify_v2_phase2b.py` must pass again against the external raw and export roots.
`environment/prepare_v2_phase2c.py` then rechecks the committed Phase 2B hashes, all six split
inventories, the complete 397-episode/53,297-frame inventory, exported sidecars, real LeRobot
readback, and the exact 203-episode/26,968-frame training root. Existing Phase 2B files are read
only and are never regenerated.

The train view is the already materialized `train` LeRobot root, not a frame-level resplit. It
excludes validation, all four test splits, rejected demonstrations, replay copies, pick-and-place,
and privileged state. The preparation command recomputes state/action moments by streaming all
26,968 training frames and requires those moments to match the statistics exposed to the official
LeRobot trainer. It records mean, standard deviation, minimum, maximum, and low-variance axes and
checks a normalization round trip.

## Official SmolVLA audit

The required implementation is LeRobot 0.6.0 `SmolVLAPolicy`, not vendored model code. The public
base model resolved on 2026-07-22 as follows:

| Field | Frozen value |
| --- | --- |
| repository | `lerobot/smolvla_base` |
| revision | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| `config.json` SHA-256 | `650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d` |
| model file | `model.safetensors`, 906,712,520 bytes before download verification |
| action chunk | 50 |
| official queued action steps | 50 |
| flow integration steps | 10 |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| language | newline, right padding, maximum 48 tokens |
| normalization | visual identity, state/action mean-standard deviation |

LeRobot creates `RenameObservations`, batch-dimension, newline-task, tokenizer, device, and
normalizer preprocessing steps. Postprocessing applies the serialized action unnormalizer and
moves output to CPU. `predict_action_chunk` bypasses the policy's internal one-action queue;
`select_action` fills and consumes that queue. LangMani deliberately calls `predict_action_chunk`
and owns a true-prefix `ActionChunk`, making the selected execution horizon distinct from the
50-action model chunk. A policy reset clears the official queue and every processor state.

The published checkpoint describes three 256x256 cameras, 6D state, and 6D action. Phase 2C
explicitly replaces only these feature descriptors with the accepted LangMani one-camera,
9D-state, 8D-action contract before loading the pretrained weights. SmolVLA's fixed padded maxima
remain 32, so this changes no weight tensor and is not a model fork. The exact override is frozen in
`configs/langmani_v2/phase2c_smolvla_push.json`.

## Deployable feature map

| Key | Runtime contract |
| --- | --- |
| `observation.images.base_camera` | RGB CHW `[3,256,256]`; uint8 is mapped to float32 `[0,1]`; official resize-with-pad produces 512x512 model input |
| `observation.state` | `PandaPolicyStateV0`, float32 `[9]` |
| `task` | the scheduled non-empty instruction; official tokenizer processing |
| `action` | `PandaJointPositionActionV0`, float32 `[8]`, `pd_joint_pos`, 20 Hz |

Object/target positions or identities, success/failure state, expert phase, planner values,
segmentation, future observations, and corruption metadata are rejected. NaN, infinity, malformed
chunks, and out-of-environment-bound actions are hard failures. The Phase 2C evaluator uses the
existing audited bound processor in `reject` mode; it never clips or projects learned actions.

## Training protocol

The frozen protocol uses the official pretrained weights, BF16, frozen vision encoder,
expert-only training plus the state projection, AdamW at `1e-4`, gradient clipping at 10, and the
official warmup/cosine schedule. One-batch smoke screens batch sizes 16, 8, 4, 2, and 1 in that
order and accepts the largest with finite gradients and at most 28 GiB peak allocated VRAM.

Gates are:

1. one real batch forward/backward/optimizer step, save/reload, adapter inference, and environment
   step;
2. a labelled 16-episode/500-step pipeline-only micro-overfit;
3. seed-0 full training to 20,000 steps with 5k/10k/20k checkpoints;
4. offline metrics on the external Phase 2B validation root only;
5. at most two checkpoints times execution horizons 1/4/8, with at least 40 fixed validation
   episodes per candidate;
6. sealed testing only after the complete development competence gate passes.

The official `lerobot-train` process owns optimization and checkpoint serialization.
`environment/launch_v2_smolvla_training.py` validates the preparation gate, caps steps and sample
updates, supports dry-run and exact checkpoint resume, records the full argv and runtime paths, and
never enables Hub upload. `environment/finalize_v2_smolvla_checkpoint.py` hashes the complete
checkpoint and processor artifacts before it can be loaded by the generic adapter.

`environment/audit_v2_smolvla_base.py` downloads only the pinned official model files, verifies the
resolved Hub revision and reviewed config bytes, hashes the complete tensor and processor files,
and records the installed API signatures. `environment/validate_v2_smolvla_offline.py` evaluates
each frozen checkpoint on all 77 validation episodes. Its predeclared selection rule is lowest full
validation flow-matching loss, with inverse-normalized first-action MAE as the tie-break; it retains
at most two checkpoints and refuses partial smoke reports. No test outcome enters this selection.

`environment/evaluate_v2_phase2c_baselines.py` runs either accepted Candidate E or the trivial
hold-position policy on the exact sealed identities. The latter uses only the current 9D deployed
proprioceptive vector to hold the seven arm joints and mean gripper position. It is a benchmark
sanity check, not a learned baseline. Both outputs bind the same schedule bytes used by SmolVLA.

## Outcome-free evaluation identities

Preparation freezes 40 balanced validation episodes and a sealed schedule containing 30 separate
train-distribution sanity episodes plus 50 episodes for each required held-out split. Stored held-
out identities are retained once. When a Phase 2B split has fewer than 50 identities, supplemental
seeds and scene-group IDs are generated deterministically and must satisfy that split's unchanged
assignment rule. They are never duplicated stored trials.

`test_visual_shift` applies the frozen deployable-RGB transform
`gamma=0.92, gains=[0.82,0.96,1.08], offset=0.025, clamp=[0,1]`. The transform was fixed before
learned outcomes and changes neither simulator state nor task predicates. Other splits use the
base camera. Held-out language uses the Phase 2B instruction stored in the schedule rather than
the canonical task string.

## Current execution status

The local source, adapter, training/validation launchers, sealed schedules, paired baselines, and
CPU contract tests are implemented. The available RTX 5090 host has LeRobot 0.6.0 but not the
accepted external Phase 2B roots. The prior LangMani data host is currently unreachable. Therefore
no Phase 2B rerun, real batch, checkpoint, or
closed-loop learned-policy result is claimed here yet. Training must not start from copied,
regenerated, or unverified substitutes.
