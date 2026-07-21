# LangMani 2.0 Phase 1: stable evaluation foundation

Phase 1 adds a versioned evaluation boundary around one recovered v1 controller without changing
v1 behavior, data, weights, schedules, gates, or evidence.

## Taxonomy

`pick_and_place` is one skill family. Its six task instances are the Cartesian product of three
object IDs (`red_cube`, `green_cube`, `blue_cube`) and two destination IDs (`left_bin`,
`right_bin`). A task instance is therefore a parameterization of one skill, not a new skill.

The canonical catalog is `configs/langmani_v2/tasks/pick_and_place_v0.json`. Project-owned immutable
types validate that catalog and expose stable skill-family and task-instance IDs.

## Generic policy contract

The `PolicyAdapter` boundary accepts an `ObservationBatch` and returns an `ActionChunk`. An action
chunk carries a finite `[horizon, action_dimension]` tensor plus a prefix-valid mask. The unified
evaluator never imports or calls a language router.

`ActPerTaskPolicyAdapter` is the first real adapter. It reuses the existing v1 LeRobot ACT loader,
preprocessor, inference path, postprocessor, and explicit action-bound processor. It does not copy
ACT internals. Its runtime manifest records:

- adapter and policy identity;
- checkpoint path and cryptographic hashes;
- supported skill family and task instance;
- observation and action contracts;
- normalization, preprocessing, and postprocessing identities;
- execution horizon, device, dtype, and historical source configuration.

Only the recovered red-to-left controller is registered in Phase 1. Future architectures must
implement the same adapter contract; they are not pre-registered or faked here.

## Unified evaluation

`scripts/evaluate_policy_v2.py` loads a policy by registry ID, validates task compatibility, creates
the M1 environment, and evaluates deterministic seeds with `num_envs=1` and `pd_joint_pos`.
Observations are reconstructed through the existing M4 policy-observation path. Runtime actions are
projected by `BoundedActionEnvPostprocessorV0(mode="project")`; raw and executed actions are never
silently conflated.

Each run atomically promotes these files:

- `runtime_manifest.json`
- `episodes.jsonl`
- `summary.json`
- `complete.json`

Per-episode evidence includes success, timeout, wrong-object interaction, target loss, invalid
action, episode length, projection counts, policy-query count, inference latency, environment-step
latency, task identity, skill identity, policy identity, run fingerprint, and checkpoint
fingerprint.

## Commands

```bash
python scripts/validate_v1_release.py

python scripts/evaluate_policy_v2.py \
  --policy-config configs/langmani_v2/policies/act_per_task_red_left_v0.json \
  --task-config configs/langmani_v2/tasks/pick_and_place_v0.json \
  --task-id langmani-pick-place-task-v0:red_cube:left_bin:canonical_v0 \
  --seeds 41001 41002 41003 \
  --output-root outputs/diagnostics/v2/phase1/act-red-left-three-seed
```

The second command is a native-target command. A Windows/CPU run or a cloud instance without a
visible GPU/Vulkan device must not be reported as a real ManiSkill policy smoke.

## Acceptance boundary

Phase 1 implementation acceptance requires the v1 release validator, CPU-safe tests, formatting,
lint, and build. Runtime acceptance additionally requires one canonical controller on one task over
three deterministic seeds on the native target, with real model output and all four persisted run
files. Runtime success rate is reported honestly; a low policy success rate is not itself an
infrastructure failure.

## Observed Phase 1 validation (2026-07-21)

- The release validator passed locally and on the Linux artifact host. `v1.0.0` resolves to
  `58434cb17a7234b6d4b2c4fb15aecf8df0621487`; all seven frozen files and declared identities
  validated.
- The three deployable checkpoint files on the artifact host matched the SHA-256 values in the
  repository-controlled adapter configuration.
- The full CPU-safe suite passed with 1396 tests, 15 platform-capability skips, and 16 explicitly
  deselected GPU/rendering tests. Ruff formatting/checks and isolated sdist/wheel construction
  passed.
- The requested three-seed real-policy smoke has **not run yet**. The available artifact host was
  booted without `/dev/nvidia*`, and `nvidia-smi` was unavailable; the second known cloned host was
  unreachable. This is an infrastructure block, not a policy-quality result. No historical v1
  success count is reused as Phase 1 output.

Consequently, the source implementation and artifact recovery are validated, while Phase 1 runtime
acceptance remains pending the documented three-seed command on a GPU/Vulkan-capable native host.
