# LangMani 2.0 Phase 2: precondition status

## Status

Phase 2 is authorized to begin but is not yet implemented. The required Phase 1 native smoke and
independent entry gate now pass. This status is not a claim that the multi-skill benchmark or
SmolVLA baseline already exists.

The first implementation checkpoint defines an immutable `PushTaskSpec` with two contact
geometries (`blue_cube`, `orange_cylinder`), four parameterized target regions (`left`, `right`,
`forward_left`, `forward_right`), and explicit `standard`/`hard` difficulty. Its batched evaluation
logic keeps stable-success count, full-region containment, workspace exit, lift, topple, grasp,
wrong-object displacement/contact, overshoot, action validity, projection, collision, and stall as
separate numeric/tensor events. Simulator integration and physical validation remain pending.

## Minimum recovery implemented

`environment/verify_v2_phase1.py` independently validates:

- the immutable `v1.0.0` release manifest;
- the repository-controlled canonical ACT adapter configuration;
- the three deployable checkpoint artifact hashes;
- the exact four-file Phase 1 evidence layout;
- at least three unique deterministic seeds in exact order;
- compatible policy, checkpoint, task, skill, action shape, and execution horizon identities;
- at least one real policy query and one real environment step in every episode;
- observed policy-inference and environment-step latency;
- internally consistent summary and completion records.

The verifier is intentionally independent of policy loading and simulator execution. It cannot
create missing evidence and cannot convert historical v1 results or infrastructure attempts into a
Phase 1 rollout. Any missing, malformed, incompatible, or changed input leaves
`phase2_authorized=false` and returns a nonzero exit code.

## Current observed result

On 2026-07-21 the accepted native run produced 3/3 successful episodes, 40 real policy queries, and
392 real environment steps. The independently recomputed flags were:

```text
v1_release_validated=true
phase1_runtime_evidence_validated=true
three_seed_schedule_validated=true
real_policy_inference_validated=true
real_environment_steps_validated=true
phase2_authorized=true
passed=true
```

No push task, pushing expert, pushing dataset, SmolVLA integration, model training, or closed-loop
Phase 2 evaluation existed at the moment of authorization. Phase 2 must now proceed in its declared
order beginning with the standard `push_to_region` task.

## Exact unblock sequence

On a native Linux host with one visible NVIDIA GPU, Vulkan rendering, the canonical checkpoint,
and the pinned LangMani runtime:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/verify_install.py --target

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_policy_v2.py \
  --policy-config configs/langmani_v2/policies/act_per_task_red_left_v0.json \
  --task-config configs/langmani_v2/tasks/pick_and_place_v0.json \
  --task-id langmani-pick-place-task-v0:red_cube:left_bin:canonical_v0 \
  --seeds 41001 41002 41003 \
  --output-root outputs/diagnostics/v2/phase1/act-red-left-three-seed

python environment/verify_v2_phase1.py \
  --evidence-root outputs/diagnostics/v2/phase1/act-red-left-three-seed \
  --output outputs/diagnostics/v2/phase1/precondition-verification.json
```

Resume Phase 2 at the standard `push_to_region` task only when the final command returns zero and
reports `phase2_authorized=true`. Do not start SmolVLA work before that gate and before the pushing
expert and dataset stop conditions are independently satisfied.
