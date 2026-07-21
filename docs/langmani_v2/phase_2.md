# LangMani 2.0 Phase 2: pushing expert gate

## Status

Phase 2 is authorized and its pushing task plus privileged expert are implemented. The required
Phase 1 native smoke and independent entry gate pass. The real pushing environment was created,
stepped, rendered, reset deterministically, and exercised with vectorized RGB observations on the
RTX 5090 target. The fixed expert gate is now complete and failed conjunctively: standard reached
41/50 (82%) against 90%, while hard reached 21/30 (70%) against 70%. The workflow stops before
demonstration collection, unified data, SmolVLA integration, or training.

The first implementation checkpoint defines an immutable `PushTaskSpec` with two contact
geometries (`blue_cube`, `orange_cylinder`), four parameterized target regions (`left`, `right`,
`forward_left`, `forward_right`), and explicit `standard`/`hard` difficulty. Its batched evaluation
logic keeps stable-success count, full-region containment, workspace exit, lift, topple, grasp,
wrong-object displacement/contact, overshoot, action validity, projection, collision, and stall as
separate numeric/tensor events. The expert uses explicit semantic handles, high precontact transit,
low planar contact, a full-containment-derived endpoint, and at most two corrections. No expert
state is added to visual policy observations.

The cylinder is an intentional horizontal rolling geometry: its local-`x` symmetry axis must
remain approximately horizontal, while rotation about that axis is normal push motion. Target
markers are flat on the table. Left/right region centers use `x=0.12`; forward-left/forward-right
use the still-distinct but Panda-reachable `x=0.22`. Standard and hard containment radii remain
`0.11` and `0.085`, respectively.

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
Phase 2 evaluation existed at the moment of authorization. The environment and expert were then
implemented in that declared order. The later expert gate result below supersedes the historical
authorization-time snapshot without changing the Phase 1 evidence.

## Pushing expert validation

The accepted native producer is Git `500b09ce9d1cf10f7ac2f6f585a5fb8efed9e686`, using the pinned
mplib side runtime (`numpy==1.26.4`, `mplib==0.1.1`) with PhysX CUDA and the NVIDIA Vulkan ICD.
An earlier launcher mistakenly used the main NumPy 2.2.6 runtime and produced 16 zero-step
initialization failures; that artifact is invalid infrastructure evidence and is not included in
the rates below.

| Schedule | Standard | Hard | Conjunctive gate |
| --- | ---: | ---: | --- |
| All-task smoke | 8/8 (100%) | 5/8 (62.5%) | failed |
| Fixed target validation | 41/50 (82%) | 21/30 (70%) | failed |

Wilson 95% intervals for the target validation are 69.2%–90.2% for standard and 52.1%–83.3% for
hard. Standard failures were five timeouts, three verification failures, and one planning failure.
Hard failures were two timeouts, three wrong-object interactions, two workspace exits, one
planning failure, and one out-of-bounds execution failure. Forward standard tasks reached 24/24;
the main weakness was lateral pushing, especially the rolling cylinder.

Two bounded diagnostic probes tested a deeper endpoint/longer settle and a compact correction
path. Each produced only 10/17 successes and introduced new workspace or baseline regressions, so
neither was committed and neither is quality evidence. The stop condition remains active.

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

The Phase 1 command already returns zero with `phase2_authorized=true`. Resume Phase 2 only by
making a bounded controller-quality change, rerunning the all-task smoke, and then producing a new
fixed 50/30 report with standard at least 90% and hard at least 70%. Do not start demonstration
collection or SmolVLA work before that expert gate passes.

## Phase 2A side-push stabilization protocol

The bounded recovery starts with a behavior-neutral replay of exactly the failed lateral episodes
from the fixed report:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/diagnose_push_expert.py \
  --baseline-report outputs/diagnostics/v2/phase2/push-expert-target-validation-500b09c.json \
  --output outputs/diagnostics/v2/phase2/side-push-failure-replay.json
```

The diagnostic trace is phase-boundary expert metadata, never a policy observation. A candidate
controller must first improve those fixed failures and the complete lateral subset, then pass the
unchanged 16-task smoke and 50-standard/30-hard gate. The forward standard subset must remain
24/24. Rejected candidate code is removed; diagnostic tooling and its machine-readable root-cause
table may remain.

The intermediate full lateral check is explicitly non-gating and preserves the exact target
schedule order before filtering only `left` and `right`:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/benchmark_push_expert.py \
  --lateral-subset --sim-backend physx_cuda \
  --output outputs/diagnostics/v2/phase2/push-expert-lateral-subset.json
```

It contains 26 standard and 16 hard episodes. Its report sets
`quality_gate_applicable=false`; it cannot replace either official gate.
