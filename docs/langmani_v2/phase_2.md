# LangMani 2.0 Phase 2: pushing expert gate

## Status

Phase 2 is authorized and its pushing task plus privileged expert are implemented. The required
Phase 1 native smoke and independent entry gate pass. The real pushing environment was created,
stepped, rendered, reset deterministically, and exercised with vectorized RGB observations on the
RTX 5090 target. Phase 2A's unchanged fixed expert gate now passes conjunctively: standard reached
46/50 (92%) against 90%, while hard reached 23/30 (76.7%) against 70%. The exact smoke reached 8/8
standard and 6/8 hard, the standard forward subset remained 24/24, and workspace-exit/action-bound
counts were both zero.

Demonstration collection is the next separately authorized stage. No pushing data, unified dataset,
SmolVLA integration, or training was started in Phase 2A.

## Frozen task contract

`PushTaskSpec` defines two contact geometries (`blue_cube`, `orange_cylinder`), four parameterized
target regions (`left`, `right`, `forward_left`, `forward_right`), and explicit `standard`/`hard`
difficulty. Scene seed remains separate from semantic task identity.

The batched evaluation contract keeps stable success, full-region containment, workspace exit,
lift, topple, grasp, wrong-object displacement/contact, overshoot, action validity, projection,
collision, and stall as separate tensor events. The expert uses explicit semantic handles, a high
precontact transit, low planar contact, and a full-containment-derived endpoint. No expert state is
added to visual policy observations.

The orange cylinder intentionally lies with its local-`x` symmetry axis horizontal. Rotation about
that axis is normal rolling and is not a topple. Target markers lie flat on the table. Left/right
region centers use `x=0.12`; forward-left/forward-right use the distinct reachable `x=0.22`.
Standard and hard containment radii remain `0.11` and `0.085`. Phase 2A changed none of these task,
success, or schedule contracts.

## Phase 1 entry gate

The accepted native Phase 1 run used Git `aa599dd`, seeds 41001/41002/41003, one RTX 5090, EGL and
the NVIDIA Vulkan ICD. It completed 3/3 episodes, 40 real ACT queries, and 392 real environment
steps. `environment/verify_v2_phase1.py` independently rehashed the canonical checkpoint and
returned:

```text
v1_release_validated=true
phase1_runtime_evidence_validated=true
three_seed_schedule_validated=true
real_policy_inference_validated=true
real_environment_steps_validated=true
phase2_authorized=true
passed=true
```

The bounded Docker/Windows attempts and one GLX launch failure remain invalid infrastructure
diagnostics. They are not physical evidence and do not affect the accepted native result.

## Pushing expert validation

The baseline producer at Git `500b09ce9d1cf10f7ac2f6f585a5fb8efed9e686` used the pinned planner
runtime (`numpy==1.26.4`, `mplib==0.1.1`) with PhysX CUDA and the NVIDIA Vulkan ICD. It reached 8/8
standard and 5/8 hard on smoke, then 41/50 and 21/30 on the fixed target gate. An earlier NumPy
2.2.6 launcher artifact remains invalid zero-step infrastructure evidence.

Phase 2A accepted Candidate E at Git `59ca88e9f0514187252a6286ab1b8e06c4318fb4` in the same pinned
runtime and on the same fixed schedules:

| Schedule | Standard | Hard | Conjunctive gate |
| --- | ---: | ---: | --- |
| Full lateral diagnostic | 22/26 (84.6%) | 11/16 (68.8%) | non-gating |
| All-task smoke | 8/8 (100%) | 6/8 (75%) | passed |
| Fixed target validation | 46/50 (92%) | 23/30 (76.7%) | passed |

Wilson 95% intervals for target validation are 81.2%-96.8% for standard and 59.1%-88.2% for hard.
Standard has four remaining cylinder verification failures. Hard has three cylinder verification
failures and four wrong-object precontact interactions. Standard forward tasks remain 24/24. The
target report records zero timeouts, planner failures, workspace exits, and action-bound events.

Two earlier uncommitted probes remained rejected at 10/17. Candidate A was rejected after creating
six cylinder workspace exits; Candidate B after three standard workspace exits; Candidate C after
its endpoint bound proved too late to prevent the re-contact sweep; and Candidate D after the smoke
remained 5/8 hard. Those artifacts are diagnostic rather than accepted quality evidence.

## Phase 2A side-push protocol

The bounded recovery first replays failed lateral episodes from the immutable baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/diagnose_push_expert.py \
  --baseline-report outputs/diagnostics/v2/phase2/push-expert-target-validation-500b09c.json \
  --output outputs/diagnostics/v2/phase2/side-push-failure-replay.json
```

The phase-boundary diagnostic trace is expert metadata, never a policy observation. It records
contact geometry, progress, contact loss, workspace margin, planner/action events, final distance,
and a non-timeout root cause. The public interface does not expose a stable contact-manifold point,
so the recorded first-contact location is explicitly a TCP-center proxy.

The full lateral check preserves the exact target schedule before selecting only `left` and `right`:

```bash
CUDA_VISIBLE_DEVICES=0 python environment/benchmark_push_expert.py \
  --lateral-subset --sim-backend physx_cuda \
  --output outputs/diagnostics/v2/phase2/push-expert-lateral-subset.json
```

It contains 26 standard and 16 hard episodes and sets `quality_gate_applicable=false`. A candidate
may proceed only from that diagnostic to the unchanged exact smoke and fixed 50/30 target gate.

Candidate E uses a small deterministic scored approach set. Cube lateral pushes retain the measured
15-degree signed compensation. Cylinder lateral pushes keep ideal geometry and disable the unsafe
re-contact sweep; incomplete pushes still fail unchanged verification. Planned joint actions are
checked against an explicit expert-only copy of the environment bounds before execution. Only a
zero-step precheck rejection may try the next already-scored approach. No action is clipped,
projected, or hidden.

## Next ordered stage

The expert gate no longer blocks Phase 2. The next separately authorized step is deterministic
pushing demonstration collection under the unchanged task, seed, action, and success contracts.
No dataset work was run in Phase 2A, and SmolVLA remains downstream of an accepted data archive.
