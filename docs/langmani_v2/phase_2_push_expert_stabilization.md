# LangMani 2.0 Phase 2A: side-push stabilization

## Frozen baseline

The accepted NumPy 1.26.4 / mplib 0.1.1 producer at Git `500b09c` completed the fixed protocols:

| Protocol | Standard | Hard | Decision |
| --- | ---: | ---: | --- |
| 16-task smoke | 8/8 | 5/8 | rejected by the joint gate |
| 50/30 target validation | 41/50 | 21/30 | rejected; Standard below 45/50 |

Standard forward targets are the protected 24/24 baseline. The main concentration is lateral
pushing, especially `orange_cylinder/left` at 2/6. The baseline terminal counts are five standard
timeouts, three standard verification failures, one standard planning failure, two hard timeouts,
three hard wrong-object interactions, two hard workspace exits, one hard planning failure, and one
hard action-bound failure.

## Diagnostic contract

Phase 2A first replays only the failed lateral seeds from the immutable target report. Each fresh
episode records initial and phase-boundary object/TCP poses, intended contact normal, chosen
precontact/contact points, per-push displacement and progress, phase-boundary contact losses,
planner/action/workspace events, final distance, and one non-timeout root cause. The public
ManiSkill interface does not provide a stable contact-manifold point, so the recorded first-contact
location is clearly labeled as a TCP-center proxy.

## Experiment ledger

| Change | Hypothesis | Lateral replay | Lateral subset | Forward | Smoke | Fixed 50/30 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | geometry-corrected expert | 16/16 reproduced | pending | 24/24 standard | 8/8, 5/8 | 41/50, 21/30 | blocked |
| A | compensate signed lateral drift by 15 degrees; add safe fixed fallback list | 8/16 recovered; 3 new standard workspace exits | pending | pending | pending | pending | provisional |

The behavior-neutral replay classified the 16 lateral failures as seven contact losses, two
verification-boundary cases, two workspace-margin violations, and one each of overshoot, primary
planner failure, unreachable initial approach, premature wrong-object contact, and action-bound
violation. Failed left pushes deviated toward world `+x` by approximately 14--18 degrees; failed
right pushes showed the opposite signed angular error but the same world-`+x` bias. Candidate A is
the single predeclared response to that measured asymmetry, not a parameter sweep.

Earlier uncommitted probes that combined a deeper endpoint with longer settling, or removed the
separate correction lift, each reached only 10/17 and introduced workspace or forward regressions.
They remain rejected and are not silently restored.

## Gate

Dataset collection and SmolVLA remain unauthorized unless the unchanged fixed evaluation reaches
Standard at least 45/50, Hard at least 21/30, the 16-task joint gate passes, the standard forward
subset remains 24/24, and no systemic workspace/action-bound regression appears.
