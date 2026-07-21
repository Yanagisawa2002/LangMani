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
| A | 15-degree compensation plus 4 cm cylinder contact | 8/16 recovered; 3 new standard workspace exits | Standard 20/26, Hard 11/16; 6 workspace exits | not run | not run | not run | rejected |
| B | cube-only 15-degree compensation; baseline cylinder contact/direction | pending | pending | protected | pending | pending | provisional |
| C | Candidate B plus 3 cm workspace-bounded cylinder corrections | pending | pending | protected | pending | pending | provisional |
| D | Candidate B with unsafe cylinder re-contact disabled | pending | pending | protected | pending | pending | provisional |
| E | Candidate D plus pre-execution joint-bound fallback | pending | pending | protected | pending | pending | provisional |

The behavior-neutral replay classified the 16 lateral failures as seven contact losses, two
verification-boundary cases, two workspace-margin violations, and one each of overshoot, primary
planner failure, unreachable initial approach, premature wrong-object contact, and action-bound
violation. Failed left pushes deviated toward world `+x` by approximately 14--18 degrees; failed
right pushes showed the opposite signed angular error but the same world-`+x` bias. Candidate A is
the single predeclared response to that measured asymmetry, not a parameter sweep.

Earlier uncommitted probes that combined a deeper endpoint with longer settling, or removed the
separate correction lift, each reached only 10/17 and introduced workspace or forward regressions.
They remain rejected and are not silently restored.

Candidate A recovered five of nine failed standard lateral seeds and three of seven failed hard
lateral seeds, but the complete subset exposed five new standard cylinder workspace exits and one
hard cylinder workspace exit. Its protected forward and official gates were therefore not run.
The next bounded diagnostic replays only those fixed candidate-A failures to separate the signed
direction compensation from the newly deeper cylinder contact and correction behavior.

That replay localized all six workspace exits to the cylinder: two occurred during the primary
push and four during correction. Candidate B therefore retains compensation only for the cube and
restores the cylinder's original 4.5 cm ideal-direction contact. This is a geometry-specific
ablation from phase evidence, not a new angle or distance sweep.

Candidate B improved the complete subset to Standard 22/26 and Hard 10/16, but it still increased
standard workspace-exit terminal labels from zero to three, including one new regression at seed
44044. It is rejected before smoke. Candidate C retains its primary-push geometry but bounds only
lateral-cylinder corrections to the smaller of 3 cm, remaining target progress, and footprint-safe
workspace headroom. The existing two-correction budget can cover the observed 3.5--5 cm residual.

Candidate C reproduced all ten Candidate-B failures at the same terminal steps, proving that the
cylinder was swept before the bounded endpoint was reached. Its unused controls are removed.
Candidate D makes lateral-cylinder correction phases no-motion fail-safe phases; incomplete primary
pushes must still pass the unchanged final verification and therefore remain honest failures.

Candidate D retained the 22/26 and 10/16 lateral result with zero workspace exits, but smoke stayed
8/8 and 5/8. Candidate E addresses the smoke's cylinder-right action violation at its originating
layer: reject the complete joint plan before stepping it, then try the next fixed safe approach.
No action is clipped or projected.

## Gate

Dataset collection and SmolVLA remain unauthorized unless the unchanged fixed evaluation reaches
Standard at least 45/50, Hard at least 21/30, the 16-task joint gate passes, the standard forward
subset remains 24/24, and no systemic workspace/action-bound regression appears.
