# Phase 2B.3 comparison with Candidate E

Candidate E remains the historical best formal baseline. It completed an independent frozen
100-episode gate and was rejected at 75/100. The geometry expert did not reach an independent
development gate: it stopped during an already-consumed historical failure regression diagnostic.
The two success rates are therefore not directly comparable.

| Metric | Candidate E formal gate | Geometry expert |
| --- | ---: | ---: |
| Formal/independent episodes | 100 | 0 |
| Formal/independent successes | 75 | not run |
| Historical diagnostic episodes | not applicable | 12 |
| Historical diagnostic successes | not applicable | 2 |
| Planning failures | 7 | 0 in the 12-case diagnostic |
| Timeouts | 7 | 9 in the 12-case diagnostic |
| Verification failures | 7 | 0 in the 12-case diagnostic |
| Wrong-object interactions | 4 | 1 in the 12-case diagnostic |
| Workspace violations | 0 | 0 in the 12-case diagnostic |
| Action violations | 0 | 0 in the 12-case diagnostic |
| Simulator errors | 0 | 0 in the 12-case diagnostic |
| Mean episode length | 134.00 | 215.25 in the diagnostic |
| Mean recovery attempts | 0.12 | 0.75 in the diagnostic |

Candidate E's 25 failures were 7 planning failures, 7 timeouts, 7 verification failures, and 4
wrong-object interactions. The geometry expert's 10 diagnostic failures were 9 timeouts and 1
wrong-object interaction. Its two successes were cylinders; boxes were 0/6 and cylinders 2/6.
The small, selected, historical regression prefix cannot support a shape-generalization claim.

The geometry implementation passed a stronger explicit engineering contract than Candidate E:
shape-aware support geometry, a shrunken safe workspace, pre-execution path checks, a versioned
state machine, closed-loop verification, and bounded recovery. That engineering improvement did
not translate into an acceptable expert. It consumed more episode steps, retained systematic
horizon/contact-loss failures, and introduced one zero-tolerance wrong-object displacement. The
fair conclusion is rejection and pivot, not that 2/12 is an alternative formal score.
