# Phase 2B.3 expert report

## Outcome

Phase 2B.3 is **Result C - MPC architecture invalid**. The native simulator clone/restore
precondition failed, so expert implementation and quality evaluation stopped before the 24-episode
development pilot. This is an architecture-validity result, not an expert success-rate result.

## Required questions

1. Candidate E/F/P were insufficient because they reached 75/100, 11/100, and diagnostic 2/5;
   none passed the unchanged 95/100 plus zero-safety gate.
2. Public simulator state cloning is deterministic between cold sandboxes but does not reproduce a
   live continuation. The final error was 4.394 mm in target-object pose.
3. The frozen contract proposed six deterministic cube/cylinder-aware angle/distance candidates;
   it was not activated.
4. Candidates would have been evaluated by 12 real physics steps in isolated environments; zero
   candidate rollouts were admitted after the clone failure.
5. Unsafe candidates would have been hard-vetoed using the seven authoritative zero-tolerance
   fields; the veto was not exercised by a pilot.
6. The small explicit score was frozen in `scoring_manifest.json`; no physical score was used.
7. The intended real prefix was three steps followed by replanning, bounded to 40 decisions; no
   MPC decision was executed.
8. Development prediction accuracy is unavailable. Its prerequisite clone comparison failed
   despite 100% categorical agreement because continuous state diverged materially.
9. Expert runtime p50/p95 is unavailable because no expert episode was run.
10. The 95/100 formal gate was not accessed.
11. Formal safety counts are unavailable, not zero, because formal evaluation was not run.
12. Phase 2B.4 data production is not authorized; Phase 2C.2 and SmolVLA remain blocked.

## Execution boundary

Only seed 69000 was used to obtain a successful reference trace for the clone precondition. The
development pilot consumed zero scheduled episodes as a pilot, and formal seeds 66300--66399 were
never reset. No raw episode, frame, dataset, checkpoint, model load, backward pass, or optimizer
step was created.
