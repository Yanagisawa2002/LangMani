# Phase 2B.3.1-RR reset-replay feasibility plan

## Purpose and boundary

Phase 2B.3.1-RR asked one infrastructure question: can a cold sandbox reconstruct the
physically relevant transition history by resetting from the original episode contract and
replaying every original native action from step zero? A second gate would have measured whether
that reconstruction was practical enough for a separately authorized MPC pilot.

The phase began from exact MPC Result C commit
`cc31ced97e66220626ef39374b3e3acf84ccd463` on branch
`codex/langmani-v2-phase2b3-reset-replay-equivalence`. The geometry result commit
`ae6aae49c61d96c68de434e08dc1167461a29543` remained an external read-only source and is not an
ancestor of this branch. Their merge base is
`8476a72b3ce7504e7fef0a08114c6c567da77ba7`.

This phase never authorizes an MPC expert, expert qualification, demonstration collection,
LeRobot export, SmolVLA, PPO, or any training.

## Ordered gates

The planned execution order was:

1. prove repository and worktree isolation;
2. rehash the immutable MPC and geometry result packages;
3. establish that complete original `float32[8]` `pd_joint_pos` transcripts exist;
4. freeze episode, boundary, call-sequence, state, tolerance, probe, isolation, and runtime
   contracts;
5. only then create live/cold environments and execute reset-replay comparisons;
6. measure runtime only after valid transition trials survive;
7. classify exactly Result A, B, C, or D and keep authorization separate from eligibility.

Gate 3 failed. The latest evidence node retains action counts, indices, and a content hash for the
12-action clone-audit window, but not the numeric actions themselves. Searches of the relevant
Phase 2B.2/2B.3 diagnostic and dataset paths found no authoritative executable action transcript.
The protocol therefore stopped before episode selection or simulator execution.

## Implemented audit surface

`langmani.v2.phase2b3_rr` is pure CPU contract code. It validates source lineage, transcript
completeness, native action shape/dtype/order and hashing, reset/call identity, state
serialization, quaternion error, numeric and categorical comparisons, first-divergence
detection, isolation fingerprints, runtime projections, result classification, and the mandatory
authorization state. It exposes no simulator, expert, collection, or training runner.

`environment/verify_v2_phase2b3_rr.py` rehashes and verifies the compact Result D package without
constructing a ManiSkill environment.

## Hard-stop outcome

The exact outcome is `RESULT_D / HISTORICAL_PREFIX_UNAVAILABLE`.

This does not alter the historical Phase 2B.3 `RESULT_C / MPC_ARCHITECTURE_INVALID` result and is
not a new physical nondeterminism result. No executable trial protocol was frozen, because the
declared order stops immediately when original action prefixes are unavailable.

All future execution requires a new explicit user instruction.
