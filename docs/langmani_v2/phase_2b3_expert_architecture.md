# Phase 2B.3 simulator-MPC architecture disposition

## Intended boundary

`SimulatorMPCPushExpertV1` was specified as a deterministic receding-horizon expert. At each
decision it would generate at most six geometry-aware candidates, reject invalid full plans,
physically simulate 12 control steps in isolated sandboxes, hard-veto every safety event, score the
remaining rollouts, execute only three real steps, and replan. Cube and horizontal-cylinder contact
angles, distractor clearance, and deterministic tie-breaking were separate explicit contracts.

The frozen state-machine vocabulary was `initialize`, `select_contact_mode`, `plan_precontact`,
`evaluate_candidates`, `execute_prefix`, `assess_progress`, `replan`, `verify`, and `terminate`.
The proposed bounds were 40 MPC decisions, two recontacts, three no-progress replans, and one
planner retry inside the unchanged 250-step episode budget.

## Why it was not implemented

Physical candidate evaluation is valid only if a sandbox initialized from state S predicts the
main environment's continuation from the same S. The clone audit proved cold-sandbox determinism
but disproved live-continuation equivalence. Implementing the state machine after that failure
would turn the sandbox into a different transition model and violate the main research question.

No hand-written transition estimate, looser tolerance, state teleport, action projection, or
seed-specific fallback was substituted. The candidate generator, score, and safety-veto code in
`phase2b3.py` remains contract scaffolding only; no `SimulatorMPCPushExpertV1` runtime exists and no
physical candidate was scored.
