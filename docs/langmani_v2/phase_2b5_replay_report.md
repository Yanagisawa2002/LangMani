# Phase 2B.5 Official Replay Report

The authoritative protocol and results are:

- `artifacts/langmani_v2/phase_2b5/replay_protocol.json`
- `artifacts/langmani_v2/phase_2b5/bounded_replay_results.json`
- `artifacts/langmani_v2/phase_2b5/strong_replay_results.json`

Every replay uses the official recorded action sequence. It first performs the episode's semantic
reset and records its continuous state error against the official initial state. If and only if
that reset differs, it applies the official first-state fallback; an already exact reset is left
untouched. It then executes every action. It does not invoke a planner, expert, repair, clipping, or
altered success predicate.

The final report records per-task agreement, alignment, invalid actions, simulator exceptions,
first-success timing, semantic-reset error, and terminal continuous error. A source task is selected
only if its 20-episode gate and its stronger 100-episode gate pass unchanged.
