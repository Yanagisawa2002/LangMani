# Phase 2B.6.1-v2 failure analysis

## Localized failure

The exact earliest failed sub-gate is:

```text
canonical_terminal_success_gate
```

The primary category is:

```text
canonical_success_failure
```

There are no secondary contributing factors in the bounded evidence.

The failure is not source ambiguity, Vulkan preflight, environment construction, reset, action
contract, simulator exception, or repetition-level intermittency. The source/action identities
were exact, both accepted controls passed, all three target runs submitted all 105 actions, and
the three target outcomes and terminal pose diagnostics were identical.

The replay did reach the official canonical predicate, but one action step later than the source:
source first success is one-based step 100, whereas every replay first success is step 101. Source
success occupies its six terminal steps and remains true at step 105. Replay success occupies
action indices 100--103 and becomes false at the final action index 104. Under the unchanged
producer rule `final_step_canonical_success`, this is a failure. The later
`source_replay_outcome_agreement_gate` also fails because source final success is true and replay
final success is false.

The committed producer contains no separate stable-success acceptance gate. Therefore this result
is not `PHYSICAL_REPLAY_VALID_BUT_PRODUCTION_SUCCESS_GATE_FAILED` under an invented consecutive
success threshold. Consecutive success is diagnostic only.

## Historical relationship

Phase 2B.6's aggregate wrapper did not preserve the inner rejected record, so this phase does not
claim byte-for-byte proof that the old wrapper failed for the same internal reason. It does show
that the exact source/action input reproducibly fails the explicit fresh-process frozen
final-success contract after infrastructure recovery. The historical Result C remains immutable;
the new bounded result is independently Result B.

Because Result B is conclusive, Mode B and Mode C were prohibited. Observation, alignment,
serialization, and writer paths therefore do not explain the new failure and remain untested for
episode 938.
