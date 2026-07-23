# Phase 2B.3.1-RR call-sequence audit

Reset-replay must reproduce environment construction, reset and options, controller
initialization, observation ordering, action submission, physics steps/substeps,
termination checks, sensor/render/camera updates, elapsed-step handling, dtype conversions, and
backend identity.

The surviving Phase 2B.3 reports contain useful runtime and action-window metadata, but do not bind
a complete call sequence to executable original action arrays. In particular, the original
observation-read and sensor/render timing cannot be reconstructed with content identity from the
compact reports alone.

Consequently:

- `physics_only_replay` was not defined or run;
- `full_call_sequence_replay` was not defined or run;
- no assumption was made that sensors or rendering are physically irrelevant;
- no reset option, call, or action was regenerated from current expert code;
- no protocol identity was frozen.

The ambiguity is material because the research question explicitly depends on exact historical
execution semantics. The transcript hard stop therefore precedes all reset and simulator work.
