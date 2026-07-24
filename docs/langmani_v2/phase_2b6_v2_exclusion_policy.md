# Phase 2B.6-v2 versioned exclusion policy

The authoritative machine-readable policy is
`configs/langmani_v2/phase2b6_v2_exclusion_policy.yaml`. Its SHA-256 is bound by
the production specification before the first replay.

## Classes

- `ACCEPTED_REPLAY`: every source, action, construction, reset, step,
  final-success, observation/alignment, completeness, writer, and serialization
  gate passes.
- `EXCLUDED_DETERMINISTIC_PHYSICAL`: exact actions complete without
  infrastructure or writer failure, but an explicit frozen physical or final
  success gate fails unambiguously.
- `EXCLUDED_SOURCE_CONTRACT`: source identity, action identity, reset metadata,
  or schema is missing or invalid.
- `RETRYABLE_INFRASTRUCTURE`: a declared environment, Vulkan, transient file,
  writer-initialization, or storage failure. Only one identical retry is
  permitted and the first attempt remains recorded.
- `UNCLASSIFIED_FAILURE`: evidence does not support another class. One such
  failure terminates production.

No episode is silently discarded or replaced.

## Episode 938

`StackCube-v1/traj_938`, seed 962, action SHA-256
`5fc50bc7beeb91e55b2eeccf3f016414811d314142e7a80ad6037d4f83d92eb8`
is preregistered as eligible for `EXCLUDED_DETERMINISTIC_PHYSICAL`, referencing
the immutable Phase 2B.6.1-v2 Result B evidence. It must still receive one
normal v2 production attempt. It is excluded only if that attempt again fails
the final canonical-success gate consistently with the forensic evidence.

The trajectory is not shortened at transient success, modified, repaired, or
retried for physical quality.

## Dataset thresholds

- Overall accepted-source rate: at least 99.7%.
- Per-task accepted-source rate: at least 99.5%.
- At most five exclusions per task and nine overall.
- Three accepted skill families.
- Zero unclassified failures.
- Zero invalid or non-finite actions.
- Zero accepted-episode simulator, alignment, writer, or readback failures.

These values cannot change after production begins.
