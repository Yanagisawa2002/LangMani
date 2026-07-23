# Phase 2B.6.1 failure analysis

## Historical evidence

The frozen producer log observes a generic `episode replay gate failed` stop for StackCube source
episode 938. The complete returned inner replay record was not persisted. The episode NPZ is
absent; internal sub-gate, frames, timestamps, writer state, terminal state, and any observed
simulator exception are missing or not recorded. The surrogate historical
`simulator_error_count=1` remains a wrapper field, not an observed exception.

Controls 936 and 937 are historically accepted and retain action equality, successful replay,
frame/action alignment, and zero recorded simulator errors. Those historical facts justified
selecting them as controls; they do not count as new Phase 2B.6.1 control runs.

## New first failure

The earliest new failing layer is `environment_construction`, before the explicit
`reset_identity_gate`. Because no environment existed, the task-level sub-gate order was never
entered. The allowed primary classification is therefore `insufficient_evidence`; the secondary
factor is `environment_construction_failure`.

The failure is infrastructure/runtime configuration related: the imported SAPIEN process selected
its packaged `nvidia_icd.json`, which failed Vulkan instance creation. The legacy system GLX ICD
also failed, while the separately pinned EGL ICD passed `vulkaninfo`. This is sufficient to reject
the forensic pipeline execution, but not sufficient to infer the old episode-938 rejection cause.

## What is and is not established

Established:

- exact source episode 938 and producer input identity are proven;
- immutable Phase 2B.6 and Phase 2B.5 evidence still passes;
- the new Mode-A control attempt made zero action submissions;
- the new runtime failed before physical replay;
- hard-stop scope was preserved;
- no production, package, or training work started.

Not established:

- whether controls reproduce under a valid SAPIEN/Vulkan environment;
- whether episode 938 passes or fails physics-only replay;
- whether episode 938 is deterministic;
- whether canonical or stable success occurs in replay;
- whether source and replay outcomes agree;
- whether observation, alignment, state, timestamp, serialization, or writer layers fail;
- the exact historical Phase 2B.6 inner sub-gate.

The old rejection therefore was not reclassified. Phase 2B.6 remains immutable `RESULT_C`;
Phase 2B.6.1 separately ends as `RESULT_D`.
