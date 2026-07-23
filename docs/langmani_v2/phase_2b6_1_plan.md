# LangMani 2.0 Phase 2B.6.1 forensic plan

## Scope

Phase 2B.6.1 is a single-trajectory, no-training forensic phase. Its only target is official
`StackCube-v1` source episode 938, with accepted episodes 936 and 937 reserved as Mode-A controls.
It starts from Phase 2B.6 source commit
`617e222b706ac8ba40410f817b71fd4b014bf41f` and preserves physical producer commit
`73b2cd1517e7c3d2bfc08bd2898e46a3195eb1c9` as provenance.

The preregistered execution budget was:

1. one fresh-process physics-only Mode-A run for control 936;
2. one fresh-process physics-only Mode-A run for control 937, only if 936 passed;
3. three fresh-process Mode-A runs for target 938, only if both controls passed;
4. two observation Mode-B runs only if every target Mode-A run passed;
5. one isolated-writer Mode-C run only if both Mode-B runs passed.

Every run had explicit reset, action, step, simulator, canonical-success, source-agreement,
alignment, observation, state, timestamp, temporary-writer, and serialization sub-gates. The
producer has no separate stable-success acceptance gate: final-step canonical success is the
frozen rule, while consecutive-success timing is diagnostic.

## Isolation and stop rules

The frozen production root
`/root/autodl-tmp/langmani-external/phase2b6/production_v1` was opened read-only. New execution was
directed to
`/root/autodl-tmp/langmani-external/phase2b6_1/stack938_forensics_v1`. Stack episodes 939 and later,
PushCube, LeRobot conversion, normalization, archive/restore, accepted-package creation, policy
loading, optimizer creation, backward, and training were inaccessible.

The first control attempt stopped during `gym.make`, before reset or action submission, because
the SAPIEN process could not create a Vulkan instance. The mandatory control-failure stop rule
therefore ended physical execution immediately. Control 936 was not retried; 937 and 938 were not
run; Modes B and C were not run. The terminal classification is Result D, not a conclusion about
episode 938.

## Evidence and verification

Compact evidence is under `artifacts/langmani_v2/phase_2b6_1/`. The independent verifier rehashes
all listed files, recomputes Result D and its eligibility state, confirms the hard stop, and
requires every production, package, and training authorization to remain closed:

```bash
python environment/verify_v2_phase2b6_1.py
```

Any future attempt requires a separate authorization and a new forensic run identity. It must
first add a native Vulkan/SAPIEN construction gate using the working ICD and must not reinterpret
this Result D as permission to resume Phase 2B.6.
