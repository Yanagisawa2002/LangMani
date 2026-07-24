# Phase 2B.6.1-v2 control report

The recovered Phase 2B.6.1-R process contract passed before replay. It used the explicit EGL
NVIDIA ICD, selected the RTX 5090, and passed `vulkaninfo`, the minimal SAPIEN probe, and one
zero-step `StackCube-v1` construction. That preflight submitted no action and performed no
explicit reset or step.

Both preregistered accepted controls then passed in their sole permitted fresh-process Mode A
run:

| Control | Source actions | First canonical success | Terminal successful steps | Final success | First failed gate |
| --- | ---: | ---: | ---: | --- | --- |
| StackCube 936 | 82 | step 77 | 6 | true | none |
| StackCube 937 | 92 | step 88 | 5 | true | none |

For both controls, environment construction, configuration, reset, exact source action contract,
all-step execution, simulator-exception, final canonical success, source/replay agreement, and
action-count gates passed. The official producer has no separate stable-success acceptance gate;
the terminal consecutive-success counts above are diagnostics and
`stable_success_gate=not_applicable`.

The controls used distinct process identities, fresh environments, the same accepted launcher,
and exact official source actions. Their success authorized only the fixed three Mode A
repetitions for episode 938.
