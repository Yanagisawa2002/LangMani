# Phase 2B.6.1-v2 physical replay report

## Exact input

The official target is `StackCube-v1`, `traj_938`, source episode 938, seed 962. The source input
was independently proven before construction:

```text
source ZIP sha256:f9b7d34b9aa418a04aa8e4322d4dea5aa27e8ae81757f60b210c1ffc54bf9c1b
source HDF5 sha256:61d07df7b52f303d2088e977df5f863be7d2274348ff05791f3a815359815523
source JSON sha256:48cb1119c9f5a8f0d7187fc83674a2c401b41466cb1930b2a5c26b706ba3bd81
reset identity sha256:e396b1aea2fe017ae24ab29115f5e3702fa1a37d6d34f0f9d0629f188f44410b
trajectory identity sha256:f5c25c7b4f1f7f5ed0cd8b3641f7a451f758c1a47eb9aa31bf96b0081848f7a3
actions float32[105,8]
action sha256:5fc50bc7beeb91e55b2eeccf3f016414811d314142e7a80ad6037d4f83d92eb8
```

The source first becomes canonically successful at one-based step 100, remains successful for
the six terminal steps, and has final success true.

## Three fresh-process Mode A results

All three repetitions executed all 105 source actions without simulator exception and produced
the same categorical and continuous diagnostics:

| Repetition | First success | Successful interval | Final success | Maximum terminal translation error | First failed gate |
| --- | ---: | --- | --- | ---: | --- |
| A1 | step 101 | action indices 100--103 | false | 0.892462 mm | `canonical_terminal_success_gate` |
| A2 | step 101 | action indices 100--103 | false | 0.892462 mm | `canonical_terminal_success_gate` |
| A3 | step 101 | action indices 100--103 | false | 0.892462 mm | `canonical_terminal_success_gate` |

The first gripper/cube contact was action index 39, first lift was 51, first canonical placement
relation was 97, first cube/cube contact was 99, first canonical success was 100, and final action
index was 104 in all three repetitions. At first canonical success the upper cube height was
0.059997309 m. Robot qpos/qvel, TCP pose, both cube poses and velocities, gripper/cube forces,
cube/cube force, relative displacement, official task diagnostics, and canonical success were
retained at the required event snapshots.

The official predicate is authoritative. Diagnostic geometry does not replace it. The replay
reached transient canonical success for four steps but lost success at the final action, so it
fails the frozen producer rule. There is no separate frozen stable-success gate: the observed
four-step interval is recorded, while `stable_success_gate` remains `not_applicable`.

The repetitions are classified `deterministic_failure`. They have no categorical or recorded
task-transition divergence and no terminal-pose drift between repetitions. Source final success
true versus replay final success false makes `source_replay_outcome_agreement_gate` fail after the
earlier final-success gate.
