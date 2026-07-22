# Phase 2B.2 push expert report

## Current gate status

The Phase 2B.2 expert gate has not passed, so formal v2 data collection remains prohibited. No
SmolVLA training, backward pass, optimizer step, tiny overfit, Phase 2C.2, or Phase 2D work has
started.

## Candidate E

Candidate E completed the first frozen 100-episode v2 gate at source
`cd97c4f7cb48be5d45569a5729926fe7ddaa3075`. It achieved 75/100 and was rejected. Its immutable
full report SHA-256 is `a6a8296a9df7c716698920d782c8e6f0aead6f61f18c228dcadba0a13a1c9489`.

## Candidate F

Candidate F was implemented at `6277d77227a51d39d675e328c4df7c233f302484` and evaluated from
clean source `f441b1b035007f0876dafd2a54886b26f6350e40` on the untouched seed interval 66100 through
66199. All 100 episodes completed, but only 11 succeeded. The status counts were 74 timeouts, 10
wrong-object interactions, four planning failures, and one contact failure. All 74 timeouts reached
the unchanged 250-step limit. Eighty failures occurred in `primary_push` and nine in
`move_to_precontact`.

The exact zero-tolerance safety checks remained clean: simulator errors, non-finite actions,
action-bound violations, and workspace violations were all zero. Candidate F is nevertheless
rejected because its 11% success rate is below the frozen 95% gate. The bounded 4 cm primary-push
segments repeatedly execute complete planner trajectories and dominate the available horizon.
Formal collection did not start.

The compact rejection summary is
`artifacts/langmani_v2/phase_2b2/expert_evaluation_candidate_f_rejected.json`. The immutable full
remote report SHA-256 is `94975a838063c4e20891cf478bf775127eaf6a04fbcfcd990caef415fad3087d`.

## Candidate G diagnostic rejection

Candidate G was locally validated at `24709f1702914e72131245484d4405ac68a6303c` and ran only the
diagnostic seed interval 66200 through 66231 from clean source
`3db2c057a4b9be76a76e3ab3eb9f6874a3d1eb6a`. It achieved 22/32. There were four planning
failures, two timeouts, two wrong-object interactions, one correction failure, and one target
workspace exit. Six failures occurred during `move_to_precontact`; four occurred during the two
corrective phases.

The probe was explicitly `formal_gate_eligible=false` and never consumed the frozen formal seeds
66300 through 66399. Its full report SHA-256 is
`e049ec4b5e0a4ab0a6a187fb938fa402759a0a61e79b8a0f3fd54fafa15a6ba9`. Candidate G is rejected
without a formal gate, and collection remains closed. The compact summary is
`artifacts/langmani_v2/phase_2b2/expert_probe_candidate_g_rejected.json`.

## Candidate H diagnostic rejection

Candidate H was locally validated at `62db322a6954a362078badcb5f298501cb533bcf` and ran only the
diagnostic seeds 66400 through 66463 from clean source
`48ab184cb172a5d4533dcff8e2effb16006c2dbd`. It achieved 38/64. Fifteen timeouts and three
correction failures exposed the cost of full-resolution free-space approach execution: successful
precontact phases averaged 104.9 steps. Five verification failures, one planning failure, one
wrong-object interaction, and one target workspace exit were also preserved.

Candidate H is rejected without using formal seeds 66300 through 66399. The full report SHA-256 is
`a896a8e5645b7e9c63237c94f9cc56359d2b5db9e703eed0cb815c2d3eadbc4c`; the compact summary is
`artifacts/langmani_v2/phase_2b2/expert_probe_candidate_h_rejected.json`. Formal collection remains
closed.

## Candidate I pending gate

Candidate I is locally validated at `3da6ecf5afccc09eae5fe69ece9170647a3357a0`. Its diagnostic
range is 66500 through 66563 and cannot promote the expert. The untouched formal gate remains
66300 through 66399. A diagnostic run writes an explicit partial report and stops once a
zero-tolerance failure or an arithmetic success ceiling below 95% makes formal promotion
impossible. Collection remains closed.

## Candidate I diagnostic rejection

Candidate I stopped after 16/64 diagnostic episodes as predeclared: 12 successes plus all 48
remaining episodes could reach only 60/64, below 95%. Its two timeouts, one verification failure,
and one initial-lift planning failure were preserved; simulator, non-finite action, action-bound,
and workspace counts were all zero. The formal 66300--66399 schedule remains untouched. The full
partial report SHA-256 is `599ff261a1e05c6cd5c5405586fa3e4e0484c7a3bda3154a13bf40a637dd519a`,
and the compact summary is
`artifacts/langmani_v2/phase_2b2/expert_probe_candidate_i_rejected.json`. Collection remains closed.

## Candidate J pending diagnostic

Candidate J is locally validated at `d757d91fa7c12f8c341f692118d8d45882623834`. Its isolated
diagnostic range is 66600--66663; it is not eligible for promotion. Formal seeds 66300--66399 and
all collection seeds remain sealed. Collection cannot start until a later complete formal gate
reaches at least 95/100 with every zero-tolerance count at zero.

## Candidate J diagnostic rejection

Candidate J ran from clean source `750d71bc00ba0c5db3d67cc15e0391b125df03d0` on diagnostic
seeds beginning at 66600. It stopped after 14/64 episodes with 10 successes and four failures. One
rightward cylinder correction pushed the target outside the workspace, so the predeclared
zero-tolerance rule stopped the probe immediately. The other failures were one cube contact event,
one corrective TCP tracking failure, and one zero-step contact-planning failure. The full report
SHA-256 is `246b77dfd380a32562afaf53345aba3a8e1612257117373770fcfd60dcead6b3`.
Formal seeds and collection remain untouched.

## Candidate K pending gate

Candidate K restores dense controller execution for contact establishment and correction re-contact,
adds a fixed two-waypoint contact fallback only after a zero-step direct plan failure, uses a 2 cm
cylinder interior margin, and bounds the rare segmented fallback to eight 6 cm advances. It changes
neither the 250-step limit nor any task, action, or acceptance gate.

Candidate K passed 39 focused runtime-contract tests and Ruff validation at implementation commit
`413710deb15a1139b3ac830387c7543d7839c7f7`. The v2 contract binds that exact source. Its isolated
diagnostic range is 66700--66763; it is not eligible for promotion. Formal seeds 66300--66399 and
all collection seeds remain sealed.

## Candidate K diagnostic rejection

Candidate K ran from clean source `1ec0001f505e33686af01e1d013c1db0aa3881bd` on diagnostic
seeds beginning at 66700. It stopped after 5/64 episodes with three successes, one timeout, and one
target-workspace exit. The zero-tolerance workspace event occurred during the first cylinder/left
standard task and stopped the probe immediately. The full report SHA-256 is
`268b925950e7e6e97fe717bb69b7403d579e278910af653fc58429dabc0c9c5f`. Formal seeds and
collection remain untouched.

## Candidate L pending diagnostic

Candidate L adds object-specific pre-containment braking to the unchanged Candidate K motion. It
uses the existing privileged target-distance evaluation only inside the expert, stops the current
planner action stream before overshoot, holds the current joint state, and lets the existing
correction phases finish the task. It does not alter full-containment success, the 250-step horizon,
action bounds, or any formal schedule.

## Candidate K diagnostic rejection

Candidate K ran from clean source `1ec0001f505e33686af01e1d013c1db0aa3881bd` and stopped after
five diagnostic episodes. Three succeeded, one exhausted 250 steps during re-contact, and one
leftward cylinder coasted outside the workspace while the first correction was repositioning.
The zero-tolerance stop preserved formal seeds and collection. The full report SHA-256 is
`9502be405b1cbb7839392e16136ce56401ce7c75df5aa9ff13e55ad6c4e104ea`.

## Candidate L pending gate

Candidate L adds a fixed object-specific precontainment braking margin. During a physical push, the
expert stops planner execution before full containment, commands the current measured joint state
for the existing settle window, and lets the unchanged environment decide whether stable success
forms. No target state, success flag, action bound, horizon, or acceptance criterion changes.

Candidate L passed 39 focused runtime-contract tests, Ruff, format, and changed-module mypy at
implementation commit `c7361c60daeaa50740bccee213de9485a4c078ef`. Diagnostic seeds
66800--66863 are isolated from the untouched formal and collection schedules.
