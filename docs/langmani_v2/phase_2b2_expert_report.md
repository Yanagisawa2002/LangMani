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
