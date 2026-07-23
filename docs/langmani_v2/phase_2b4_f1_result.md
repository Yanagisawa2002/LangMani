# Phase 2B.4-F1 result

F1 is `RESULT_C - PPO redesign fails`.

The residual mapping is mathematically and operationally valid and materially improves local
initial motion. The complete F1 pipeline is also valid. It nevertheless produced zero training
successes and its fixed Probe A evaluation had zero correct contacts, zero target-progress episodes,
and seven wrong-object interactions. The hard stop blocked Probe B. This is an algorithm/design
failure for the custom PPO teacher under the current task formulation, not infrastructure failure
or an assertion that PPO cannot solve arbitrary pushing tasks.

The final state is:

```text
ppo_safe_curriculum_eligible = false
ppo_full_training_authorized = false
expert_qualification_authorized = false
data_collection_authorized = false
smolvla_training_authorized = false
demonstration_source_validated = false
student_policy_training_started = false
```

The custom PPO teacher route is frozen. There is no PPO F2, Probe C, reward/architecture retry, or
budget increase. A future separately authorized route should use a standard task with an existing
validated expert or official demonstrations; independently validated teleoperation is the fallback
when no such source exists. Full PPO training, formal qualification, collection, LeRobot, SmolVLA,
and student work remain unauthorized.
