# LangManiOfficialMultiSkill-v2 dataset card

## Summary

`LangManiOfficialMultiSkill-v2` is a replay-derived, nonprivileged,
LeRobot 0.6-compatible dataset for three Panda manipulation skills:
PickCube-v1 (`pick_and_place`), StackCube-v1 (`stacking`), and PushCube-v1
(`planar_pushing`).

It contains 2,998 accepted episodes and 254,200 primary policy frames from
3,000 immutable official source episodes. Two source episodes are retained in
the source/exclusion inventories but omitted from policy roots under the
versioned deterministic-physical exclusion class.

Accepted package fingerprint:
`sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`.

## Data roots

| Dataset identity | Episodes | Frames | Tree digest |
| --- | ---: | ---: | --- |
| `langmani/official-pickcube-v2` | 1,000 | 77,976 | `sha256:7d0a5665cd8337d8f3664bc11cc029e5cbe6a0b3ec3c909c772b330e6fcb4738` |
| `langmani/official-stackcube-v2` | 999 | 107,315 | `sha256:d0546aa0edf64463f8d35b69cc2250a011b387e6dd9fdc4c9093b9841a375c36` |
| `langmani/official-pushcube-v2` | 999 | 68,909 | `sha256:cd9156115e1fa8416c26ea29b14a64a4e0cbcb720472d9f13b9ada59ac32b2cf` |

The immutable unified multi-root index fingerprint is
`sha256:191c81c75c1985cfbdba5cd77ac7d906498dd77ef5284471afb3b5a2969e8253`.
It references the three task roots rather than destructively concatenating
them.

## Features

| Feature | Contract |
| --- | --- |
| `observation.images.base_camera` | uint8 RGB video, 256 x 256, 20 Hz |
| `observation.state` | `PandaPolicyStateV0`, float32[9] |
| `action` | native Panda `pd_joint_pos`, float32[8] |
| `task` | deterministic split-controlled instruction |
| metadata | task, skill, source/derived episode, frame, timestamp, template, split identities |

The frame convention is `observation[t]` immediately before `action[t]`.
There is no fabricated terminal action. Privileged object/goal state, full
simulator state, reward, success internals, and contacts are not student
features.

## Splits and evaluation views

Primary source counts per task are 700 train, 100 validation, 100 unseen
reset, 50 unseen task language, and 50 visual-shift source episodes.
Accepted counts differ only where Stack episode 938 is excluded from unseen
reset and Push episode 202 is excluded from train.

The separate post-render visual-shift roots contain 150 evaluation-only
episodes and 12,841 frames. They do not change physics, actions, policy state,
or timestamps and contain zero training episodes.

Three metadata-only skill-holdout folds are supplied. They duplicate no media
and do not by themselves support an unseen-skill claim.

## Quality and lineage

- source identity and action identity are exact for every accepted episode
- 2,998/2,998 source-to-derived action sequences are exactly equal
- 2,033,600 action scalar values checked without approximate tolerance
- full LeRobot API readback and decoding: 3,148 episodes, 267,041 frames,
  zero failures
- primary split overlap: zero
- complete source accounting: 3,000 = 2,998 accepted + 2 excluded
- content-addressed archive and independent clean restore: passed

The two exclusions are:

- StackCube-v1 episode 938, final canonical success false after transient
  success at action indices 100–103
- PushCube-v1 episode 202, final canonical success false

Both executed all exact source actions in a valid environment, had no
simulator/action/writer failure, were not retried, and first failed
`canonical_terminal_success_gate`.

## Intended use

The dataset is eligible as input to separately authorized ACT-baseline,
SmolVLA, and VLA-JEPA work. Initial multi-skill experiments should prefer
uniform-task sampling because natural frame sampling gives StackCube 42.14%
of train frames.

Eligibility is not authorization. This phase contains no trained model,
checkpoint, optimizer state, backward pass, or learned-policy result. All
training authorization flags remain false.

## Limitations

- The dataset covers only three official ManiSkill tasks and one Panda
  `pd_joint_pos` action contract.
- The two deterministic physical exclusions expose source/replay final-success
  incompatibilities; they must not be silently reintroduced as successful
  policy data.
- Visual shift is a deterministic post-render evaluation view, not a new
  physical domain.
- Skill-holdout folds are metadata only; future models and evaluations are
  required for transfer claims.
- Dataset acceptance is an integrity result, not a policy-quality result.

## Evidence

The compact, source-controlled evidence lives in
`artifacts/langmani_v2/phase_2b6_v2/`. The artifact manifest fingerprint is
`sha256:aff53479e161a3aba19035364c147bc3926d906efa85b9f7d0ccc9db681cd20b`;
the independent verifier fingerprint is
`sha256:80a50c759425c999e02812e95249d3ef606be9526a6a64fdba6ffecc4b734b6e0`.
