# Phase 2B.6.1-R zero-step StackCube report

## Environment contract

After the explicit Vulkan and minimal SAPIEN gates passed, each fresh process constructed exactly
one `StackCube-v1` environment using:

```text
robot: Panda task default
num_envs: 1
control_mode: pd_joint_pos
obs_mode: rgb
reward_mode: none
render_mode: None
sim_backend: physx_cpu
render_backend: sapien_cuda
sensor width/height: 256/256
```

The constructor was allowed to initialize its scene internally, as ManiSkill requires, but the
phase issued no explicit task reset, physics step, control step, action, policy query, or frame
acquisition.

## Three fresh-process runs

| Run | PID | `gym.make` | Native action contract | SAPIEN render device | Close |
| --- | ---: | --- | --- | --- | --- |
| 1 | 6490 | passed | `float32[8]`, `pd_joint_pos` | RTX 5090, `0000:27:00.0` | passed |
| 2 | 6818 | passed | `float32[8]`, `pd_joint_pos` | RTX 5090, `0000:27:00.0` | passed |
| 3 | 7148 | passed | `float32[8]`, `pd_joint_pos` | RTX 5090, `0000:27:00.0` | passed |

All three observation spaces contained the expected Panda agent state, TCP pose, and 256-by-256
RGB base-camera and hand-camera structures. Every recorded contract check passed, and the
environment closed normally in each independent process.

The enforced execution counters were:

```text
explicit_reset_count=0
explicit_step_count=0
action_submission_count=0
policy_frame_count=0
forensic_replay_started=false
production_resumed=false
```

No episode 936, 937, or 938 replay was started. The original forensic runner was not invoked for
physical replay, and no frozen Phase 2B.6 byte changed.

## Repeatability and cleanup

Fresh-process repeatability passed 3/3. Every run used the same ICD hash
`sha256:faa5543269860bb750bab4c3a9cb08a8c749f6d27500780b773dfcd334c188f8`,
the same RTX 5090, the same PCI binding, and the same environment/action-space contract.

Post-run cleanup found no surviving phase Python or renderer process, no replay or dataset writer,
no export/archive/restore/training process, no phase-created tmux session, no NVIDIA compute
process, and zero MiB reported GPU memory use. The frozen production files selected for direct
before/after hashing remained byte-identical.

This is an infrastructure construction result only. It establishes neither StackCube replay
compatibility nor episode-938 determinism, task success, source agreement, alignment, timestamp,
serialization, or writer behavior.
