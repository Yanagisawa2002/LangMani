# Phase 2B.5 ManiSkill Demonstration Mechanism Audit

## Pinned mechanism

The qualification is pinned to ManiSkill `3.0.1`, upstream source revision
`a4a4f9272ad64b1564035874b605ceb687b63ed8`, and the official Hugging Face dataset
`haosulab/ManiSkill_Demonstrations` at revision
`d674485bbffdd533914e52d272fdda34c0515608`. The dataset card declares Apache-2.0 for
demonstration data. The selected tasks use built-in procedural rigid-body assets; no third-party
asset byte is treated as demonstration data.

The installed download registry, not a documentation example, was inspected. The five required
task archives exist. Every downloaded ZIP is retained unchanged, byte-sized, SHA-256 addressed,
and safely expanded into a separate working directory only after rejecting absolute paths,
parent traversal, and symlink members.

## Actual formats

The official JSON sidecar contains `env_info` and per-episode records including episode ID, seed,
control mode, elapsed steps, reset kwargs, and source success. HDF5 contains one
`traj_<episode_id>` group per record. Each group stores the actual executed action array, transition
arrays of length `T`, and recursive actor/articulation environment-state arrays of length `T+1`.
Compressed motion-planning sources intentionally contain no policy RGB observation.

The installed replay command is:

```text
python -m mani_skill.trajectory.replay_trajectory
```

It can replay actions, use the first stored environment state, replay state sequences, render
cameras, record a changed observation mode, and perform supported CPU-side control-mode conversion.
Phase 2B.5 uses no conversion. CPU and GPU simulation can randomize or continue differently, so the
qualification separately records semantic-reset error and then uses the official
`--use-first-env-state` equivalent before replaying actions on `physx_cpu`.

## Boundaries

Official replay success is read from each unmodified task's own `info["success"]`. A source is not
made compatible by changing its task predicate, action bounds, action timing, or terminal handling.
The compact authoritative mechanism record is
`artifacts/langmani_v2/phase_2b5/maniskill_demo_system_manifest.json`.
