# Phase 2B.5 LeRobot 0.6 Pilot

The conversion target is a separate environment with LeRobot `0.6.0` and PyTorch `>=2.7`; no
dependency is installed into the planner runtime. Only five replay-validated episodes per selected
task are converted.

Each LeRobot frame contains video RGB, Panda state, action, task ID, and the canonical task string.
LeRobot owns episode index, frame index, and 20 Hz timestamp. The real public API creates and
finalizes the pilot, after which every frame is read back to verify video decoding, shapes,
finiteness, tasks, episode boundaries, frame order, and timestamps. This pilot is not the full
dataset.

The exact environment identity, episode/frame totals, conversion manifest, readback result, and
future action-padding arithmetic are recorded in:

- `lerobot_060_environment_manifest.json`
- `conversion_pilot_manifest.json`
- `lerobot_readback_result.json`
- `padding_audit.json`

Historical ACT used a different frozen data distribution and `H=50`. The official multi-skill
tasks, reset distribution, camera content, lengths, and language supervision differ, so a fair
future LeRobot 0.6 comparison requires retraining under one identical contract; no ACT retraining
occurs here.
