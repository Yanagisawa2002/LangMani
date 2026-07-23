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

The isolated environment was
`/root/autodl-tmp/conda-envs/langmani-lerobot060-phase2b5`, with Python 3.12.13, LeRobot 0.6.0,
Torch 2.11.0+cu128, and PyAV 15.1.0. The public writer produced 15 episodes and 1,546 frames. PyAV
readback decoded and checked every frame; image, state, action, task ID, index, and timestamp
failure counts were all zero.

For a future explicit `action_is_pad` mask, `H=10` would pad 135/1,546 chunks and 675/15,460
timesteps, while `H=50` would pad 735/1,546 chunks and 18,375/77,300 timesteps. This arithmetic
does not authorize or execute training. Historical ACT remains context only and would require a
new same-contract training run before comparison.
