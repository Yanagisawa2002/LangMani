# LangMani 2.0 Phase 2B results

## Recruiter summary

LangMani 2.0 now includes a production-grade demonstration pipeline for a second manipulation skill:
a Panda pushes either a cube or a rolling cylinder into one of four target regions under standard
and hard scene conditions. The pipeline preserved all failed attempts, admitted only successful
trajectories that passed independent action replay, exported nonprivileged RGB/state/action data
through the real LeRobot 0.6 API, and proved compatibility with the frozen pick-and-place dataset.

The final result is 397 independently action-replayed demonstrations across two object geometries,
four regions, two difficulty levels, and three language-template groups. No SmolVLA model was
implemented or trained in this phase.

## Results at a glance

| Item | Result |
| --- | ---: |
| Corrected pilot | 22/32 accepted; 22/22 replayed; 2,914 LeRobot frames |
| Full plus bounded top-up attempts | 516 |
| Independently validated demonstrations | 397 |
| Preserved rejected attempts | 119 |
| Cube / cylinder | 208 / 189 |
| Standard / hard | 283 / 114 |
| Left / right / forward-left / forward-right | 90 / 107 / 100 / 100 |
| Action-replay categorical match | 397/397 (100%) |
| Transition-label / frame-count match | 397/397 / 397/397 |
| LeRobot export | 397 episodes / 53,297 frames / 141,422,866 bytes |
| Split leakage | passed; all eight overlap counters are zero |
| Pick/push compatibility | passed; immutable multi-root index ready |
| Phase 2C authorization | data gates satisfied; final independent verifier pending |

## Why the dataset is trustworthy

The authoritative layer is a ManiSkill-native archive with T original actions, T terminal and
success/failure labels, and T+1 states. Collection success alone is insufficient: a separate
process recreates each reset, executes every recorded action without the expert, and independently
recomputes success, failures, terminal pose drift, frame counts, and per-transition labels. Only
replay-passing episodes enter the derived LeRobot dataset.

Splits use stable scene and language groups rather than random frames. The audit checks repeated
seeds, initial-state hashes, trajectory hashes, episode IDs, scene groups, held-out template
placement, and file overlap. Failed attempts remain separately addressable for future safety work;
they are not silently deleted or mixed with successful demonstrations.

## Multi-skill readiness

The historical pick-and-place dataset contains 360 episodes and 64,548 frames. The compatibility
audit loaded one real sample from both sources and confirmed the same 256x256 RGB, 9D Panda state,
8D `pd_joint_pos` action, 20 Hz control rate, and LeRobot task-string path. TaskSpec sidecars remain
skill-specific and normalization must be recomputed from a future exact train-only multi-root view.
The sources themselves remain immutable and are joined only by a content-bound multi-root index.

## Evidence and next step

The detailed audit is [`phase_2b_dataset_audit.md`](phase_2b_dataset_audit.md). Machine-readable
release identities are in `phase_2b_result_manifest.json` and exact source/test/build evidence is in
`phase_2b_source_validation.json`.

The exact next stage is a separately invoked Phase 2C SmolVLA implementation/training milestone
after the final independent Phase 2B verifier passes. Dataset readiness does not imply
learned-policy success, and Phase 2B makes no SmolVLA training or control-quality claim.
