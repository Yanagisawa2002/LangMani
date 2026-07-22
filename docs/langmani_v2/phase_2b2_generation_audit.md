# Phase 2B.2 generation-chain audit

## Audited path

The frozen v2 path is `LangMani-PushToRegion-v0` -> a content-bound `PushToRegionExpert`
candidate -> ManiSkill-native HDF5/JSON recording -> independent fresh-environment action replay
-> state-restored RGB/9D-state/8D-action LeRobot 0.6 export. It has no runtime dependency on the
lost v1 bytes. Phase 2B.2 stopped at the expert gate before any recording or export began.

## Contract answers

1. Environment ID: `LangMani-PushToRegion-v0`, registered with a 250-step limit.
2. Embodiment/control: Panda, `pd_joint_pos`, 100 Hz simulation and 20 Hz control.
3. Camera: exactly `observation.images.base_camera`, RGB 256x256, uint8 before LeRobot video
   encoding and CHW uint8 on readback.
4. State: `PandaPolicyStateV0`, ordered as Panda joints 1-7 then finger joints 1-2, float32[9].
5. Action: Panda joints 1-7 position targets in radians followed by the normalized gripper mimic
   command, float32[8]. Frozen bounds come from the native controller; violations are rejected,
   never clipped.
6. Language: three training templates, one validation template, and one held-out paraphrase.
   Rendered instructions bind object and region; canonical TaskSpec remains in sidecars.
7. Expert: each candidate plans and executes native controller actions. No candidate teleports an
   object, writes success, or mutates task state. Candidates E and F ran complete formal gates;
   G--L ran only disjoint diagnostic probes.
8. Success: the target is fully contained, static, upright, ungrasped, stably successful, and has
   no latched failure.
9. Termination: canonical success, canonical failure, or the 250-step limit. Runtime exceptions
   remain execution errors.
10. The frozen data design retains failed trajectories separately and excludes them from training.
    No v2 trajectory was recorded because the expert gate failed.
11. v1 had 397 accepted episodes because 516 bounded attempts yielded 397 real successes after the
    full schedule and one predeclared top-up; no episode was duplicated.
12. Schedule, seed, task, language, and episode identities are deterministic. Physical execution
    must still be revalidated.
13. The unexecuted exporter is implemented against LeRobot 0.6 and would create official metadata,
    parquet/chunks, videos, feature schemas, timing, and loader readback from real v2 episodes.
14. Dataset action equals the executed native `pd_joint_pos` action; no normalization, retargeting,
    padding, clipping, or repair is allowed.
15. No generator or exporter path consumes v1 bytes. v1 is historical comparison evidence only.

## Result

The final clean Candidate L generation audit at
`fc478d6364e5264b6bceb188f4497da02ad991fe` passed with collection fingerprint
`sha256:615102a337041b642a92e818f7e6c7776bd0acecf4b72ad959962cced7f861e9`. Candidate L then
failed its isolated diagnostic success ceiling. The mandatory formal expert gate remains false, so
the atomic collector, replay, exporter, archive, replication, and SmolVLA loader paths were not
executed.
