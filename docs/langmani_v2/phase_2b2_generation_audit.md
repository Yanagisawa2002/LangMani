# Phase 2B.2 generation-chain audit

## Audited path

Phase 2B.2 extends the accepted v1 path: `PushToRegionEnv` → `PushToRegionExpert/CandidateE` →
ManiSkill `RecordEpisode` native HDF5/JSON → independent fresh-environment action replay →
state-restored RGB/9D-state/8D-action LeRobot export. It has no runtime dependency on lost v1 files.

## Contract answers

1. Environment ID: `LangMani-PushToRegion-v0`, registered with a 250-step limit.
2. Embodiment/control: Panda, `pd_joint_pos`, 100 Hz simulation and 20 Hz control.
3. Camera: exactly `observation.images.base_camera`, RGB 256x256, uint8 before LeRobot video
   encoding and CHW uint8 on readback.
4. State: `PandaPolicyStateV0` ordered as Panda joints 1–7 then finger joints 1–2, float32[9].
5. Action: Panda joints 1–7 position targets in radians followed by the normalized gripper mimic
   command, float32[8]. The frozen numerical bounds are copied from the native controller action
   space into `dataset_contract.yaml`; actions are rejected rather than clipped.
6. Language: three training templates, one validation template, and one held-out paraphrase
   template. Rendered instructions bind object and region; canonical TaskSpec stays in sidecars.
7. Expert: Candidate E plans and executes controller actions through the environment. It never
   teleports an object, writes success, or mutates task state.
8. Success: the target is fully contained, static, upright, ungrasped, stably successful, and has
   no latched failure.
9. Termination: canonical success, canonical failure, or the 250-step time limit. Runtime exceptions
   remain execution errors.
10. Failed trajectories are retained as a separate failure corpus but excluded from training.
11. v1 had 397 accepted episodes because 516 bounded attempts yielded 397 real successes after the
    full schedule and one predeclared top-up; it was not an integer target fabricated by duplication.
12. The schedule, seed, task, language, and episode identities are deterministic. Physical
    execution is revalidated rather than assumed from determinism.
13. The exporter uses the official LeRobot 0.6 writer to create `meta/info.json`, episode/task
    metadata, parquet/chunk data, H.264 videos, features, timing, and real loader readback.
14. Dataset action is the original executed native `pd_joint_pos` action. It is not clipped,
    normalized, retargeted, padded, or repaired.
15. No generator or exporter path consumes v1 bytes. v1 is historical comparison evidence only.

## Phase 2B.2 additions

The v2 collector commits every completed native episode through an owned staging directory and
atomic rename, verifies completed episode manifests during resume, and never exposes a partial
episode as accepted. Dataset and archive identities are newly computed from v2 bytes.
