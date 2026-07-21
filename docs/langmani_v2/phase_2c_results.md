# Phase 2C results: execution status

This page is intentionally not a performance claim yet. Phase 2C source and experiment contracts
are implemented, but the accepted external Phase 2B dataset is not currently available on the
reachable RTX 5090 host. The stop rule therefore prevents model training and closed-loop testing.

## Recruiter-readable status

1. **Model:** the official LeRobot 0.6.0 `lerobot/smolvla_base`, pinned to revision
   `c83c3163b8ca9b7e67c509fffd9121e66cb96205`; fine-tuning has not started.
2. **Push data:** the accepted training contract is 203 episodes and 26,968 frames, but its external
   bytes must be reverified before use.
3. **Observation/action:** one RGB 256x256 base camera, 9D Panda joint state, language task string,
   and 8D `pd_joint_pos` action at 20 Hz. No privileged simulator state is allowed.
4. **Closed-loop result:** not available; no trained checkpoint exists.
5. **Generalization:** not evaluated. The frozen suites are unseen scene, unseen language, hard,
   and visual shift, 50 episodes each, plus 30 train-distribution sanity episodes.
6. **Failures:** no learned-policy failures exist yet. The required taxonomy is documented in
   `phase_2c_failure_analysis.md`.
7. **Inference speed:** not measured on the final checkpoint.
8. **Pretraining value:** not established; a real pretrained fine-tune is mandatory before this can
   be answered.
9. **Multi-skill decision:** not authorized. Phase 2D requires a recoverable checkpoint, complete
   sealed evaluation, and at least Level 2 quality.

## Validated implementation evidence

- The explicit one-camera/9D-state/8D-action adapter rejects malformed or non-finite output and
  uses action bounds in reject mode.
- Model chunk length 50 is separate from validation-selected execution horizons 1, 4, or 8.
- Validation-only checkpoint selection and the development competence gate are machine-enforced.
- Sealed identities and the paired Candidate E/hold-position comparisons are content-bound before
  outcomes.
- Local focused tests pass; the real GPU integration test remains a target-runtime gate.

This page must be replaced with measured checkpoint, validation, sealed-test, latency, confidence
interval, expert-comparison, and quality-level evidence after the external data gate opens.
