# Phase 2C results: execution status

This page is intentionally not a performance claim. Phase 2C.1 exhaustively audited accessible
asset locations but did not recover the original byte-identical accepted Phase 2B dataset. It
therefore closes as **Result B** with zero optimizer steps, no checkpoint, no learned-policy
evaluation, and no Phase 2D entry. See `phase_2c1_dataset_asset_audit.md`.

## Recruiter-readable status

1. **Model:** the official LeRobot 0.6.0 `lerobot/smolvla_base`, pinned to revision
   `c83c3163b8ca9b7e67c509fffd9121e66cb96205`, with the nested SmolVLM2 dependency pinned to
   `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; fine-tuning has not started.
2. **Push data:** the accepted training contract is 203 episodes and 26,968 frames. The original
   accepted bytes were not found; equivalent regeneration is prohibited.
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

- Official base bytes and the nested VLM bytes are hashed. A strict CUDA construction loaded
  450,046,176 parameters under the exact single-camera/9D-state/8D-action contract.
- The official LeRobot CLI accepted the frozen training command and reached dataset creation. The
  deliberate empty-root probe then failed at missing `meta/info.json` with zero optimizer steps.
- The explicit one-camera/9D-state/8D-action adapter rejects malformed or non-finite output and
  uses action bounds in reject mode.
- Model chunk length 50 is separate from validation-selected execution horizons 1, 4, or 8.
- Validation-only checkpoint selection and the development competence gate are machine-enforced.
- Sealed identities and the paired Candidate E/hold-position comparisons are content-bound before
  outcomes.
- Local and Linux focused tests pass; the real data/model/simulator integration test remains gated
  on the accepted external Phase 2B roots and a trained checkpoint.

Machine-readable sanitized evidence is in `phase_2c_base_model_manifest.json`. This is a base-model
and CLI preflight result, not a learned-policy result.

Measured checkpoint, validation, sealed-test, latency, confidence-interval, expert-comparison, and
quality-level claims remain unavailable. They may be added only in a separately authorized stage
after the exact immutable dataset bytes are recovered and independently reverified.
