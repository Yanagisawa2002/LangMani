# LangMani 2.0 Phase 2C plan

## Objective

Fine-tune the official LeRobot 0.6.0 `lerobot/smolvla_base` model on only the independently
accepted Phase 2B pushing training split, then evaluate the frozen learned policy through the
Phase 1 policy interface in real closed-loop ManiSkill pushing episodes.

## Frozen starting point

- Source branch: `codex/langmani-v2-phase2b-data`
- Source commit: `7394faba1ed2e29ab26b70dbc4058f46e5164041`
- Working branch: `codex/langmani-v2-phase2c-smolvla-push`
- Accepted pushing expert: `59ca88e9f0514187252a6286ab1b8e06c4318fb4`
- Official base model: `lerobot/smolvla_base`
- Resolved base revision: `c83c3163b8ca9b7e67c509fffd9121e66cb96205`

## Gates and execution order

1. Rerun the full Phase 2B verifier against the immutable raw and exported roots.
2. Content-bind the six split identities and prove the exact 203-episode/26,968-frame train view.
3. Recompute state/action normalization from that train view only and inspect low-variance axes.
4. Export image, language, state, and action sanity evidence from real LeRobot samples.
5. Load the pinned official model with the explicit one-camera/9D-state/8D-action feature override.
6. Pass one real batch forward/backward/optimizer/save/reload/adapter/simulator-step smoke.
7. Pass a deterministic 16-episode, 500-step micro-overfit gate.
8. Freeze the largest stable batch size and run the seed-0 20,000-step schedule with 5k/10k/20k
   checkpoints.
9. Select at most two checkpoints and one of execution horizons 1/4/8 using validation only.
10. Open sealed test identities only if the complete development competence gate passes.
11. Run paired Candidate E and no-op references on the exact final identities, classify failures,
    and decide the Phase 2D gate without test tuning.

## Assumptions

- The Phase 2B raw and exported datasets remain external to Git and byte-identical to the
  accepted result manifest.
- Training and simulator environments may be isolated, but their exact versions, checkpoint
  transfer, and inference latency are recorded.
- The official model feature descriptors are overridden to the accepted LangMani embodiment;
  model source is not forked or vendored.

## Exclusions

- No pick-and-place or failed demonstrations enter training.
- No multi-skill work, LatentGuard integration, task/success-predicate change, expert change,
  privileged input, action clipping/projection, manual action repair, or test-set tuning.
- Checkpoints, raw data, image caches, and videos remain outside Git.

## Current external dependency

The online execution host has an RTX 5090 and LeRobot 0.6.0 but does not contain the accepted Phase
2B roots. The previously used LangMani data host is currently unreachable. Real data verification
and all model gates must wait for access to the accepted roots; this is an external-artifact
availability condition, not a failed data or model gate.
