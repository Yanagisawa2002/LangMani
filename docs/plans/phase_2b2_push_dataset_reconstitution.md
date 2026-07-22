# Phase 2B.2 push dataset reconstitution plan

## Objective

Create a newly collected and independently identified `langmani/phase2b-push-v2` dataset without
recovering or impersonating v1. Keep the native ManiSkill episodes as raw authority, action-replay
every generation-accepted trajectory in a fresh environment, export only replay-passing episodes
through LeRobot 0.6.0, then content-address and restore-test the data before any future training.

## Ordered gates

1. Freeze the v2 contract and retire v1 from runtime use.
2. Pass a disjoint 100-episode expert gate at at least 95% success with zero simulator, finite-action,
   action-bound, and workspace violations.
3. Collect a resumable 650-attempt initial schedule and, only after replay, one bounded 160-attempt
   top-up if the 500 preferred target or minimum strata remain deficient.
4. Fresh-environment action replay of every generation-accepted episode.
5. Export at least 400 accepted successes and 50,000 frames with real LeRobot metadata/readback.
6. Run full quality, integrity, duplicate, split-leakage, and native/archive identity checks.
7. Build content-addressed raw and LeRobot archives and restore-test them.
8. Verify replication and one real SmolVLA loader batch with zero optimizer steps.

## Assumptions

- The accepted Candidate E implementation remains an ancestor of the clean collection commit.
- The target runtime remains NumPy 1.26.4, mplib 0.1.1, ManiSkill 3.0.1, LeRobot 0.6.0, and one
  native NVIDIA GPU.
- Six historical semantic splits are retained because they are stricter than a single 80/10/10
  random split and preserve the existing Phase 2C development/external-evaluation boundary.

## Exclusions

- No v1 byte reconstruction or identity reuse.
- No failed episode enters the default imitation-learning view.
- No metadata editing after export.
- No SmolVLA backward pass, optimizer step, tiny overfit, formal training, checkpoint work,
  closed-loop evaluation, final access, Phase 2C.2, or Phase 2D.
- Raw episodes, videos, LeRobot data, archives, caches, and full runtime logs remain outside Git.
