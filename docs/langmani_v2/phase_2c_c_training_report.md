# Phase 2C-C training report

Status: implementation and pre-training audit in progress.

The only authorized new optimizer is the seed-0 Pick relative-action SmolVLA. It uses the exact
Phase 2C-B official base/VLM revisions and unchanged 20,000-step optimization configuration.

The final report will record:

- static preparation and 100k transform audit;
- real-batch smoke, backward/optimizer step, save/reload, bounded generation, and one real step;
- 500-step micro-overfit loss, residual error, and reconstructed physical error;
- 5k/10k/20k checkpoint hashes and validation-only diagnostics;
- the selected checkpoint and complete runtime identity.

No Shared, Stack, Push, VLA-JEPA, ACT, or second-seed training is permitted.
