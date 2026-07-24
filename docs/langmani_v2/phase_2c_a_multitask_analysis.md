# Phase 2C-A multitask analysis

The shared seed-0 ACT training pipeline completed and used strict uniform-task sampling: each of
8,712 batches contained 136 Pick, 136 Stack, and 136 Push examples. Each task contributed exactly
1,184,832 effective samples. Its selected validation loss was 0.107751 and its selected per-task
raw L1 values were 0.032349 Pick, 0.044145 Stack, and 0.012217 Push.

These numbers show that the shared model learned nonconstant task-conditioned offline predictions;
they do not establish closed-loop task competence or task-condition use. Phase 2C-A stopped on the
selected Pick per-task policy before shared closed-loop evaluation. Consequently:

- shared success is unavailable;
- per-task versus shared success differences are unavailable;
- multi-task interference is unavailable;
- correct/incorrect/shuffled Task-ID sensitivity is unavailable;
- shared seed 1 was not authorized or started.

No natural-language generalization claim is made. The shared condition remains a public three-way
task ID only.
