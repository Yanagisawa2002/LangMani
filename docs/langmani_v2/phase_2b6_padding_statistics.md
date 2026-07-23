# Phase 2B.6 Padding Statistics

The padding audit was computed from the complete frozen source-length and split metadata before
physical production. It defines a future consumer contract but does not authorize training and is
not normalization evidence.

| Horizon | Chunks | Chunks with padding | Chunk fraction | Padded timesteps | Timestep fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 254,374 | 27,000 | 0.106143 | 135,000 | 0.053071 |
| 16 | 254,374 | 45,000 | 0.176905 | 360,000 | 0.088452 |
| 50 | 254,374 | 147,000 | 0.577889 | 3,675,000 | 0.288945 |

The explicit mask feature is `action_is_pad` with boolean dtype. Padding uses a zero action vector,
but zero values alone never identify padding. Any future masked objective must exclude all masked
timesteps. The audit covers distribution by task, split, and chunk position in
`artifacts/langmani_v2/phase_2b6/padding_audit.json`.

No canonical train-only state or action normalization was computed because production stopped
before the statistics stage. The task-balance metadata is source-schedule advice only:
train episodes are 700 per task, while natural train-frame proportions are approximately 30.72%
PickCube, 42.12% StackCube, and 27.16% PushCube. Uniform-task sampling was the preregistered future
recommendation, but no sampler or optimizer was instantiated.
