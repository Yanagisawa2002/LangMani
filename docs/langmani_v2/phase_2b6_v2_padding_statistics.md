# LangMani 2.0 Phase 2B.6-v2 padding and statistics

All padding statistics use the 2,998 accepted policy episodes only. Canonical
normalization uses only the 2,099 accepted primary-train episodes and their
177,718 frames. Validation, test, visual-shift copies, and excluded sources do
not influence normalization.

## Action padding

The mask feature is boolean `action_is_pad`; the padding value is the zero
action vector. LeRobot 0.6-compatible masked loss behavior passed.

| Horizon | Total chunks | Chunks with padding | Chunk fraction | Padded timesteps | Timestep fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 254,200 | 26,982 | 10.6145% | 134,910 / 2,542,000 | 5.3072% |
| 16 | 254,200 | 44,970 | 17.6908% | 359,760 / 4,067,200 | 8.8454% |
| 50 | 254,200 | 146,902 | 57.7899% | 3,672,550 / 12,710,000 | 28.8950% |

Padding fingerprint:
`sha256:04d75352e7e407a4278527aaafa32c3a0439f14f3d0a445daf1698ec50dde342`.

## Episode lengths

| Task | Episodes | Transitions | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PickCube-v1 | 1,000 | 77,976 | 49 | 78 | 77.976 | 103 |
| StackCube-v1 | 999 | 107,315 | 80 | 106 | 107.4224 | 412 |
| PushCube-v1 | 999 | 68,909 | 61 | 69 | 68.9780 | 81 |

## Canonical normalization

Identity:
`LangManiOfficialMultiSkill-v2-primary-train-pooled-natural-frame-v0`.
The weighting is natural primary-train frame frequency. The state and action
statistics each contain 177,718 samples. Neither pooled state nor pooled
action has a low-variance dimension under the frozen rule.

State mean:

```text
[0.00266113, 0.58468150, -0.00071537, -1.91761752,
 0.00058514, 2.50063071, 0.13200161, 0.02166682, 0.02165595]
```

State standard deviation:

```text
[0.12429607, 0.20230728, 0.06040285, 0.29919135,
 0.05839265, 0.16763679, 0.84573646, 0.01582167, 0.01582364]
```

Action mean:

```text
[0.00280315, 0.59370121, -0.00072968, -1.91179071,
 0.00058449, 2.50328037, 0.13235500, -0.25654126]
```

Action standard deviation:

```text
[0.12660424, 0.20815069, 0.06151716, 0.30811306,
 0.05921227, 0.17065222, 0.84848610, 0.96653328]
```

The gripper action is binary in the accepted train view: 62.8271% negative,
37.1729% positive, and 0% zero. Its endpoint rate is 100%.

Normalization fingerprint:
`sha256:9e08cc0cc6f719298655d250e75b59fda85f9a8414fa3d53e99397a6c4ee58a6`.
Dataset-statistics fingerprint:
`sha256:4a08aab7231e6ea5e3caa68f3aabd8b466bb0b4e2efc0b9eca3d6613b32ae92f`.

## Task balance

| Task | Train episodes | Train frames | Natural-frame probability | Uniform-task frame multiplier |
| --- | ---: | ---: | ---: | ---: |
| PickCube-v1 | 700 | 54,608 | 30.7273% | 1.0848 |
| StackCube-v1 | 700 | 74,891 | 42.1404% | 0.7910 |
| PushCube-v1 | 699 | 48,219 | 27.1323% | 1.2285 |

Uniform-episode probabilities are 33.3492%, 33.3492%, and 33.3016% for Pick,
Stack, and Push respectively. The recommended initial multi-skill sampling
scheme is uniform-task sampling at one third per task. Natural-frame sampling
would over-represent the longer StackCube episodes.

Task-balance fingerprint:
`sha256:b8438fbe5246ba695c22ef71ae0b5d18af2038a22d94ff2d25815a55fc258d5b`.
This recommendation is metadata only; no sampler or optimizer was run.
