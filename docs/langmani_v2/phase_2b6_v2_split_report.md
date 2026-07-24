# LangMani 2.0 Phase 2B.6-v2 split report

The primary split is an episode-level partition over immutable source
identities. Excluded sources remain in the complete source inventory and are
not replaced.

## Source and accepted counts

Each cell is `accepted / source`.

| Task | Train | Validation | Unseen reset | Unseen language | Visual-shift source | Total accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PickCube-v1 | 700/700 | 100/100 | 100/100 | 50/50 | 50/50 | 1,000 |
| StackCube-v1 | 700/700 | 100/100 | 99/100 | 50/50 | 50/50 | 999 |
| PushCube-v1 | 699/700 | 100/100 | 100/100 | 50/50 | 50/50 | 999 |
| Total | 2,099/2,100 | 300/300 | 299/300 | 150/150 | 150/150 | 2,998 |

Accepted frame counts are:

| Split | Frames |
| --- | ---: |
| Train | 177,718 |
| Validation | 25,251 |
| Unseen reset | 25,579 |
| Unseen task language | 12,811 |
| Primary visual-shift source | 12,841 |
| Total | 254,200 |

Stack episode 938 accounts for the one unseen-reset exclusion. Push episode
202 accounts for the one train exclusion.

## Language

Every task has six training templates, two validation templates, and four
held-out paraphrase templates. Assignment is deterministic from episode and
split identity. No trajectory is copied into another primary split to attach a
different sentence. The language manifest fingerprint is
`sha256:5a0774b90847b4ea1a43c4c603d9f3cbd283622f47cb1733880d6356fc457997`.

## Visual shift

The physical-equivalence pilot passed 3/3. Full visual-shift materialization
contains 150 evaluation-only episodes and 12,841 frames:

| Task | Episodes | Frames | Repository identity |
| --- | ---: | ---: | --- |
| PickCube-v1 | 50 | 3,941 | `langmani/official-pickcube-v2-visual-shift` |
| StackCube-v1 | 50 | 5,441 | `langmani/official-stackcube-v2-visual-shift` |
| PushCube-v1 | 50 | 3,459 | `langmani/official-pushcube-v2-visual-shift` |

The shift is post-render appearance only. Physics, source actions, policy
states, and timestamps are unchanged. These roots contribute zero training
episodes.

## Metadata-only cross-skill folds

No media is duplicated.

| Fold | Train skills | Train episodes | Held-out skill | Test episodes |
| --- | --- | ---: | --- | ---: |
| `fold_holdout_pick` | stacking, planar_pushing | 1,399 | pick_and_place | 300 |
| `fold_holdout_stack` | pick_and_place, planar_pushing | 1,399 | stacking | 299 |
| `fold_holdout_push` | pick_and_place, stacking | 1,400 | planar_pushing | 300 |

These are alternate metadata views. They do not establish unseen-skill
performance; a separately authorized model would have to be trained for each
fold before such a claim.

## Leakage audit

Primary split overlap is zero across source trajectory IDs, derived episode
IDs, reset IDs, action hashes, relative media paths, and media file hashes.
Train/held-out language-template overlap is zero. Visual-shift lineage matches
all 150 expected source episodes and has zero training-source overlap.
Cross-skill folds are metadata-only and introduce no primary leakage.

Key fingerprints:

- accepted split:
  `sha256:5bb231f808008630ef81417d8a4cc15d1ec1e132293844546d8755d0b7ee285c6`
- cross-skill folds:
  `sha256:06369e4d5d8653e57dfe6d482540c625667b0067ad7fd51ce002d666e1d8383f`
- leakage audit:
  `sha256:19515e727f0654415a3125b79ee916ab97eec7080efadeba0c8337946dcad7bc`
