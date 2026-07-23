# Phase 2B.4-F1 safe curriculum design

## Stage 0 - contact acquisition

F1 may run only Stage 0:

- target object: blue cube;
- directions: left and forward-right;
- difficulty: Standard;
- distractor: unchanged Standard distractor in its canonical safe location;
- environment: `LangMani-PushToRegion-v0`;
- robot/control: Panda, native `pd_joint_pos`;
- physics, object/target geometry, stable-success predicate, and 250-step limit: unchanged.

Probe A advances only with zero wrong-object and workspace events, action integrity, at least 25%
correct-contact episodes, and at least 20% target-directed-progress episodes. Probe B's final gate
requires at least 20% success, at least one cube success in both directions, at least 50% contact,
at least 40% progress, and zero safety/action-integrity events.

## Stage 1 - geometry and direction expansion

Stage 1 would add the cylinder, all canonical directions, and full Standard variation. It is
defined for design completeness but is not runnable or authorized in F1.

## Stage 2 - full task distribution

Stage 2 would add Hard scenes, distractors across the full task distribution, and complete
Standard/Hard variation. It is not runnable or authorized in F1.

Any zero-tolerance event or failed gate rolls back by hard-stopping F1. Prohibited
simplifications include changing physics, success, or episode length; direct object control;
scripted fallback; motion planning; and deleting the distractor from the environment
implementation.

## F1 outcome

Stage 0 was the only stage executed. Probe A failed its independent gate with 0/32 correct-contact
episodes, 0/32 target-progress episodes, seven wrong-object interactions, and zero workspace exits.
Stage 1, Stage 2, and Probe B therefore remained sealed and were not run.
