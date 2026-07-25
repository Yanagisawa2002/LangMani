# LangMani 2.0 Phase 2C-B language analysis

## What was tested

The shared-language GPU micro-overfit completed and produced different actions for different
instructions on at least one identical observation
(`same_observation_instruction_max_abs_difference=1.921348`). The shared input contract contained
natural-language task text and no one-hot, task index, or other task-ID model feature.

This is evidence that the bounded shared micro path carries language into the model. It is not
evidence that language changed behavior correctly or improved task success.

## What was not tested

Full shared SmolVLA training was conditional on the Pick competence gate. Pick remained 0/30 after
the only permitted repair, so the following stages were not authorized and did not run:

- correct/wrong-skill/blank/shuffled language interventions;
- same-observation action comparisons on a full shared checkpoint;
- held-out task-language paraphrases;
- shared validation or final closed-loop evaluation.

The language-intervention identities were frozen before validation results under fingerprint
`sha256:5ef9cd0fdfc75dfdf72f719eabe68a9eb40d852ab86c1c408119838f09aaab6b`,
but they were never opened for execution.

## Conclusion

Phase 2C-B demonstrated micro-level instruction sensitivity only. It did not establish instruction
grounding, paraphrase generalization, correct skill selection, or language-conditioned robotic
success. Those metrics are unavailable, not zero. Result D prohibits another SmolVLA tuning phase
or automatic VLA-JEPA/LatentGuard work.
