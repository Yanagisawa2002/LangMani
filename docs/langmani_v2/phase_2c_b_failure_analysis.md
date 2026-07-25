# LangMani 2.0 Phase 2C-B failure analysis

## Failure boundary

The Phase 2C-B pipeline was valid:

- the frozen dataset identity passed;
- the official base and nested VLM loaded strictly with no reinitialized tensor;
- the embodiment, padding, processor, and bounded-action contracts passed;
- GPU smoke submitted a real action to `env.step`;
- Pick/shared micro-overfits passed;
- Pick full training completed 20,000 optimizer steps;
- all retained checkpoints reloaded strictly;
- all closed-loop actions were finite and bounded;
- no evaluated episode had an invalid action or simulator error.

This rules out Result E and an infrastructure-failure interpretation.

## Initial behavior

The frozen selector chose 20k/H=1 after all six checkpoint/horizon cells tied at zero success. The
initial 30-episode gate had 26 no-motion and four failed-grasp timeouts. The model queried on every
step, but the resulting one-step receding behavior rarely produced object motion.

## Demonstrated configuration defect and repair

On the same checkpoint and frozen six-episode subset, H=8 produced five contact-bearing grasp
attempts while H=1 produced six no-motion failures. This supported one bounded
action-chunk-execution-horizon repair. H=8 reduced no-motion failures from 26/30 to 13/30 and
increased failed-grasp attempts from 4/30 to 16/30. One episode reached an object-drop failure.

The repair therefore changed behavior in the predicted direction: it enabled more coherent
multi-step motion and contact. It did not solve the task.

## Dominant competence failure

The repaired gate had:

- 16 failed grasps;
- 13 no-initial-motion failures;
- 1 object drop;
- 0 successes;
- 0 invalid actions;
- 0 simulator errors.

The dominant residual failure was grasp acquisition and stabilization, with a substantial secondary
no-motion mode. Offline action error continued to improve through 20k, but lower supervised/flow
error did not translate into successful closed-loop Pick behavior.

## Classification

Because Pick remained 0/30 after the sole permitted repair, Phase 2C-B is Result D: generic SmolVLA
competence failure. The evidence suggests a deeper observation/action/demonstration formulation
issue, but this is a research hypothesis rather than a proven cause. No further SmolVLA repair,
model-size or learning-rate sweep, new data phase, VLA-JEPA, or LatentGuard work is authorized.
