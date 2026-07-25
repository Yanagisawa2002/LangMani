# Phase 2C-C evaluation report

Status: schedules are being frozen before model results.

The experiment uses:

- the frozen absolute 20k Pick checkpoint on 30 accepted training resets;
- the relative selected checkpoint on the same 30 training resets;
- a fixed six-reset validation screen for H=1 versus H=8;
- 30 Pick validation resets with the selected horizon;
- 50 unseen resets only if relative validation reaches at least 3/30.

Each closed-loop report records success, grasp-region entry, grasp, lift, failed grasp, no motion,
object drop, timeout, invalid action, simulator error, episode length, and inference latency.

The final report will distinguish training-reset fit, validation generalization, and any unopened
test metric. An unopened final metric is unavailable, not zero.
