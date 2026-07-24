# Phase 2C-A failure analysis

## Observed failure

The only closed-loop attempt was the one-episode infrastructure smoke. It failed as
`invalid_policy_output` before any environment step because the selected Pick ACT predicted
gripper commands above the native `pd_joint_pos` upper bound. The observed 16-value gripper range
was 1.018094 to 1.059146 against the exact bound `[-1, 1]`. Arm commands, tensor shape, dtype, and
finiteness were valid; the simulator reset succeeded and reported no error.

This is not evidence that Pick, Stack, Push, or shared ACT has zero task success. No development
episode executed an action, so task-specific categories such as failed grasp, failed stack
alignment, insufficient push, timeout, or wrong-task behavior were not measured.

## Ruled-out explanations

- The accepted package, 168-file payload, 169-file live tree, sidecar, archive, restore, splits,
  exclusions, and normalization remained unchanged.
- The declared action bounds exactly matched the live environment action space.
- The selected checkpoint and processors reloaded from their content-bound manifests.
- The action was not nonfinite or malformed.
- The error was not a Vulkan, reset, or simulator exception.

## Boundary

Clipping, projection, gripper thresholding, and outcome-informed checkpoint replacement were
forbidden. The Result D hard stop therefore occurred before horizon selection, development,
bounded zero-success repair, final evaluation, task-condition intervention, or shared seed 1.
