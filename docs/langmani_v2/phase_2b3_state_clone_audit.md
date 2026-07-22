# Phase 2B.3 simulator state clone audit

## Contract

ManiSkill 3.0.1 `BaseEnv.get_state_dict()` captures actor states as position, quaternion, linear
velocity, and angular velocity, plus complete articulation state. For the configured Panda
`pd_joint_pos` controller, `agent.get_controller_state()` is empty. ManiSkill's default
`set_state_dict()` does not restore task counters, latched failure state, elapsed steps, or random
generator state, so Phase 2B.3 adds an expert-only project snapshot covering those fields.

The physical audit must still run on the native CUDA/Vulkan target. It will compare a fixed
contact-bearing action sequence before and after restore, and across isolated main/sandbox
environments. Categorical observations, contacts, success, and failure must agree exactly;
continuous state tolerances and the resulting hashes will be recorded here after execution.

Status: implementation in progress; no MPC qualification is authorized by this document alone.

