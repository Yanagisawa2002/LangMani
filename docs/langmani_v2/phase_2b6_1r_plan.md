# LangMani 2.0 Phase 2B.6.1-R plan

## Scope

Phase 2B.6.1-R is an infrastructure-only Vulkan/EGL recovery and zero-step environment gate. It
starts from immutable Phase 2B.6.1 commit
`4f96a0f42dbaf0adb58c4c3aad79fdd7bccfde84` on the new branch
`codex/langmani-v2-phase2b6-1r-vulkan-recovery`.

The phase may:

1. rehash prior compact evidence and the directly relevant frozen partial-production files;
2. inventory Vulkan ICD, EGL vendor, GLVND, loader, driver, Python-package, GPU, and process state;
3. run `vulkaninfo --summary` under a frozen process-scoped ICD contract;
4. construct and release a minimal SAPIEN `RenderSystem`;
5. call `gym.make("StackCube-v1", ...)`, inspect static contracts, and close the environment;
6. repeat that sequence in three independent processes;
7. record one safe default-loader negative control after the primary gate passes.

It may not reset an environment, call a control/physics step, submit any robot action, replay
episodes 936--938, write a demonstration, resume Phase 2B.6 production, create a dataset, load a
policy, create an optimizer, or train any model.

## Immutable inputs

- Prior result: Phase 2B.6.1 `RESULT_D`, primary `insufficient_evidence`, first new failure at
  `environment_construction`.
- Phase 2B.6 partial root:
  `/root/autodl-tmp/langmani-external/phase2b6/production_v1`.
- Official source root:
  `/root/autodl-tmp/langmani-external/phase2b5/`
  `maniskill_d674485bbffdd533914e52d272fdda34c0515608`.
- Pinned runtime:
  `/root/autodl-tmp/conda-envs/langmani`.
- Task contract: `StackCube-v1`, Panda, `pd_joint_pos`, `obs_mode=rgb`,
  `sim_backend=physx_cpu`, `render_backend=sapien_cuda`, one environment, 256-by-256 sensor.

The parent process verifies source-commit ancestry, target branch, clean status, upstream/GitHub
identity, prior manifests, frozen files, process state, GPU identity, and disk capacity before
running a loader probe.

## Frozen candidates

| Candidate | Vulkan selector | EGL selector | Allowed work |
| --- | --- | --- | --- |
| A | `/etc/vulkan/icd.d/my_nvidia_icd.json` | `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` | primary Vulkan, SAPIEN, and zero-step gate |
| B | `/etc/vulkan/icd.d/nvidia_icd.json` | `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` | `vulkaninfo` audit only |
| C | no explicit Vulkan selector | no explicit EGL selector | negative-control Vulkan and SAPIEN probe only |

No arbitrary combinations or system-file edits are allowed. Candidate A uses
`VK_ICD_FILENAMES`; `VK_DRIVER_FILES` must remain unset.

## Execution order and hard stops

1. Audit repository, evidence, frozen bytes, host, and renderer inventory.
2. Freeze the three candidates before any `vulkaninfo --summary` execution.
3. Launch Candidate A in a clean allowlisted child process.
4. Run `vulkaninfo`, minimal SAPIEN construction/release, `gym.make`, static inspection, and close.
5. Audit child-process and GPU cleanup.
6. Stop on any failure. Only after a pass, repeat in two additional fresh processes.
7. Require identical ICD, Vulkan device, SAPIEN device, PCI identity, action space, control mode,
   and observation mode across 3/3 runs.
8. Only after 3/3, run Candidate B `vulkaninfo` and Candidate C negative control.
9. Rehash the frozen partial root, classify A/B/C/D, write compact evidence, and run the independent
   no-simulator verifier.

An internal constructor initialization is recorded as a third-party construction behavior. The
project code itself records and requires exactly zero explicit resets, zero steps, zero action
submissions, and zero policy frames.

## Result boundary

Result A requires the exact ICD identity, successful Vulkan and SAPIEN probes, 3/3 matching
zero-step StackCube constructions, the intended RTX 5090, clean environment close, and no process
or GPU-workload leakage. Result A may set only
`phase2b6_1_forensic_restart_eligible=true`.

Eligibility is not authorization. Every result keeps:

```text
phase2b6_1_forensic_restart_authorized=false
phase2b6_production_resume_authorized=false
accepted_multiskill_dataset_validated=false
act_training_eligible=false
smolvla_training_eligible=false
vla_jepa_training_eligible=false
act_training_authorized=false
smolvla_training_authorized=false
vla_jepa_training_authorized=false
optimizer_created=false
backward_passes=0
optimizer_steps=0
student_policy_training_started=false
```
