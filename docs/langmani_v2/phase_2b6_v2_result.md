# LangMani 2.0 Phase 2B.6-v2 result

## Result A — Versioned multi-skill dataset accepted

`accepted_multiskill_dataset_validated=true`.

The clean production accounted for all 3,000 official source identities,
accepted 2,998 episodes with 254,200 primary policy frames, and retained two
explicit deterministic-physical exclusions. Overall acceptance is 99.9333%;
PickCube, StackCube, and PushCube acceptance is 100.0%, 99.9%, and 99.9%.
Every frozen conjunctive gate passed.

Accepted package:

- identity: `LangManiOfficialMultiSkill-v2`
- fingerprint:
  `sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`
- result fingerprint:
  `sha256:81ebfbdc0547e09fd896d9b9eb13213b3040c8fed6f61495500a5d11b2dab2c7`

## Passed gates

- three skill families retained
- 99.7% overall and 99.5% per-task thresholds passed
- all 3,000 sources accounted
- zero unclassified failure
- accepted replay outcome agreement and action/frame alignment: 100%
- zero invalid/non-finite actions and accepted simulator errors
- privileged student fields excluded
- real LeRobot 0.6 full readback passed
- exact independent source-to-derived verification passed
- primary split and language/media leakage: zero
- accepted-only padding and train-only normalization passed
- content-addressed archive and clean restore passed

## Eligibility and authorization

| Flag | Value |
| --- | --- |
| `act_baseline_training_eligible` | true |
| `smolvla_training_eligible` | true |
| `vla_jepa_training_eligible` | true |
| `act_training_authorized` | false |
| `smolvla_training_authorized` | false |
| `vla_jepa_training_authorized` | false |
| `student_policy_training_started` | false |
| `optimizer_created` | false |
| `optimizer_steps` | 0 |
| `backward_passes` | 0 |

No ACT, SmolVLA, VLA-JEPA, Diffusion Policy, SARM, PPO, or other policy was
instantiated or loaded.

## Verification

- complete compact artifact set: 49 files, all declared sizes and SHA-256
  values matched
- artifact manifest:
  `sha256:aff53479e161a3aba19035364c147bc3926d906efa85b9f7d0ccc9db681cd20b`
- independent artifact verifier:
  `sha256:80a50c759425c999e02812e95249d3ef606be9526a6a64fdba6ffecc4b734b6e0`
- targeted Phase 2B.6-v2 tests: 23 passed after the final archive fix
- earlier complete production-target suite: 39 passed
- Ruff on the complete source tree: 404 files clean
- changed source modules: mypy clean
- isolated sdist and wheel build: passed

The full CPU-safe pytest baseline produced 1,733 passes, 18 deselections, and
five failures in existing physical smoke tests because the unwrapped default
SAPIEN loader could not find a rendering device. These are explicitly retained
environment diagnostics, not accepted-launcher physical results. The required
explicit Vulkan launcher subsequently passed `vulkaninfo`, minimal SAPIEN,
all three zero-step tasks, and the complete physical production. The full
mypy baseline likewise retains 56 pre-existing third-party-import or duplicate
script-module errors across 25 files; none are in the changed modules.

## Remaining boundary

There is no unresolved dataset-integrity, readback, leakage, archive, or
restore blocker. The restored scratch tree is intentionally retained for
inspection. Policy training remains unauthorized and did not start.

The recommended next step is a separately authorized Phase 2C consumer-design
and training-preflight milestone that pins one model family, sampling policy,
train/validation protocol, compute budget, and stop gates against this package.
It must not begin automatically from this Result A.
