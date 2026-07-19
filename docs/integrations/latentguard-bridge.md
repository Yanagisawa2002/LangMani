# LatentGuard initial-state proposal bridge

This optional bridge connects the Python 3.12 LangMani runtime to the separate Python 3.11
LatentGuard runtime through canonical JSON files. Neither project imports the other. Normal
LangMani rollout behavior is unchanged.

The bridge selects the lexically first eligible canonical task from the accepted PerTask
controller registry. Eligibility requires the reconstructed registry and runtime locators to match
the frozen registry exactly. The binding includes the task, instruction, checkpoint, model,
processors, normalization, active action bounds, control frequency, ACT chunk length, and queue
horizon, but never includes runtime paths, checkpoint bytes, images, hosts, or credentials.

The initial proposal command performs exactly one seeded reset and one policy query. It exports the
raw postprocessed `[50,8]` chunk, the LangMani-projected `[50,8]` chunk, and the first ten executable
actions. Image and state bytes are represented only by digests. A counting environment proxy makes
any `env.step` call a hard failure.

Candidate projection is a second JSON-only operation. It opens no simulator and performs no reset:
the action-space values are already content-bound by the policy binding, and the operation invokes
the existing `BoundedActionEnvPostprocessorV0`. Every candidate retains separate raw and projected
digests plus the complete correction mask and count.

The future bounded M6B design starts every candidate from the same seeded initial reset, executes a
ten-action candidate prefix, and then replays the previously recorded projected nominal
continuation beginning at index ten. The policy is not requeried after divergence. This continuation
is reproducible but is not faithful same-policy continuation, intermediate exact replay, serialized
ACT queue restoration, or closed-loop shielding.

M6A.1 makes no performance claim and executes no action or outcome. The action-only LatentGuard
scorer is intentionally task-out-of-distribution; structured state, RGB, instruction, task identity,
provenance, fault metadata, and outcomes are prohibited scorer inputs.

Commands:

```text
python scripts/export_latentguard_policy_binding.py ...
python scripts/export_latentguard_initial_proposal.py ...
python scripts/project_latentguard_candidates.py ...
python environment/audit_latentguard_bridge.py --strict ...
```

## M6A.1 preflight result

The implementation and CPU fixture gates passed. The real single-seed probe did not run because the
current execution server does not contain the accepted controller registry, runtime-selection
record, ACT checkpoint, or processor artifacts required by `export_policy_binding`. The bridge
failed closed before creating an environment: reset, policy-query, `env.step`, candidate-projection,
and outcome counts are all zero.

The local frozen registry declares
`langmani-pick-place-task-v0:blue_cube:left_bin:canonical_v0` as the lexical selection, with
checkpoint digest
`sha256:0aa8add02a8675a802fde163a32e0c3a48dd97113a141d6faede28cc6167f6e4`, but
this is not claimed as an artifact-validated binding on the current server. M6B remains blocked and
was not started.
