# M5B LatentGuard proposal bridge plan

## Goal

Add an optional, JSON-only bridge that exports one deterministically selected accepted PerTask ACT
controller, performs one seeded initial reset and one 50-action policy query, and projects a frozen
four-candidate request with the existing LangMani action-bound processor.

## Frozen scope

- Branch: `codex/m5b-latentguard-proposal-bridge` from
  `b0fd9115496d3cc3d700c4eac7fb9f3f3ad401fd`.
- Exactly one canonical task, selected lexically from digest-valid accepted registry entries.
- One reset, one policy query, zero `env.step` calls, zero candidate outcomes.
- Raw postprocessed, projected executable, and actually executed actions remain separate; M5B
  produces no actually executed candidate action.
- Projection reuses `BoundedActionEnvPostprocessorV0`; no second simulator reset is needed.
- Cross-process payloads are canonical finite UTF-8 JSON with strict schemas and content digests.

## Validation

Run the complete repository validation, the bridge unit suite, a marked reset-only integration
probe at the pushed revision, and a final Linux suite once. Keep all raw images, checkpoints, and
runtime paths outside committed evidence.

The requested repository-wide `mypy src` command is run and retained, but the starting revision has
no mypy dependency or configuration and reports 516 pre-existing errors in 55 legacy modules under
the actual Python 3.12 environment. M5B does not suppress or bulk-repair that unrelated debt. The
new `src/langmani/integrations/latentguard_bridge.py` module contributes no errors to that result;
ruff, the complete pytest suite, build, and installation diagnostics remain mandatory passing gates.

## Exclusions

No training, rollout, action execution, task outcome, model selection, threshold fitting, VLM/LLM,
multi-GPU work, policy-state restoration, or automatic M6B execution is authorized.

## Final readiness disposition

Implementation commit `9713f7503dae6612309d0ef8a499d93f6c3899a3` was pushed and validated in a
clean isolated Linux worktree. The bridge unit subset passed, but the current server has no accepted
LangMani controller registry, runtime-selection record, ACT checkpoint, or processor artifacts.
The required artifact rebuild and digest comparison could not begin, so the real proposal probe
stopped before environment construction.

Observed bridge counts are reset `0`, policy query `0`, `env.step` `0`, candidate projection `0`,
and outcome `0`. No canonical real proposal was exported and no M6B action was executed. The
deterministic registry declaration remains useful for the exact expected identity, but is not
reported as a current-server artifact-validated binding. Bounded M6B readiness remains `blocked`.
The server remains online and GPU-idle.
