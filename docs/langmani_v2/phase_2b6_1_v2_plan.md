# LangMani 2.0 Phase 2B.6.1-v2 Plan

## Scope

This phase is a bounded, instrumented forensic replay of official
`StackCube-v1` source episode 938. It starts from
`799e62cc9f7d5590116d236242c4328af1812e1c` and uses only:

- StackCube controls 936 and 937, once each in Mode A;
- StackCube episode 938, at most three fresh-process Mode A runs;
- two Mode B runs only after every Mode A prerequisite passes;
- one Mode C run only after every Mode B prerequisite passes.

The recovered Phase 2B.6.1-R Vulkan/EGL process contract is mandatory. The
historical default-loader failure is immutable evidence and is not rerun.

## Immutable inputs

- Phase 2B.6, Phase 2B.6.1, and Phase 2B.6.1-R committed artifacts;
- physical producer commit
  `73b2cd1517e7c3d2bfc08bd2898e46a3195eb1c9`;
- official source archive
  `sha256:f9b7d34b9aa418a04aa8e4322d4dea5aa27e8ae81757f60b210c1ffc54bf9c1b`;
- target action array `float32[105,8]`,
  `sha256:5fc50bc7beeb91e55b2eeccf3f016414811d314142e7a80ad6037d4f83d92eb8`;
- reset identity
  `sha256:e396b1aea2fe017ae24ab29115f5e3702fa1a37d6d34f0f9d0629f188f44410b`;
- source trajectory identity
  `sha256:f5c25c7b4f1f7f5ed0cd8b3641f7a451f758c1a47eb9aa31bf96b0081848f7a3`.

The frozen partial-production root is read-only. New outputs belong only under
`/root/autodl-tmp/langmani-external/phase2b6_1_v2/stack938_forensics_v1`.

## Frozen success rule

The committed producer code and evidence show:

```text
producer_success_rule = final_step_canonical_success
separate_stable_success_gate_present_in_producer = false
```

The source has first canonical success at action index 99 (one-based step 100),
six successful terminal steps, and final success true. Consecutive success is
recorded as a diagnostic, but `stable_success_gate` is explicitly
`not_applicable`; this phase does not invent a new acceptance rule.

## Gate order

Each replay retains construction, reset, action, per-step, physical outcome,
observation/alignment, and writer/serialization gates independently. Mode A
requires only construction/reset/action/physical gates. Mode B additionally
requires complete pre-action RGB and `PandaPolicyStateV0`, exact
`observation[t] -> action[t]` alignment, and timestamps. Mode C additionally
requires isolated atomic NPZ finalization and readback.

The earliest failed gate terminates the next richer mode. Controls must pass
before episode 938. Three Mode A repetitions are compared categorically without
averaging.

## Result boundary

- Result A: episode 938 is reproducibly physically valid; any observed failure
  is non-physical, or the historical aggregate rejection is not reproduced by
  the explicit path.
- Result B: controls pass and all three Mode A runs reproducibly fail the same
  frozen physical or final-success gate.
- Result C: Mode A categorical outcomes are intermittent.
- Result D: source identity, recovered rendering preflight, controls, or
  instrumentation is invalid or insufficient.

No result authorizes production continuation, package acceptance, dataset
conversion, archive/restore, or model work. All optimizer and backward counters
remain zero.
