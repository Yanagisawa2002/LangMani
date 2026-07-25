# LangMani 2.0 Phase 2C-B plan

Phase 2C-B evaluates the official pretrained SmolVLA as the modern VLA
baseline on the frozen `LangManiOfficialMultiSkill-v2` package. It does not
modify or retrain ACT, does not modify the accepted dataset, and does not
authorize VLA-JEPA or LatentGuard.

## Immutable inputs

- source branch: `codex/langmani-v2-phase2c-a1-bounded-act`;
- source commit: `4c38190a36ec41349ef39774ee95dc666537eed6`;
- working branch: `codex/langmani-v2-phase2c-b-smolvla`;
- package fingerprint:
  `sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04`;
- accepted policy rows: 2,998 episodes and 254,200 frames;
- train rows: Pick 700/54,608, Stack 700/74,891, Push 699/48,219;
- train-only normalization:
  `LangManiOfficialMultiSkill-v2-primary-train-pooled-natural-frame-v0`;
- LeRobot: exactly `0.6.0`;
- base checkpoint: `lerobot/smolvla_base` at
  `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.

The Phase 2C-A.1 `RESULT_B` is a valid closed-loop ACT pipeline with weak
policies, not an infrastructure failure. All four bounded ACT baselines
finished their frozen evaluation and remained at 0% success. ACT is closed.

## Policy and embodiment contract

All four models consume only:

- `observation.images.base_camera`: uint8 RGB `[3,256,256]`;
- `observation.state`: `PandaPolicyStateV0` float32 `[9]`;
- `task`: natural-language instruction.

They produce 50-step float32 `[8]` action chunks for the native 20 Hz
`pd_joint_pos` controller. The shared model receives no task-ID feature.

The reviewed official SmolVLA 0.6 flow-matching decoder is unbounded. Phase
2C-B therefore uses the isolated parameter-free
`smolvla_bounded_action_latent_v1` representation:

```text
physical
-> affine native bounds to [-1,1]
-> atanh((1-epsilon) * normalized)
-> official SmolVLA flow target

official SmolVLA generated latent
-> tanh
-> affine native Panda bounds
-> physical action
```

`epsilon=1e-6`. Exact dataset endpoints map to finite, distinguishable latent
targets and decode to the deterministic nearest interior value. There is no
clipping, projection, rejection sampling, zero/previous-action replacement, or
environment correction.

The official feature override uses the published 32-dimensional internal
state/action padding path, so changing the public state/action feature shapes
does not require changing any pretrained tensor shape. Strict checkpoint
loading must prove the actual loaded ratio and reinitialized parameter list
before training.

## Frozen primary training configuration

The authoritative JSON is
`configs/langmani_v2/phase2c_b_smolvla.json`.

- chunk size: 50;
- validation-only execution horizons: 1, 4, 8;
- batch size: 4;
- gradient accumulation: 4;
- effective batch: 16;
- precision: BF16;
- optimizer: AdamW, learning rate `1e-4`, betas `(0.9,0.95)`;
- weight decay: `1e-10`;
- gradient clipping: 10;
- scheduler: cosine decay with 1,000 warmup steps;
- vision encoder frozen;
- action expert trained;
- state projection trained;
- augmentation: none;
- seed: 0;
- primary training: 20,000 optimizer steps;
- retained checkpoints: 5,000, 10,000, 20,000.

Model P uses only Pick train rows. Model S uses Stack. Model U uses Push.
Model M uses all three roots with a deterministic exact long-run 1/3 task
sampler. Language remains the only task condition entering Model M.

## Ordered gates

1. Compactly revalidate package identity and ACT closure.
2. Resolve/download the already pinned official base and nested VLM bytes.
3. Strictly load the pretrained policy and record embodiment adaptation.
4. Run 100,000 synthetic latent chunks through the physical decoder.
5. Verify official padding semantics, processor save/reload, and model views.
6. Run one real Pick batch forward/backward/optimizer/save/reload/inference and
   submit at least one bounded action to a real simulator `env.step`.
7. Run deterministic Pick and shared micro-overfit gates.
8. Train Model P for 20,000 optimizer steps.
9. Screen at most two P checkpoints and compare horizons only on a fixed
   validation subset.
10. Run 30 Pick validation episodes.

Model P must achieve at least 3/30 with zero invalid-action and zero simulator
error episodes. If it is 0/30, one repair is allowed only after a specific
implementation/configuration defect is demonstrated. If the repaired policy
also remains 0/30, the phase stops as `RESULT_D`; Models M/S/U are not fully
trained.

Only a passing Pick gate authorizes Model M full training. Valid shared
closed-loop execution then authorizes Models S and U. Final policies,
processors, action transforms, horizons, and reset identities are frozen
before any final split is opened.

## Evaluation

Development uses 30 validation episodes per matching task. Horizon comparison
uses a fixed small validation-only subset. The shared policy additionally gets
fixed correct, wrong-skill, blank, and shuffled language interventions plus
same-observation action comparisons.

Final task-specific evaluation uses 50 unseen-reset and 50 visual-shift
episodes per matching task. Shared evaluation uses 50 episodes per task for
unseen reset, held-out task-language paraphrases, and visual shift.

Success, confidence intervals, timeouts, failure class, action saturation,
smoothness, query count, p50/p95 inference latency, peak VRAM, and runtime are
reported separately. Training or offline loss alone is never a robotics
result.

## Hard stops

Stop before further model work if the package identity changes, an excluded
episode enters a view, a privileged/task-ID field enters the shared model,
the physical path requires clipping/projection, processor/checkpoint reload
changes semantics, or final settings change after results are visible.

VLA-JEPA, LatentGuard, SARM, PPO, Diffusion Policy, and any new data phase
remain unauthorized regardless of the Phase 2C-B result.
