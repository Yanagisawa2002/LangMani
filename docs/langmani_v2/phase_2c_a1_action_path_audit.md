# LangMani 2.0 Phase 2C-A.1 action-path audit

## Scope and evidence

This audit explains the Phase 2C-A `RESULT_D` terminal observation. It does not
reinterpret the four Phase 2C-A checkpoints as evaluated policies. The primary
immutable evidence is:

- Phase 2C-A training commit
  `c0b5106cb1fff5b43f46500321264cec33613673`;
- selected Pick run fingerprint
  `sha256:ab6af2aa23406cc05af0397fc7becc50f3bfa35008f814eb580d92012f36319a`;
- selected Pick checkpoint fingerprint
  `sha256:da7128d019689d190523c1f4c6f7f1e4060371ef24f203821afd98dbe316e0f3`;
- independently verified Phase 2C-A result fingerprint
  `sha256:d8b42978845249ccfcb0019c05b5e6a73fd16b01c993029a62864dd3fac1d12b`;
- closed-loop diagnostic episode
  `phase2c-a:validation:pickcube:000`.

The exact machine-readable stage record is
`artifacts/langmani_v2/phase_2c_a1/action_path_audit.json`.

## Exact path

| Stage | Representation | Bound guarantee | Correction |
| --- | --- | --- | --- |
| Dataset | physical `float32[B,16,8]` | dataset values were valid, but storage alone does not constrain a model | none |
| Preprocessor | train-only `MEAN_STD` coordinates | no | none |
| ACT target/loss | normalized action with `action_is_pad` mask | no | padded rows have zero loss |
| ACT head | `nn.Linear(dim_model, 8)` | no; output is unbounded | none |
| Checkpoint output | normalized `float32[B,16,8]` | no | none |
| Postprocessor | `output * train_std + train_mean` | no | inverse normalization only |
| `ActionChunk` | physical `float32[16,8]` | no in Phase 2C-A | none |
| Chunk iteration | first `H_exec` rows unchanged | no | no queue scaling |
| Evaluator | native lower/upper hard check | yes, by rejection | rejection, not correction |
| Environment | native `float32[8]` | not reached | none relied upon |

The Pick train-view gripper statistics stored in both the preprocessor and
postprocessor were:

```text
mean = 0.09075593203306198
std  = 0.9958731532096863
min  = -1.0
max  = 1.0
```

The first Phase 2C-A query produced normalized gripper outputs from
`0.9311804931643738` to `0.9724027139967077`. Inverse normalization mapped
them to physical values from `1.0180935859680176` to
`1.0591456890106201`, above the native upper bound `1.0`.

## Root cause

The overflow originated in the unbounded linear ACT head. `MEAN_STD` inverse
normalization then mapped a finite but unconstrained normalized prediction
outside the physical action space. The saved preprocessor and postprocessor
agreed exactly, so this was not a processor or checkpoint mismatch. There was
no gripper-only scale, clip, projection, threshold, action-queue conversion,
or environment correction.

The evaluator behaved correctly: it recorded one policy query, zero submitted
actions, and zero `env.step` calls. Consequently, Phase 2C-A established a
consumer/deployment contract failure, not policy quality.

## Authorized repair boundary

Phase 2C-A.1 replaces only the unbounded output path with
`bounded_action_head_v1`. It trains and infers directly in physical
`pd_joint_pos` coordinates:

```text
u = Linear(decoder_token)
z = tanh(u)
a = lerp(lower, upper, 0.5 * (z + 1))
```

The algebraic mapping is inside the trainable policy and uses the authoritative
eight-dimensional native bounds. Action normalization is `IDENTITY`; the loss
is padding-masked physical-action L1 plus the unchanged ACT KL term. No
post-hoc clipping, projection, gripper thresholding, or replacement action is
introduced.
