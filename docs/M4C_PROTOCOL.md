# M4C fixed-budget language-diversity experiment

M4B.1 was completed and separately pushed at `3104578386242bcf62a2b09ea97a86f1b2a92dc0`
before this protocol was created. Its observable destination approach is distinct from red-object
interaction and full task success. M4A `717a07d` and M4B `99139de` stay frozen. No new task,
architecture, environment change or online language interpreter is introduced.

## Controlled comparison

Run L1, L5 and L10 separately, training seed 0, from the same pinned upstream SmolVLA base.
Each run uses the same 96 red-object training trajectories (48 scene pairs), 17,149 frames,
train-only normalization, 20,000 optimizer steps, batch 8, learning rate 1e-4, optimizer,
scheduler, parameter-freezing configuration and 0 DataLoader workers as M4B. No robot data
is added. Validation/test robot trajectories remain excluded. Do not increase steps or choose
hyperparameters based on these evaluations. No L20 or extra seeds are launched in this stage.

Language levels are nested: first 1, 5 or 10 catalog templates per semantic goal, including
canonical wording at every level. L5 includes canonical plus lexical, syntactic, referential
and polite examples; L10 also adds alternate natural phrasing and additional realizations.
Thus diversity changes both wording count and family coverage; this experiment does not isolate
individual families. Every expression explicitly identifies the red cube and exactly one side.

Manifest rows enumerate every allowed (training trajectory, text realization) pair: 96/480/960
label rows for L1/L5/L10, respectively, always only 96 unique robot trajectories. Each row stores
semantic_goal_id, instruction_text, paraphrase_family, template_id, scene_seed, trajectory_source
and the original episode/frame metadata. These label rows are not independent demonstrations.

The official sampler still samples the original frame dataset. A read-only dataset wrapper
changes only the returned `task` string. For sample ordinal n, select a text using a SHA-256
of language seed, n, source episode and source frame, modulo the group's diversity. This draws
from all allowed realizations with replacement, without touching Python/NumPy/Torch RNG. The
wrapper logs every actual sample to CSV. All groups must have identical robot-sample sequence
hashes and 159,973 actual sample presentations; text realization counts are reported separately.
The configured batch size is 8, but upstream single-process `drop_last=False` preserves a final
5-frame batch each epoch: 2,144 updates traverse 17,149 frames. In 20,000 updates there are nine
short batches, giving 159,973 actual samples versus the nominal `20,000 * 8 = 160,000`.
This corrects sample accounting without padding, dropping frames or changing the optimizer budget.
The original LeRobot files and physical observation/action tensors remain unchanged.

Accelerate prefetches one batch even with zero workers. A FIFO logs samples at optimizer use;
unused lookahead is excluded from presentation counts. Each invocation retains its complete CSV.
A compatible interrupted run resumes the official optimizer/RNG and sampler state from a saved
checkpoint; a separate consolidated CSV keeps each logical sample prefix once and preserves
abandoned post-checkpoint evidence. Never overwrite later checkpoints or repeat a completed run.
Align Accelerate's loader epoch counter with the official resumed sampler epoch before iteration;
otherwise it would reset a later-epoch sampler to epoch zero. CSV optimizer-step indices and actual
sample ordinals account for short batches. Sample-order equivalence does not by itself establish
bitwise equivalence of all optimizer/model randomness after interruption.

## Leakage and initialization gates

Persist one language manifest containing the complete training and held-out catalogs, the
three trajectory-label views and their hashes. NFKC/casefold normalization replaces punctuation
with spaces and collapses whitespace. Reject duplicate normalized expressions, forbidden
train/held-out normalized-text overlap, forbidden template-ID overlap, wrong side labels and
missing source identities. Also forbid all four original M4B held-out paraphrase strings from
training, even when not used by the new evaluation. Seen canonical reuse is explicitly allowed.

Validate token lengths with the pinned tokenizer before training; no instruction may lose its
goal through the existing 48-token limit. Validate all original dataset bytes against M4B's
exhaustive decoded receipt and recheck pinned model/tokenizer asset inventories. Save the actual
pre-update model-state hash and compare it across groups, as well as pretrained revision/hash,
configuration, normalization and robot-sample hashes. Reuse official model and trainer components.
An L1 wrapper smoke must produce the same model checkpoint as an unwrapped official smoke
with the same real data, initialization and budget. Fixture tests alone cannot pass this gate.

## Common evaluation protocol

All groups use the same frozen M4B 20 scene seeds, initial state/RGB/qpos identities and expert
reference (39/40; retain the failure), plus the same per-scene flow-noise schedule. Reuse the
validated M4B.1 diagnostic observer without changing thresholds. Keep 200 control steps, the
original success predicate, termination rules, gripper saturation and strict arm bounds.

The same nine physical prompts per scene are used for every model: common canonical left/right,
held-out lexical left/right, syntactic left/right, natural left/right, and one blank. Each
held-out family has two templates alternating by frozen scene index. The primary seen condition
uses the common canonical expression, which belongs to all three training sets; it is not an
average across each group's larger seen-language catalog. This keeps evaluation texts identical
across groups. Swapped reuses canonical trajectories but scores the opposite original request;
blank is scored against both requested goals. There are 180 physical rollouts and 240 scored
rows per model, not 240 independent trials. The three runs total 540 physical model rollouts.

For every condition, report requested/supplied-goal success and destination approach, bin first
contact, shared-red first contact/grasp, placement attempt, timeout, cumulative funnel and
deterministic failure taxonomy. Blank supplied-goal and instruction-switch metrics are null;
retain neutral physical failure categories. Pair-switch means both first destination approaches
correspond to their respective supplied goals. It is separate from both full successes. Preserve
raw numerators/denominators, per-goal counts, all action arrays, telemetry, videos and hashes.

## Execution and interpretation

Validate the implementation and real-data smokes first. Then run complete L1/L5/L10 seed 0 and
evaluate all three without changing protocol or reacting to intermediate result quality.
Training/inference smoke results are never counted as full experiments. If an operational error
occurs, retain evidence and recover only the affected compatible stage into appropriate fresh
artifacts; do not repeat completed long training or tune on final outcomes.

Publish separate full-success and destination-approach tables, with contact/grasp/failure data
alongside them. Language diversity can affect the pickup/execution sequence even where the
destination proxy is already high. A negative result is valid. One training seed and 20 paired
scenes cannot establish broad language understanding, open-vocabulary behavior or transfer.
After all models, evaluations, local verified recovery and A-Q delivery are complete, recommend
the next action but do not start another milestone or an additional seed sweep automatically.
