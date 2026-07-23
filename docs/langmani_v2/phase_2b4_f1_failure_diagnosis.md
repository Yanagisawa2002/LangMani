# Phase 2B.4-F1 failure diagnosis

## What immutable F0 proves

F0 completed 1,048,576 environment steps and 21,034 training episodes with zero training success.
Its fixed 48-episode deterministic evaluation achieved 0/48 and ended with 22 wrong-object
displacements, 12 target workspace exits, and 14 timeouts. Vector resets, GAE/PPO updates,
checkpoint reconstruction, action bounds, CUDA simulation, and reward-hacking checks passed.
This is a failure of the frozen F0 learned teacher, not a simulator or action-integrity failure.

## Retention boundary

F0 retained 33 iteration snapshots of transition reward mean/range, advantage range, PPO losses,
approximate KL, clip fraction, gradient norm, and cumulative episode/success counts. It retained
first/last-quartile episode return and task-progress aggregates. It did not retain per-episode
training returns, a training termination-reason history, per-component reward timelines, value
explained variance, action statistics by iteration, state-value distributions, or contact-event
frequencies. Those quantities cannot be reconstructed from the compact package.

A later read-only evaluation of the frozen F0 checkpoint on new F1 diagnostic seeds may measure
current behavior and reward density. It is explicitly not a reconstruction of the missing F0
training transitions.

## Current diagnosis before target audits

The F0 Normal standard deviation did not obviously collapse: the final checkpoint remains near its
initial log standard deviation of -2.5. F0 also was not simply an uncentered zero-mean policy at
initialization. Its Normal mean added a current-qpos inverse-tanh reference, while the learned actor
could add an unbounded latent offset. A credible, not-yet-proven failure mechanism is therefore
drift from state-centered initialization into large absolute joint-target offsets, combined with
sparse useful contact/progress and immediate zero-tolerance failures.

Terminal semantics are structurally correct: terminated failures do not bootstrap, time-limit
truncations do bootstrap but stop GAE continuation, and failed slots reset on the same transition.

## Target diagnostic result

The 100,000-sample initial audit found that F0's actual initial distribution was state-referenced,
not an uncentered zero-mean controller. Its uncapped latent offset nevertheless produced a mean arm
target displacement of 0.499 rad and a displacement above 0.25 rad on 96.98% of samples. It also
commanded a mean gripper displacement of 1.899 native units because the F0 reference fixed the
gripper near closed while reset state was open.

The read-only F0 checkpoint diagnostic used 32 new F1 diagnostic episodes. It entered the
behind-object region in 20/32 episodes but made zero correct target contacts, increased containment
in zero episodes, and reduced target distance by at least 5 mm in only 1/32. Outcomes were 14
wrong-object interactions, nine target workspace exits, and nine timeouts. Numerous positive scalar
events came from approach and small distance fluctuations, but target-progress reward summed
negative and no contact or containment credit occurred. This is partial approach behavior on new
diagnostic seeds, not proof that the historical F0 training run learned useful contact.

The primary supported mechanism is therefore unsafe/nonlocal target exploration plus a failure to
convert approach-side motion into correct contact. Reward cancellation after correct contact is
not supported because correct contact never occurred. Reward revision 0 was retained.
