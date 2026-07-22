# Phase 2B.3 simulator-backed MPC push expert plan

Phase 2B.3 starts from immutable commit
`8476a72b3ce7504e7fef0a08114c6c567da77ba7`. It does not continue the Candidate E--P offset
sequence. Its only purpose is to decide whether a simulator-backed receding-horizon expert can pass
the unchanged 95/100 expert gate with every zero-tolerance safety count at zero.

The bounded order is input verification, complete simulator-state clone audit, sandbox/main
configuration equivalence, closed-loop expert implementation, contract tests, one fixed 24-episode
development pilot, at most one predeclared bounded repair, and at most one formal 100-episode gate.
Formal seeds remain 66300--66399 and are inaccessible until the development success and prediction
agreement gates pass.

The clone/restore precondition failed on the native RTX 5090 target before expert implementation.
Three preserved audits showed deterministic agreement between two cold-restored sandboxes, but the
same restored state did not reproduce the live environment's contact-bearing continuation. The
final contact-free-boundary attempt diverged by 4.394 mm in target-object pose against a 0.020 mm
tolerance. This is material at the task's 5 mm containment-clearance scale.

The stop rule therefore fired at step 5 of the bounded order. Steps 6--18 were not run: there is no
MPC expert, development pilot, prediction-agreement gate, bounded repair, or formal qualification.
The phase is `RESULT_C`; seeds 66300--66399 remain sealed.

This phase recorded no demonstrations, exported no LeRobot data, created no accepted dataset
package, loaded no SmolVLA model, executed no optimizer step, and began no Phase 2C/2D work.
