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

This phase cannot record demonstrations, export LeRobot data, create an accepted dataset package,
load SmolVLA, execute an optimizer step, or begin Phase 2C/2D. Result A authorizes only a later
separately invoked Phase 2B.4 collection phase.

