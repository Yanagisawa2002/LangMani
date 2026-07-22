# Phase 2B.3 comparison with prior push experts

| Expert | Evidence scope | Success | Safety/gate disposition |
| --- | --- | ---: | --- |
| Candidate E | formal 100 | 75/100 | below 95%; rejected |
| Candidate F | formal 100 | 11/100 | below 95%; rejected |
| Candidate P | isolated diagnostic | 2/5 | workspace violation; formal gate not reached |
| SimulatorMPCPushExpertV1 | clone precondition | not evaluated | Result C; architecture invalid |

Candidate E remains the strongest measured formal expert. Phase 2B.3 cannot claim improvement or
regression because no expert episode was run. Its contribution is a negative systems result: the
installed public state API is insufficient to validate simulator-backed receding-horizon prediction
under the task's precision contract.

Formal seeds 66300--66399 and all collection identities remain untouched. Consequently there is no
Standard/Hard, geometry, direction, failure-taxonomy, planner-call, or wall-clock comparison for a
new expert. Null fields in the machine-readable artifacts mean “not run”, never zero failures.
