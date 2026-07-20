# Case 4 — safe unsupported-action rejection

| Field | Evidence |
| --- | --- |
| Probe | `unsupported_action` |
| Command | “Push the red cube toward the left bin.” |
| Status | `reject_unsupported` |
| Reason | `unsupported_action` |
| Object / bin / TaskSpec | `null` / `null` / `null` |
| Controller lookup | 0 |
| Policy reset | 0 |
| Environment reset | 0 |
| `env.step` | 0 |
| Probe fingerprint | `sha256:a7abfdf256115addb59f531fd04b15e25ed2d2f08b113da058008fc009e2e236` |

This is a correct rejection. “Push” is outside the six supported pick-and-place TaskSpecs. The
decision carries no executable target and the dispatcher proves that the robot runtime remains
untouched.
