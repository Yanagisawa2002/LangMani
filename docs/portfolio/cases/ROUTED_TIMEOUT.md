# Case 3 — correct route followed by control timeout

| Field | Evidence |
| --- | --- |
| Episode index | 2 |
| Scene | `langmani-pick-place-scene-v0:2000000012` |
| Task | `green_cube` → `left_bin` |
| Command | “Next, Use only the named assignment: green cube into the left bin.” |
| Route | Correct |
| Controller identity | Oracle and NeuroSymbolic matched |
| Initial state | Exactly paired |
| Outcome | Oracle timeout; NeuroSymbolic timeout |
| Attribution | `timeout` |
| Language evidence ID | `langmani-m5a-sealed-final-example-4c0fa983a47cf106e11ead91070e1872ac1fde6e9e641c4dfe1ec9c0bb927f3e` |

Because route, controller identity, and initial state all match, this failure belongs to continuous
control rather than language understanding. The frozen PerTask controller failed on the same
physical episode under both routing conditions.
