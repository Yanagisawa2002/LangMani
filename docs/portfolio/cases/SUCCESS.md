# Case 1 — successful route and control

| Field | Evidence |
| --- | --- |
| Episode index | 0 |
| Scene | `langmani-pick-place-scene-v0:2000000012` |
| Task | `red_cube` → `left_bin` |
| Command | “In this command, The desired result is the red cube resting in the left bin.” |
| Route | Correct |
| Controller identity | Oracle and NeuroSymbolic matched |
| Initial state | Exactly paired |
| Outcome | Oracle success; NeuroSymbolic success |
| Attribution | `routing_correct_control_success` |
| Language evidence ID | `langmani-m5a-sealed-final-example-ccab626da8a0bf478c56dddac60c3da94d4cba9e23ca9d9aeb5117d1deb39f4b` |

This is the intended end-to-end path: free-form command → canonical TaskSpec → frozen PerTask
controller → successful physical evaluation. Exact pairing removes scene-layout differences as an
alternative explanation.
