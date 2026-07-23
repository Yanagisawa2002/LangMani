# Phase 2B.3.1-RR runtime report

## Runtime environment audit

The latest evidence/code node was clean at `cc31ced...` and had no GPU device. The other node had
an idle NVIDIA GeForce RTX 5090 with 32,607 MiB, but its repository remained at the older
`0e79fa9...` M5A-era commit. Neither node had an active experiment. No checkout was updated, no
server process was started, and no power state changed.

Local static validation used the repository's Python 3.12 virtual environment. An initial system
Python 3.13 invocation could not import the project and performed no project work; all reported
verification and tests use Python 3.12.

## Measurements

Reset, replay-step, 25/100/250-step prefix, 12-step probe, two-sandbox boundary, CPU, memory, and
GPU runtime measurements are all unavailable because the transcript hard stop forbade simulator
construction.

## Projections

No numerical MPC runtime was projected. Although the intended future calculation used six
candidates, a 12-step horizon, a three-step executed prefix, and a 250-step episode, substituting
unmeasured timing inputs would create a fabricated estimate.

The prospective viability limits remain documented but untested:

- projected p95 episode wall time no greater than 600 seconds;
- projected 400-episode production time no greater than 72 continuous hours.

`runtime_viability_validated=false` means unassessed under Result D, not that a measured workload
exceeded either threshold.
