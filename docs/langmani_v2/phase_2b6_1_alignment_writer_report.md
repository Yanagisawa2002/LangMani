# Phase 2B.6.1 alignment and writer report

The instrumentation defines the intended contracts:

- Mode B/C observation `t` is acquired immediately before action `t`;
- action and frame indices must both be exactly `0..T-1`;
- timestamps must be finite, strictly increasing, and exactly 20 Hz;
- the post-final-action image is one diagnostic frame and never a policy frame;
- Mode C uses the Phase 2B.6 compressed-NPZ staging, flush/fsync, atomic rename, cleanup, and
  shape/hash readback path in an independent destination.

None of these runtime gates was reached. Control 936 failed while constructing the environment in
Mode A. Under the preregistered progressive protocol, target Mode B and Mode C were therefore
forbidden.

As a result:

| Question | Finding |
| --- | --- |
| Did observation generation alter physics? | unverified; no Mode B/C replay |
| Was there an off-by-one alignment failure? | unverified; no policy frames generated |
| Was a terminal frame inserted into policy data? | no forensic policy data exists |
| Did state extraction fail? | unverified |
| Did timestamp generation fail? | unverified |
| Did NPZ serialization fail? | unverified |
| Did writer finalization fail? | unverified |
| Did the forensic writer touch frozen Phase 2B.6 bytes? | no |

The old Phase 2B.6 episode-938 NPZ is missing, and its frame count, timestamps, writer status, and
terminal state were not recorded. The generic historical rejection cannot establish an alignment
or writer failure.
