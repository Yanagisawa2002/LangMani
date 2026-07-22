# Phase 2B v1 runtime retirement

| Field | Value |
| --- | --- |
| dataset ID | `langmani/phase2b-push-v1` |
| historical acceptance | `PASSED` |
| bytes currently available | `false` |
| runtime usable | `false` |
| superseded by | `langmani/phase2b-push-v2` |
| reason | original byte-level dataset asset irrecoverable |

The v1 historical experiment conclusions and their committed evidence remain valid and are not
withdrawn or rewritten. The unavailable v1 bytes must not be used as a current training dependency,
and no v1 digest, file hash, byte count, acceptance package, or acceptance status transfers to v2.

Phase 2B.2 performs a new simulator collection. Even where the logical observation/action schema is
unchanged, v2 is a distinct dataset with new seeds, episodes, raw archives, LeRobot bytes, source
commit, manifests, tree digest, archive hashes, and acceptance decision. Every downstream report
must identify `langmani/phase2b-push-v2` explicitly.
