# Phase 2C.1 accepted-dataset asset audit

## Result

Phase 2C.1 closes as **Result B: the original byte-identical accepted Phase 2B dataset was not
recoverable from any accessible location**. No candidate was copied, regenerated, re-exported,
re-encoded, repaired, or promoted. Optimizer steps remain exactly zero, all learned-policy gates
remain closed, and Phase 2D was not entered.

This result is an external-asset availability failure. It does not revoke the historical Phase 2B
acceptance and it is not evidence that SmolVLA learned or failed to learn the push skill.

## Frozen identity sought

The search targeted only the accepted `langmani/phase2b-push-v1` export named
`phase2b-push-lerobot-v1`: LeRobot 0.6.0 v3, 397 episodes, 53,297 frames, 141,422,866 bytes, one
256x256 `base_camera` RGB stream, 9D Panda state, 8D `pd_joint_pos` action, task metadata, and
20 Hz control. The six accepted split counts and all retained sidecar hashes are frozen in
`phase_2c1_recovery_plan.md`.

A path, matching aggregate counts, or a logically equivalent regenerated export was never enough.
The accepted evidence did not preserve a complete per-file digest inventory, a combined tree
digest, or the six original `meta/info.json` byte hashes. Therefore only the original bytes could
have acquired a new complete tree digest after re-passing every retained Phase 2B gate.

## Read-only search coverage

The audit inspected the current repository outputs, dataset roots, user-accessible local storage,
the local Git object history and LFS index, GitHub releases, the reachable AutoDL workspace and run
roots, the AutoDL persistent-data mount, temporary storage, and Hugging Face caches. The former
historical endpoint was also checked and was unreachable. Searches were limited to filenames,
directory structure, archive names, LeRobot metadata, and candidate manifests; no source or asset
was mutated.

| Classification | Count | Meaning |
| --- | ---: | --- |
| `EXACT_ACCEPTED_DATASET` | 0 | no candidate passed the complete identity contract |
| potentially matching | 0 | no candidate reached metadata or sidecar verification |
| incomplete copy | 0 | no partial export with recoverable accepted identity was found |
| `NOT_FOUND` on explicit remote roots | 4 | expected repo, persistent mount, and cache roots were absent |

The GitHub repository has no release asset containing the dataset and no tracked LFS object for it.
The search found only source/configuration references to Phase 2B, never dataset media or complete
LeRobot split roots.

## Integrity conclusion

Because `exact_accepted_dataset_count=0`, no `AcceptedDatasetPackage` exists. Copy integrity,
post-copy identity, six-split revalidation, real-batch loading, micro-overfit, checkpoint/resume,
formal training, checkpoint selection, development evaluation, sealed evaluation, and policy
promotion are all recorded as not run. This is the required fail-closed outcome.

The reachable RTX 5090 remained idle for training. The server was left online.
