# Phase 2C.1 accepted-dataset recovery plan

## Objective and stop rule

Recover the original accepted Phase 2B bytes without regenerating, rewriting, or repairing them.
No real batch and no optimizer step may run until one candidate independently re-passes the
complete identity contract below. If the original bytes cannot be found, Phase 2C.1 ends as Result
B with zero optimizer steps.

## Frozen expected identity

Every value below comes from committed Phase 2B acceptance evidence or the accepted producer
contract. A value that was not retained is marked unavailable rather than inferred.

| Field | Frozen value | Evidence |
| --- | --- | --- |
| `expected_dataset_identity` | `langmani/phase2b-push-v1`; accepted export fingerprint must be read from the recovered exact export manifest | Phase 2B collection config and export contract |
| `expected_dataset_root_name` | `phase2b-push-lerobot-v1` | Phase 2B README and accepted audit |
| `expected_lerobot_version` | `0.6.0`; v3 layout | Phase 2B result and runtime audit |
| `expected_episode_count` | 397 total; train 203, validation 77, unseen scene 26, unseen language 40, hard 23, visual shift 28 | accepted split manifest summary |
| `expected_frame_count` | 53,297 total; 26,968 / 10,425 / 3,382 / 5,277 / 3,210 / 4,035 by the split order above | accepted split manifest summary |
| `expected_feature_schema` | `observation.images.base_camera`, `observation.state`, `action`; task strings via LeRobot `task_index`/task metadata; no privileged policy fields | Phase 2B dataset audit and producer source |
| `expected_camera_keys` | exactly `observation.images.base_camera`, RGB HWC `uint8/video [256,256,3]` at rest and CHW `[3,256,256]` on readback | Phase 2B dataset audit and feature contract |
| `expected_state_dimension` | 9, `PandaPolicyStateV0`, float32 | Phase 2B dataset audit |
| `expected_action_dimension` | 8, `PandaJointPositionActionV0`, float32 `pd_joint_pos` | Phase 2B dataset audit |
| `expected_instruction/task schema` | non-empty rendered push instruction, stored through LeRobot task metadata; canonical TaskSpec remains a sidecar | Phase 2B dataset audit |
| `expected_fps` | 20 Hz control; 100 Hz simulation in the raw authority | Phase 2B config and audit |
| `expected_video/image encoding` | PyAV H.264, `yuv444p`, CRF 18, GOP 2, medium preset, one encoder thread | frozen `VideoCodecConfig` used by the accepted producer |
| `expected_split or episode grouping` | six stable episode-level splits; all eight overlap counters zero; 397 unique episode assignments | Phase 2B dataset audit |
| `expected source commit` | accepted Phase 2B result `7394faba1ed2e29ab26b70dbc4058f46e5164041`; collection producer `8dc9f4e24598d4a0266d8afe15f50603eae14515`; top-up producer `f081275478a72aa90b7b6529874c974a02e695dc` | Phase 2B result manifest |
| accepted total bytes | 141,422,866 | Phase 2B result manifest |

The accepted evidence did not preserve a complete per-file digest list, a combined dataset-tree
digest, or the six `meta/info.json` byte digests. Those values are therefore unavailable before
recovery. A candidate cannot be accepted by path or counts alone. It must match the retained
sidecar bytes, complete tree/count/schema/task/timing evidence, and a newly recomputed full tree
digest, then re-pass the independent Phase 2B verifier.

## Frozen manifest hashes

| Artifact | SHA-256 |
| --- | --- |
| committed Phase 2B result manifest | `a724f518bef1bcbc4b5fc11a91e3af1758619903be976f4bfbe509176225c2ad` |
| accepted replay validation | `dacb7db3b980997f104b3b580b8ea27a58dbacaec8e5307ad3b16d06b1638c65` |
| accepted export manifest | `4bf966a83da6bcf6af7c4d53ed45e4efca8dea659d53618fef846c4ab42147f2` |
| accepted dataset statistics | `cd16ad2179a2c225f28bf9801d1ab0cd0ec2103e7c6eb756cdddac5875b627c7` |
| accepted split manifest | `5d9df54e3ac06c4e031c450f9677f069392d2639d0314b75d7477827567d1833` |
| accepted leakage audit | `231f061c9362ad7345b419c58be084536cc65ab35feabd3b90577719ad7916b5` |
| accepted export completion marker | `91835efab01effc5331e6560001b84993d4448e98390a9a93a1ea8cd0a1e552c` |
| accepted unified multi-root index | `864b8d74593143fb907f1d838b243c27b0d2e239745fdf90ed3ac9bbb84b9313` |
| accepted compatibility report | `00a2b579ca5e7c4bad4f1796d2f61c57a3f455432167f12f9ccbdac9a2483858` |
| accepted independent verifier report | `b07f97c8ade07e23eac8b710dfee801ebd8f8ef5df44e92c967958122b0310ee` |
| Phase 2C frozen protocol | `adae906d77acfc91a44b042d8ac6d3bccc5f4f796b9695c94ed4c790ae6e73b5` |
| Phase 2C base preflight manifest | `00f9f5419d92e504a15da316ce767a18190adcd09d17fd0e30801e1cd2c3b1f1` |

## Required `meta/info.json` evidence

Every split must retain at least the LeRobot v3 fields `codebase_version`, `robot_type`,
`total_episodes`, `total_frames`, `fps`, and `features`. The exact accepted split counts must match;
the only video feature must be `observation.images.base_camera`; state and action must remain
float32 `[9]` and `[8]`; `task_index` must remain int64 and `meta/tasks.parquet` must exist. Missing
or changed metadata is a hard failure. Metadata is never synthesized or repaired.

## Recovery protocol

1. Audit every explicitly reachable local, remote, cache, archive, LFS, and release location as
   read-only.
2. Classify candidates only as one of the seven Phase 2C.1 statuses.
3. For a plausible candidate, rerun the accepted sidecar, split, readback, and independent Phase 2B
   verifier gates at the source location.
4. Build `AcceptedDatasetPackage` only after every gate is true and hash every file into a complete,
   path-independent tree identity.
5. Copy through an unaddressed hidden staging directory, rehash it, atomically rename it to the
   expected destination, then rehash the activated destination.
6. Run data-summary revalidation. Only then may the real-batch smoke execute.

The unchanged downstream order is real-batch smoke, 16-episode/500-step micro-overfit, checkpoint
save/reload, interrupted resume, zero-work resume, the seed-0 20,000-step formal run, validation-only
selection, development evaluation, the frozen development gate, and sealed evaluation only after
promotion. Phase 2D is never launched by this stage.

## Exclusions

- No data regeneration, re-export, metadata repair, video re-encoding, or logical-equivalence claim.
- No use of failed attempts, pick-and-place data, test identities, or privileged simulator fields
  for training.
- No repeated base-model download unless the existing pinned bytes fail their digest check.
- No training or simulator evaluation before exact dataset recovery.
