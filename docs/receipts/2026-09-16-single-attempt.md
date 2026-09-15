# Bounded attempt receipt — 2026-09-16

## Result and scope

**Learned ACT demonstration: SKIPPED at dependency preflight.** Documentation and byte-integrity
auditing are the delivered scope. The existing checkpoint was found and verified; the target
runtime was unavailable. No model was loaded, no simulator was created, no episode/seed was run,
and no new video exists. Success/failure rate is **unavailable**, with zero executed episodes.

- Run ID: `langmani-single-attempt-20260916`.
- START: `2026-09-15T18:50:48Z`; total deadline: `2026-09-15T20:20:48Z`.
- Base main: `6920b52c1f48c278e669cd71b69b8949dd900f3a` (fresh origin fetch).
- Branch: `codex/single-attempt-20260916-langmani`.
- Isolated, clean v1 source inspected: `58434cb17a7234b6d4b2c4fb15aecf8df0621487`.
- No runtime, checkpoint, dataset, historical schedule, or quality threshold was modified.
- The sole preflight resolution command was bounded to 180 seconds and terminated with **124**.
- No second dependency attempt, alternative version, training run, or rollout followed.

## Target preflight

At `2026-09-15T18:51:57Z`, the granted native Ubuntu target's identity matched the coordinator's
record. A real `flock` covered each remote command and all of the dependency command's child
processes. NVIDIA RTX 5090: 32,607 MiB, 2 MiB used, zero compute processes; driver 595.58.03.
The task only wrote its own campaign subdirectory and redirected cache/temp files there.

| Dependency | Base runtime observed | Frozen v1 declaration |
| --- | --- | --- |
| Python | 3.12.3 | 3.12.13 in environment.yml; package requires >=3.12,<3.13 |
| PyTorch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 | 2.11.0+cu128 / 0.26.0+cu128 |
| NumPy | 2.3.2 | 2.2.6 |
| ManiSkill / SAPIEN | missing / missing | 3.0.1 / 3.0.3 |
| LeRobot | missing | 0.6.0 with dataset extra |
| Gymnasium / h5py | Gymnasium missing; h5py not probed | 1.2.3 / 3.16.0 |
| PyAV | missing | resolved transitively through the declared stack |

One preliminary resolver command tested critical frozen runtime packages. It was not the full
environment acceptance command and did not install them:

```bash
timeout --signal=TERM --kill-after=10s 180s "$BASE_PYTHON" -m pip install \
  --dry-run --ignore-installed --only-binary=:all: --report "$OWN_ROOT/preflight/resolution.json" \
  --retries 0 --timeout 20 --extra-index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' 'numpy==2.2.6' \
  'gymnasium==1.2.3' 'h5py==3.16.0' 'Pillow==12.3.0' 'sapien==3.0.3' \
  'mani-skill==3.0.1' 'lerobot[dataset]==0.6.0' 'safetensors==0.8.0' \
  'transformers==5.4.0'
```

The log reached the 101.7 MB ManiSkill wheel after SAPIEN's 51.3 MB wheel; observed downloads
were approximately 0.4–0.5 MB/s. Timeout stopped the route before completed dependency resolution.
This proves an unavailable environment within this attempt's bound; it does **not** establish
that RTX 5090 is generally incompatible with LangMani or that the ACT model failed.

Original downloaded `resolution.log` SHA-256:
`f23636f22b4a9062092abb95adb0965baba4aa52d64f7f4906e29d0545726c18`.
Original `resolution.exit` contains `124` plus LF, SHA-256:
`ca2ebdf97d7469496b1f4b78958f9dc8447efdcb623953fee7b6996b762f6fff`.

At `2026-09-15T18:58:00Z`, a separate identity-checked closeout acquired the same lock and found
zero task-owned processes and zero GPU compute processes. Task directory size was 159,034,483
bytes; available disk was 323 GiB. The closeout exited 0 and released its lock. No shutdown or
deletion of server assets was performed by this task.

## Verified checkpoint recovery identity

Recovery paths below are relative to the preserved local source checkout, not downloads or
claims that model files are hosted in Git. This is the single existing red-to-left controller;
there is no claim of six-controller recovery or fresh model deserialization.

- Checkpoint directory: `outputs/models/act/4e089b39ed36b55cd68a0a8ad00012ed24d0788199c6eb6675ac39d7fad30c53/checkpoints/step-00095000-01cbd124005e`.
- Checkpoint fingerprint: `sha256:01cbd124005e12c04acc67258a48f19b9327364ab78fa7954a359f03152853d9`.
- Training producer: `ceb73db1a0fe88fa7f58f347b51c542aa76663a0`; step 95,000.
- Task: `langmani-pick-place-task-v0:red_cube:left_bin:canonical_v0`.
- `checkpoint_manifest.json`: `bcb1701b11dee720126dfa5ff06ecdf363b57147995cb0ffbc4e3d1594951449`.
- `complete.json`: `63da3d8d9dae0d84a8a362d9bbfe771daac203e28569c4a88e5edcccfbf00b96`.
- Completion marker manifest hash matched; every manifest-covered size/hash matched.

| File | Bytes | Verified SHA-256 |
| --- | ---: | --- |
| `pretrained_model/config.json` | 1501 | `7f09ff69ef42326a6ab5c79f0785ebd40cd9b976fa69e2596e82f5742c004621` |
| `pretrained_model/model.safetensors` | 206515416 | `56a30fda4c0157946fcdad990bad5157f7c2a317c7005aeba9fa36b805608301` |
| `pretrained_model/policy_postprocessor.json` | 660 | `ad2d49a0f67981d2e0a1bee69d85a816654b3159365829093227baa46a74d8bb` |
| `pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors` | 1524 | `c043c76eab4e1d2ee4040c8892923c1aae0a0c47b4c252a8fb1d1118778bf781` |
| `pretrained_model/policy_preprocessor.json` | 1160 | `eaff91dc5004c22473578eecd7cbe14566a0ab1d5c63f9edc41d3cc4d0bc270b` |
| `pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors` | 1524 | `5cbdf752bf5ac112b1e8188a3bb53964852fa2c6795e644476e96cd70e68758c` |
| `training_state/optimizer_param_groups.json` | 3263 | `fe7ff611a203c8aeee519daa2d8960d97cefd5d592fca26d460b0b11f2106f90` |
| `training_state/optimizer_state.safetensors` | 412653828 | `93e126d1624e52b878a5432bf30e7eb2549a46ab1d18c4a8a584a8110cdb3fcd` |
| `training_state/rng_state.pt` | 14645 | `3a62a74881eb4dfc7338796ce0253c64f3f963a6e4d1bf07400344063cd203af` |

## M5A historical evidence recovery audit

Only existing historical reports and file integrity were audited. This work did not regenerate
sealed schedules, run final examples, or perform policy selection.

Source-relative directory: `outputs/diagnostics/m5a/sealed-final-synced-0c5bb7d`.

Manifest SHA-256: `9fc9007556bfb1c4331fe2aef4b4f5d660bc79e2484f190b361794536513b270`.

Saved independent verifier SHA-256: `2d50136806021281e541133d43f46658b05851a63378c29f5a1e4844a749734f`
(`sealed-final-independent-verification-b2489c7.json`).

The report records `paired_initial_state_count=61`, `final_paired_states_validated=false`,
`final_pipeline_validated=true`, `physical_target_validated=true`, `final_quality_gate_passed=false`,
and `smolvla_go=false`. The source-controlled correction is in
[Versions and evidence](../VERSIONS_AND_EVIDENCE.md).

**24 manifest entries: 16 MATCH, 8 MISSING, 0 MISMATCH.** Missing entries have an expected hash
only; those hashes are not claims that missing bytes were independently verified.

| File | Status | Manifest SHA-256 |
| --- | --- | --- |
| `control_quality_gate.json` | MATCH | `36139e089c5ab9b745ac3e6f74383d587b5e1ef56d5d47aa428ade9b3aa121c1` |
| `control_schedule.json` | MATCH | `b601c99561ff4e7838592120877e4830287565f2c8c01a5ab396c2e62bb2e29a` |
| `controller_registry.json` | MATCH | `9b2edb98b974079389d4b97db423ba11ddab60262f9cf2a44ade86b45912455a` |
| `failure_attribution.json` | MATCH | `2b951f4e29f272a7808566db5b44768603d744e10e17d8e92bac741b5cc83a7f` |
| `final_authorization.json` | MATCH | `92022568fd93b0e35e9b48a9d9c5d5e91aa186b8770fa8ea3dafa9a62b203492` |
| `final_result.json` | MATCH | `7a16e1e5e933a08744ca15ee6500c2b2536f4f1fba6ead26709fff79a6fef115` |
| `final_run_identity.json` | MATCH | `77a752302bd487fd238b5947e5e3ca5fb48603b731e3da838f1a084767d6e195` |
| `language/classifier_negative_baseline.json` | MISSING | `2e76c7518d7bf4d7b305841381c70b703f226484d2f583e5092a973f505afd59` |
| `language/model_identities.json` | MATCH | `21494d3a7ef0226fd83080b1db3f1b394fb6d75dc4c2e3c942f94e9861c1ebdf` |
| `language/neuro_symbolic_final.json` | MISSING | `d3190cfe99259fed3077ef746f1200ec37fe2a17798eae79db29d86c8ee6d8e4` |
| `language/qwen17b_negative_baseline.json` | MISSING | `e0a287ea2eccb91b98e3181f217b466896a7b1fe3b1b3fa354943cc8f05053ac` |
| `language/qwen4b_negative_baseline.json` | MISSING | `79254fabb0f036ce6988035a18c126ddc7855d34b5e0df12dc6166a4c215c6ea` |
| `language/rule_router.json` | MISSING | `3ea11beb9aa41914fabb32473cb5235b4c76bd528a2fa681a0dd4ef1b43f324e` |
| `language/system_metrics.json` | MATCH | `28aaefc264601cf7f0a13c7ee1d89b4382ff19df8d0dde9b9ee04f7608e51532` |
| `language_quality_gate.json` | MATCH | `01fe1f5c3773e95dd4bd9f1ea7063c6a91ae687e11b8dc49618d9a867a373463` |
| `language_schedule.json` | MISSING | `9f07d8e4ab85bd0a88664edac32d6d821b13a61747bd75724218e1c07a46d5cb` |
| `neuro_symbolic_episodes.json` | MISSING | `eec672c5ebff19fff38e0a5f4d0b62e662a0cbd521fb6d9f09610ea7db6c83b4` |
| `oracle_episodes.json` | MISSING | `1c5b389769e1485589d322c45af4ce219f0a8dd9f5c3083eacefdd01ac7dd190` |
| `owner.json` | MATCH | `d32560b1111c4ed7153cd30c4b1b0e1c2d4a644a56ae7ab15295602f56a6f647` |
| `paired_results.json` | MATCH | `f7cca9e6362933c2ddaa18d6ae7f8c8365e3d2830b0cd78cc09957c9ef1178ba` |
| `rejection_probes.json` | MATCH | `dce52edcfe8538eb0059261d502d64ec7e627ef11c246d4f3a633eff3bda73db` |
| `router_identity.json` | MATCH | `014ca64cf78ab5e0ab7f7102feee05fead005daaa26165fc1043197b05a73dbe` |
| `runtime_identity.json` | MATCH | `8463a81b5475dc0c0bc386abb443164c6ddc628326ce384177667901a598c284` |
| `summary.md` | MATCH | `e5a66e5b655867a8b2f4ed719f839d68c173c5f07359d1e632535f62989b82c4` |

## Audit method and local evidence

The read-only asset audit uses SHA-256 over original file bytes and compares each recorded size.
For the checkpoint it additionally hashes the manifest and compares that digest with the
completion marker. Missing files are reported without substitution. File-integrity matches do
not mean a checkpoint was deserialized or a historical benchmark was rerun.

The task's local evidence directory contains the original resolver log, remote inspection and
closeout transcripts, read-only audit script, and the two detailed JSON audit reports. The final
coordinator receipt provides their absolute paths and SHA-256 values. The compact audit tables
above are the reviewable repository record; generated datasets, weights, and raw experiment
outputs are not committed.

## Frozen local verification

Pending the single local validation pass. This section will receive only the observed command
results after the documentation candidate is committed. No physical target acceptance is claimed.
