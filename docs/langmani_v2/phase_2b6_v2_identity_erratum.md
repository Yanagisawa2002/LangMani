# LangMani 2.0 Phase 2B.6-v2 identity erratum

This erratum resolves a human-document transcription error without changing
the accepted dataset, its package identity, or any dataset bytes.

## Package fingerprint

The incorrect transcribed value was:

```text
sha256:77675e2134e4886a97e4bdac2230c64c3da30cc080e647433b7c701a79544ed04
```

It contains 65 hexadecimal characters and cannot be a SHA-256 digest. The
canonical value is:

```text
sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04
```

The canonical value is established independently by
`artifacts/langmani_v2/phase_2b6_v2/accepted_multiskill_dataset_package.json`
field `fingerprint` and
`artifacts/langmani_v2/phase_2b6_v2/result_classification.json` field
`accepted_package_fingerprint`.

The package fingerprint is recomputed by removing the top-level `fingerprint`
field, serializing the remaining JSON with sorted keys, UTF-8,
`ensure_ascii=false`, `allow_nan=false`, and separators `(",", ":")`, then
applying SHA-256 and adding the `sha256:` scheme prefix. The recomputed value
equals the canonical machine-readable value.

## Payload tree and control sidecar

The immutable archive and its clean restore establish the frozen payload:

- file count: 168
- total bytes: 2,779,801,643
- tree digest:
  `sha256:b0e255d8b3f7f77fe33538cb2c313620d1d00adf90562b33dad06b10674474b3`

The live primary tree contains that exact payload plus the post-finalization
self-describing control-plane sidecar
`metadata/accepted_multiskill_dataset_package.json`:

- live file count: 169
- live total bytes: 2,782,385,671
- live tree digest:
  `sha256:ef7b246d4a43d2ab473b9e710425dbaeabdcb70ff9dcbc841099d719ea8db5c2`
- sidecar bytes: 2,584,028
- sidecar SHA-256:
  `sha256:940272b9dc6386a356f27288870a5e0667d22678fb268f141d15c0c100462a59`

The sidecar was written by
`langmani.v2.phase2b6_v2_finalize.finalize_acceptance`, whose implementation
was introduced at
`40161abc310995b0d8d7bf444df7ba454c1619fc`. Its bytes match the accepted
package committed with the final Phase 2B.6-v2 evidence at
`19de56781e029c20859839a1e3493920449ff9bf`.

The semantic relationship is:

```text
frozen_payload_tree
-
post_finalization_control_sidecar
```

The sidecar is excluded only from the frozen payload-tree identity. It is not
deleted and the 169-file primary tree is not redefined as the archived
payload. A complete common-file hash comparison confirms that no media,
state, action, split, statistics, normalization, or source-lineage bytes
changed.

The compact authorization-bound resolution is
`artifacts/langmani_v2/phase_2c_a/identity_resolution.json`; the independent
verifier is `environment/verify_v2_phase2c_a_identity.py`.
