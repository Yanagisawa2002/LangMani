# Phase 2B.6 Archive and Restore Report

## Result

Archive creation and restore validation were not started. This is not an archive failure
(`RESULT_D`); the earlier physical replay hard stop already classified the phase as `RESULT_C`.

## Locations

- Partial production root:
  `/root/autodl-tmp/langmani-external/phase2b6/production_v1`
- Reserved archive root:
  `/root/autodl-tmp/langmani-external/phase2b6-archive`
- Reserved restore scratch:
  `/root/autodl-tmp/langmani-external/phase2b6-restore-scratch`

At closeout the partial production root occupied 20,770,745,988 bytes. The archive and restore
paths did not exist. Consequently there is no content-addressed archive path, archive size,
archive SHA-256, restored tree digest, or valid recovery command for an accepted package.

The partial replay bytes remain in place and were not deleted or repackaged. They must not be
described as a recoverable accepted dataset. The compact Result C evidence is independently
hashed by `artifact_manifest.json`; its manifest fingerprint is
`sha256:a4b4ace15b3d5bd5dfb2215507abd14fcbe1a0a21dce0f875d6d91dcc676f8a0`.
