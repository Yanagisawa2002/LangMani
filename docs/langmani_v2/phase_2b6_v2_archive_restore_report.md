# LangMani 2.0 Phase 2B.6-v2 archive and restore report

The accepted primary tree was archived outside the production root and
restored into a new physical scratch directory. Neither archive nor restore
uses a symlink to the primary tree.

## Primary tree

- location:
  `/root/autodl-tmp/langmani-external/phase2b6-v2/production_v2/primary`
- file count: 168
- size: 2,779,801,643 bytes (approximately 2.59 GiB)
- tree digest:
  `sha256:b0e255d8b3f7f77fe33538cb2c313620d1d00adf90562b33dad06b10674474b3`

The primary tree contains the task/split roots, unified metadata, accepted and
excluded inventories, language/split/fold metadata, statistics,
normalization, visual-shift roots, and verification references. Official
source ZIP bytes are not duplicated; their immutable references and hashes
are included.

## Content-addressed archive

- location:
  `/root/autodl-tmp/langmani-external/phase2b6-v2-archive/LangManiOfficialMultiSkill-v2-b0e255d8b3f7f77fe33538cb2c313620d1d00adf90562b33dad06b10674474b3.tar.gz`
- size: 2,709,340,363 bytes (approximately 2.52 GiB)
- SHA-256:
  `sha256:a43af6568fa46e074efb3d94d4e370a1e5d7c2d7d1ad300c1d4d20f65b7d94f3`
- archive prefix: `LangManiOfficialMultiSkill-v2`
- archive manifest fingerprint:
  `sha256:034a703904e747c16ef10a7d28d847d8adccd9b86546bb45a96d952c72687f5a`

The first archive report exposed a serialization defect that duplicated the
`sha256:` scheme prefix. The archive bytes themselves were not overwritten or
deleted. Commit `8fc84141aaf137169917286f8ddd6243fc88bcae` added a fail-closed
repair path that reused the archive only after the prior manifest fingerprint,
path, tree digest, prefix, size, and actual file SHA all matched. The corrected
manifest contains one scheme prefix, and an independent full-file hash equals
the recorded value.

## Clean restore

- location:
  `/root/autodl-tmp/langmani-external/phase2b6-v2-restore-scratch/b0e255d8b3f7f77fe33538cb2c313620d1d00adf90562b33dad06b10674474b3/LangManiOfficialMultiSkill-v2`
- archive SHA verified: true
- complete restored tree digest verified: true
- restored files: 168
- restored bytes: 2,779,801,643
- restored primary episodes/frames: 2,998 / 254,200
- task/split roots: 15
- stratified LeRobot API samples: 45/45 passed
- symlink to primary: false
- restore manifest fingerprint:
  `sha256:7a5b7407581e22a50482e1556dc5da1a12c67f09522da44f2de2cb138fdf54c9`

The scratch restore remains present for inspection. It was not deleted after
evidence finalization.

## Reproduction commands

```bash
bash environment/run_v2_phase2b6_v2_production.sh \
  --conda-prefix /root/autodl-tmp/conda-envs/langmani \
  archive \
  --production-root /root/autodl-tmp/langmani-external/phase2b6-v2/production_v2 \
  --evidence-root /root/autodl-tmp/langmani-external/phase2b6-v2/production_v2/evidence \
  --archive-root /root/autodl-tmp/langmani-external/phase2b6-v2-archive

bash environment/run_v2_phase2b6_v2_production.sh \
  --conda-prefix /root/autodl-tmp/conda-envs/langmani \
  restore \
  --production-root /root/autodl-tmp/langmani-external/phase2b6-v2/production_v2 \
  --evidence-root /root/autodl-tmp/langmani-external/phase2b6-v2/production_v2/evidence \
  --restore-root /root/autodl-tmp/langmani-external/phase2b6-v2-restore-scratch
```
