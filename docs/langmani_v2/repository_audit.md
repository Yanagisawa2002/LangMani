# LangMani 2.0 repository audit

This audit is descriptive. It does not reproduce or reinterpret the frozen v1 experiments.

## Frozen v1 boundary

- Release tag: `v1.0.0`
- Peeled release commit: `58434cb17a7234b6d4b2c4fb15aecf8df0621487`
- Frozen claims: `final_pipeline_validated=true`, `physical_target_validated=true`,
  `final_quality_gate_passed=false`, and `smolvla_go=false`.
- The source-controlled release manifest is `releases/langmani_v1/manifest.yaml`. Its validator checks
  the exact tag, commit, frozen claims, stable artifact identities, and SHA-256 hashes of the listed
  release sources.

The validator deliberately does not treat a historical metric as newly reproduced evidence. It
only verifies the source-controlled release index against the frozen release commit.

## Historical artifact inventory

The target artifact host contains the authoritative M3B dataset, eight M4 ACT run directories,
the M4.2 runtime selection, and later M4/M5 evidence. The local Windows checkout intentionally does
not contain the generated dataset or model directories.

| Required artifact | Classification | Verified location or source | Audit conclusion |
| --- | --- | --- | --- |
| six selected PerTask ACT checkpoints | present but ambiguous as a single v2 default | `outputs/models/act/<run-fingerprint>` on the target artifact host | all six run manifests are complete and each has a validation-only `checkpoint_selection.json`; Phase 1 explicitly chooses one below |
| canonical ACT model weights | repository-configured and valid on the artifact host | selected checkpoint `pretrained_model/model.safetensors` | raw file hash matches the v2 policy config |
| policy preprocessing | repository-configured and valid on the artifact host | selected checkpoint `pretrained_model/policy_preprocessor.json` and processor state | loaded by the existing LeRobot processor API; normalizer state hash is pinned |
| policy postprocessing | repository-configured and valid on the artifact host | selected checkpoint `pretrained_model/policy_postprocessor.json` and processor state | loaded by the existing LeRobot processor API; unnormalizer state hash is pinned |
| normalization statistics | repository-configured and valid on the artifact host | checkpoint processor state plus `run_manifest.json` train-statistics fingerprint | validation uses the exact train-only statistics recorded by v1 |
| M3B LeRobot metadata | present and valid on the artifact host | `outputs/datasets/m3b/langmani-pick-place-lerobot-v1` | completion, split, video, and source-provenance records exist; export and split identities are frozen in the release manifest |
| controller registry | repository-controlled and valid | `src/langmani/language/controller_registry.py` | six-entry v1 deployment registry remains unchanged; v2 adds a policy-neutral registry instead of changing it |
| M4.2 runtime selection | present and valid on the artifact host | `outputs/diagnostics/m42/runtime_ablation/runtime_selection.json` | identity is frozen in the v1 release manifest; it is not reselected by Phase 1 |
| M5A final evaluation | repository-indexed; compact evidence present locally and full evidence present on the artifact host | `outputs/diagnostics/m5a/sealed-final-*` and `docs/RESULTS_INDEX.md` | final pipeline/physical claims are accepted, quality is false, and the historical evaluation is not rerun |
| sealed split definitions | repository-controlled and valid | `src/langmani/policies/m42_schedules/m42_dev_v0.json` and `m42_final_v0.json`; M3B split digest in the release manifest | both schedule files are frozen by raw SHA-256; Phase 1 does not open either schedule |
| PerTask reported-result runner | repository-controlled and executable subject to target artifacts | `environment/verify_m4.py`, `scripts/evaluate_act.py`, `scripts/compare_act_baselines.py` | historical 31/36 and 143/180 values remain reported v1 evidence, not Phase 1 output |
| Oracle/routed final runner | repository-controlled and executable subject to final authorization/evidence | `scripts/run_m5a_sealed_final.py`, `environment/verify_m5a_final.py` | historical 55/72 and 46/72 values remain sealed v1 evidence and are not rematerialized |
| local Windows generated artifacts | referenced but missing locally | `outputs/models/act` and canonical M3B dataset | expected by repository policy; generated artifacts are intentionally untracked and must be consumed on the artifact host |
| checkpoint regeneration recipe | reproducible from existing data, but not exercised | M4 full command and frozen M3B data/configuration | regeneration is unnecessary because the original accepted checkpoint exists; a retrained model would be a new artifact, not the original |
| files irrecoverable without an external artifact host | none among the selected baseline's required runtime files | target artifact host retains the selected checkpoint and processors | loss of that host/cache would make the original weight bytes unavailable from Git; Git intentionally does not vendor them |

The source tree contains several policy-selection records for different milestones (M4 checkpoint
selection, M4.2 runtime selection, and M5A controller binding). They are not interchangeable or
contradictory: each owns a different stage. Phase 1 resolves the apparent duplication by naming one
exact checkpoint in one v2 configuration while leaving every historical selector untouched.

Historical `run_manifest.json` files can contain the absolute dataset path used on the training
machine. That path is preserved as provenance, not used for loading the deployed checkpoint. The
new canonical runtime configuration contains only repository-relative paths and rejects absolute
or parent-traversing paths.

The six PerTask selections are one policy family evaluated over six task instances. They must not
be described as six different manipulation skills. Each selection is validation-only and points
to an immutable checkpoint under `outputs/models/act/<run-fingerprint>/checkpoints/`.

The Phase 1 canonical baseline is the existing `red_cube` to `left_bin` PerTask ACT checkpoint:

- run fingerprint: `sha256:4e089b39ed36b55cd68a0a8ad00012ed24d0788199c6eb6675ac39d7fad30c53`
- selected step: `95000`
- selected checkpoint fingerprint:
  `sha256:01cbd124005e12c04acc67258a48f19b9327364ab78fa7954a359f03152853d9`
- model file SHA-256: `56a30fda4c0157946fcdad990bad5157f7c2a317c7005aeba9fa36b805608301`
- preprocessor normalizer SHA-256:
  `5cbdf752bf5ac112b1e8188a3bb53964852fa2c6795e644476e96cd70e68758c`
- postprocessor normalizer SHA-256:
  `c043c76eab4e1d2ee4040c8892923c1aae0a0c47b4c252a8fb1d1118778bf781`

Existing project validation accepted the checkpoint, processor, and component fingerprints on the
artifact host. The runtime configuration uses repository-relative paths. Absolute paths retained
inside the historical training configuration are provenance only and are not runtime dependencies.

## Missing and unreproduced items

- Generated datasets, models, checkpoints, video, and experiment outputs remain outside Git.
- No v1 training or sealed-final evaluation was rerun for Phase 1.
- No historical metric was fabricated from compact metadata.
- The real three-seed Phase 1 policy smoke requires a native Linux NVIDIA/Vulkan target with the
  canonical artifacts. A no-GPU cloud boot can validate source and artifact identity but is not
  physical or rendering evidence.

On 2026-07-21 the artifact host had 109 GB free and all canonical ACT files, but it exposed no
`/dev/nvidia*` device; the alternate cloned host was unreachable. The new smoke result is therefore
`not_run_infrastructure_gpu_unavailable`, not a reproduced success rate.
