# LangMani v1.0 frozen results index

This index is the source-controlled entry point for the LangMani v1 release. It freezes the
interpretation of the existing evidence; it does not rewrite, regenerate, or promote any model,
dataset, checkpoint, or experimental result.

## Release identity

| Field | Frozen value |
| --- | --- |
| Release | `v1.0.0` |
| Release scope | M0 through M5A, plus the structurally complete but target-blocked M5B bridge |
| M6 scope | Release and portfolio only; no v1 runtime or experiment changes |
| M5A final producer Git | `0c5bb7de045e936cd6a3ce850d620bba1d923a3d` |
| M5A final verifier Git | `b2489c73a10546ed1d4a2956649096d6255cd980` |
| M5A final run fingerprint | `sha256:d4d176d43cd9b6602f0a0e6c352e95a950a2fdbe96913ea0b12401f28712d2e5` |
| M5A artifact fingerprint | `sha256:2bfeaefdfb9a4ccb6a6e9a34a57ec053b06c1cb9cfc5ed0e98f5ab67863157c1` |
| M5A verifier fingerprint | `sha256:01c7add759f3e6691b5c8aa011c23db56389bb8ee10a8f964edb9aedacc52c08` |
| M3B export fingerprint | `sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4` |

The Git tag identifies the release materials and source tree. Large generated evidence, datasets,
checkpoints, PDF files, and videos remain outside Git by repository policy.

## End-to-end result

The separately authorized M5A sealed final is the terminal v1 evaluation. Its pipeline and physical
execution were valid, while its conjunctive quality gate failed. That distinction is intentional.

| Measure | Result |
| --- | ---: |
| Final language examples | 600 |
| Oracle / NeuroSymbolic control atoms | 72 / 72 |
| Exactly paired initial states | 72 / 72 |
| NeuroSymbolic correct routes | 61 / 72 (84.72%) |
| NeuroSymbolic end-to-end success | 46 / 72 (63.89%) |
| Oracle controller ceiling | 55 / 72 (76.39%) |
| Safe false rejections of routeable commands | 11 |
| Correctly routed control timeouts | 15 |
| Unsafe wrong-object / wrong-bin routes | 0 / 0 |
| Off-table / infrastructure failures | 0 / 0 |
| Rejection probes with zero runtime work | 10 / 10 |

Frozen flags:

```text
final_pipeline_validated=true
physical_target_validated=true
final_language_safety_quality_passed=false
final_rejection_taxonomy_quality_passed=false
final_control_quality_passed=false
final_quality_gate_passed=false
smolvla_go=false
```

## Milestone ledger

| Milestone | Frozen outcome | Main evidence claim |
| --- | --- | --- |
| M0–M2 | Complete | Native CUDA/Vulkan/PhysX environment and 177/180 expert benchmark |
| M3A | Complete | 60 complete counterfactual groups, 360 action-replayed raw episodes |
| M3B | Complete | 360 aligned LeRobot episodes, exact 288/36/36 scene-group split |
| M4 | Complete, quality gate false | Eight reproducible ACT runs; experiment and physical validation passed |
| M4.2 | Complete, TaskToken rejected | 15/72 development successes; final schedule remained sealed |
| M4.3a | Complete | Shared-policy semantic-alignment audit on frozen observations |
| M4.3b | Complete, FactorFiLM rejected | 38/72 development successes; physical verifier passed |
| M5A | Complete, final quality gate false | Safe modular language routing over six frozen PerTask ACT controllers |
| M5B | Structural implementation complete; target probe blocked | Zero-action JSON bridge fails closed without accepted controller artifacts |
| M6 | Release/portfolio | No v1 behavior changes |

## Router comparison on the sealed final language set

| Router | TaskSpec accuracy | False-route rate | False-rejection rate | Final schema-valid | Status accuracy | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RuleRouterV0 | 83.33% | 7.92% | 16.67% | 100% | 74.50% | 0.19 ms |
| Frozen classifier | 98.06% | 0% | 1.94% | 100% | 93.00% | 5.25 ms |
| Qwen3-1.7B | 100% | 24.58% | 0% | 96.83% | 81.83% | 460.12 ms |
| Qwen3-4B | 99.17% | 3.33% | 0.83% | 98.00% | 94.50% | 634.81 ms |
| NeuroSymbolicRouterV0 | 78.33% | 0% | 21.67% | 100% | 74.17% | 2056.98 ms |

Only `NeuroSymbolicRouterV0` reached the sealed physical evaluation. The other rows are descriptive
offline baselines, not alternative deployed systems.

## Control baseline summary

| Control | Locked test | Historical fresh seeds | Interpretation |
| --- | ---: | ---: | --- |
| Six ACT-PerTask policies | 31/36 | 143/180 | Strongest v1 control library |
| ACT-Mixed-Unconditioned | 5/36 | 18/180 | Counterfactual one-to-many conflict is severe |
| ACT-Mixed-TaskOneHot | 27/36 | 101/180 | Oracle task conditioning resolves much, not all, conflict |
| ACT-Mixed-TaskToken | — | 15/72 development | Rejected |
| ACT-Mixed-FactorFiLM | — | 38/72 development | Rejected |

## Portfolio entry points

- [Technical report source](portfolio/TECHNICAL_REPORT.md)
- [System architecture](portfolio/ARCHITECTURE.md)
- [Demo video script and shot list](portfolio/DEMO_VIDEO_SCRIPT.md)
- [Generated artifact identities](portfolio/ARTIFACTS.md)
- [Resume, project description, and interview talk track](portfolio/RESUME_AND_INTERVIEW.md)
- [Success case](portfolio/cases/SUCCESS.md)
- [Safe false-rejection case](portfolio/cases/FALSE_REJECTION.md)
- [Correct-route timeout case](portfolio/cases/ROUTED_TIMEOUT.md)
- [Safe unsupported-action rejection](portfolio/cases/SAFE_REJECTION.md)

## Evidence locations

The synchronized compact M5A release evidence is retained locally under:

```text
outputs/diagnostics/m5a/sealed-final-synced-0c5bb7d/
```

The M6 generated artifacts are retained locally under:

```text
outputs/portfolio/v1/
```

Neither directory is a substitute for the source-controlled contracts. Generated outputs remain
untracked by design.

## Release validation

- Ruff formatting and lint: passed.
- CPU-safe suite: 1,388 passed, 15 platform/privilege skips, 16 GPU/rendering deselections.
- Isolated sdist and wheel build: passed in `outputs/build/m6-release/`.
- Markdown local-link audit and frozen-evidence assertions: passed.
- Technical report: 10/10 pages rendered and visually inspected.
- Demo video: 230 seconds, 2,760 frames, H.264 + AAC, ten timeline points decoded and inspected.
