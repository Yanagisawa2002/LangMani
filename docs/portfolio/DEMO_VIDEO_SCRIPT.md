# LangMani v1 demo video script

Final duration: 3:50 (230 seconds). The generated demo uses verified M3B target-smoke policy-camera footage
only as a source-data demonstration. It does not present that footage as an M5A sealed-final learned
rollout.

| Time | Visual | Narration / caption |
| --- | --- | --- |
| 0:00–0:15 | Title and six-task grid | “LangMani is an auditable language-conditioned manipulation pipeline for a Panda robot.” |
| 0:15–0:40 | Task scene and typed TaskSpec | “Three cubes and two bins yield six semantic tasks. Scene and task identities are separate.” |
| 0:40–1:10 | Architecture flow | “Language routing, controller selection, action bounds, environment truth, and verification are explicit boundaries.” |
| 1:10–1:50 | Real M3B target-smoke footage | “This is verified source-demonstration footage used by the data pipeline; it is not relabeled as learned final control.” |
| 1:50–2:15 | M3A/M3B counters | “Sixty complete counterfactual groups produced 360 action-replayed raw episodes and 360 aligned LeRobot episodes.” |
| 2:15–2:45 | ACT comparison | “Eight ACT runs show the one-to-many conflict: PerTask 143/180 fresh, unconditioned 18/180, oracle OneHot 101/180.” |
| 2:45–3:10 | Sealed final result | “The final pipeline and physical verifier passed, while the quality gate failed: 46/72 learned versus 55/72 Oracle.” |
| 3:10–3:50 | Four cases | Success; safe false rejection; correct-route timeout; unsupported-action zero-execution rejection. |
| 3:50–4:05 | Honest conclusion | “Zero unsafe wrong-object or wrong-bin routes, but coverage and controller timeouts remain unresolved. SmolVLA was not started.” |

## On-screen source labels

- `REAL M3B TARGET-SMOKE SOURCE FOOTAGE — NOT M5A FINAL ROLLOUT`
- `final_pipeline_validated=true`
- `physical_target_validated=true`
- `final_quality_gate_passed=false`
- `smolvla_go=false`

## Asset provenance

The source clip retained under `outputs/portfolio/v1/source_media/` has SHA-256
`3f658057526e047683a4c0ea7e41afa5be319c52c5aa4b8a44cf79ce8aa1518a`, duration 53.4 seconds,
resolution 256×256, 20 FPS, and H.264 encoding.
