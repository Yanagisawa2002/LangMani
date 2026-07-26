# Phase 2C-C result

Status: complete as **Case B**.

The consumer pipeline was valid, the relative transform was exact and structurally bounded, and
the sole relative Pick SmolVLA completed its frozen 20,000-step run. The selected policy achieved
1/30 success on accepted training resets but 0/30 on validation. This satisfies the pre-registered
Case-B condition:

```text
action_formulation_partially_validated = true
relative_action_formulation_validated = false
absolute_action_material_bottleneck = false
langmani_generalization_route_blocked = true
langmani_custom_model_route_eligible = false
```

## Answers to the decision questions

1. **Could the absolute model solve training resets?** Yes, but only 1/30 at H=8. Its accepted
   validation baseline remained 0/30.
2. **Was the relative transform exact and bounded?** Yes. The 100,000-chunk audit had zero
   non-finite values, bound violations, clipping, projection, or replacement; all accepted Pick
   frames reconstructed within `1e-6`, with maximum error `7.152557373046875e-07`.
3. **Did relative training improve phase-conditioned action error?** Yes offline. Overall physical
   MAE improved from `0.0644338` to `0.0525535`, and first-action MAE from `0.0370820` to
   `0.0192859`. Every frozen phase's first-action and full-chunk MAE improved.
4. **Did relative SmolVLA solve training resets?** It solved 1/30, the pre-registered minimum for
   meaningful fit, while 29 timed out.
5. **Did relative SmolVLA solve validation resets?** No: 0/30, despite 22 grasps and four lift
   events. The unopened 50-reset test result is unavailable.
6. **Was absolute action formulation the main bottleneck?** No material-bottleneck conclusion was
   established. Relative actions improved offline error and contact behavior, but did not produce
   validation success and did not improve the 1/30 training-reset success count over the absolute
   model.
7. **Should LangMani continue or pivot?** Pivot to a standard benchmark with an established
   working policy/data contract. Case B permits a separately authorized observation/view
   ablation, but this final formulation experiment does not justify more custom LangMani
   model-family training.

No automatic next stage is authorized. Shared/Stack/Push SmolVLA, VLA-JEPA, ACT changes, a second
seed, new data, a repair phase, and final unseen-reset evaluation did not run and remain
unauthorized.
