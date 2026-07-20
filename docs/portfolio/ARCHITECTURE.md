# LangMani v1 architecture

LangMani separates physical task truth, language interpretation, policy selection, continuous
control, and verification. This prevents language strings from leaking into visual observations and
makes rejection a first-class no-dispatch outcome.

```mermaid
flowchart LR
    U["Natural-language command"] --> R["NeuroSymbolicRouterV0"]
    R -->|"route: typed TaskSpec"| G["Frozen six-controller registry"]
    R -->|"reject"| N["No lookup · no reset · no env.step"]
    G --> P["Selected ACT-PerTask policy"]
    O["RGB base_camera + 9D Panda state"] --> P
    P --> A["Raw 8D action chunk"]
    A --> B["Versioned bound handler"]
    B --> E["LangMani M1 environment"]
    T["Oracle EpisodeSpec"] --> E
    E --> V["Batched task evaluation"]
    V --> Q["Provenance + failure attribution"]
    R --> Q
    G --> Q
    B --> Q

    M2["M2 privileged expert"] --> M3A["M3A raw archive"]
    M3A --> M3B["M3B LeRobotDataset v3"]
    M3B --> P
```

## Trust boundaries

1. **M1 task truth** — reset `TaskSpec` and `EpisodeSpec` define the physical episode; natural
   language never enters numeric observations or per-step info.
2. **Language safety** — a rejection contains no executable TaskSpec and returns before controller
   lookup, reset, policy inference, or simulation.
3. **Frozen skill library** — the deployed low-level library consists only of six selected PerTask
   ACT controllers. Language confidence never changes continuous actions.
4. **Action audit** — raw, binary-transformed, projected, and executed actions remain separate.
5. **Counterfactual evaluation** — the same physical scene is paired across tasks and across Oracle
   and learned routing, enabling exact attribution of language versus control failure.
6. **Immutable evidence** — dataset, run, checkpoint, prompt, parser, schedule, and verifier
   identities use canonical serialization and cryptographic fingerprints.

## What v1 deliberately does not claim

- It is not a general-purpose robot foundation model.
- It does not train on free-form internet language or new robot tasks.
- It does not hide unsafe behavior behind clipping or aggregate success.
- M5A physical validity does not imply that the final quality gate passed.
- M5B's structural bridge does not imply a real LatentGuard proposal or executed action.
- SmolVLA was not started because the sealed final quality gate failed.
