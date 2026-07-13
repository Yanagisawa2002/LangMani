# LangMani environment specification

This document fixes the M1 environment contract consumed by the M2 expert and M3A collector.

## Identity and episode contract

- Gym environment ID: `LangMani-PickPlaceByInstruction-v0`
- Robot: Panda only
- Reward modes: `sparse` and `none` only
- Episode time limit: 200 control steps
- Determinism: `enhanced_determinism=True`, Panda joint reset noise fixed to `0.0`
- Geometry: project-owned primitive boxes and cubes; no downloaded assets

Every scene contains `red_cube`, `green_cube`, `blue_cube`, `left_bin`, and `right_bin`. A scene is
identified only by its non-negative scene seed. A task is a separate semantic selection; changing a
task never reconfigures or removes geometry.

## Scene geometry and reset

Each cube has half extent `0.025 m`. Cube source slots use x in `[-0.18, -0.07] m` and y centers
`[-0.12, 0.00, 0.12] m` with at most `0.012 m` y jitter. A seeded random permutation assigns the
three semantic cubes to these separated slots, and yaw is sampled in `[-pi, pi]`. This construction
prevents cube-cube and cube-bin overlap while keeping every cube reachable.

The shallow bins are static. Their centers are `(0.08, 0.18)` and `(0.08, -0.18)` in table x/y.
Each bin has interior half size `0.085 m`, wall thickness `0.008 m`, wall height `0.05 m`, and bottom
thickness `0.008 m`. All cube linear and angular velocities are reset to zero.

The normal reset is:

```python
observation, info = env.reset(seed=123)
```

The exact task override is:

```python
observation, info = env.reset(
    seed=123,
    options={
        "task_spec": {
            "target_object_id": "red_cube",
            "target_bin_id": "right_bin",
            "instruction_template_id": "canonical_v0",
        }
    },
)
```

`task_spec` must contain exactly those three fields. Unknown object, bin, template, option, missing
field, or extra field is an error. `env_idx` and `reconfigure` remain the only other accepted
ManiSkill reset options. `reset_to_env_states` is rejected because simulator state does not encode
LangMani task metadata. M3A state audit first performs the semantic reset and then calls the state
API explicitly.

## Language and metadata

The canonical object order is red, green, blue; the canonical bin order is left, right. The only
template produces these six English instructions:

- `Pick up the red cube and place it in the left bin.`
- `Pick up the red cube and place it in the right bin.`
- `Pick up the green cube and place it in the left bin.`
- `Pick up the green cube and place it in the right bin.`
- `Pick up the blue cube and place it in the left bin.`
- `Pick up the blue cube and place it in the right bin.`

`TaskSpec` and `EpisodeSpec` are immutable JSON-ready project types. `get_task_texts()` and
`get_episode_specs()` return tuples for both one and many environments. Strings and stable IDs are
not inserted into numeric observations or every-step `info`.

## Observation boundary

Standard ManiSkill observation modes are used without an M1 wrapper. All modes may expose normal
Panda proprioception and the TCP pose. Only modes whose ManiSkill observation structure uses
privileged state add `cube_poses`, `bin_centers`, `target_object_index`, and `target_bin_index`.

Visual-only modes must not contain cube or bin ground-truth poses, target indices, semantic color
IDs, task IDs, instruction embeddings, language strings, or success-oracle inputs. The M2 expert
uses separate `get_expert_task_context()` and `get_expert_evaluation()` accessors; those accessors do
not alter the policy observation contract and require `num_envs=1`.

M3A additionally calls `get_expert_initial_scene_state()` once immediately after reset and before
the first expert action. This privileged metadata accessor returns fresh JSON-ready values for all
three cube poses, both actual static-bin poses, and the nine-value Panda qpos. It exists because
ManiSkill 3.0.1 state dictionaries omit static actors. It is never called from observation,
evaluation, reward, or per-step info construction and therefore does not weaken the visual
no-leakage contract.

## Cameras

| Camera | Eye -> target (m) | Resolution | FOV | Near/far | Shader |
| --- | --- | --- | --- | --- | --- |
| `base_camera` | `(0.65,-0.75,0.70)` -> `(-0.04,0,0.08)` | 256 x 256 | 1.05 rad | 0.01/10 m | `minimal` |
| `render_camera` | `(0.78,-0.90,0.82)` -> `(-0.04,0,0.08)` | 512 x 512 | 1.0 rad | 0.01/10 m | `default` |

The first camera is the fixed policy camera. The second is a separate human diagnostic camera.
There is no wrist camera and no domain randomization in M1-M3A.

## Evaluation and reward

`evaluate()` returns batched boolean tensors:

- `target_in_target_bin`
- `target_in_wrong_bin`
- `wrong_object_in_target_bin`
- `target_is_grasped`
- `target_is_static`
- `target_off_table`
- `success`
- `fail`

Containment uses a sphere enclosing the cube, a `0.002 m` wall clearance, and a `0.005 m` resting
height tolerance. Success requires the selected target inside the selected bin, no wrong object in
that bin, the target released, the target static (`0.01 m/s` linear and `0.1 rad/s` angular
thresholds), and the target not off-table. A target in the wrong bin or held above a bin is not
success. `fail` is reserved for the target falling below the table threshold. Sparse reward is the
success tensor converted to float; `none` emits no task reward.

Native Linux simulator, CUDA, Vulkan, and camera visibility acceptance remains governed by
`environment/verify_m1.py --target`; Windows structural checks are not physical validation.
