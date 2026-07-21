"""Deterministic, vectorized planar-pushing environment for LangMani 2.0."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose

from langmani.environments.pick_place_by_instruction import (
    TABLE_TOP_Z,
    PickPlaceByInstructionEnv,
    _PrimitiveTableSceneBuilder,
    _validated_env_indices,
)
from langmani.environments.push_logic import (
    build_push_observation_extra,
    evaluate_push_state,
    update_progress_state,
    update_stable_success_count,
)
from langmani.environments.push_specs import (
    PUSH_DIFFICULTIES,
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    PushEpisodeSpec,
    PushTaskSpec,
)

ENV_ID = "LangMani-PushToRegion-v0"
MAX_EPISODE_STEPS = 250

CUBE_HALF_SIZE = 0.025
CYLINDER_RADIUS = 0.025
CYLINDER_HALF_LENGTH = 0.025
OBJECT_PLANAR_RADII = (math.sqrt(2.0) * CUBE_HALF_SIZE, CYLINDER_RADIUS)
OBJECT_RESTING_HEIGHTS = (CUBE_HALF_SIZE, CYLINDER_HALF_LENGTH)

STANDARD_REGION_RADIUS = 0.11
HARD_REGION_RADIUS = 0.085
TARGET_REGION_CENTERS_XY = (
    (0.12, 0.22),
    (0.12, -0.22),
    (0.28, 0.14),
    (0.28, -0.14),
)
TARGET_REGION_JITTER = 0.012

SOURCE_X_RANGE = (-0.22, -0.12)
SOURCE_Y_RANGE = (-0.075, 0.075)
STANDARD_DISTRACTOR_XY = (-0.28, 0.24)
HARD_DISTRACTOR_PATH_FRACTION = 0.52
HARD_DISTRACTOR_LATERAL_OFFSET = 0.085

WORKSPACE_BOUNDS_XY = (-0.43, 0.43, -0.38, 0.38)
CONTAINMENT_CLEARANCE = 0.005
LIFT_TOLERANCE = 0.015
TOPPLE_ALIGNMENT_THRESHOLD = 0.75
WRONG_OBJECT_DISPLACEMENT_THRESHOLD = 0.03
CONTACT_FORCE_THRESHOLD = 0.2
ARM_COLLISION_FORCE_THRESHOLD = 2.0
REQUIRED_STABLE_SUCCESS_STEPS = 5
MINIMUM_PROGRESS = 0.002
STALL_STEPS = 50
ROBOT_INIT_QPOS_NOISE = 0.01

INITIAL_OBJECT_BUILD_POSITIONS = (
    (-0.18, -0.08, CUBE_HALF_SIZE),
    (-0.18, 0.08, CYLINDER_HALF_LENGTH),
)
INITIAL_TARGET_BUILD_POSITION = (0.18, 0.0, 0.001)

_ALLOWED_RESET_OPTION_KEYS = frozenset({"env_idx", "reconfigure", "task_spec"})


def _normalize_push_reset_options(options: Mapping[str, object] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"reset options must be a mapping, got {type(options).__name__}")
    if not all(isinstance(key, str) for key in options):
        raise TypeError("reset option keys must be strings")
    unknown = sorted(set(options) - _ALLOWED_RESET_OPTION_KEYS)
    if unknown:
        raise ValueError("unsupported reset option keys: " + ", ".join(unknown))
    normalized = dict(options)
    if "reconfigure" in normalized and not isinstance(normalized["reconfigure"], bool):
        raise TypeError("reset option 'reconfigure' must be a bool")
    if "env_idx" in normalized:
        try:
            env_idx = torch.as_tensor(normalized["env_idx"])
        except (TypeError, ValueError) as error:
            raise TypeError("reset option 'env_idx' must be a 1-D integer sequence") from error
        if env_idx.ndim != 1 or env_idx.numel() == 0:
            raise ValueError("reset option 'env_idx' must be a non-empty 1-D sequence")
        if env_idx.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("reset option 'env_idx' must contain integers")
        normalized["env_idx"] = env_idx
    if "task_spec" in normalized:
        raw = normalized["task_spec"]
        if not isinstance(raw, Mapping):
            raise TypeError(f"reset option 'task_spec' must be a mapping, got {type(raw).__name__}")
        normalized["task_spec"] = PushTaskSpec.from_mapping(raw).to_dict()
    return normalized


@register_env(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
class PushToRegionEnv(BaseEnv):
    """Push one selected primitive object fully into one marked planar region."""

    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_REWARD_MODES = ("sparse", "none")
    agent: Panda

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = ROBOT_INIT_QPOS_NOISE,
        enhanced_determinism: bool = True,
        **kwargs: Any,
    ) -> None:
        if robot_uids != "panda":
            raise ValueError(f"{ENV_ID} supports only robot_uids='panda'")
        if robot_init_qpos_noise != ROBOT_INIT_QPOS_NOISE:
            raise ValueError(f"{ENV_ID} fixes robot_init_qpos_noise at {ROBOT_INIT_QPOS_NOISE}")
        if not enhanced_determinism:
            raise ValueError(f"{ENV_ID} requires enhanced_determinism=True")
        if max(OBJECT_PLANAR_RADII) + CONTAINMENT_CLEARANCE >= HARD_REGION_RADIUS:
            raise ValueError("hard target region cannot contain the configured objects")
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(
            *args,
            robot_uids=robot_uids,
            enhanced_determinism=enhanced_determinism,
            **kwargs,
        )

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        normalized = _normalize_push_reset_options(options)
        if "env_idx" in normalized:
            normalized["env_idx"] = _validated_env_indices(
                normalized["env_idx"],
                num_envs=self.num_envs,
                device=self.device,
            )
        return super().reset(seed=seed, options=normalized)

    @staticmethod
    def policy_camera_config() -> CameraConfig:
        return PickPlaceByInstructionEnv.policy_camera_config()

    @staticmethod
    def human_camera_config() -> CameraConfig:
        return PickPlaceByInstructionEnv.human_camera_config()

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        return [self.policy_camera_config()]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        return self.human_camera_config()

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, TABLE_TOP_Z]))

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        self.table_scene = _PrimitiveTableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        cube = actors.build_cube(
            self.scene,
            half_size=CUBE_HALF_SIZE,
            color=[0.08, 0.28, 0.92, 1.0],
            name=PUSH_OBJECT_IDS[0],
            body_type="dynamic",
            initial_pose=sapien.Pose(p=INITIAL_OBJECT_BUILD_POSITIONS[0]),
        )
        cylinder = actors.build_cylinder(
            self.scene,
            radius=CYLINDER_RADIUS,
            half_length=CYLINDER_HALF_LENGTH,
            color=[0.94, 0.38, 0.04, 1.0],
            name=PUSH_OBJECT_IDS[1],
            body_type="dynamic",
            initial_pose=sapien.Pose(p=INITIAL_OBJECT_BUILD_POSITIONS[1]),
        )
        self.push_objects: tuple[Actor, ...] = (cube, cylinder)
        self.target_regions: tuple[Actor, ...] = tuple(
            actors.build_red_white_target(
                self.scene,
                radius=radius,
                thickness=2e-4,
                name=f"push_target_region_{difficulty}",
                add_collision=False,
                body_type="kinematic",
                initial_pose=sapien.Pose(p=INITIAL_TARGET_BUILD_POSITION),
            )
            for difficulty, radius in zip(
                PUSH_DIFFICULTIES,
                (STANDARD_REGION_RADIUS, HARD_REGION_RADIUS),
                strict=True,
            )
        )
        self._objects_by_id = dict(zip(PUSH_OBJECT_IDS, self.push_objects, strict=True))

    def _ensure_task_buffers(self) -> None:
        if hasattr(self, "_scene_seeds"):
            return
        float_kwargs = {"device": self.device, "dtype": torch.float32}
        self._scene_seeds = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._target_object_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._target_region_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._difficulty_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_region_centers = torch.zeros((self.num_envs, 3), **float_kwargs)
        self._target_region_radii = torch.full(
            (self.num_envs,), STANDARD_REGION_RADIUS, **float_kwargs
        )
        self._initial_object_positions = torch.zeros(
            (self.num_envs, len(PUSH_OBJECT_IDS), 3), **float_kwargs
        )
        self._initial_target_positions = torch.zeros((self.num_envs, 3), **float_kwargs)
        self._stable_success_count = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._best_target_distance = torch.full((self.num_envs,), torch.inf, **float_kwargs)
        self._steps_without_progress = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        for name in (
            "wrong_object_contact",
            "wrong_object_displaced",
            "target_outside_workspace",
            "target_overshoot",
            "target_toppled",
            "target_lifted",
            "invalid_action",
            "action_out_of_bounds",
            "no_progress_stall",
            "robot_collision",
        ):
            setattr(
                self,
                f"_event_{name}",
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
        self._last_wrong_object_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_invalid_action = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_action_out_of_bounds = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_robot_collision = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        self._ensure_task_buffers()
        self.table_scene.initialize(env_idx)
        batch_size = len(env_idx)
        indices = env_idx.detach().cpu().tolist()
        episode_rng = self._batched_episode_rng[indices]
        float_kwargs = {"device": self.device, "dtype": torch.float32}
        scene_seeds = torch.as_tensor(
            self._episode_seed[indices], dtype=torch.long, device=self.device
        )
        self._scene_seeds[env_idx] = scene_seeds

        raw_task = options.get("task_spec")
        if raw_task is None:
            task_choice = torch.remainder(
                scene_seeds,
                len(PUSH_OBJECT_IDS) * len(TARGET_REGION_IDS) * len(PUSH_DIFFICULTIES),
            )
            difficulty_indices = torch.remainder(task_choice, len(PUSH_DIFFICULTIES))
            task_choice = torch.div(task_choice, len(PUSH_DIFFICULTIES), rounding_mode="floor")
            region_indices = torch.remainder(task_choice, len(TARGET_REGION_IDS))
            object_indices = torch.div(task_choice, len(TARGET_REGION_IDS), rounding_mode="floor")
        else:
            task_spec = PushTaskSpec.from_mapping(raw_task)
            object_indices = torch.full(
                (batch_size,),
                PUSH_OBJECT_IDS.index(task_spec.target_object_id),
                dtype=torch.long,
                device=self.device,
            )
            region_indices = torch.full(
                (batch_size,),
                TARGET_REGION_IDS.index(task_spec.target_region_id),
                dtype=torch.long,
                device=self.device,
            )
            difficulty_indices = torch.full(
                (batch_size,),
                PUSH_DIFFICULTIES.index(task_spec.difficulty),
                dtype=torch.long,
                device=self.device,
            )
        self._target_object_indices[env_idx] = object_indices
        self._target_region_indices[env_idx] = region_indices
        self._difficulty_indices[env_idx] = difficulty_indices

        source_x = torch.as_tensor(
            episode_rng.uniform(*SOURCE_X_RANGE, size=1), **float_kwargs
        ).reshape(batch_size)
        source_y = torch.as_tensor(
            episode_rng.uniform(*SOURCE_Y_RANGE, size=1), **float_kwargs
        ).reshape(batch_size)
        source_xy = torch.stack((source_x, source_y), dim=1)

        region_table = torch.tensor(TARGET_REGION_CENTERS_XY, **float_kwargs)
        region_xy = region_table[region_indices]
        region_jitter = torch.as_tensor(
            episode_rng.uniform(-TARGET_REGION_JITTER, TARGET_REGION_JITTER, size=2),
            **float_kwargs,
        )
        region_xy = region_xy + region_jitter
        target_centers = torch.zeros((batch_size, 3), **float_kwargs)
        target_centers[:, :2] = region_xy
        target_centers[:, 2] = 0.001
        self._target_region_centers[env_idx] = target_centers
        self._target_region_radii[env_idx] = torch.where(
            difficulty_indices == PUSH_DIFFICULTIES.index("hard"),
            torch.full((batch_size,), HARD_REGION_RADIUS, **float_kwargs),
            torch.full((batch_size,), STANDARD_REGION_RADIUS, **float_kwargs),
        )

        standard_distractor = torch.tensor(STANDARD_DISTRACTOR_XY, **float_kwargs).expand(
            batch_size, -1
        )
        path = region_xy - source_xy
        direction = path / torch.linalg.vector_norm(path, dim=1, keepdim=True).clamp_min(1e-6)
        perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        side = torch.where(
            torch.remainder(scene_seeds, 2) == 0,
            torch.ones(batch_size, **float_kwargs),
            -torch.ones(batch_size, **float_kwargs),
        )
        hard_distractor = (
            source_xy
            + HARD_DISTRACTOR_PATH_FRACTION * path
            + side[:, None] * HARD_DISTRACTOR_LATERAL_OFFSET * perpendicular
        )
        distractor_xy = torch.where(
            (difficulty_indices == PUSH_DIFFICULTIES.index("hard"))[:, None],
            hard_distractor,
            standard_distractor,
        )

        object_positions = torch.zeros((batch_size, len(PUSH_OBJECT_IDS), 3), **float_kwargs)
        object_positions[..., :2] = distractor_xy[:, None, :]
        resting_heights = torch.tensor(OBJECT_RESTING_HEIGHTS, **float_kwargs)
        object_positions[..., 2] = resting_heights[None, :]
        batch_index = torch.arange(batch_size, device=self.device)
        object_positions[batch_index, object_indices, :2] = source_xy
        self._initial_object_positions[env_idx] = object_positions
        self._initial_target_positions[env_idx] = object_positions[batch_index, object_indices]

        cube_yaw = torch.as_tensor(
            episode_rng.uniform(-math.pi, math.pi, size=1), **float_kwargs
        ).reshape(batch_size)
        cube_quaternion = torch.zeros((batch_size, 4), **float_kwargs)
        cube_quaternion[:, 0] = torch.cos(cube_yaw / 2)
        cube_quaternion[:, 3] = torch.sin(cube_yaw / 2)
        cylinder_quaternion = torch.zeros((batch_size, 4), **float_kwargs)
        cylinder_quaternion[:, 0] = 1.0
        object_quaternions = (cube_quaternion, cylinder_quaternion)
        zero_velocity = torch.zeros((batch_size, 3), **float_kwargs)
        for object_index, actor in enumerate(self.push_objects):
            actor.set_pose(
                Pose.create_from_pq(
                    p=object_positions[:, object_index],
                    q=object_quaternions[object_index],
                )
            )
            actor.set_linear_velocity(zero_velocity)
            actor.set_angular_velocity(zero_velocity)
        hidden_target_centers = torch.zeros_like(target_centers)
        hidden_target_centers[:, 2] = -1.0
        for difficulty_index, target_region in enumerate(self.target_regions):
            active = difficulty_indices == difficulty_index
            marker_centers = torch.where(active[:, None], target_centers, hidden_target_centers)
            target_region.set_pose(Pose.create_from_pq(p=marker_centers, q=[1, 0, 0, 0]))

        self._stable_success_count[env_idx] = 0
        self._best_target_distance[env_idx] = torch.linalg.vector_norm(source_xy - region_xy, dim=1)
        self._steps_without_progress[env_idx] = 0
        for name in (
            "wrong_object_contact",
            "wrong_object_displaced",
            "target_outside_workspace",
            "target_overshoot",
            "target_toppled",
            "target_lifted",
            "invalid_action",
            "action_out_of_bounds",
            "no_progress_stall",
            "robot_collision",
        ):
            getattr(self, f"_event_{name}")[env_idx] = False
        self._last_wrong_object_contact[env_idx] = False
        self._last_invalid_action[env_idx] = False
        self._last_action_out_of_bounds[env_idx] = False
        self._last_robot_collision[env_idx] = False

    def _object_up_alignment(self) -> torch.Tensor:
        cube_rotation = self.push_objects[0].pose.to_transformation_matrix()[:, :3, :3]
        cylinder_rotation = self.push_objects[1].pose.to_transformation_matrix()[:, :3, :3]
        cube_up = torch.abs(cube_rotation[:, 2, 2])
        cylinder_axis_up = torch.abs(cylinder_rotation[:, 2, 2])
        return torch.stack((cube_up, cylinder_axis_up), dim=1)

    def _robot_contact_magnitude(self, actor: Actor, *, arm_only: bool) -> torch.Tensor:
        magnitudes: list[torch.Tensor] = []
        for link in self.agent.robot.get_links():
            name = link.name.lower()
            if arm_only and any(token in name for token in ("hand", "finger")):
                continue
            force = self.scene.get_pairwise_contact_forces(link, actor)
            magnitudes.append(torch.linalg.vector_norm(force, dim=1))
        if not magnitudes:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        return torch.stack(magnitudes, dim=1).amax(dim=1)

    def _current_wrong_object_contact(self) -> torch.Tensor:
        force = torch.stack(
            [self._robot_contact_magnitude(actor, arm_only=False) for actor in self.push_objects],
            dim=1,
        )
        object_index = torch.arange(len(PUSH_OBJECT_IDS), device=self.device)[None, :]
        wrong_mask = object_index != self._target_object_indices[:, None]
        return ((force > CONTACT_FORCE_THRESHOLD) & wrong_mask).any(dim=1)

    def _current_robot_collision(self) -> torch.Tensor:
        arm_force = torch.stack(
            [self._robot_contact_magnitude(actor, arm_only=True) for actor in self.push_objects],
            dim=1,
        )
        return (arm_force > ARM_COLLISION_FORCE_THRESHOLD).any(dim=1)

    def _raw_evaluation(self) -> dict[str, torch.Tensor]:
        positions = torch.stack([actor.pose.p for actor in self.push_objects], dim=1)
        is_static = torch.stack(
            [actor.is_static(lin_thresh=0.03, ang_thresh=0.5) for actor in self.push_objects],
            dim=1,
        )
        is_grasped = torch.stack(
            [self.agent.is_grasping(actor) for actor in self.push_objects], dim=1
        )
        radii = torch.tensor(OBJECT_PLANAR_RADII, dtype=torch.float32, device=self.device).expand(
            self.num_envs, -1
        )
        resting = torch.tensor(
            OBJECT_RESTING_HEIGHTS, dtype=torch.float32, device=self.device
        ).expand(self.num_envs, -1)
        return dict(
            evaluate_push_state(
                object_positions=positions,
                object_up_alignment=self._object_up_alignment(),
                object_is_static=is_static,
                object_is_grasped=is_grasped,
                initial_object_positions=self._initial_object_positions,
                initial_target_position=self._initial_target_positions,
                target_region_center=self._target_region_centers,
                target_object_index=self._target_object_indices,
                object_planar_radii=radii,
                object_resting_heights=resting,
                target_region_radius=self._target_region_radii,
                stable_success_count=self._stable_success_count,
                required_stable_steps=REQUIRED_STABLE_SUCCESS_STEPS,
                workspace_bounds_xy=WORKSPACE_BOUNDS_XY,
                containment_clearance=CONTAINMENT_CLEARANCE,
                lift_tolerance=LIFT_TOLERANCE,
                topple_alignment_threshold=TOPPLE_ALIGNMENT_THRESHOLD,
                wrong_object_displacement_threshold=WRONG_OBJECT_DISPLACEMENT_THRESHOLD,
                wrong_object_contact=self._last_wrong_object_contact,
                action_out_of_bounds=self._last_action_out_of_bounds,
                invalid_action=self._last_invalid_action,
                progress_stalled=self._event_no_progress_stall,
                robot_collision=self._last_robot_collision,
            )
        )

    def _with_latched_events(self, metrics: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(metrics)
        current_overshoot = result["target_overshoot"]
        for name in (
            "wrong_object_contact",
            "wrong_object_displaced",
            "target_outside_workspace",
            "target_overshoot",
            "target_toppled",
            "target_lifted",
            "invalid_action",
            "action_out_of_bounds",
            "no_progress_stall",
            "robot_collision",
        ):
            result[name] = result[name] | getattr(self, f"_event_{name}")
        fail = (
            result["invalid_action"]
            | result["target_outside_workspace"]
            | result["target_lifted"]
            | result["target_toppled"]
            | result["target_is_grasped"]
        )
        result["fail"] = fail
        result["stable_success_steps"] = self._stable_success_count
        result["success"] = (
            (self._stable_success_count >= REQUIRED_STABLE_SUCCESS_STEPS)
            & result["target_inside_region"]
            & result["target_is_static"]
            & ~current_overshoot
            & ~result["wrong_object_displaced"]
            & ~fail
        )
        return result

    def evaluate(self) -> dict[str, torch.Tensor]:
        return self._with_latched_events(self._raw_evaluation())

    def _inspect_action(self, action: Any) -> tuple[Any, torch.Tensor, torch.Tensor]:
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        out_of_bounds = torch.zeros_like(invalid)
        if action is None:
            return action, invalid, out_of_bounds
        value = action.get("action") if isinstance(action, Mapping) else action
        if not isinstance(value, (np.ndarray, torch.Tensor)):
            return action, invalid, out_of_bounds
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        was_unbatched = tensor.ndim == 1
        batched = tensor[None, :] if was_unbatched else tensor
        finite = torch.isfinite(batched)
        invalid = ~finite.all(dim=1)
        sanitized = torch.where(finite, batched, torch.zeros_like(batched))
        low = torch.as_tensor(
            self._orig_single_action_space.low, dtype=torch.float32, device=self.device
        )
        high = torch.as_tensor(
            self._orig_single_action_space.high, dtype=torch.float32, device=self.device
        )
        out_of_bounds = ((sanitized < low) | (sanitized > high)).any(dim=1)
        prepared_value = sanitized[0] if was_unbatched else sanitized
        if isinstance(action, Mapping):
            prepared = dict(action)
            prepared["action"] = prepared_value
            return prepared, invalid, out_of_bounds
        return prepared_value, invalid, out_of_bounds

    def step(self, action: Any) -> tuple[Any, ...]:
        prepared_action, invalid, out_of_bounds = self._inspect_action(action)
        self._last_invalid_action = invalid
        self._last_action_out_of_bounds = out_of_bounds
        observation, _reward, _terminated, truncated, _info = super().step(prepared_action)

        self._last_wrong_object_contact = self._current_wrong_object_contact()
        self._last_robot_collision = self._current_robot_collision()
        metrics = self._raw_evaluation()
        stable_condition = (
            metrics["target_inside_region"]
            & metrics["target_is_static"]
            & ~metrics["target_overshoot"]
            & ~metrics["target_outside_workspace"]
            & ~metrics["target_lifted"]
            & ~metrics["target_toppled"]
            & ~metrics["target_is_grasped"]
            & ~metrics["invalid_action"]
        )
        self._stable_success_count = update_stable_success_count(
            self._stable_success_count, stable_condition
        )
        (
            self._best_target_distance,
            self._steps_without_progress,
            stalled,
        ) = update_progress_state(
            distance=metrics["target_distance"],
            best_distance=self._best_target_distance,
            steps_without_progress=self._steps_without_progress,
            minimum_improvement=MINIMUM_PROGRESS,
            stall_steps=STALL_STEPS,
        )
        metrics["no_progress_stall"] = stalled
        for name in (
            "wrong_object_contact",
            "wrong_object_displaced",
            "target_outside_workspace",
            "target_overshoot",
            "target_toppled",
            "target_lifted",
            "invalid_action",
            "action_out_of_bounds",
            "no_progress_stall",
            "robot_collision",
        ):
            event = getattr(self, f"_event_{name}") | metrics[name]
            setattr(self, f"_event_{name}", event)
        info = self._with_latched_events(metrics)
        info["elapsed_steps"] = self.elapsed_steps
        terminated = info["success"] | info["fail"]
        reward = (
            info["success"].to(torch.float32) - info["fail"].to(torch.float32)
            if self.reward_mode == "sparse"
            else torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        )
        return observation, reward, terminated, truncated, info

    def _get_obs_extra(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del info
        if not self.obs_mode_struct.use_state:
            return build_push_observation_extra(
                tcp_pose=self.agent.tcp.pose.raw_pose,
                use_privileged_state=False,
            )
        return build_push_observation_extra(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            use_privileged_state=True,
            object_poses=torch.cat([actor.pose.raw_pose for actor in self.push_objects], dim=1),
            target_region_center=self._target_region_centers,
            target_object_index=self._target_object_indices,
            target_region_index=self._target_region_indices,
            difficulty_index=self._difficulty_indices,
        )

    def get_episode_specs(self) -> tuple[PushEpisodeSpec, ...]:
        seeds = self._scene_seeds.detach().cpu().tolist()
        objects = self._target_object_indices.detach().cpu().tolist()
        regions = self._target_region_indices.detach().cpu().tolist()
        difficulties = self._difficulty_indices.detach().cpu().tolist()
        return tuple(
            PushEpisodeSpec.create(
                scene_seed=int(seed),
                task_spec=PushTaskSpec(
                    target_object_id=PUSH_OBJECT_IDS[object_index],
                    target_region_id=TARGET_REGION_IDS[region_index],
                    difficulty=PUSH_DIFFICULTIES[difficulty_index],
                ),
            )
            for seed, object_index, region_index, difficulty_index in zip(
                seeds, objects, regions, difficulties, strict=True
            )
        )

    def get_task_texts(self) -> tuple[str, ...]:
        return tuple(spec.canonical_instruction for spec in self.get_episode_specs())

    def get_policy_rollout_evaluation(self) -> dict[str, torch.Tensor]:
        return dict(self.evaluate())

    def get_policy_rollout_diagnostics(self) -> dict[str, torch.Tensor]:
        positions = torch.stack([actor.pose.p for actor in self.push_objects], dim=1)
        return {
            **self.evaluate(),
            "object_positions": positions,
            "object_linear_velocities": torch.stack(
                [actor.linear_velocity for actor in self.push_objects], dim=1
            ),
            "object_angular_velocities": torch.stack(
                [actor.angular_velocity for actor in self.push_objects], dim=1
            ),
            "target_region_center": self._target_region_centers,
            "target_object_index": self._target_object_indices,
            "target_region_index": self._target_region_indices,
            "difficulty_index": self._difficulty_indices,
            "tcp_position": self.agent.tcp.pose.p,
        }


__all__ = [
    "ENV_ID",
    "MAX_EPISODE_STEPS",
    "PushToRegionEnv",
]
