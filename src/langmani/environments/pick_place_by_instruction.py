"""Deterministic language-conditioned pick-and-place environment for M1."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from langmani.environments.expert_state import (
    ExpertContextError,
    ExpertTaskContext,
    SemanticBinHandle,
    SemanticObjectHandle,
)
from langmani.environments.specs import (
    BIN_IDS,
    OBJECT_IDS,
    EpisodeSpec,
    TaskSpec,
)
from langmani.environments.task_logic import build_observation_extra, evaluate_task_state

ENV_ID = "LangMani-PickPlaceByInstruction-v0"
MAX_EPISODE_STEPS = 200

CUBE_HALF_SIZE = 0.025
CUBE_BOUNDING_RADIUS = math.sqrt(3.0) * CUBE_HALF_SIZE

BIN_INTERIOR_HALF_SIZE = 0.085
BIN_WALL_THICKNESS = 0.008
BIN_WALL_HEIGHT = 0.05
BIN_BOTTOM_THICKNESS = 0.008
BIN_X = 0.08
BIN_Y = (0.18, -0.18)

SOURCE_X_RANGE = (-0.18, -0.07)
SOURCE_Y_CENTERS = (-0.12, 0.0, 0.12)
SOURCE_Y_JITTER = 0.012
INITIAL_CUBE_BUILD_POSITIONS = tuple((-0.12, y, CUBE_HALF_SIZE) for y in SOURCE_Y_CENTERS)

TABLE_BOUNDS_XY = (-0.70, 0.60, -0.55, 0.55)
TABLE_TOP_Z = 0.0
TABLE_THICKNESS = 0.08

CONTAINMENT_CLEARANCE = 0.002
RESTING_HEIGHT_TOLERANCE = 0.005
OFF_TABLE_HEIGHT = -CUBE_BOUNDING_RADIUS

POLICY_CAMERA_EYE = (0.65, -0.75, 0.70)
POLICY_CAMERA_TARGET = (-0.04, 0.0, 0.08)
POLICY_CAMERA_RESOLUTION = (256, 256)
POLICY_CAMERA_FOV = 1.05
POLICY_CAMERA_NEAR = 0.01
POLICY_CAMERA_FAR = 10.0

HUMAN_CAMERA_EYE = (0.78, -0.90, 0.82)
HUMAN_CAMERA_TARGET = (-0.04, 0.0, 0.08)
HUMAN_CAMERA_RESOLUTION = (512, 512)
HUMAN_CAMERA_FOV = 1.0
HUMAN_CAMERA_NEAR = 0.01
HUMAN_CAMERA_FAR = 10.0

_ALLOWED_RESET_OPTION_KEYS = frozenset({"env_idx", "reconfigure", "task_spec"})


class _PrimitiveTableSceneBuilder(TableSceneBuilder):
    """Use ManiSkill's Panda reset convention with project-owned box geometry."""

    _table_pose = sapien.Pose(p=[-0.05, 0.0, -TABLE_THICKNESS / 2])

    def build(self, build_config_idxs: list[int] | None = None) -> None:
        del build_config_idxs
        table_builder = self.scene.create_actor_builder()
        table_half_size = [
            (TABLE_BOUNDS_XY[1] - TABLE_BOUNDS_XY[0]) / 2,
            (TABLE_BOUNDS_XY[3] - TABLE_BOUNDS_XY[2]) / 2,
            TABLE_THICKNESS / 2,
        ]
        table_builder.add_box_collision(half_size=table_half_size)
        table_builder.add_box_visual(
            half_size=table_half_size,
            material=sapien.render.RenderMaterial(base_color=[0.32, 0.34, 0.38, 1.0]),
        )
        table_builder.initial_pose = self._table_pose
        self.table = table_builder.build_kinematic(name="table-workspace")

        ground_builder = self.scene.create_actor_builder()
        ground_builder.add_box_collision(half_size=[5.0, 5.0, 0.02])
        ground_builder.add_box_visual(
            half_size=[5.0, 5.0, 0.02],
            material=sapien.render.RenderMaterial(base_color=[0.12, 0.12, 0.14, 1.0]),
        )
        ground_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -TABLE_THICKNESS - 0.02])
        self.ground = ground_builder.build_static(name="ground")

        self.table_length = TABLE_BOUNDS_XY[1] - TABLE_BOUNDS_XY[0]
        self.table_width = TABLE_BOUNDS_XY[3] - TABLE_BOUNDS_XY[2]
        self.table_height = TABLE_THICKNESS
        self.scene_objects = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor) -> None:
        # The parent owns the installed ManiSkill 3.0.1 Panda qpos/base reset convention.
        super().initialize(env_idx)
        self.table.set_pose(self._table_pose)


def _normalize_reset_options(options: Mapping[str, object] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"reset options must be a mapping, got {type(options).__name__}")

    option_keys = tuple(options.keys())
    if not all(isinstance(key, str) for key in option_keys):
        raise TypeError("reset option keys must be strings")
    unknown = sorted(set(option_keys) - _ALLOWED_RESET_OPTION_KEYS)
    if unknown:
        if "reset_to_env_states" in unknown:
            raise ValueError(
                "reset_to_env_states is not supported because M1 task metadata is not part "
                "of ManiSkill simulator state"
            )
        raise ValueError("unsupported reset option keys: " + ", ".join(unknown))

    normalized = dict(options)
    if "reconfigure" in normalized and not isinstance(normalized["reconfigure"], bool):
        raise TypeError("reset option 'reconfigure' must be a bool")
    if "env_idx" in normalized:
        try:
            env_idx = torch.as_tensor(normalized["env_idx"])
        except (TypeError, ValueError) as error:
            raise TypeError(
                "reset option 'env_idx' must be a 1-D integer tensor or sequence"
            ) from error
        if env_idx.ndim != 1:
            raise TypeError("reset option 'env_idx' must be a 1-D integer tensor or sequence")
        if env_idx.numel() == 0:
            raise ValueError("reset option 'env_idx' must not be empty")
        if env_idx.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("reset option 'env_idx' must be a 1-D integer tensor or sequence")
        normalized["env_idx"] = env_idx
    if "task_spec" in normalized:
        raw_task_spec = normalized["task_spec"]
        if not isinstance(raw_task_spec, Mapping):
            raise TypeError(
                f"reset option 'task_spec' must be a mapping, got {type(raw_task_spec).__name__}"
            )
        normalized["task_spec"] = TaskSpec.from_mapping(raw_task_spec).to_dict()
    return normalized


def _validated_env_indices(
    env_idx: torch.Tensor,
    *,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    indices = env_idx.detach().cpu().tolist()
    if len(set(indices)) != len(indices):
        raise ValueError("reset option 'env_idx' must not contain duplicates")
    if indices != sorted(indices):
        raise ValueError("reset option 'env_idx' must be strictly increasing")
    if any(index < 0 or index >= num_envs for index in indices):
        raise ValueError(f"reset option 'env_idx' entries must be in [0, {num_envs})")
    return torch.tensor(indices, dtype=torch.long, device=device)


@register_env(ENV_ID, max_episode_steps=MAX_EPISODE_STEPS)
class PickPlaceByInstructionEnv(BaseEnv):
    """Pick one instructed cube and place it in the instructed shallow bin."""

    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_REWARD_MODES = ("sparse", "none")
    agent: Panda

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.0,
        enhanced_determinism: bool = True,
        **kwargs: Any,
    ) -> None:
        if robot_uids != "panda":
            raise ValueError(f"{ENV_ID} supports only robot_uids='panda'")
        if robot_init_qpos_noise != 0.0:
            raise ValueError(f"{ENV_ID} fixes robot_init_qpos_noise at 0.0 in M1")
        if not enhanced_determinism:
            raise ValueError(f"{ENV_ID} requires enhanced_determinism=True")
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
        """Reset after validating the M1 options contract before simulator mutation."""
        normalized_options = _normalize_reset_options(options)
        if "env_idx" in normalized_options:
            normalized_options["env_idx"] = _validated_env_indices(
                normalized_options["env_idx"],
                num_envs=self.num_envs,
                device=self.device,
            )
        return super().reset(seed=seed, options=normalized_options)

    @staticmethod
    def policy_camera_config() -> CameraConfig:
        """Return the fixed policy-camera configuration."""
        return CameraConfig(
            uid="base_camera",
            pose=sapien_utils.look_at(POLICY_CAMERA_EYE, POLICY_CAMERA_TARGET),
            width=POLICY_CAMERA_RESOLUTION[0],
            height=POLICY_CAMERA_RESOLUTION[1],
            fov=POLICY_CAMERA_FOV,
            near=POLICY_CAMERA_NEAR,
            far=POLICY_CAMERA_FAR,
            shader_pack="minimal",
        )

    @staticmethod
    def human_camera_config() -> CameraConfig:
        """Return the separate, higher-resolution diagnostic camera configuration."""
        return CameraConfig(
            uid="render_camera",
            pose=sapien_utils.look_at(HUMAN_CAMERA_EYE, HUMAN_CAMERA_TARGET),
            width=HUMAN_CAMERA_RESOLUTION[0],
            height=HUMAN_CAMERA_RESOLUTION[1],
            fov=HUMAN_CAMERA_FOV,
            near=HUMAN_CAMERA_NEAR,
            far=HUMAN_CAMERA_FAR,
            shader_pack="default",
        )

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

        cube_colors = (
            [0.90, 0.08, 0.08, 1.0],
            [0.08, 0.72, 0.16, 1.0],
            [0.08, 0.22, 0.92, 1.0],
        )
        self.cubes: tuple[Actor, ...] = tuple(
            actors.build_cube(
                self.scene,
                half_size=CUBE_HALF_SIZE,
                color=color,
                name=object_id,
                body_type="dynamic",
                initial_pose=sapien.Pose(p=initial_position),
            )
            for object_id, color, initial_position in zip(
                OBJECT_IDS,
                cube_colors,
                INITIAL_CUBE_BUILD_POSITIONS,
                strict=True,
            )
        )

        self.bins: tuple[Actor, ...] = (
            self._build_bin("left_bin", [BIN_X, BIN_Y[0], BIN_BOTTOM_THICKNESS / 2]),
            self._build_bin("right_bin", [BIN_X, BIN_Y[1], BIN_BOTTOM_THICKNESS / 2]),
        )
        self._objects_by_id = dict(zip(OBJECT_IDS, self.cubes, strict=True))
        self._bins_by_id = dict(zip(BIN_IDS, self.bins, strict=True))

    def _build_bin(self, name: str, position: list[float]) -> Actor:
        builder = self.scene.create_actor_builder()
        wall_half_thickness = BIN_WALL_THICKNESS / 2
        wall_half_height = BIN_WALL_HEIGHT / 2
        outer_half_size = BIN_INTERIOR_HALF_SIZE + BIN_WALL_THICKNESS
        wall_offset = BIN_INTERIOR_HALF_SIZE + wall_half_thickness
        wall_z = BIN_BOTTOM_THICKNESS / 2 + wall_half_height
        material = sapien.render.RenderMaterial(base_color=[0.78, 0.72, 0.42, 1.0])

        components = (
            (
                sapien.Pose(),
                [outer_half_size, outer_half_size, BIN_BOTTOM_THICKNESS / 2],
            ),
            (
                sapien.Pose(p=[-wall_offset, 0.0, wall_z]),
                [wall_half_thickness, outer_half_size, wall_half_height],
            ),
            (
                sapien.Pose(p=[wall_offset, 0.0, wall_z]),
                [wall_half_thickness, outer_half_size, wall_half_height],
            ),
            (
                sapien.Pose(p=[0.0, -wall_offset, wall_z]),
                [BIN_INTERIOR_HALF_SIZE, wall_half_thickness, wall_half_height],
            ),
            (
                sapien.Pose(p=[0.0, wall_offset, wall_z]),
                [BIN_INTERIOR_HALF_SIZE, wall_half_thickness, wall_half_height],
            ),
        )
        for component_pose, half_size in components:
            builder.add_box_collision(pose=component_pose, half_size=half_size)
            builder.add_box_visual(
                pose=component_pose,
                half_size=half_size,
                material=material,
            )
        builder.initial_pose = sapien.Pose(p=position)
        return builder.build_static(name=name)

    def _ensure_task_buffers(self) -> None:
        if hasattr(self, "_scene_seeds"):
            return
        self._scene_seeds = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._target_object_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._target_bin_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        self._ensure_task_buffers()
        self.table_scene.initialize(env_idx)
        batch_size = len(env_idx)

        # This reset-path synchronization is exclusively for deterministic RNG and
        # metadata; it never occurs in evaluate(), get_obs(), or step().
        env_indices = env_idx.detach().cpu().tolist()
        episode_rng = self._batched_episode_rng[env_indices]
        device_kwargs = {"device": self.device, "dtype": torch.float32}

        slot_x = torch.as_tensor(episode_rng.uniform(*SOURCE_X_RANGE, size=3), **device_kwargs)
        slot_y = torch.as_tensor(
            episode_rng.uniform(-SOURCE_Y_JITTER, SOURCE_Y_JITTER, size=3),
            **device_kwargs,
        )
        slot_y += torch.tensor(SOURCE_Y_CENTERS, **device_kwargs)
        slot_z = torch.full((batch_size, len(OBJECT_IDS)), CUBE_HALF_SIZE, **device_kwargs)
        slot_positions = torch.stack((slot_x, slot_y, slot_z), dim=-1)

        permutation_keys = torch.as_tensor(episode_rng.uniform(0.0, 1.0, size=3), **device_kwargs)
        permutation = torch.argsort(permutation_keys, dim=1)
        cube_positions = torch.gather(
            slot_positions,
            dim=1,
            index=permutation[:, :, None].expand(-1, -1, 3),
        )

        yaw = torch.as_tensor(episode_rng.uniform(-math.pi, math.pi, size=3), **device_kwargs)
        cube_quaternions = torch.zeros((batch_size, len(OBJECT_IDS), 4), **device_kwargs)
        cube_quaternions[..., 0] = torch.cos(yaw / 2)
        cube_quaternions[..., 3] = torch.sin(yaw / 2)

        zero_velocity = torch.zeros((batch_size, 3), **device_kwargs)
        for object_index, cube in enumerate(self.cubes):
            cube.set_pose(
                Pose.create_from_pq(
                    p=cube_positions[:, object_index],
                    q=cube_quaternions[:, object_index],
                )
            )
            cube.set_linear_velocity(zero_velocity)
            cube.set_angular_velocity(zero_velocity)

        scene_seeds = torch.as_tensor(
            self._episode_seed[env_indices], dtype=torch.long, device=self.device
        )
        self._scene_seeds[env_idx] = scene_seeds

        raw_task_spec = options.get("task_spec")
        if raw_task_spec is None:
            task_choice = torch.remainder(scene_seeds, len(OBJECT_IDS) * len(BIN_IDS))
            target_object_indices = torch.div(task_choice, len(BIN_IDS), rounding_mode="floor")
            target_bin_indices = torch.remainder(task_choice, len(BIN_IDS))
        else:
            task_spec = TaskSpec.from_mapping(raw_task_spec)
            target_object_indices = torch.full(
                (batch_size,),
                OBJECT_IDS.index(task_spec.target_object_id),
                dtype=torch.long,
                device=self.device,
            )
            target_bin_indices = torch.full(
                (batch_size,),
                BIN_IDS.index(task_spec.target_bin_id),
                dtype=torch.long,
                device=self.device,
            )
        self._target_object_indices[env_idx] = target_object_indices
        self._target_bin_indices[env_idx] = target_bin_indices

    def _bin_floor_centers(self) -> torch.Tensor:
        centers = torch.stack([bin_actor.pose.p for bin_actor in self.bins], dim=1)
        floor_offset = torch.zeros_like(centers)
        floor_offset[..., 2] = BIN_BOTTOM_THICKNESS / 2
        return centers + floor_offset

    def evaluate(self) -> dict[str, torch.Tensor]:
        cube_positions = torch.stack([cube.pose.p for cube in self.cubes], dim=1)
        cube_is_grasped = torch.stack([self.agent.is_grasping(cube) for cube in self.cubes], dim=1)
        cube_is_static = torch.stack(
            [cube.is_static(lin_thresh=1e-2, ang_thresh=0.1) for cube in self.cubes],
            dim=1,
        )
        return dict(
            evaluate_task_state(
                cube_positions=cube_positions,
                bin_floor_centers=self._bin_floor_centers(),
                cube_is_grasped=cube_is_grasped,
                cube_is_static=cube_is_static,
                target_object_index=self._target_object_indices,
                target_bin_index=self._target_bin_indices,
                cube_bounding_radius=CUBE_BOUNDING_RADIUS,
                cube_resting_height=CUBE_HALF_SIZE,
                bin_interior_half_size=BIN_INTERIOR_HALF_SIZE,
                containment_clearance=CONTAINMENT_CLEARANCE,
                resting_height_tolerance=RESTING_HEIGHT_TOLERANCE,
                off_table_height=OFF_TABLE_HEIGHT,
            )
        )

    def _get_obs_extra(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del info
        tcp_pose = self.agent.tcp.pose.raw_pose
        if not self.obs_mode_struct.use_state:
            return build_observation_extra(
                tcp_pose=tcp_pose,
                use_privileged_state=False,
            )

        return build_observation_extra(
            tcp_pose=tcp_pose,
            use_privileged_state=True,
            cube_poses=torch.cat([cube.pose.raw_pose for cube in self.cubes], dim=1),
            bin_centers=self._bin_floor_centers().reshape(self.num_envs, -1),
            target_object_index=self._target_object_indices,
            target_bin_index=self._target_bin_indices,
        )

    def get_episode_specs(self) -> tuple[EpisodeSpec, ...]:
        """Materialize immutable per-environment metadata outside the step path."""
        scene_seeds = self._scene_seeds.detach().cpu().tolist()
        object_indices = self._target_object_indices.detach().cpu().tolist()
        bin_indices = self._target_bin_indices.detach().cpu().tolist()
        return tuple(
            EpisodeSpec.create(
                scene_seed=int(scene_seed),
                task_spec=TaskSpec(
                    target_object_id=OBJECT_IDS[object_index],
                    target_bin_id=BIN_IDS[bin_index],
                    instruction_template_id="canonical_v0",
                ),
            )
            for scene_seed, object_index, bin_index in zip(
                scene_seeds, object_indices, bin_indices, strict=True
            )
        )

    def get_task_texts(self) -> tuple[str, ...]:
        """Return canonical English instructions for single or batched environments."""
        return tuple(spec.canonical_instruction for spec in self.get_episode_specs())

    def get_policy_rollout_evaluation(self) -> dict[str, torch.Tensor]:
        """Return M1 evaluation plus rollout-only wrong-grasp diagnostics.

        The extra field is an evaluator oracle used only after a learned policy
        chooses its action.  It is absent from observations and from the stable
        M1/M3A step-info contract, so it cannot become a policy input or change
        historical demonstration schemas.
        """
        evaluation = dict(self.evaluate())
        cube_is_grasped = torch.stack([self.agent.is_grasping(cube) for cube in self.cubes], dim=1)
        object_indices = torch.arange(len(self.cubes), device=self.device)[None, :]
        evaluation["wrong_object_is_grasped"] = (
            cube_is_grasped & (object_indices != self._target_object_indices[:, None])
        ).any(dim=1)
        return evaluation

    def get_policy_rollout_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return privileged rollout-analysis tensors outside the policy input path.

        M4.2 samples this accessor at reset and after executed actions to diagnose
        post-grasp phases.  The values are absent from observations, processor
        inputs, rewards, and step ``info``; therefore this accessor does not alter
        the M1 no-leakage contract or provide a deployable policy input.
        """
        cube_positions = torch.stack([cube.pose.p for cube in self.cubes], dim=1)
        cube_velocities = torch.stack([cube.linear_velocity for cube in self.cubes], dim=1)
        cube_is_grasped = torch.stack([self.agent.is_grasping(cube) for cube in self.cubes], dim=1)
        gather_xyz = self._target_object_indices[:, None, None].expand(-1, 1, 3)
        target_position = torch.gather(cube_positions, dim=1, index=gather_xyz).squeeze(1)
        target_velocity = torch.gather(cube_velocities, dim=1, index=gather_xyz).squeeze(1)
        bin_centers = self._bin_floor_centers()
        target_bin_center = torch.gather(
            bin_centers,
            dim=1,
            index=self._target_bin_indices[:, None, None].expand(-1, 1, 3),
        ).squeeze(1)
        return {
            "target_position": target_position,
            "target_linear_velocity": target_velocity,
            "target_bin_floor_center": target_bin_center,
            "target_to_bin_center_distance": torch.linalg.vector_norm(
                target_position - target_bin_center, dim=1
            ),
            "object_is_grasped": cube_is_grasped,
        }

    def get_expert_task_context(self) -> ExpertTaskContext:
        """Return the explicit privileged context for a single-environment expert.

        This accessor is intentionally unavailable for vectorized rollouts.  It
        materializes simulator handles and a small amount of immutable metadata on
        demand and is never called by observation, reward, evaluation, or step code.
        """
        if self.num_envs != 1:
            raise ValueError(
                f"LangMani M2 experts require num_envs=1, got num_envs={self.num_envs}"
            )
        if not hasattr(self, "_scene_seeds"):
            raise ExpertContextError("reset the environment before requesting expert task context")
        if not hasattr(self, "_objects_by_id") or not hasattr(self, "_bins_by_id"):
            raise ExpertContextError(
                "expert actor handles are unavailable because the scene is incomplete"
            )
        if not hasattr(self, "agent") or not hasattr(self.agent, "robot"):
            raise ExpertContextError(
                "expert robot handles are unavailable because Panda is incomplete"
            )

        episode_spec = self.get_episode_specs()[0]
        target_object_id = episode_spec.task_spec.target_object_id
        target_bin_id = episode_spec.task_spec.target_bin_id
        if target_object_id not in self._objects_by_id:
            raise ExpertContextError(f"missing semantic actor handle for {target_object_id!r}")
        if target_bin_id not in self._bins_by_id:
            raise ExpertContextError(f"missing semantic actor handle for {target_bin_id!r}")
        target_actor = self._objects_by_id[target_object_id]

        floor_centers = self._bin_floor_centers()[0]
        object_handles = tuple(
            SemanticObjectHandle(object_id=object_id, actor=self._objects_by_id[object_id])
            for object_id in OBJECT_IDS
        )
        bin_handles = tuple(
            SemanticBinHandle(
                bin_id=bin_id,
                actor=self._bins_by_id[bin_id],
                floor_center=floor_centers[bin_index].detach().clone(),
            )
            for bin_index, bin_id in enumerate(BIN_IDS)
        )
        target_bin_handle = next(
            bin_handle for bin_handle in bin_handles if bin_handle.bin_id == target_bin_id
        )
        return ExpertTaskContext(
            environment_id=ENV_ID,
            episode_spec=episode_spec,
            agent=self.agent,
            robot=self.agent.robot,
            target_object=SemanticObjectHandle(
                object_id=target_object_id,
                actor=target_actor,
            ),
            target_bin=target_bin_handle,
            objects=object_handles,
            bins=bin_handles,
            cube_half_extent=CUBE_HALF_SIZE,
            bin_interior_half_size=BIN_INTERIOR_HALF_SIZE,
            bin_wall_height=BIN_WALL_HEIGHT,
            bin_bottom_thickness=BIN_BOTTOM_THICKNESS,
            table_top_z=TABLE_TOP_Z,
        )

    def get_expert_evaluation(self) -> dict[str, torch.Tensor]:
        """Return a fresh batched evaluation mapping for privileged expert checks."""
        if self.num_envs != 1:
            raise ValueError(
                f"LangMani M2 experts require num_envs=1, got num_envs={self.num_envs}"
            )
        return dict(self.evaluate())

    def get_expert_initial_scene_state(self) -> dict[str, object]:
        """Materialize the physical reset state needed by M3A group audits.

        ManiSkill's simulator state dictionary intentionally omits static actors,
        which includes both LangMani bins.  The collector calls this privileged
        metadata accessor exactly once immediately after reset and before the
        expert executes an action.  It is never used by observations, rewards,
        evaluation, or the per-step ``info`` path.
        """
        context = self.get_expert_task_context()

        def pose_values(actor: Actor) -> list[float]:
            values = actor.pose.raw_pose.detach().cpu().reshape(-1).tolist()
            if len(values) != 7:
                raise ExpertContextError(
                    f"expected one 7D actor pose, received {len(values)} values"
                )
            return [float(value) for value in values]

        qpos = context.robot.get_qpos().detach().cpu().reshape(-1).tolist()
        if len(qpos) != 9:
            raise ExpertContextError(f"expected one 9D Panda qpos, received {len(qpos)} values")
        return {
            "object_poses": {
                handle.object_id: pose_values(handle.actor) for handle in context.objects
            },
            "bin_poses": {handle.bin_id: pose_values(handle.actor) for handle in context.bins},
            "panda_qpos": [float(value) for value in qpos],
        }
