"""Diagnose the shared LangMani robotics-learning environment.

The default mode performs CPU-safe package and ManiSkill checks, then attempts
rendering as an optional diagnostic. ``--target`` turns native Linux, CUDA,
Vulkan, GPU simulation, RGB observations, rendering, and frame persistence into
hard requirements. The script never installs or changes system packages.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import os
import platform
import shutil
import subprocess
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_FRAME = PROJECT_ROOT / "outputs" / "diagnostics" / "pickcube_rgb.png"

EXPECTED_VERSIONS = {
    "gymnasium": "1.2.3",
    "lerobot": "0.6.0",
    "mani-skill": "3.0.1",
    "numpy": "2.2.6",
    "Pillow": "12.3.0",
    "torch": "2.11.0",
}


@dataclass
class Reporter:
    """Collect human-readable checks while preserving a nonzero failure result."""

    failures: list[str] = field(default_factory=list)

    def info(self, name: str, detail: str) -> None:
        """Print an informational value."""
        print(f"[INFO] {name}: {detail}")

    def check(self, name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        """Record a required check or an optional diagnostic warning."""
        status = "PASS" if ok else "FAIL" if required else "WARN"
        print(f"[{status}] {name}: {detail}")
        if not ok and required:
            self.failures.append(name)

    def exception(
        self,
        name: str,
        error: BaseException,
        *,
        required: bool = True,
        verbose: bool = False,
    ) -> None:
        """Report an exception explicitly and optionally include its traceback."""
        detail = f"{type(error).__name__}: {error}"
        self.check(name, False, detail, required=required)
        if verbose:
            formatted = "".join(traceback.format_exception(error)).rstrip()
            print(formatted)


def parse_args() -> argparse.Namespace:
    """Parse diagnostic mode and verbosity options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="store_true",
        help=(
            "Require native Linux, CUDA, Vulkan, explicit ManiSkill GPU simulation, "
            "RGB observation and rendering, a successful step, and a saved frame."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tracebacks for failed checks.",
    )
    return parser.parse_args()


def distribution_version(distribution: str) -> str | None:
    """Return installed distribution metadata without importing the package."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def public_version(version: str) -> str:
    """Remove a wheel's local build label, such as ``+cu128`` or ``+cpu``."""
    return version.partition("+")[0]


def native_linux_detected() -> bool:
    """Return whether this process is on native Linux rather than WSL."""
    release = platform.release().lower()
    is_wsl = "microsoft" in release or bool(os.environ.get("WSL_INTEROP"))
    return platform.system() == "Linux" and not is_wsl


def inspect_versions(reporter: Reporter) -> None:
    """Report and enforce the project-owned direct dependency pins."""
    for distribution, expected in EXPECTED_VERSIONS.items():
        installed = distribution_version(distribution)
        if installed is None:
            reporter.check(
                f"{distribution} version",
                False,
                f"not installed; expected {expected}",
            )
            continue
        reporter.check(
            f"{distribution} version",
            public_version(installed) == expected,
            f"installed {installed}; expected public version {expected}",
        )

    for distribution in ("sapien", "torchvision"):
        installed = distribution_version(distribution)
        reporter.info(f"{distribution} version", installed or "not installed")


def inspect_python_and_os(reporter: Reporter, *, target: bool) -> bool:
    """Report interpreter and operating-system identity."""
    python_version = platform.python_version()
    reporter.info("Python version", python_version)
    reporter.check(
        "Python 3.12",
        sys.version_info[:2] == (3, 12),
        f"running {python_version}; the reproducible environment requires Python 3.12",
    )
    reporter.info("Operating system", platform.platform())

    native_linux = native_linux_detected()
    reporter.check(
        "Native Linux",
        native_linux,
        "native Linux detected" if native_linux else "native Linux not detected",
        required=target,
    )
    return native_linux


def inspect_imports_and_cuda(
    reporter: Reporter,
    *,
    target: bool,
    verbose: bool,
) -> tuple[Any | None, bool]:
    """Import the three core packages and inspect PyTorch CUDA state."""
    torch_module: Any | None = None
    for module_name in ("torch", "mani_skill", "lerobot"):
        try:
            module = importlib.import_module(module_name)
            reporter.check(f"import {module_name}", True, "import succeeded")
            if module_name == "torch":
                torch_module = module
        except Exception as error:  # noqa: BLE001 - diagnostics must report third-party failures
            reporter.exception(
                f"import {module_name}",
                error,
                verbose=verbose,
            )

    if torch_module is None:
        reporter.check(
            "CUDA availability",
            False,
            "PyTorch did not import, so CUDA could not be inspected",
            required=target,
        )
        reporter.info("CUDA runtime reported by PyTorch", "unavailable")
        reporter.info("Detected GPU name", "unavailable")
        return None, False

    cuda_available = False
    try:
        cuda_available = bool(torch_module.cuda.is_available())
        reporter.check(
            "CUDA availability",
            cuda_available,
            f"torch.cuda.is_available() returned {cuda_available}",
            required=target,
        )
        cuda_runtime = torch_module.version.cuda
        reporter.check(
            "CUDA runtime reported by PyTorch",
            cuda_runtime is not None,
            str(cuda_runtime) if cuda_runtime is not None else "none (CPU-only PyTorch build)",
            required=target,
        )

        gpu_name = torch_module.cuda.get_device_name(0) if cuda_available else None
        reporter.check(
            "Detected GPU name",
            gpu_name is not None,
            gpu_name or "no CUDA GPU detected",
            required=target,
        )
    except Exception as error:  # noqa: BLE001 - diagnostics must report CUDA failures
        reporter.exception("CUDA inspection", error, required=target, verbose=verbose)

    return torch_module, cuda_available


def inspect_opencv(reporter: Reporter, *, verbose: bool) -> None:
    """Expose the unavoidable SAPIEN/LeRobot OpenCV wheel collision risk."""
    gui_version = distribution_version("opencv-python")
    headless_version = distribution_version("opencv-python-headless")
    reporter.info("opencv-python distribution", gui_version or "not installed")
    reporter.info("opencv-python-headless distribution", headless_version or "not installed")
    reporter.check(
        "OpenCV single-wheel packaging",
        not (gui_version and headless_version),
        (
            "both OpenCV wheels are installed because SAPIEN and LeRobot declare conflicting "
            "package names; runtime behavior must be validated"
            if gui_version and headless_version
            else "no dual-wheel collision detected"
        ),
        required=False,
    )
    try:
        cv2 = importlib.import_module("cv2")
        reporter.check("import cv2", True, f"runtime version {cv2.__version__}")
    except Exception as error:  # noqa: BLE001 - diagnostics must report third-party failures
        reporter.exception("import cv2", error, verbose=verbose)


def inspect_vulkan(reporter: Reporter, *, target: bool, verbose: bool) -> bool:
    """Locate and run the standard Vulkan diagnostic tool without modifying the host."""
    executable = shutil.which("vulkaninfo")
    reporter.check(
        "Vulkan tooling visible",
        executable is not None,
        executable or "vulkaninfo was not found on PATH",
        required=target,
    )
    if executable is None:
        return False

    try:
        probe = subprocess.run(
            [executable, "--summary"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (probe.stdout or probe.stderr).strip()
        preview = " | ".join(output.splitlines()[:4]) or "no output"
        reporter.check(
            "Vulkan probe",
            probe.returncode == 0,
            f"exit code {probe.returncode}; {preview}",
            required=target,
        )
        return probe.returncode == 0
    except Exception as error:  # noqa: BLE001 - diagnostics must report probe failures
        reporter.exception("Vulkan probe", error, required=target, verbose=verbose)
        return False


def validate_reset_result(result: Any) -> tuple[Any, Mapping[str, Any]]:
    """Validate and return the standard Gymnasium reset result."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(f"reset returned {type(result).__name__}, expected a 2-tuple")
    observation, info = result
    if observation is None:
        raise ValueError("reset returned a null observation")
    if not isinstance(info, Mapping):
        raise TypeError(f"reset info is {type(info).__name__}, expected a mapping")
    return observation, info


def shape_tuple(value: Any) -> tuple[int, ...]:
    """Return an array/tensor shape as plain integers."""
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{type(value).__name__} does not expose a shape")
    return tuple(int(dimension) for dimension in shape)


def validate_state_observation(value: Any, *, label: str) -> None:
    """Require ManiSkill's single-environment batched state tensor/array."""
    shape = shape_tuple(value)
    if len(shape) != 2 or shape[0] != 1 or shape[1] < 1:
        raise ValueError(f"{label} has shape {shape}, expected (1, D) with D > 0")
    if getattr(value, "dtype", None) is None:
        raise TypeError(f"{label} does not expose a dtype")


def validate_single_value_batch(value: Any, *, label: str, boolean: bool) -> None:
    """Require a one-element batched reward or termination tensor/array."""
    shape = shape_tuple(value)
    if shape != (1,):
        raise ValueError(f"{label} has shape {shape}, expected (1,)")

    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise TypeError(f"{label} does not expose a dtype")
    dtype_text = str(dtype).lower()
    if boolean:
        if "bool" not in dtype_text:
            raise TypeError(f"{label} has dtype {dtype}, expected bool")
        return

    is_floating = bool(getattr(dtype, "is_floating_point", False)) or (
        getattr(dtype, "kind", None) == "f"
    )
    if not is_floating:
        raise TypeError(f"{label} has dtype {dtype}, expected a floating-point reward")


def validate_step_result(result: Any) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    """Validate and return ManiSkill's batched Gymnasium step result."""
    if not isinstance(result, tuple) or len(result) != 5:
        raise TypeError(f"step returned {type(result).__name__}, expected a 5-tuple")
    observation, reward, terminated, truncated, info = result
    if isinstance(observation, Mapping):
        if not observation:
            raise ValueError("step returned an empty observation mapping")
    else:
        validate_state_observation(observation, label="step observation")
    validate_single_value_batch(reward, label="step reward", boolean=False)
    validate_single_value_batch(terminated, label="step terminated", boolean=True)
    validate_single_value_batch(truncated, label="step truncated", boolean=True)
    if not isinstance(info, Mapping):
        raise TypeError(f"step info is {type(info).__name__}, expected a mapping")
    return observation, reward, terminated, truncated, info


def describe_tree(value: Any, path: str = "observation") -> list[str]:
    """Describe nested observation keys plus tensor/array shapes and dtypes."""
    if isinstance(value, Mapping):
        if not value:
            return [f"{path}: empty mapping"]
        lines: list[str] = []
        for key, child in value.items():
            lines.extend(describe_tree(child, f"{path}.{key}"))
        return lines

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return [f"{path}: type={type(value).__name__}, shape={shape}, dtype={dtype}"]


def report_observation(reporter: Reporter, observation: Any, label: str) -> None:
    """Print observation keys and leaf shapes without dumping values."""
    lines = describe_tree(observation)
    reporter.info(f"{label} observation keys and shapes", f"{len(lines)} leaf value(s)")
    for line in lines:
        print(f"       {line}")


def find_sensor_rgb_leaves(
    observation: Any,
    path: str = "observation",
) -> list[tuple[str, Any]]:
    """Find only policy RGB leaves under ManiSkill's sensor_data contract."""
    if not isinstance(observation, Mapping):
        return []
    sensor_data = observation.get("sensor_data")
    if not isinstance(sensor_data, Mapping):
        return []
    leaves: list[tuple[str, Any]] = []
    for camera_uid, textures in sensor_data.items():
        if isinstance(textures, Mapping) and "rgb" in textures:
            leaves.append(
                (f"{path}.sensor_data.{camera_uid}.rgb", textures["rgb"]),
            )
    return leaves


def validate_rgb_batch(value: Any, *, label: str) -> tuple[int, ...]:
    """Require one non-empty batched uint8 RGB image with channels last."""
    shape = shape_tuple(value)
    if len(shape) != 4 or shape[0] != 1 or shape[1] < 1 or shape[2] < 1 or shape[3] != 3:
        raise ValueError(f"{label} has shape {shape}, expected (1, H, W, 3) with H,W > 0")
    dtype = getattr(value, "dtype", None)
    if dtype is None or "uint8" not in str(dtype).lower():
        raise TypeError(f"{label} has dtype {dtype}, expected uint8")
    return shape


def to_uint8_frame(value: Any) -> Any:
    """Remove the validated single batch dimension from a rendered RGB frame."""
    validate_rgb_batch(value, label="render frame")
    numpy = importlib.import_module("numpy")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    frame = numpy.asarray(value)
    return frame[0]


def save_diagnostic_frame(frame: Any) -> Path:
    """Persist one rendered RGB frame under the ignored diagnostics directory."""
    image_module = importlib.import_module("PIL.Image")
    DIAGNOSTIC_FRAME.parent.mkdir(parents=True, exist_ok=True)
    image_module.fromarray(to_uint8_frame(frame)).save(DIAGNOSTIC_FRAME)
    return DIAGNOSTIC_FRAME


def import_environment_modules() -> Any:
    """Import Gymnasium and register ManiSkill environments."""
    gym = importlib.import_module("gymnasium")
    importlib.import_module("mani_skill.envs")
    return gym


def close_environment(
    reporter: Reporter,
    env: Any,
    *,
    required: bool,
    verbose: bool,
) -> None:
    """Close an environment and expose cleanup errors."""
    try:
        env.close()
        reporter.check("ManiSkill environment close", True, "close succeeded", required=required)
    except Exception as error:  # noqa: BLE001 - diagnostics must report cleanup failures
        reporter.exception(
            "ManiSkill environment close",
            error,
            required=required,
            verbose=verbose,
        )


def run_cpu_maniskill_check(reporter: Reporter, *, verbose: bool) -> None:
    """Run the required CPU-safe reset and single-step diagnostic."""
    try:
        gym = import_environment_modules()
    except Exception as error:  # noqa: BLE001 - diagnostics must report import failures
        reporter.exception("ManiSkill CPU environment imports", error, verbose=verbose)
        return

    env: Any | None = None
    try:
        env = gym.make(
            "PickCube-v1",
            num_envs=1,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            sim_backend="physx_cpu",
            render_backend="none",
        )
        reporter.check("ManiSkill CPU environment creation", True, "PickCube-v1 created")

        observation, _ = validate_reset_result(env.reset(seed=0))
        validate_state_observation(observation, label="CPU reset observation")
        reporter.check("ManiSkill CPU reset", True, "reset returned a valid 2-tuple")
        report_observation(reporter, observation, "CPU reset")

        validate_step_result(env.step(env.action_space.sample()))
        reporter.check("ManiSkill CPU one-step execution", True, "step returned a valid 5-tuple")
    except Exception as error:  # noqa: BLE001 - diagnostics must report simulator failures
        reporter.exception("ManiSkill CPU environment execution", error, verbose=verbose)
    finally:
        if env is not None:
            close_environment(reporter, env, required=True, verbose=verbose)


def run_visual_maniskill_check(
    reporter: Reporter,
    *,
    target: bool,
    verbose: bool,
) -> None:
    """Attempt RGB observation, viewer rendering, persistence, and one step."""
    try:
        gym = import_environment_modules()
    except Exception as error:  # noqa: BLE001 - diagnostics must report import failures
        reporter.exception(
            "ManiSkill visual environment imports",
            error,
            required=target,
            verbose=verbose,
        )
        return

    env: Any | None = None
    try:
        env = gym.make(
            "PickCube-v1",
            num_envs=1,
            obs_mode="rgb",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array",
            sim_backend="physx_cuda" if target else "physx_cpu",
            render_backend="sapien_cuda" if target else "sapien_cpu",
        )
        reporter.check(
            "ManiSkill visual environment creation",
            True,
            "PickCube-v1 created",
            required=target,
        )

        if target:
            gpu_enabled = bool(env.unwrapped.gpu_sim_enabled)
            reporter.check(
                "ManiSkill explicit GPU simulation",
                gpu_enabled,
                f"env.unwrapped.gpu_sim_enabled={gpu_enabled}",
            )

        observation, _ = validate_reset_result(env.reset(seed=0))
        reporter.check(
            "ManiSkill visual reset",
            True,
            "reset returned a valid 2-tuple",
            required=target,
        )
        report_observation(reporter, observation, "Visual reset")

        rgb_leaves = find_sensor_rgb_leaves(observation)
        if not rgb_leaves:
            reporter.check(
                "ManiSkill RGB observation",
                False,
                "no observation.sensor_data.<camera>.rgb leaf found",
                required=target,
            )
        else:
            try:
                rgb_details = []
                for path, value in rgb_leaves:
                    shape = validate_rgb_batch(value, label=path)
                    rgb_details.append(f"{path} shape={shape}, dtype={value.dtype}")
                reporter.check(
                    "ManiSkill RGB observation",
                    True,
                    ", ".join(rgb_details),
                    required=target,
                )
            except Exception as error:  # noqa: BLE001 - report invalid third-party output
                reporter.exception(
                    "ManiSkill RGB observation",
                    error,
                    required=target,
                    verbose=verbose,
                )

        frame = env.render()
        render_shape = validate_rgb_batch(frame, label="env.render() frame")
        reporter.check(
            "ManiSkill RGB-array rendering",
            True,
            f"render returned shape={render_shape}, dtype={frame.dtype}",
            required=target,
        )

        saved_path = save_diagnostic_frame(frame)
        reporter.check(
            "Diagnostic frame saved",
            saved_path.is_file() and saved_path.stat().st_size > 0,
            str(saved_path),
            required=target,
        )

        validate_step_result(env.step(env.action_space.sample()))
        reporter.check(
            "ManiSkill visual one-step execution",
            True,
            "step returned a valid 5-tuple",
            required=target,
        )
    except Exception as error:  # noqa: BLE001 - diagnostics must report simulator failures
        reporter.exception(
            "ManiSkill visual environment execution",
            error,
            required=target,
            verbose=verbose,
        )
    finally:
        if env is not None:
            close_environment(reporter, env, required=target, verbose=verbose)


def main() -> int:
    """Run diagnostics and return nonzero when required checks fail."""
    faulthandler.enable()
    args = parse_args()
    reporter = Reporter()
    mode = "TARGET (strict native Linux GPU/rendering)" if args.target else "DIAGNOSTIC (CPU-safe)"
    print(f"LangMani installation verification — {mode}")
    print(f"Repository root: {PROJECT_ROOT}")

    native_linux = inspect_python_and_os(reporter, target=args.target)
    inspect_versions(reporter)
    _, cuda_available = inspect_imports_and_cuda(
        reporter,
        target=args.target,
        verbose=args.verbose,
    )
    inspect_opencv(reporter, verbose=args.verbose)
    vulkan_available = inspect_vulkan(reporter, target=args.target, verbose=args.verbose)

    if args.target:
        if native_linux and cuda_available and vulkan_available:
            run_visual_maniskill_check(reporter, target=True, verbose=args.verbose)
        else:
            reporter.check(
                "ManiSkill target environment execution",
                False,
                "not attempted because native Linux, CUDA, or Vulkan prerequisites failed",
            )
    else:
        if native_linux:
            run_cpu_maniskill_check(reporter, verbose=args.verbose)
        else:
            reporter.check(
                "ManiSkill CPU environment execution",
                False,
                "not attempted outside the repository's native Linux execution boundary",
                required=False,
            )

        if native_linux and vulkan_available:
            run_visual_maniskill_check(reporter, target=False, verbose=args.verbose)
        else:
            reporter.check(
                "ManiSkill optional visual environment execution",
                False,
                "not attempted without native Linux and a successful Vulkan probe",
                required=False,
            )

    print("\nVerification summary")
    if reporter.failures:
        print(f"FAILED: {len(reporter.failures)} required check(s) failed")
        for failure in reporter.failures:
            print(f"  - {failure}")
        return 1

    print("PASSED: all required checks succeeded")
    if not args.target:
        print("Note: optional CUDA/Vulkan/rendering warnings do not validate the target machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
