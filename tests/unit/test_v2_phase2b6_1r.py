from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from langmani.v2 import phase2b6_1r as contracts
from langmani.v2 import phase2b6_1r_runtime as runtime


def test_source_commit_and_target_branch_are_exact() -> None:
    contracts.enforce_starting_point(
        branch=contracts.TARGET_BRANCH,
        source_commit=contracts.SOURCE_COMMIT,
        source_is_ancestor=True,
    )
    with pytest.raises(contracts.Phase2B61RContractError, match="expected branch"):
        contracts.enforce_starting_point(
            branch=contracts.SOURCE_BRANCH,
            source_commit=contracts.SOURCE_COMMIT,
            source_is_ancestor=True,
        )
    with pytest.raises(contracts.Phase2B61RContractError, match="source commit"):
        contracts.enforce_starting_point(
            branch=contracts.TARGET_BRANCH,
            source_commit="0" * 40,
            source_is_ancestor=True,
        )
    with pytest.raises(contracts.Phase2B61RContractError, match="not an ancestor"):
        contracts.enforce_starting_point(
            branch=contracts.TARGET_BRANCH,
            source_commit=contracts.SOURCE_COMMIT,
            source_is_ancestor=False,
        )


def test_icd_manifest_parsing_and_hashing(tmp_path: Path) -> None:
    path = tmp_path / "nvidia.json"
    path.write_text(
        json.dumps(
            {
                "file_format_version": "1.0.1",
                "ICD": {
                    "library_path": "/lib/libEGL_nvidia.so.0",
                    "api_version": "1.4.312",
                },
            }
        ),
        encoding="utf-8",
    )
    parsed = contracts.parse_icd_manifest(path)
    assert parsed == {
        "file_format_version": "1.0.1",
        "library_path": "/lib/libEGL_nvidia.so.0",
        "api_version": "1.4.312",
    }
    assert contracts.sha256_file(path).startswith("sha256:")
    malformed = tmp_path / "bad.json"
    malformed.write_text('{"ICD": {}}', encoding="utf-8")
    with pytest.raises(contracts.Phase2B61RContractError, match="file_format"):
        contracts.parse_icd_manifest(malformed)


def test_candidate_protocol_is_small_frozen_and_noncombinatorial() -> None:
    rows = contracts.candidate_protocol()
    assert [row["candidate_id"] for row in rows] == [
        contracts.CandidateKind.PRIMARY.value,
        contracts.CandidateKind.SECONDARY.value,
        contracts.CandidateKind.DEFAULT.value,
    ]
    assert rows[0]["may_run_zero_step_stackcube"] is True
    assert rows[1]["may_run_zero_step_stackcube"] is False
    assert rows[2]["vk_icd_filenames"] is None
    with pytest.raises(contracts.Phase2B61RContractError, match="unregistered"):
        contracts.candidate_by_id("try-until-it-works")


def test_clean_environment_requires_explicit_primary_device_variables(tmp_path: Path) -> None:
    prefix = tmp_path / "envs" / "langmani"
    environment = contracts.clean_child_environment(
        conda_prefix=prefix,
        home=tmp_path,
        xdg_runtime_dir=tmp_path / "xdg",
        candidate_id=contracts.CandidateKind.PRIMARY.value,
        inherited={
            "DISPLAY": ":99",
            "VK_DRIVER_FILES": "/untrusted.json",
            "LD_LIBRARY_PATH": "/driver/lib",
        },
    )
    assert environment["VK_ICD_FILENAMES"] == contracts.PRIMARY_ICD_PATH
    assert (
        environment["__EGL_VENDOR_LIBRARY_FILENAMES"]
        == contracts.EGL_VENDOR_PATH
    )
    assert "VK_DRIVER_FILES" not in environment
    assert "DISPLAY" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    contracts.validate_primary_environment(environment)
    for name in ("VK_ICD_FILENAMES", "__EGL_VENDOR_LIBRARY_FILENAMES"):
        broken = dict(environment)
        broken.pop(name)
        with pytest.raises(contracts.Phase2B61RContractError, match=name):
            contracts.validate_primary_environment(broken)


def test_wrong_or_software_device_is_rejected() -> None:
    accepted = {
        "name": "NVIDIA GeForce RTX 5090",
        "is_cuda": True,
        "can_render": True,
    }
    assert contracts.device_identity_matches(accepted)
    assert not contracts.device_identity_matches(
        {**accepted, "name": "llvmpipe (LLVM 19)"}
    )
    assert not contracts.device_identity_matches(
        {**accepted, "name": "NVIDIA GeForce RTX 4090"}
    )
    assert not contracts.device_identity_matches({**accepted, "is_cuda": False})


@pytest.mark.parametrize(
    ("reset_count", "step_count", "action_count"),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
)
def test_zero_step_enforcement_rejects_any_forbidden_operation(
    reset_count: int,
    step_count: int,
    action_count: int,
) -> None:
    with pytest.raises(contracts.Phase2B61RContractError, match="zero-step"):
        contracts.enforce_zero_step_counts(
            explicit_reset_count=reset_count,
            explicit_step_count=step_count,
            action_submission_count=action_count,
        )
    contracts.enforce_zero_step_counts(
        explicit_reset_count=0,
        explicit_step_count=0,
        action_submission_count=0,
    )


class _FakeDevice:
    name = "NVIDIA GeForce RTX 5090"
    pci_string = "0000:01:00.0"
    cuda_id = 0
    is_cpu = False
    is_cuda = True
    can_present = True
    can_render = True


class _FakeSpace:
    shape = (8,)
    dtype = "float32"

    def __repr__(self) -> str:
        return "Box(-1.0, 1.0, (8,), float32)"


class _FakeEnvironment:
    def __init__(self, *, close_fails: bool = False) -> None:
        self.spec = SimpleNamespace(id=contracts.TASK_ID)
        self.action_space = _FakeSpace()
        self.observation_space = {"sensor_data": "static-uninitialized-contract"}
        self.unwrapped = SimpleNamespace(
            control_mode=contracts.EXPECTED_CONTROL_MODE,
            control_freq=20,
            obs_mode=contracts.EXPECTED_OBS_MODE,
            render_system=SimpleNamespace(device=_FakeDevice()),
        )
        self.closed = False
        self.close_fails = close_fails
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, *args: object, **kwargs: object) -> None:
        self.reset_calls += 1
        raise AssertionError("zero-step probe called reset")

    def step(self, *args: object, **kwargs: object) -> None:
        self.step_calls += 1
        raise AssertionError("zero-step probe called step")

    def close(self) -> None:
        if self.close_fails:
            raise RuntimeError("close failed")
        self.closed = True


def _install_fake_maniskill(
    monkeypatch: pytest.MonkeyPatch,
    environment: _FakeEnvironment,
) -> None:
    gymnasium = ModuleType("gymnasium")

    def make(task_id: str, **kwargs: object) -> _FakeEnvironment:
        assert task_id == contracts.TASK_ID
        assert kwargs["control_mode"] == contracts.EXPECTED_CONTROL_MODE
        assert kwargs["obs_mode"] == contracts.EXPECTED_OBS_MODE
        assert kwargs["sim_backend"] == contracts.EXPECTED_SIM_BACKEND
        assert kwargs["render_backend"] == contracts.EXPECTED_RENDER_BACKEND
        return environment

    gymnasium.make = make  # type: ignore[attr-defined]
    mani_skill = ModuleType("mani_skill")
    mani_skill.__path__ = []  # type: ignore[attr-defined]
    mani_skill_envs = ModuleType("mani_skill.envs")
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "mani_skill", mani_skill)
    monkeypatch.setitem(sys.modules, "mani_skill.envs", mani_skill_envs)


def _write_source_metadata(root: Path) -> None:
    path = root / "expanded" / contracts.TASK_ID / "motionplanning" / "trajectory.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "env_info": {
                    "env_kwargs": {
                        "robot_uids": "panda",
                        "control_mode": contracts.EXPECTED_CONTROL_MODE,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_zero_step_probe_never_calls_reset_step_or_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _FakeEnvironment()
    _install_fake_maniskill(monkeypatch, environment)
    _write_source_metadata(tmp_path)
    report = runtime._zero_step_stackcube_probe(tmp_path)
    assert report["passed"] is True
    assert report["explicit_reset_count"] == 0
    assert report["explicit_step_count"] == 0
    assert report["action_submission_count"] == 0
    assert report["policy_frame_count"] == 0
    assert environment.reset_calls == 0
    assert environment.step_calls == 0
    assert environment.closed is True


def test_zero_step_probe_fails_when_environment_does_not_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment = _FakeEnvironment(close_fails=True)
    _install_fake_maniskill(monkeypatch, environment)
    _write_source_metadata(tmp_path)
    report = runtime._zero_step_stackcube_probe(tmp_path)
    assert report["passed"] is False
    assert report["close_succeeded"] is False
    assert environment.reset_calls == 0
    assert environment.step_calls == 0


def _repeatability_rows() -> list[dict[str, object]]:
    return [
        {
            "pid": 100 + index,
            "passed": True,
            "icd_sha256": "sha256:icd",
            "vulkan_device_name": "NVIDIA GeForce RTX 5090",
            "render_device_name": "NVIDIA GeForce RTX 5090",
            "render_device_pci": "0000:01:00.0",
            "action_shape": [8],
            "control_mode": "pd_joint_pos",
            "obs_mode": "rgb",
        }
        for index in range(3)
    ]


def test_fresh_process_comparison_requires_three_distinct_matching_runs() -> None:
    rows = _repeatability_rows()
    assert contracts.repeatability_passed(rows)
    rows[2]["pid"] = rows[1]["pid"]
    assert not contracts.repeatability_passed(rows)
    rows = _repeatability_rows()
    rows[2]["render_device_pci"] = "0000:02:00.0"
    assert not contracts.repeatability_passed(rows)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        (
            {
                "host_evidence_sufficient": True,
                "vulkan_passed": True,
                "sapien_passed": True,
                "zero_step_passed": True,
                "repeatability_validated": True,
                "correct_device_selected": True,
                "cleanup_passed": True,
                "first_failure_layer": None,
            },
            "RESULT_A",
        ),
        (
            {
                "host_evidence_sufficient": True,
                "vulkan_passed": True,
                "sapien_passed": True,
                "zero_step_passed": False,
                "repeatability_validated": False,
                "correct_device_selected": False,
                "cleanup_passed": True,
                "first_failure_layer": "stackcube_environment_construction",
            },
            "RESULT_B",
        ),
        (
            {
                "host_evidence_sufficient": True,
                "vulkan_passed": False,
                "sapien_passed": False,
                "zero_step_passed": False,
                "repeatability_validated": False,
                "correct_device_selected": False,
                "cleanup_passed": True,
                "first_failure_layer": "vulkaninfo",
            },
            "RESULT_C",
        ),
        (
            {
                "host_evidence_sufficient": False,
                "vulkan_passed": False,
                "sapien_passed": False,
                "zero_step_passed": False,
                "repeatability_validated": False,
                "correct_device_selected": False,
                "cleanup_passed": False,
                "first_failure_layer": None,
            },
            "RESULT_D",
        ),
    ),
)
def test_result_classification_is_exact(
    kwargs: dict[str, object],
    expected: str,
) -> None:
    report = contracts.classify_result(**kwargs)  # type: ignore[arg-type]
    assert report["result"] == expected
    assert report["phase2b6_1_forensic_restart_eligible"] is (expected == "RESULT_A")
    assert report["phase2b6_1_forensic_restart_authorized"] is False
    assert report["phase2b6_production_resume_authorized"] is False


def test_authorization_state_never_opens_replay_production_or_training() -> None:
    state = contracts.authorization_state()
    assert state["phase2b6_1_forensic_restart_authorized"] is False
    assert state["phase2b6_production_resume_authorized"] is False
    assert state["accepted_multiskill_dataset_validated"] is False
    assert state["act_training_eligible"] is False
    assert state["smolvla_training_eligible"] is False
    assert state["vla_jepa_training_eligible"] is False
    assert state["act_training_authorized"] is False
    assert state["smolvla_training_authorized"] is False
    assert state["vla_jepa_training_authorized"] is False
    assert state["optimizer_created"] is False
    assert state["backward_passes"] == 0
    assert state["optimizer_steps"] == 0
    assert state["student_policy_training_started"] is False


def test_cleanup_audit_detects_live_child_or_gpu_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_process_audit",
        lambda **_: {"passed": True, "prohibited_active_processes": []},
    )
    monkeypatch.setattr(
        runtime,
        "_gpu_audit",
        lambda: {"compute_processes": [], "passed": True},
    )
    report = runtime._post_child_cleanup(2**30)
    assert report["passed"] is True
    monkeypatch.setattr(
        runtime,
        "_gpu_audit",
        lambda: {"compute_processes": ["123, python, 100"], "passed": True},
    )
    assert runtime._post_child_cleanup(2**30)["passed"] is False


def test_frozen_manifest_verification_detects_any_byte_change(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    payload = contracts.fingerprinted({"schema_version": "test-v0", "passed": True})
    data_path = root / "data.json"
    data_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest = contracts.fingerprinted(
        {
            "schema_version": "manifest-v0",
            "files": [
                {
                    "path": "data.json",
                    "size_bytes": data_path.stat().st_size,
                    "sha256": contracts.sha256_file(data_path),
                }
            ],
        }
    )
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    assert runtime._verify_artifact_manifest(root)["passed"] is True
    data_path.write_text("{}\n", encoding="utf-8")
    assert runtime._verify_artifact_manifest(root)["passed"] is False
