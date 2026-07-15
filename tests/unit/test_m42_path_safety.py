from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment import verify_m42
from scripts import evaluate_m42, run_m42_runtime_ablation, train_act_task_token


def test_task_token_default_model_root_is_consistent_across_commands() -> None:
    expected = verify_m42.DEFAULT_M42_MODEL_ROOT
    assert expected == train_act_task_token.DEFAULT_OUTPUT_ROOT
    assert expected == evaluate_m42.DEFAULT_TASK_TOKEN_ROOT


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=tmp_path / "dataset",
        m4_model_root=tmp_path / "m4-models",
        task_token_model_root=tmp_path / "task-token-model",
        m4_diagnostics_root=tmp_path / "m4-diagnostics",
        output_root=tmp_path / "m42-diagnostics",
    )


def test_verifier_path_validation_does_not_create_output(tmp_path: Path) -> None:
    args = _args(tmp_path)
    output = verify_m42._validate_paths(args)
    assert output == args.output_root.resolve()
    assert not args.output_root.exists()


@pytest.mark.parametrize(
    "protected_attribute",
    (
        "dataset_root",
        "m4_model_root",
        "task_token_model_root",
        "m4_diagnostics_root",
    ),
)
def test_verifier_output_rejects_bidirectional_protected_overlap(
    tmp_path: Path, protected_attribute: str
) -> None:
    args = _args(tmp_path)
    protected = getattr(args, protected_attribute)
    args.output_root = protected / "nested-output"
    with pytest.raises(RuntimeError, match="must not overlap"):
        verify_m42._validate_paths(args)

    args = _args(tmp_path)
    container = tmp_path / f"contains-{protected_attribute}"
    setattr(args, protected_attribute, container / "protected")
    args.output_root = container
    with pytest.raises(RuntimeError, match="must not overlap"):
        verify_m42._validate_paths(args)


def test_verifier_output_rejects_source_and_git_overlap(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.output_root = verify_m42.PROJECT_ROOT / "tests" / "unsafe-m42-output"
    with pytest.raises(RuntimeError, match="must not overlap"):
        verify_m42._validate_paths(args)

    args.output_root = verify_m42.PROJECT_ROOT
    with pytest.raises(RuntimeError, match="must not overlap"):
        verify_m42._validate_paths(args)


@pytest.mark.parametrize("relation", ("equal", "child", "parent"))
@pytest.mark.parametrize(
    "immutable_attribute", ("dataset_root", "m4_model_root", "m4_diagnostics_root")
)
def test_verifier_task_token_output_rejects_immutable_overlap(
    tmp_path: Path, immutable_attribute: str, relation: str
) -> None:
    args = _args(tmp_path)
    immutable = getattr(args, immutable_attribute)
    if relation == "equal":
        args.task_token_model_root = immutable
    elif relation == "child":
        args.task_token_model_root = immutable / "task-token"
    else:
        container = tmp_path / f"task-token-contains-{immutable_attribute}"
        setattr(args, immutable_attribute, container / "immutable")
        args.task_token_model_root = container
    with pytest.raises(RuntimeError, match="TaskToken model output must not overlap"):
        verify_m42._validate_paths(args)


def test_verifier_task_token_output_rejects_source_overlap(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.task_token_model_root = verify_m42.PROJECT_ROOT / "src" / "unsafe-models"
    with pytest.raises(RuntimeError, match="TaskToken model output must not overlap"):
        verify_m42._validate_paths(args)


def test_verifier_rejects_linked_output_ancestor(tmp_path: Path) -> None:
    args = _args(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    args.output_root = linked / "m42"
    with pytest.raises(RuntimeError, match="symlink or junction"):
        verify_m42._validate_paths(args)


@pytest.mark.parametrize(
    "resolver",
    (
        run_m42_runtime_ablation._resolved_unlinked,
        train_act_task_token._resolved_unlinked,
        evaluate_m42._resolved_unlinked,
        verify_m42._resolved_unlinked,
    ),
)
def test_all_m42_entrypoints_check_lexical_link_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolver: Callable[..., Path],
) -> None:
    linked = (tmp_path / "synthetic-link").absolute()
    path_type = type(linked)
    original = path_type.is_symlink
    monkeypatch.setattr(
        path_type,
        "is_symlink",
        lambda self: self == linked or original(self),
    )
    with pytest.raises(RuntimeError, match="symlink or junction"):
        resolver(linked / "child", label="fixture")
