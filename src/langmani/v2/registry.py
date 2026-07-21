"""Repository-controlled policy adapter loading for LangMani 2.0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langmani.v2.act_adapter import ACT_ADAPTER_NAME, ActPerTaskPolicyAdapter
from langmani.v2.policy import PolicyAdapter, PolicyContractError, PolicyRegistry


def default_policy_registry() -> PolicyRegistry:
    """Return registrations implemented today, without fake future adapters."""

    registry = PolicyRegistry()
    registry.register(ACT_ADAPTER_NAME, ActPerTaskPolicyAdapter.from_config)
    return registry


def load_policy_adapter(
    *,
    config_path: Path,
    project_root: Path,
    registry: PolicyRegistry | None = None,
) -> PolicyAdapter:
    """Load exactly the adapter named by one repository-controlled JSON config."""

    try:
        value: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyContractError(f"cannot read policy config {config_path}: {error}") from error
    if not isinstance(value, dict):
        raise PolicyContractError("policy config must be a JSON object")
    adapter_name = value.get("adapter_name")
    if not isinstance(adapter_name, str):
        raise PolicyContractError("policy config lacks adapter_name")
    selected = registry or default_policy_registry()
    return selected.create(
        adapter_name,
        config_path=config_path,
        project_root=project_root,
    )


__all__ = ["default_policy_registry", "load_policy_adapter"]
