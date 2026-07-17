"""Strict, dependency-free validation for structured M5A router output.

The language model is never allowed to provide an executable task directly.
Its UTF-8 JSON text first passes this exact schema; a later adapter may then
construct the project-owned :class:`RouterDecision`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from langmani.environments.specs import BIN_IDS, OBJECT_IDS

ROUTER_OUTPUT_STATUSES = (
    "route",
    "reject_ambiguous",
    "reject_unsupported",
    "reject_malformed",
)
ROUTER_OUTPUT_FIELDS = frozenset({"status", "target_object_id", "target_bin_id", "reason"})
_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class RouterSchemaError(ValueError):
    """Raised when generated router text is not one exact safe JSON object."""


@dataclass(frozen=True, slots=True)
class StrictRouterPayload:
    """Schema-validated model output without executable environment authority."""

    status: str
    target_object_id: str | None
    target_bin_id: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.status not in ROUTER_OUTPUT_STATUSES:
            raise RouterSchemaError(f"unsupported router status: {self.status!r}")
        if not isinstance(self.reason, str) or _REASON_PATTERN.fullmatch(self.reason) is None:
            raise RouterSchemaError("reason must be a short lowercase machine-readable identifier")
        if self.status == "route":
            if self.target_object_id not in OBJECT_IDS:
                raise RouterSchemaError("a route requires one supported target_object_id")
            if self.target_bin_id not in BIN_IDS:
                raise RouterSchemaError("a route requires one supported target_bin_id")
            return
        if self.target_object_id is not None or self.target_bin_id is not None:
            raise RouterSchemaError("a rejected output must not contain an executable target")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "target_object_id": self.target_object_id,
            "target_bin_id": self.target_bin_id,
            "reason": self.reason,
        }


def _reject_constant(value: str) -> object:
    raise RouterSchemaError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RouterSchemaError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def parse_strict_router_json(raw_text: str) -> StrictRouterPayload:
    """Parse one JSON object without fences, trailing text, duplicates, or NaN.

    Leading and trailing JSON whitespace is harmless. Markdown code fences and
    prose are rejected naturally because the entire string must be one JSON
    value.
    """

    if not isinstance(raw_text, str):
        raise TypeError("router output must be text")
    if not raw_text.strip():
        raise RouterSchemaError("router output is empty")
    try:
        decoded = json.loads(
            raw_text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RouterSchemaError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RouterSchemaError(f"router output is not one strict JSON value: {error}") from error
    if not isinstance(decoded, dict):
        raise RouterSchemaError("router output must be one JSON object")
    payload = cast(dict[str, object], decoded)
    fields = frozenset(payload)
    if fields != ROUTER_OUTPUT_FIELDS:
        missing = sorted(ROUTER_OUTPUT_FIELDS - fields)
        extra = sorted(fields - ROUTER_OUTPUT_FIELDS)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise RouterSchemaError(
            "router output fields differ from schema (" + "; ".join(details) + ")"
        )
    status = payload["status"]
    reason = payload["reason"]
    target_object_id = payload["target_object_id"]
    target_bin_id = payload["target_bin_id"]
    if not isinstance(status, str):
        raise RouterSchemaError("status must be a string")
    if not isinstance(reason, str):
        raise RouterSchemaError("reason must be a string")
    if target_object_id is not None and not isinstance(target_object_id, str):
        raise RouterSchemaError("target_object_id must be a string or null")
    if target_bin_id is not None and not isinstance(target_bin_id, str):
        raise RouterSchemaError("target_bin_id must be a string or null")
    return StrictRouterPayload(
        status=status,
        target_object_id=cast(str | None, target_object_id),
        target_bin_id=cast(str | None, target_bin_id),
        reason=reason,
    )


__all__ = [
    "ROUTER_OUTPUT_FIELDS",
    "ROUTER_OUTPUT_STATUSES",
    "RouterSchemaError",
    "StrictRouterPayload",
    "parse_strict_router_json",
]
