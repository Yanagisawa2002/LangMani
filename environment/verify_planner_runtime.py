"""Verify the isolated NumPy-1 runtime required by mplib 0.1.1."""

from __future__ import annotations

import json
import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langmani.environments.specs import TaskSpec
from langmani.experts.command_support import create_expert_environment, task_reset_options
from langmani.experts.planner import MplibPandaPlannerAdapter
from langmani.experts.runtime import (
    EXPECTED_PLANNER_RUNTIME_VERSIONS,
    planner_runtime_matches_expected,
    query_planner_runtime_versions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "outputs" / "diagnostics" / "m2" / "planner_runtime.json"


@dataclass(slots=True)
class Report:
    """Collect explicit side-runtime evidence."""

    checks: list[dict[str, Any]] = field(default_factory=list)

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "pass" if condition else "fail"
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(check["status"] == "fail" for check in self.checks)

    def write(self, *, versions: dict[str, str | None]) -> None:
        payload = {
            "schema_version": "langmani-planner-runtime-verification-v0",
            "interpreter": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "runtime_versions": versions,
            "expected_versions": dict(EXPECTED_PLANNER_RUNTIME_VERSIONS),
            "checks": self.checks,
            "passed": not self.failed,
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    REPORT_PATH.unlink(missing_ok=True)
    report = Report()
    versions: dict[str, str | None] = {}
    env: Any | None = None
    adapter: MplibPandaPlannerAdapter | None = None
    try:
        versions = query_planner_runtime_versions(sys.executable)
        report.check(
            "planner runtime package pins",
            planner_runtime_matches_expected(versions),
            json.dumps(versions, sort_keys=True),
        )
        report.check(
            "planner runtime Python",
            platform.python_version() == "3.12.13",
            f"installed={platform.python_version()!r}, expected='3.12.13'",
        )
        if not report.failed:
            task_spec = TaskSpec(
                target_object_id="red_cube",
                target_bin_id="left_bin",
                instruction_template_id="canonical_v0",
            )
            env = create_expert_environment(
                diagnostic_rendering=False,
                sim_backend="physx_cpu",
            )
            env.reset(seed=0, options=task_reset_options(task_spec))
            adapter = MplibPandaPlannerAdapter(env)
            adapter.synchronize()
            report.check(
                "mplib Panda construction and synchronization",
                True,
                "public mplib Planner constructed from installed Panda URDF/SRDF",
            )
    except Exception as error:  # noqa: BLE001 - verifier command boundary
        traceback.print_exc()
        report.check(
            "planner runtime execution",
            False,
            f"{type(error).__name__}: {str(error) or repr(error)}",
        )
    finally:
        if adapter is not None:
            adapter.close()
        if env is not None:
            try:
                env.close()
            except Exception as error:  # noqa: BLE001 - verifier command boundary
                traceback.print_exc()
                report.check(
                    "planner runtime close",
                    False,
                    f"{type(error).__name__}: {str(error) or repr(error)}",
                )
    report.write(versions=versions)
    print(f"[INFO] report: {REPORT_PATH}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
