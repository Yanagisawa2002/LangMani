"""Hash named raw and LeRobot replicas and verify their restoration reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langmani.v2.phase2b2 import build_replication_report, write_json_once
from langmani.v2.push_archive import sha256_file
from langmani.v2.push_dataset import PushDatasetContractError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-raw-archive", type=Path, required=True)
    parser.add_argument("--expected-lerobot-archive", type=Path, required=True)
    parser.add_argument("--raw-replica", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--raw-restore-report", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--lerobot-replica", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument(
        "--lerobot-restore-report", action="append", default=[], metavar="LABEL=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _pairs(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise PushDatasetContractError(f"{label} must use LABEL=PATH")
        name, path = value.split("=", 1)
        if not name or not path or name in result:
            raise PushDatasetContractError(f"{label} labels must be non-empty and unique")
        result[name] = path
    return result


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PushDatasetContractError(f"cannot read restore report: {error}") from error
    if not isinstance(value, dict):
        raise PushDatasetContractError("restore report must be a JSON object")
    return value


def _report_passes(*, report: dict[str, Any], asset: str, expected_sha256: str) -> bool:
    if report.get("schema_version") == "langmani-v2-phase2b2-restored-copy-v0":
        return bool(
            report.get("asset") == asset
            and report.get("archive_sha256") == expected_sha256
            and report.get("passed") is True
        )
    if report.get("schema_version") == "langmani-v2-phase2b2-archive-report-v0":
        archives = report.get("archives")
        item = archives.get(asset) if isinstance(archives, dict) else None
        return bool(
            isinstance(item, dict)
            and item.get("sha256") == expected_sha256
            and item.get("restore_passed") is True
            and report.get("passed") is True
        )
    return False


def _one_asset(
    *, expected: Path, asset: str, values: list[str], restore_reports: list[str]
) -> dict[str, object]:
    expected_sha256 = f"sha256:{sha256_file(expected.resolve())}"
    replicas = _pairs(values, f"{asset} replica")
    reports = _pairs(restore_reports, f"{asset} restore report")
    if not set(reports).issubset(replicas):
        raise PushDatasetContractError("restore report labels must name declared replicas")
    tested = {
        label
        for label, report_path in reports.items()
        if _report_passes(
            report=_read(Path(report_path).expanduser().resolve()),
            asset=asset,
            expected_sha256=expected_sha256,
        )
    }
    values_for_report: list[dict[str, object]] = []
    for label, path_value in replicas.items():
        if path_value == "PENDING":
            values_for_report.append(
                {
                    "location": label,
                    "copy_completed": False,
                    "file_size": 0,
                    "sha256": None,
                    "restore_test_passed": False,
                }
            )
            continue
        path = Path(path_value).expanduser().resolve()
        values_for_report.append(
            {
                "location": label,
                "copy_completed": path.is_file(),
                "file_size": path.stat().st_size if path.is_file() else 0,
                "sha256": f"sha256:{sha256_file(path)}" if path.is_file() else None,
                "restore_test_passed": label in tested,
            }
        )
    return build_replication_report(expected_sha256=expected_sha256, replicas=values_for_report)


def main() -> int:
    args = parse_args()
    raw = _one_asset(
        expected=args.expected_raw_archive,
        asset="raw",
        values=args.raw_replica,
        restore_reports=args.raw_restore_report,
    )
    lerobot = _one_asset(
        expected=args.expected_lerobot_archive,
        asset="lerobot",
        values=args.lerobot_replica,
        restore_reports=args.lerobot_restore_report,
    )
    report = {
        "schema_version": "langmani-v2-phase2b2-replication-set-v0",
        "assets": {"raw": raw, "lerobot": lerobot},
        "minimum_temporary_persistence_gate": raw["passed"] is True and lerobot["passed"] is True,
        "three_independent_copies": raw["three_independent_copies"] is True
        and lerobot["three_independent_copies"] is True,
        "passed": raw["passed"] is True and lerobot["passed"] is True,
    }
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
