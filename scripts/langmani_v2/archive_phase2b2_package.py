"""Archive and restore-validate the compact accepted Phase 2B.2 package evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b2 import (
    AcceptedDatasetPackage,
    create_deterministic_tar_gz,
    file_tree_manifest,
    restore_validate_archive,
    write_json_once,
)
from langmani.v2.push_dataset import PushDatasetContractError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--accepted-package", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--restore-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence = args.evidence_root.resolve()
    package_path = args.accepted_package.resolve()
    try:
        package_value = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PushDatasetContractError(f"cannot read accepted package: {error}") from error
    if not isinstance(package_value, dict):
        raise PushDatasetContractError("accepted package must be a JSON object")
    package = AcceptedDatasetPackage.from_dict(package_value)
    if package.acceptance_status != "ACCEPTED":
        raise PushDatasetContractError("only an accepted package may enter the recovery archive")
    try:
        package_path.relative_to(evidence)
    except ValueError as error:
        raise PushDatasetContractError("accepted package must be inside evidence-root") from error
    tree = file_tree_manifest(evidence)
    digest = str(tree["tree_digest"])
    archive_root = args.archive_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / (
        f"phase2b2-accepted-package-{digest.removeprefix('sha256:')[:16]}.tar.gz"
    )
    created = create_deterministic_tar_gz(
        source_root=evidence,
        archive_path=archive,
        prefix="accepted-package",
    )
    restored = restore_validate_archive(
        archive_path=archive,
        destination=args.restore_root.resolve(),
        prefix="accepted-package",
        expected_tree_digest=digest,
    )
    report = {
        "schema_version": "langmani-v2-phase2b2-accepted-package-archive-v0",
        "filename": archive.name,
        "size_bytes": created["size_bytes"],
        "sha256": created["sha256"],
        "source_tree_digest": digest,
        "restored_tree_digest": restored["restored_tree_digest"],
        "accepted_package_included": True,
        "restore_passed": restored["passed"],
        "optimizer_steps": 0,
        "passed": restored["passed"],
    }
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
