"""Build the deterministic development-safe M5A language corpus archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
import uuid
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langmani.datasets.identity import sha256_hex  # noqa: E402
from langmani.language.corpus import build_language_corpus  # noqa: E402
from langmani.language.router_types import LanguageSplit  # noqa: E402
from langmani.language.schedules import build_language_schedule_locks  # noqa: E402
from langmani.policies.act_runtime import atomic_write_json  # noqa: E402

COMMAND_SCHEMA = "langmani-m5a-build-language-corpus-command-v1"
ARCHIVE_SCHEMA = "langmani-m5a-language-corpus-archive-v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "m5a" / "langmani-language-corpus-v1"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "corpus-build.json"
PROTECTED_ROOTS = tuple(
    PROJECT_ROOT / name for name in ("src", "scripts", "environment", "tests", "docs", ".git")
)
PROTECTED_GENERATED_ROOTS = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3a",
    PROJECT_ROOT / "outputs" / "datasets" / "m3b",
    PROJECT_ROOT / "outputs" / "models",
    *(
        PROJECT_ROOT / "outputs" / "diagnostics" / name
        for name in ("m0", "m1", "m2", "m3a", "m3b", "m4", "m42", "m43")
    ),
)


class CorpusCommandError(RuntimeError):
    """Raised when corpus output safety or immutable validation fails."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise CorpusCommandError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _safe_paths(output_root: Path, report: Path) -> tuple[Path, Path]:
    output = _resolved_unlinked(output_root, label="M5A corpus output")
    report_path = _resolved_unlinked(report, label="M5A corpus command report")
    protected = tuple(
        _resolved_unlinked(path, label="protected repository or historical artifact")
        for path in (*PROTECTED_ROOTS, *PROTECTED_GENERATED_ROOTS)
    )
    if any(_overlaps(output, path) for path in protected):
        raise CorpusCommandError(
            "corpus output cannot overlap source, Git, historical datasets, models, or evidence"
        )
    if any(_overlaps(report_path, path) for path in (*protected, output)):
        raise CorpusCommandError(
            "corpus report cannot overlap source, Git, historical artifacts, or corpus output"
        )
    if output.exists() and (not output.is_dir() or output.is_symlink() or output.is_junction()):
        raise CorpusCommandError("corpus output must be a real directory when present")
    if report_path.exists() and (not report_path.is_file() or report_path.is_symlink()):
        raise CorpusCommandError("corpus report must be a real file when present")
    return output, report_path


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise CorpusCommandError(f"refusing to overwrite immutable corpus file: {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _records(root: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink() or path.is_junction():
            raise CorpusCommandError(f"corpus archive contains a linked entry: {path}")
        if not path.is_file() or path.name in {"artifact_manifest.json", "complete.json"}:
            continue
        result.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return result


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusCommandError(
            f"could not read immutable corpus artifact {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CorpusCommandError(f"immutable corpus artifact must be one object: {path}")
    return value


def _validate_archive(root: Path, *, corpus_fingerprint: str) -> str:
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise CorpusCommandError("completed corpus archive must be one real directory")
    manifest = _read_object(root / "artifact_manifest.json")
    complete = _read_object(root / "complete.json")
    records = _records(root)
    expected = (
        f"sha256:{sha256_hex({'corpus_fingerprint': corpus_fingerprint, 'artifacts': records})}"
    )
    if manifest.get("schema_version") != ARCHIVE_SCHEMA or manifest.get("artifacts") != records:
        raise CorpusCommandError("corpus archive checksum manifest differs")
    if manifest.get("archive_fingerprint") != expected:
        raise CorpusCommandError("corpus archive fingerprint differs")
    if complete != {
        "schema_version": ARCHIVE_SCHEMA,
        "corpus_fingerprint": corpus_fingerprint,
        "archive_fingerprint": expected,
        "passed": True,
    }:
        raise CorpusCommandError("corpus completion marker differs")
    return expected


def _example_jsonl(examples: Sequence[object]) -> bytes:
    lines: list[str] = []
    for example in examples:
        to_dict = getattr(example, "to_dict", None)
        if not callable(to_dict):
            raise CorpusCommandError("language example lacks to_dict()")
        lines.append(
            json.dumps(
                to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_archive(output_root: Path) -> tuple[str, str, bool]:
    corpus = build_language_corpus()
    language_dev, language_final = build_language_schedule_locks(corpus)
    corpus_fingerprint = corpus.manifest.corpus_fingerprint
    if output_root.exists():
        return (
            corpus_fingerprint,
            _validate_archive(output_root, corpus_fingerprint=corpus_fingerprint),
            True,
        )
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{output_root.name}.{uuid.uuid4().hex}.staging"
    if staging.exists():
        raise CorpusCommandError("fresh corpus staging path unexpectedly exists")
    staging.mkdir()
    try:
        _write_new(staging / "corpus_manifest.json", _json_bytes(corpus.manifest.to_dict()))
        _write_new(
            staging / "family_manifest.json",
            _json_bytes(
                {
                    "schema_version": ARCHIVE_SCHEMA,
                    "families": [family.to_dict() for family in corpus.families],
                }
            ),
        )
        _write_new(
            staging / "language_schedule_locks.json",
            _json_bytes(
                {
                    "schema_version": ARCHIVE_SCHEMA,
                    "development": language_dev.to_dict(),
                    "final": language_final.to_dict(),
                }
            ),
        )
        for split in (LanguageSplit.TRAIN, LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT):
            _write_new(
                staging / f"{split.value}.jsonl",
                _example_jsonl(corpus.examples_for_split(split)),
            )
        records = _records(staging)
        archive_fingerprint = (
            f"sha256:{sha256_hex({'corpus_fingerprint': corpus_fingerprint, 'artifacts': records})}"
        )
        _write_new(
            staging / "artifact_manifest.json",
            _json_bytes(
                {
                    "schema_version": ARCHIVE_SCHEMA,
                    "corpus_fingerprint": corpus_fingerprint,
                    "archive_fingerprint": archive_fingerprint,
                    "artifacts": records,
                }
            ),
        )
        _write_new(
            staging / "complete.json",
            _json_bytes(
                {
                    "schema_version": ARCHIVE_SCHEMA,
                    "corpus_fingerprint": corpus_fingerprint,
                    "archive_fingerprint": archive_fingerprint,
                    "passed": True,
                }
            ),
        )
        _validate_archive(staging, corpus_fingerprint=corpus_fingerprint)
        if staging.stat().st_dev != parent.stat().st_dev:
            raise CorpusCommandError("corpus staging and destination are on different filesystems")
        os.replace(staging, output_root)
        return (
            corpus_fingerprint,
            _validate_archive(output_root, corpus_fingerprint=corpus_fingerprint),
            False,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "passed": False,
        "dry_run": bool(args.dry_run),
        "corpus_validated": False,
        "split_isolation_validated": False,
        "final_schedule_locked": False,
        "language_final_accessed": False,
    }
    report_path: Path | None = None
    try:
        output_root, report_path = _safe_paths(args.output_root, args.report)
        corpus = build_language_corpus()
        language_dev, language_final = build_language_schedule_locks(corpus)
        payload.update(
            {
                "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
                "corpus_manifest": corpus.manifest.to_dict(),
                "language_development_schedule": language_dev.to_dict(),
                "language_final_schedule": language_final.to_dict(),
                "corpus_validated": True,
                "split_isolation_validated": corpus.isolation_report.passed,
                "final_schedule_locked": language_final.sealed,
            }
        )
        if not args.dry_run:
            corpus_fingerprint, archive_fingerprint, reused = build_archive(output_root)
            payload.update(
                {
                    "corpus_fingerprint": corpus_fingerprint,
                    "archive_fingerprint": archive_fingerprint,
                    "archive_reused": reused,
                    "output_root": str(output_root),
                }
            )
        payload["passed"] = True
    except Exception as error:  # noqa: BLE001 - command boundary preserves diagnostics
        traceback.print_exc()
        payload.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error) or repr(error),
            }
        )
    if report_path is not None:
        try:
            atomic_write_json(report_path, payload)
        except Exception:
            traceback.print_exc()
            return 1
    console_summary = {
        "schema_version": payload["schema_version"],
        "passed": payload["passed"],
        "dry_run": payload["dry_run"],
        "corpus_fingerprint": payload.get("corpus_fingerprint"),
        "corpus_validated": payload["corpus_validated"],
        "split_isolation_validated": payload["split_isolation_validated"],
        "final_schedule_locked": payload["final_schedule_locked"],
        "language_final_accessed": payload["language_final_accessed"],
        "report": None if report_path is None else str(report_path),
    }
    if payload["passed"] is not True:
        console_summary.update(
            {
                "error_type": payload.get("error_type"),
                "error_message": payload.get("error_message"),
            }
        )
    print(json.dumps(console_summary, sort_keys=True, allow_nan=False))
    return 0 if payload["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
