"""Download and content-bind the exact official LeRobot SmolVLA base revision."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from langmani.v2.phase2c import Phase2CContractError, read_json_object, sha256_file, write_json_once

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2c_smolvla_push.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = read_json_object(protocol_path, "Phase 2C protocol")
    base = protocol.get("base_model")
    if not isinstance(base, dict):
        raise Phase2CContractError("Phase 2C protocol base model is malformed")
    repo_id = str(base.get("repo_id"))
    revision = str(base.get("revision"))
    expected_config = str(base.get("config_sha256"))
    if repo_id != "lerobot/smolvla_base" or len(revision) != 40:
        raise Phase2CContractError("Phase 2C requires the pinned official SmolVLA base")
    try:
        import lerobot  # type: ignore[import-untyped]
        import torch
        from huggingface_hub import HfApi, snapshot_download
        from lerobot.policies.factory import (  # type: ignore[import-untyped]
            make_pre_post_processors,
        )
        from lerobot.policies.smolvla.configuration_smolvla import (  # type: ignore[import-untyped]
            SmolVLAConfig,
        )
        from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-untyped]
            SmolVLAPolicy,
        )
    except ImportError as error:
        raise Phase2CContractError("official SmolVLA audit dependencies are unavailable") from error
    if getattr(lerobot, "__version__", None) != "0.6.0":
        raise Phase2CContractError("installed LeRobot must be exactly 0.6.0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise Phase2CContractError("Hugging Face resolved a different SmolVLA revision")
    cache = args.cache_dir.resolve()
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache,
            allow_patterns=(
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            ),
        )
    )
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise Phase2CContractError("official base snapshot is incomplete: " + ", ".join(missing))
    if sha256_file(snapshot / "config.json") != expected_config:
        raise Phase2CContractError("official base config bytes changed from the reviewed revision")
    files = {
        name: {
            "size_bytes": (snapshot / name).stat().st_size,
            "sha256": f"sha256:{sha256_file(snapshot / name)}",
        }
        for name in required
    }
    report = {
        "schema_version": "langmani-v2-phase2c-base-model-audit-v0",
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "snapshot_path": snapshot.as_posix(),
        "files": files,
        "official_api": {
            "config_class": f"{SmolVLAConfig.__module__}.{SmolVLAConfig.__qualname__}",
            "policy_class": f"{SmolVLAPolicy.__module__}.{SmolVLAPolicy.__qualname__}",
            "forward_signature": str(inspect.signature(SmolVLAPolicy.forward)),
            "predict_action_chunk_signature": str(
                inspect.signature(SmolVLAPolicy.predict_action_chunk)
            ),
            "processor_factory_signature": str(inspect.signature(make_pre_post_processors)),
            "checkpoint_serialization": "Hugging Face config.json + model.safetensors + processor JSON",
            "action_queue": "select_action owns queue; Phase 2C adapter bypasses it via predict_action_chunk",
            "stochastic_generation": "flow-matching noise; Phase 2C seeds each query",
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "lerobot": getattr(lerobot, "__version__", None),
            "transformers": _version("transformers"),
            "tokenizers": _version("tokenizers"),
            "safetensors": _version("safetensors"),
        },
        "protocol_sha256": f"sha256:{sha256_file(protocol_path)}",
        "passed": True,
    }
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
