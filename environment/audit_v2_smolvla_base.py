"""Download and content-bind the exact official LeRobot SmolVLA base revision."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
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
    parser.add_argument("--verify-construction", action="store_true")
    return parser.parse_args(argv)


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _construct_official_policy(
    *,
    protocol: dict[str, object],
    base_snapshot: Path,
    vlm_snapshot: Path,
) -> dict[str, object]:
    """Strictly load the official weights under the exact LangMani feature contract."""

    try:
        import torch
        from lerobot.configs import FeatureType, PolicyFeature  # type: ignore[import-untyped]
        from lerobot.configs.policies import PreTrainedConfig  # type: ignore[import-untyped]
        from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-untyped]
            SmolVLAPolicy,
        )
    except ImportError as error:
        raise Phase2CContractError(
            "official SmolVLA construction dependencies are unavailable"
        ) from error
    if not torch.cuda.is_available():
        raise Phase2CContractError("strict SmolVLA construction requires a CUDA target")
    override = protocol.get("feature_override")
    if not isinstance(override, dict):
        raise Phase2CContractError("Phase 2C feature override is malformed")
    input_payload = override.get("input_features")
    output_payload = override.get("output_features")
    if not isinstance(input_payload, dict) or not isinstance(output_payload, dict):
        raise Phase2CContractError("Phase 2C feature maps are malformed")

    config = PreTrainedConfig.from_pretrained(
        base_snapshot,
        cli_overrides=["--input_features=null"],
    )
    if config.input_features is not None:
        raise Phase2CContractError("LeRobot did not replace the published input mapping with null")

    def convert(values: dict[str, object]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, payload in values.items():
            if not isinstance(payload, dict):
                raise Phase2CContractError(f"feature {name} is malformed")
            type_name = payload.get("type")
            shape = payload.get("shape")
            if not isinstance(type_name, str) or not isinstance(shape, list):
                raise Phase2CContractError(f"feature {name} type or shape is malformed")
            result[name] = PolicyFeature(
                type=FeatureType[type_name],
                shape=tuple(int(item) for item in shape),
            )
        return result

    config.input_features = convert(input_payload)
    config.output_features = convert(output_payload)
    expected_inputs = {"observation.images.base_camera", "observation.state"}
    if set(config.input_features) != expected_inputs or set(config.output_features) != {"action"}:
        raise Phase2CContractError("strict construction feature keys differ from Phase 2C")
    config.vlm_model_name = vlm_snapshot.as_posix()
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(base_snapshot, config=config, strict=True)
    loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    policy.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    policy.eval()
    torch.cuda.synchronize()
    moved = time.perf_counter()
    parameters = list(policy.parameters())
    dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in parameters})
    if dtypes != ["bfloat16"]:
        raise Phase2CContractError("strictly loaded SmolVLA parameters are not uniformly bfloat16")
    return {
        "strict_weight_load": True,
        "feature_binding_semantic": "complete_replacement_after_null_v0",
        "input_features": {
            name: {"type": feature.type.value, "shape": list(feature.shape)}
            for name, feature in config.input_features.items()
        },
        "output_features": {
            name: {"type": feature.type.value, "shape": list(feature.shape)}
            for name, feature in config.output_features.items()
        },
        "model_chunk_size": int(config.chunk_size),
        "official_n_action_steps": int(config.n_action_steps),
        "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in parameters if parameter.requires_grad
        ),
        "observed_parameter_dtypes": dtypes,
        "strict_weight_load_seconds": loaded - started,
        "cuda_move_seconds": moved - loaded,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "device": "cuda",
        "dtype": "bfloat16",
        "passed": True,
    }


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
    if _version("num2words") != "0.5.14":
        raise Phase2CContractError("official SmolVLA requires num2words==0.5.14")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise Phase2CContractError("Hugging Face resolved a different SmolVLA revision")
    cache: Path = args.cache_dir.resolve()
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            ],
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
    construction: dict[str, object] | None = None
    vlm_report: dict[str, object] | None = None
    if args.verify_construction:
        vlm = base.get("vlm_dependency")
        if not isinstance(vlm, dict):
            raise Phase2CContractError("Phase 2C base model lacks its VLM dependency binding")
        vlm_repo = str(vlm.get("repo_id"))
        vlm_revision = str(vlm.get("revision"))
        vlm_info = HfApi().model_info(vlm_repo, revision=vlm_revision, files_metadata=True)
        if vlm_info.sha != vlm_revision:
            raise Phase2CContractError("Hugging Face resolved a different nested VLM revision")
        vlm_snapshot = Path(
            snapshot_download(
                repo_id=vlm_repo,
                revision=vlm_revision,
                cache_dir=cache,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                    "processor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "chat_template.json",
                    "merges.txt",
                    "vocab.json",
                ],
            )
        )
        vlm_config = sha256_file(vlm_snapshot / "config.json")
        vlm_model = sha256_file(vlm_snapshot / "model.safetensors")
        if vlm_config != vlm.get("config_sha256") or vlm_model != vlm.get("model_sha256"):
            raise Phase2CContractError("nested VLM bytes changed from the reviewed revision")
        vlm_report = {
            "repo_id": vlm_repo,
            "requested_revision": vlm_revision,
            "resolved_revision": vlm_info.sha,
            "config_sha256": f"sha256:{vlm_config}",
            "model_sha256": f"sha256:{vlm_model}",
        }
        construction = _construct_official_policy(
            protocol=protocol,
            base_snapshot=snapshot,
            vlm_snapshot=vlm_snapshot,
        )
    report = {
        "schema_version": "langmani-v2-phase2c-base-model-audit-v1",
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "snapshot_path": snapshot.as_posix(),
        "files": files,
        "nested_vlm": vlm_report,
        "construction": construction,
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
            "num2words": _version("num2words"),
        },
        "protocol_sha256": f"sha256:{sha256_file(protocol_path)}",
        "passed": True,
    }
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
