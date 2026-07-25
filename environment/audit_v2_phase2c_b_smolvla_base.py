"""Pin, download, hash, and strictly construct the Phase 2C-B SmolVLA base."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from langmani.datasets.identity import sha256_file
from langmani.v2.phase2c_b import (
    OFFICIAL_BASE_CONFIG_SHA256,
    OFFICIAL_BASE_MODEL,
    OFFICIAL_BASE_MODEL_SHA256,
    OFFICIAL_BASE_REVISION,
    OFFICIAL_LEROBOT_VERSION,
    OFFICIAL_VLM_CONFIG_SHA256,
    OFFICIAL_VLM_MODEL,
    OFFICIAL_VLM_MODEL_SHA256,
    OFFICIAL_VLM_REVISION,
    ModelKind,
    SmolVLATrainingConfig,
    canonical_fingerprint,
)
from langmani.v2.phase2c_b_smolvla import (
    BoundedSmolVLAPolicyV1,
    build_official_config,
    load_pretrained_policy,
    load_smolvla_statistics,
    make_processors,
    processor_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _file_record(path: Path) -> dict[str, object]:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": f"sha256:{sha256_file(path)}",
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("strict SmolVLA construction requires CUDA")
    try:
        import lerobot  # type: ignore[import-untyped]
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError("official SmolVLA download dependencies are unavailable") from error
    if getattr(lerobot, "__version__", None) != OFFICIAL_LEROBOT_VERSION:
        raise RuntimeError("installed LeRobot must be exactly 0.6.0")
    if _package_version("num2words") != "0.5.14":
        raise RuntimeError("official SmolVLA requires num2words==0.5.14")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    api = HfApi()
    base_info = api.model_info(
        OFFICIAL_BASE_MODEL,
        revision=OFFICIAL_BASE_REVISION,
        files_metadata=True,
    )
    if base_info.sha != OFFICIAL_BASE_REVISION:
        raise RuntimeError("Hugging Face resolved a different SmolVLA base revision")
    cache = args.cache_dir.resolve()
    base_snapshot = Path(
        snapshot_download(
            repo_id=OFFICIAL_BASE_MODEL,
            revision=OFFICIAL_BASE_REVISION,
            cache_dir=cache,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            ],
        )
    )
    base_required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    if any(not (base_snapshot / name).is_file() for name in base_required):
        raise RuntimeError("downloaded official base snapshot is incomplete")
    if (
        sha256_file(base_snapshot / "config.json") != OFFICIAL_BASE_CONFIG_SHA256
        or sha256_file(base_snapshot / "model.safetensors") != OFFICIAL_BASE_MODEL_SHA256
    ):
        raise RuntimeError("downloaded official base bytes changed")

    vlm_info = api.model_info(
        OFFICIAL_VLM_MODEL,
        revision=OFFICIAL_VLM_REVISION,
        files_metadata=True,
    )
    if vlm_info.sha != OFFICIAL_VLM_REVISION:
        raise RuntimeError("Hugging Face resolved a different nested VLM revision")
    vlm_snapshot = Path(
        snapshot_download(
            repo_id=OFFICIAL_VLM_MODEL,
            revision=OFFICIAL_VLM_REVISION,
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
    if (
        sha256_file(vlm_snapshot / "config.json") != OFFICIAL_VLM_CONFIG_SHA256
        or sha256_file(vlm_snapshot / "model.safetensors") != OFFICIAL_VLM_MODEL_SHA256
    ):
        raise RuntimeError("downloaded nested VLM bytes changed")

    training = SmolVLATrainingConfig()
    config = build_official_config(
        base_snapshot=base_snapshot,
        vlm_snapshot=vlm_snapshot,
        device="cuda",
        training=training,
    )
    started = time.perf_counter()
    policy, loading = load_pretrained_policy(base_snapshot=base_snapshot, config=config)
    load_seconds = time.perf_counter() - started
    policy.to(device=torch.device("cuda"))
    torch.cuda.synchronize()
    statistics = load_smolvla_statistics(
        args.primary_root.resolve(),
        model_kind=ModelKind.PICK,
    )
    preprocessor, postprocessor = make_processors(config, statistics)
    processor = processor_manifest(config=config, statistics=statistics)
    output_root = args.output_root.resolve()
    processor_root = output_root / "audited_base_processors"
    preprocessor.save_pretrained(processor_root)
    postprocessor.save_pretrained(processor_root)
    policy.action_transform.save(processor_root)

    runtime = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "lerobot": getattr(lerobot, "__version__", None),
        "transformers": _package_version("transformers"),
        "diffusers": _package_version("diffusers"),
        "tokenizers": _package_version("tokenizers"),
        "safetensors": _package_version("safetensors"),
        "num2words": _package_version("num2words"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    base_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-base-model-manifest-v0",
        "repo_id": OFFICIAL_BASE_MODEL,
        "requested_revision": OFFICIAL_BASE_REVISION,
        "resolved_revision": base_info.sha,
        "snapshot_path": base_snapshot.as_posix(),
        "files": {name: _file_record(base_snapshot / name) for name in base_required},
        "nested_vlm": {
            "repo_id": OFFICIAL_VLM_MODEL,
            "requested_revision": OFFICIAL_VLM_REVISION,
            "resolved_revision": vlm_info.sha,
            "snapshot_path": vlm_snapshot.as_posix(),
            "config": _file_record(vlm_snapshot / "config.json"),
            "model": _file_record(vlm_snapshot / "model.safetensors"),
        },
        "runtime": runtime,
        "official_policy_class": (
            f"{BoundedSmolVLAPolicyV1.__mro__[1].__module__}."
            f"{BoundedSmolVLAPolicyV1.__mro__[1].__qualname__}"
        ),
        "bounded_subclass": (
            f"{BoundedSmolVLAPolicyV1.__module__}.{BoundedSmolVLAPolicyV1.__qualname__}"
        ),
        "forward_signature": str(inspect.signature(BoundedSmolVLAPolicyV1.forward)),
        "predict_action_chunk_signature": str(
            inspect.signature(BoundedSmolVLAPolicyV1.predict_action_chunk)
        ),
        "strict_load_seconds": load_seconds,
        "passed": True,
    }
    base_manifest = {**base_semantic, "fingerprint": canonical_fingerprint(base_semantic)}
    adaptation_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-embodiment-adaptation-v0",
        "pretrained_public_state_dimension": 6,
        "pretrained_public_action_dimension": 6,
        "pretrained_camera_features": [
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ],
        "langmani_state_dimension": 9,
        "langmani_action_dimension": 8,
        "langmani_camera_features": ["observation.images.base_camera"],
        "internal_max_state_dimension": int(config.max_state_dim),
        "internal_max_action_dimension": int(config.max_action_dim),
        "official_adaptation_path": (
            "public PolicyFeature replacement with unchanged 32D internal padding/projections"
        ),
        "state_projection_reinitialized": False,
        "action_input_projection_reinitialized": False,
        "action_output_projection_reinitialized": False,
        "visual_language_backbone_reinitialized": False,
        "strict_pretrained_loading": True,
        "action_transform": policy.action_transform.semantic_dict(),
        "passed": True,
    }
    adaptation = {
        **adaptation_semantic,
        "fingerprint": canonical_fingerprint(adaptation_semantic),
    }
    loading["strict_load_seconds"] = load_seconds
    loading["base_model_manifest_fingerprint"] = base_manifest["fingerprint"]
    loading_semantic = {key: value for key, value in loading.items() if key != "fingerprint"}
    loading = {
        **loading_semantic,
        "fingerprint": canonical_fingerprint(loading_semantic),
    }
    processor["serialized_root"] = processor_root.as_posix()
    processor_semantic = {key: value for key, value in processor.items() if key != "fingerprint"}
    processor = {
        **processor_semantic,
        "fingerprint": canonical_fingerprint(processor_semantic),
    }
    _write_new_or_equal(output_root / "base_model_manifest.json", base_manifest)
    _write_new_or_equal(output_root / "embodiment_adaptation_manifest.json", adaptation)
    _write_new_or_equal(output_root / "pretrained_weight_loading_audit.json", loading)
    _write_new_or_equal(output_root / "processor_contract.json", processor)
    completion_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-base-audit-complete-v0",
        "base_model_manifest_fingerprint": base_manifest["fingerprint"],
        "embodiment_adaptation_fingerprint": adaptation["fingerprint"],
        "pretrained_loading_fingerprint": loading["fingerprint"],
        "processor_contract_fingerprint": processor["fingerprint"],
        "passed": True,
    }
    completion = {
        **completion_semantic,
        "fingerprint": canonical_fingerprint(completion_semantic),
    }
    _write_new_or_equal(output_root / "base_audit_complete.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
