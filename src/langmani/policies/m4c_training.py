"""Change only dataset task strings around the frozen upstream SmolVLA training adapter."""

from __future__ import annotations

import csv
import hashlib
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from langmani.policies.m4a_data import file_digest, read_json, write_json
from langmani.policies.m4b_training import train_official
from langmani.policies.m4c_language import sample_expression


def model_state_hash(policy: Any) -> str:
    """Hash exact tensor storage, including BF16, without consuming a random number."""
    hasher = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        tensor = value.detach().contiguous()
        hasher.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        hasher.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return hasher.hexdigest()


class LanguageSamples:
    """FIFO matches Accelerate's one-batch lookahead; only optimizer-consumed rows are counted."""

    def __init__(
        self,
        path: Path,
        manifest: dict[str, Any],
        level: str,
        split: dict[str, Any],
        start_ordinal: int = 0,
    ) -> None:
        self.manifest, self.level = manifest, level
        self.episodes = {
            r["episode_id"]: r
            for r in split["episodes"]
            if r["episode_id"] in split["train_episode_ids"]
        }
        self.ordinal = start_ordinal
        self.start_ordinal = start_ordinal
        self.pending: deque[dict[str, Any]] = deque()
        self.robot_hash = hashlib.sha256()
        self.consumed = 0
        self.counts: Counter[str] = Counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.stream,
            fieldnames=[
                "ordinal",
                "episode_id",
                "frame_index",
                "semantic_goal_id",
                "template_id",
                "paraphrase_family",
                "instruction_text",
            ],
        )
        self.writer.writeheader()

    def relabel(self, row: dict[str, Any]) -> dict[str, Any]:
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("language sample audit requires the frozen num_workers=0 setting")
        episode, frame = int(row["episode_index"]), int(row["frame_index"])
        source = self.episodes[episode]
        if row["task"] != source["instruction"]:
            raise ValueError("underlying robot trajectory no longer has its canonical source text")
        chosen = sample_expression(
            self.manifest, self.level, source["semantic_goal"], self.ordinal, episode, frame
        )
        self.pending.append(
            {"ordinal": self.ordinal, "episode_id": episode, "frame_index": frame, **chosen}
        )
        self.ordinal += 1
        return {**row, "task": chosen["instruction_text"]}

    def consume(self, batch_size: int) -> None:
        if batch_size != 8 or len(self.pending) < batch_size:
            raise RuntimeError("optimizer batch or sample FIFO changed")
        for _ in range(batch_size):
            row = self.pending.popleft()
            if row["ordinal"] != self.start_ordinal + self.consumed:
                raise RuntimeError("non-contiguous sample presentation order")
            self.writer.writerow(row)
            self.robot_hash.update(f"{row['episode_id']}:{row['frame_index']}\n".encode())
            self.counts[f"{row['semantic_goal_id']}/{row['template_id']}"] += 1
            self.consumed += 1
        if self.consumed % 800 == 0:
            self.stream.flush()

    def close(self) -> None:
        self.stream.close()

    def receipt(self) -> dict[str, Any]:
        if len(self.pending) > 8:
            raise RuntimeError("unexpected sampler lookahead")
        return {
            "start_ordinal": self.start_ordinal,
            "sample_presentations": self.consumed,
            "robot_sample_sequence_sha256": self.robot_hash.hexdigest(),
            "language_counts": dict(sorted(self.counts.items())),
            "unused_prefetched_samples": len(self.pending),
            "note": "unused Accelerate lookahead is excluded from presentation counts/hashes",
        }


class LanguageDataset(torch.utils.data.Dataset):
    def __init__(self, source: Any, samples: LanguageSamples) -> None:
        self.source, self.samples = source, samples

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples.relabel(self.source[index])

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self[i] for i in indices]


def consolidate_samples(
    output: Path, manifest: dict[str, Any], level: str, split: dict[str, Any], steps: int
) -> dict[str, Any]:
    """Keep each resumed prefix once; preserve abandoned post-checkpoint rows in their files."""
    active: list[tuple[int, Path]] = []
    runtime_seconds = 0.0
    for invocation in sorted((output / "language_invocations").glob("attempt-*")):
        runtime = read_json(invocation / "runtime.json")
        runtime_seconds += runtime["seconds"]
        if not (invocation / "initial_model.json").exists():
            if runtime["completed_optimizer_updates"]:
                raise ValueError("trained attempt lost its initialization receipt")
            continue
        initial = read_json(invocation / "initial_model.json")
        start = initial["start_step"] * 8
        active = [(n, p) for n, p in active if n < start]
        active.append((start, invocation))
    if not active or active[0][0] != 0:
        raise ValueError("original initialization/sample prefix is missing")
    episodes = {
        r["episode_id"]: r
        for r in split["episodes"]
        if r["episode_id"] in split["train_episode_ids"]
    }
    hasher = hashlib.sha256()
    counts: Counter[str] = Counter()
    segments = []
    expected_ordinal = 0
    target = output / "samples-verified.csv"
    with target.open("x", encoding="utf-8", newline="") as destination:
        writer = None
        for i, (start, invocation) in enumerate(active):
            end = active[i + 1][0] if i + 1 < len(active) else steps * 8
            path = invocation / "samples.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if writer is None:
                    writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    ordinal = int(row["ordinal"])
                    if ordinal >= end:
                        break
                    if ordinal != expected_ordinal or ordinal < start:
                        raise ValueError("missing/duplicate resumed sample ordinal")
                    episode, frame = int(row["episode_id"]), int(row["frame_index"])
                    ep = episodes[episode]
                    chosen = sample_expression(
                        manifest, level, ep["semantic_goal"], ordinal, episode, frame
                    )
                    if not 0 <= frame < ep["frames"] or any(row[k] != v for k, v in chosen.items()):
                        raise ValueError("logged sample is outside frozen trajectory/language view")
                    writer.writerow(row)
                    hasher.update(f"{episode}:{frame}\n".encode())
                    counts[f"{row['semantic_goal_id']}/{row['template_id']}"] += 1
                    expected_ordinal += 1
            if expected_ordinal != end:
                raise ValueError("incomplete checkpoint sample prefix")
            segments.append(
                {
                    "path": str(path.relative_to(output)),
                    "start": start,
                    "end": end,
                    "sha256": file_digest(path),
                }
            )
    return {
        "level": level,
        "start_step": 0,
        "sample_presentations": expected_ordinal,
        "optimizer_updates": steps,
        "source_segments": segments,
        "samples_csv_sha256": file_digest(target),
        "robot_sample_sequence_sha256": hasher.hexdigest(),
        "language_counts": dict(sorted(counts.items())),
        "initial_model_state_sha256": read_json(active[0][1] / "initial_model.json")[
            "model_state_sha256"
        ],
        "language_manifest_sha256": manifest["language_manifest_sha256"],
        "language_view_sha256": manifest["views"][level]["view_sha256"],
        "total_training_attempt_seconds": runtime_seconds,
    }


def train_language(
    *,
    root: Path,
    output: Path,
    split: dict[str, Any],
    assets: dict[str, Any],
    manifest: dict[str, Any],
    level: str,
    smoke: bool = False,
    resume: Path | None = None,
) -> dict[str, Any]:
    from lerobot.scripts import lerobot_train as upstream

    steps = 20 if smoke else 20000
    start_step = 0
    if resume is not None:
        start_step = read_json(resume.parent / "training_state/training_step.json")["step"]
        if not 0 < start_step < steps:
            raise ValueError("resume checkpoint must precede fixed final step")
    invocations = output / "language_invocations"
    attempt = len(list(invocations.glob("attempt-*"))) if invocations.exists() else 0
    invocation = invocations / f"attempt-{attempt:02d}-step-{start_step:06d}"
    invocation.mkdir(parents=True, exist_ok=False)
    samples = LanguageSamples(invocation / "samples.csv", manifest, level, split, start_step * 8)
    factory, update = upstream.make_train_eval_datasets, upstream.update_policy
    factory_calls = updates = 0
    initial_hash = None

    def language_factory(config: Any) -> tuple[Any, Any]:
        nonlocal factory_calls
        if config.num_workers != 0 or config.batch_size != 8:
            raise ValueError("frozen DataLoader/batch settings changed")
        source, evaluation = factory(config)
        factory_calls += 1
        if evaluation is not None or factory_calls != 1:
            raise ValueError("unexpected extra dataset or evaluation loader")
        return LanguageDataset(source, samples), None

    def audited_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal updates, initial_hash
        policy = kwargs.get("policy", args[1] if len(args) > 1 else None)
        batch = kwargs.get("batch", args[2] if len(args) > 2 else None)
        if policy is None or batch is None:
            raise ValueError("upstream update signature changed")
        if initial_hash is None:
            initial_hash = model_state_hash(policy)
            write_json(
                invocation / "initial_model.json",
                {
                    "model_state_sha256": initial_hash,
                    "start_step": start_step,
                    "total_parameters": sum(p.numel() for p in policy.parameters()),
                    "trainable_parameters": sum(
                        p.numel() for p in policy.parameters() if p.requires_grad
                    ),
                },
            )
        samples.consume(int(batch["action"].shape[0]))
        result = update(*args, **kwargs)
        updates += 1
        return result

    started = time.perf_counter()
    completed = False
    try:
        with (
            patch.object(upstream, "make_train_eval_datasets", language_factory),
            patch.object(upstream, "update_policy", audited_update),
        ):
            result = train_official(
                root=root,
                output=output,
                split=split,
                assets=assets,
                mode="smoke" if smoke else "full",
                device="cuda",
                batch_size=8,
                steps=steps,
                seed=0,
                learning_rate=1e-4,
                wandb=False,
                resume=resume,
            )
        receipt = samples.receipt()
        if updates != steps - start_step or samples.consumed != (steps - start_step) * 8:
            raise RuntimeError("optimizer/sample budget differs")
        completed = True
    finally:
        samples.close()
        write_json(
            invocation / "runtime.json",
            {
                "seconds": time.perf_counter() - started,
                "completed": completed,
                "completed_optimizer_updates": updates,
                "start_step": start_step,
            },
        )
    receipt.update(
        {
            "samples_csv_sha256": file_digest(invocation / "samples.csv"),
            "initial_model_state_sha256": initial_hash,
            "language_manifest_sha256": manifest["language_manifest_sha256"],
            "language_view_sha256": manifest["views"][level]["view_sha256"],
            "level": level,
            "optimizer_updates": updates,
            "start_step": start_step,
        }
    )
    write_json(invocation / "receipt.json", receipt)
    combined = consolidate_samples(output, manifest, level, split, steps)
    result.update(
        language=combined,
        language_invocation=str(invocation.relative_to(output)),
        language_samples_file="samples-verified.csv",
    )
    write_json(output / "m4c_training_metadata.json", result)
    return result
