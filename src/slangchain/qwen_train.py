from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

import accelerate
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Qwen3_5ForCausalLM,
    get_cosine_schedule_with_warmup,
)

from .qwen_config import QwenExperimentConfig, load_qwen_config
from .qwen_data import PackedTokenDataset, load_packed_datasets, sha256_file

TRAINER_FORMAT_VERSION = 2
CHECKPOINT_MANIFEST_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPS_RNG_STATE_FILENAME = "mps_rng_state.pth"


def _jsonable_config(config: QwenExperimentConfig) -> dict[str, Any]:
    value = asdict(config)
    value["data"]["source"] = str(config.data.source)
    value["data"]["packed_dir"] = str(config.data.packed_dir)
    value["training"]["output_dir"] = str(config.training.output_dir)
    return value


def _loader(
    dataset: PackedTokenDataset,
    *,
    config: QwenExperimentConfig,
    shuffle: bool,
    epoch: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.training.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _weights_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _memory_summary(device: torch.device) -> str:
    if device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / 1_000_000_000
        driver = torch.mps.driver_allocated_memory() / 1_000_000_000
        return f" memory_gb={allocated:.2f} driver_gb={driver:.2f}"
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1_000_000_000
        return f" max_memory_gb={allocated:.2f}"
    return ""


def _checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "checkpoint_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != CHECKPOINT_MANIFEST_VERSION
    ):
        raise ValueError(f"Invalid checkpoint manifest: {manifest_path}")
    return manifest


def _validate_checkpoint(path: Path, *, verify_hashes: bool) -> dict[str, Any]:
    manifest = _checkpoint_manifest(path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Checkpoint manifest has no file inventory: {path}")
    required = {
        "config.json",
        "model.safetensors",
        "optimizer.bin",
        "publication.json",
        "random_states_0.pkl",
        MPS_RNG_STATE_FILENAME,
        "scheduler.bin",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "trainer_state.json",
    }
    if not required.issubset(files):
        missing = ", ".join(sorted(required - set(files)))
        raise ValueError(f"Checkpoint is incomplete ({missing}): {path}")
    for relative, details in files.items():
        if not isinstance(relative, str) or not isinstance(details, dict):
            raise ValueError(f"Checkpoint manifest has an invalid file record: {path}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Checkpoint manifest contains an unsafe path: {relative}")
        file_path = path / relative_path
        if not file_path.is_file() or file_path.stat().st_size != details.get("size"):
            raise ValueError(f"Checkpoint file is missing or has the wrong size: {file_path}")
        if verify_hashes and sha256_file(file_path) != details.get("sha256"):
            raise ValueError(f"Checkpoint file hash mismatch: {file_path}")
    state = _state(path)
    if manifest.get("global_step") != state.get("global_step"):
        raise ValueError(f"Checkpoint manifest and trainer state disagree: {path}")
    return state


def _write_checkpoint_manifest(path: Path, global_step: int) -> None:
    inventory = {}
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = str(file_path.relative_to(path))
        inventory[relative] = {
            "size": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
        }
    manifest = {
        "format_version": CHECKPOINT_MANIFEST_VERSION,
        "global_step": global_step,
        "files": inventory,
    }
    with (path / "checkpoint_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_publication_marker(path: Path, state: dict[str, Any]) -> None:
    marker = {
        "format_version": CHECKPOINT_MANIFEST_VERSION,
        "global_step": state["global_step"],
        "config_sha256": state["config_sha256"],
        "packed_manifest_sha256": state["packed_manifest_sha256"],
        "implementation": state["implementation"],
    }
    with (path / "publication.json").open("x", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_mps_rng_state(path: Path, device: torch.device) -> None:
    """Persist the device RNG omitted by Accelerate's random state bundle."""
    # Keep checkpoint shape identical on non-MPS hosts; that fallback is not
    # used for MPS restoration.
    rng_state = (
        torch.mps.get_rng_state(device)
        if device.type == "mps"
        else torch.get_rng_state()
    )
    with (path / MPS_RNG_STATE_FILENAME).open("xb") as handle:
        torch.save(rng_state.cpu(), handle)


def _restore_mps_rng_state(
    path: Path, device: torch.device, *, saved_device: str | None
) -> None:
    if device.type != "mps" or saved_device != "mps":
        return
    with (path / MPS_RNG_STATE_FILENAME).open("rb") as handle:
        rng_state = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(rng_state, torch.Tensor):
        raise ValueError(f"Invalid MPS RNG state: {path / MPS_RNG_STATE_FILENAME}")
    torch.mps.set_rng_state(rng_state, device)


def _matching_partial_checkpoint(
    output_dir: Path, step: int, state: dict[str, Any]
) -> Path | None:
    for candidate in sorted(output_dir.glob(f".checkpoint-{step:06d}.*")):
        try:
            candidate_state = _validate_checkpoint(candidate, verify_hashes=True)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if candidate_state == state:
            return candidate
    return None


def _publish_checkpoint(temporary: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing to replace checkpoint target: {destination}")
    os.symlink(temporary.name, destination, target_is_directory=True)
    _validate_checkpoint(destination, verify_hashes=True)


def _write_checkpoint_stats(
    output_dir: Path, checkpoint: Path, state: dict[str, Any]
) -> dict[str, Any]:
    target = _checkpoint_target(checkpoint, output_dir)
    if target is None:
        raise ValueError(f"Checkpoint is not a canonical symlink: {checkpoint}")
    manifest_sha256 = sha256_file(target / "checkpoint_manifest.json")
    stats_dir = output_dir / "checkpoint-stats"
    if os.path.lexists(stats_dir):
        if stats_dir.is_symlink() or not stats_dir.is_dir():
            raise ValueError(f"Checkpoint stats path is not a private directory: {stats_dir}")
    else:
        stats_dir.mkdir()
    stats_path = stats_dir / f"{checkpoint.name}.json"
    record = {
        "checkpoint": checkpoint.name,
        "target": target.name,
        "manifest_sha256": manifest_sha256,
        "global_step": state["global_step"],
        "state_identity": _checkpoint_identity(state),
        "state": state,
    }
    if os.path.lexists(stats_path):
        if stats_path.is_symlink() or not stats_path.is_file():
            raise ValueError(f"Checkpoint stats path is not a regular file: {stats_path}")
        with stats_path.open(encoding="utf-8") as handle:
            if json.load(handle) != record:
                raise ValueError(f"Checkpoint stats disagree: {stats_path}")
        return record
    _publish_json(stats_path, record)
    return record


def _checkpoint_target(path: Path, output_dir: Path) -> Path | None:
    """Return a validated private target, or None for an unsafe boundary."""
    if not path.is_symlink() or path.parent != output_dir:
        return None
    try:
        step = int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return None
    link_target = Path(os.readlink(path))
    if link_target.is_absolute() or link_target.parent != Path("."):
        return None
    expected_prefix = f".checkpoint-{step:06d}."
    if not link_target.name.startswith(expected_prefix):
        return None
    target = (output_dir / link_target).resolve(strict=True)
    if target.parent != output_dir.resolve() or target.name != link_target.name:
        return None
    if not target.is_dir():
        return None
    return target


def _prune_record_path(output_dir: Path, name: str) -> Path:
    return output_dir / ".checkpoint-prunes" / f"{name}.json"


def _prune_directory(output_dir: Path, *, create: bool) -> Path | None:
    directory = output_dir / ".checkpoint-prunes"
    if os.path.lexists(directory):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Checkpoint prune journal is not a private directory: {directory}")
        return directory
    if not create:
        return None
    directory.mkdir()
    return directory


def _checkpoint_identity(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "config_sha256", "packed_manifest_sha256", "implementation",
        "mixed_precision", "model_id", "model_revision", "tokenizer_id",
        "tokenizer_revision",
    )
    return {key: state.get(key) for key in keys}


def _reconcile_prunes(output_dir: Path) -> None:
    directory = _prune_directory(output_dir, create=False)
    if directory is None:
        return
    for record_path in sorted(directory.glob("checkpoint-*.json")):
        if record_path.is_symlink() or not record_path.is_file():
            continue
        try:
            with record_path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            name = record["checkpoint"]
            target_name = record["target"]
            if (
                record.get("format_version") != CHECKPOINT_MANIFEST_VERSION
                or not isinstance(name, str)
                or not isinstance(target_name, str)
                or not isinstance(record.get("stats_sha256"), str)
                or not isinstance(record.get("manifest_sha256"), str)
            ):
                continue
            if record_path.stem != name or not isinstance(record.get("state_identity"), dict):
                continue
            checkpoint = output_dir / name
            target = output_dir / target_name
            if checkpoint.name != name or target.name != target_name:
                continue
            try:
                step = int(name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            if target.parent != output_dir or not target_name.startswith(
                f".checkpoint-{step:06d}."
            ):
                continue
            target = target.resolve(strict=False)
            if target.parent != output_dir.resolve() or target.name != target_name:
                continue
            stats_path = output_dir / "checkpoint-stats" / f"{name}.json"
            if (
                stats_path.is_symlink()
                or not stats_path.is_file()
                or sha256_file(stats_path) != record["stats_sha256"]
            ):
                continue
            with stats_path.open(encoding="utf-8") as handle:
                stats = json.load(handle)
            if (
                not isinstance(stats, dict)
                or stats.get("checkpoint") != name
                or stats.get("global_step") != step
                or stats.get("state_identity") != record["state_identity"]
                or not isinstance(stats.get("state"), dict)
                or stats["state"].get("global_step") != step
                or _checkpoint_identity(stats["state"]) != record["state_identity"]
                or stats.get("target") != target_name
                or stats.get("manifest_sha256") != record["manifest_sha256"]
            ):
                continue
            if os.path.lexists(checkpoint):
                current = _validate_checkpoint(checkpoint, verify_hashes=True)
                if (
                    _checkpoint_target(checkpoint, output_dir) != target
                    or _checkpoint_identity(current) != record["state_identity"]
                    or sha256_file(checkpoint / "checkpoint_manifest.json")
                    != record["manifest_sha256"]
                ):
                    continue
            elif target.exists():
                manifest_path = target / "checkpoint_manifest.json"
                if (
                    manifest_path.exists()
                    and sha256_file(manifest_path) != record["manifest_sha256"]
                ):
                    continue
        except (FileNotFoundError, KeyError, TypeError, ValueError, OSError):
            continue
        # The journal record has passed all structural and boundary checks;
        # cleanup failures must stop publication so storage cannot accumulate.
        if os.path.lexists(checkpoint):
            checkpoint.unlink()
        if target.exists() or target.is_symlink():
            shutil.rmtree(target)
        record_path.unlink()
    with suppress(OSError):
        directory.rmdir()


def _retain_checkpoints(
    output_dir: Path,
    keep_checkpoints: int,
    *,
    expected_state: dict[str, Any] | None = None,
    protected: Path | None = None,
) -> None:
    _reconcile_prunes(output_dir)
    complete = _checkpoint_directories(output_dir)
    if expected_state is not None:
        identity = _checkpoint_identity(expected_state)
        complete = [
            path for path in complete
            if _checkpoint_identity(_state(path)) == identity
        ]
    for checkpoint in complete[:-keep_checkpoints]:
        if protected is not None and checkpoint.name == protected.name:
            continue
        target = _checkpoint_target(checkpoint, output_dir)
        if target is None:
            continue
        state = _validate_checkpoint(checkpoint, verify_hashes=True)
        stats = _write_checkpoint_stats(output_dir, checkpoint, state)
        record_path = _prune_record_path(output_dir, checkpoint.name)
        stats_path = output_dir / "checkpoint-stats" / f"{checkpoint.name}.json"
        _prune_directory(output_dir, create=True)
        record = {
            "format_version": CHECKPOINT_MANIFEST_VERSION,
            "checkpoint": checkpoint.name,
            "target": stats["target"],
            "stats_sha256": sha256_file(stats_path),
            "manifest_sha256": stats["manifest_sha256"],
            "state_identity": stats["state_identity"],
        }
        if os.path.lexists(record_path):
            if record_path.is_symlink() or not record_path.is_file():
                raise ValueError(f"Checkpoint prune record is not a regular file: {record_path}")
            with record_path.open(encoding="utf-8") as handle:
                if json.load(handle) != record:
                    raise ValueError(f"Checkpoint prune record disagrees: {record_path}")
        else:
            _publish_json(record_path, record)
        _reconcile_prunes(output_dir)


def _finalize_checkpoint(
    output_dir: Path,
    checkpoint: Path,
    state: dict[str, Any],
    keep_checkpoints: int,
    *,
    protected: Path | None = None,
) -> None:
    _write_checkpoint_stats(output_dir, checkpoint, state)
    _retain_checkpoints(
        output_dir,
        keep_checkpoints,
        expected_state=state,
        protected=protected,
    )


def _checkpoint_directories(output_dir: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        try:
            state = _validate_checkpoint(path, verify_hashes=False)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if state.get("global_step") != step:
            continue
        checkpoints.append((step, path))
    return [path for _, path in sorted(checkpoints)]


def latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = _checkpoint_directories(output_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No complete Qwen checkpoints found in {output_dir}")
    checkpoint = checkpoints[-1]
    _validate_checkpoint(checkpoint, verify_hashes=True)
    return checkpoint


def _state(path: Path) -> dict[str, Any]:
    with (path / "trainer_state.json").open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid trainer state: {path}")
    return value


def qwen_config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_identity() -> dict[str, Any]:
    source_digest = hashlib.sha256()
    source_paths = sorted((PROJECT_ROOT / "src" / "slangchain").glob("*.py"))
    for path in source_paths:
        source_digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        source_digest.update(b"\0")
        source_digest.update(path.read_bytes())
        source_digest.update(b"\0")
    lock_path = PROJECT_ROOT / "uv.lock"
    return {
        "format_version": 1,
        "source_sha256": source_digest.hexdigest(),
        "uv_lock_sha256": sha256_file(lock_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
    }


def _resolved_mixed_precision(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "bf16"
    if torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "no"


def _publish_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _resolve_resume(output_dir: Path, resume: str | Path | None) -> Path | None:
    _reconcile_prunes(output_dir)
    if resume is None or resume == "none":
        return None
    if resume == "auto" or resume == "latest":
        checkpoints = _checkpoint_directories(output_dir)
        return checkpoints[-1] if checkpoints else None
    path = Path(resume)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _validate_state(
    state: dict[str, Any],
    *,
    config_sha256: str,
    manifest_sha256: str,
    implementation: dict[str, Any],
    mixed_precision: str,
) -> None:
    if state.get("format_version") != TRAINER_FORMAT_VERSION:
        raise ValueError("Checkpoint trainer-state format is not supported")
    if state.get("config_sha256") != config_sha256:
        raise ValueError("Checkpoint was created with a different Qwen experiment config")
    if state.get("packed_manifest_sha256") != manifest_sha256:
        raise ValueError("Checkpoint was created from a different packed corpus")
    if state.get("implementation") != implementation:
        raise ValueError("Checkpoint was created by different source or dependency versions")
    if state.get("mixed_precision") != mixed_precision:
        raise ValueError("Checkpoint was created with a different resolved mixed precision")


@contextmanager
def _training_lock(output_dir: Path) -> Iterator[None]:
    lock = output_dir / ".train.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"Another Qwen training process, or a stale lock, exists at {lock}"
        ) from error
    try:
        yield
    finally:
        with suppress(FileNotFoundError):
            lock.rmdir()


def _save_checkpoint(
    accelerator: Accelerator,
    model: Qwen3_5ForCausalLM,
    tokenizer: Any,
    output_dir: Path,
    state: dict[str, Any],
    *,
    keep_checkpoints: int = 3,
) -> Path:
    step = state["global_step"]
    destination = output_dir / f"checkpoint-{step:06d}"
    if destination.exists():
        try:
            existing = _validate_checkpoint(destination, verify_hashes=True)
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            existing = None
        if existing is not None:
            if existing == state:
                _finalize_checkpoint(output_dir, destination, state, keep_checkpoints)
                return destination
            raise FileExistsError(f"Refusing to replace checkpoint: {destination}")

    accelerator.wait_for_everyone()
    temporary = _matching_partial_checkpoint(output_dir, step, state)
    try:
        if temporary is None:
            temporary = Path(
                tempfile.mkdtemp(prefix=f".checkpoint-{step:06d}.", dir=output_dir)
            )
            accelerator.save_state(temporary, safe_serialization=True)
            if accelerator.is_main_process:
                _save_mps_rng_state(temporary, accelerator.device)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.config.save_pretrained(temporary)
                tokenizer.save_pretrained(temporary / "tokenizer")
                with (temporary / "trainer_state.json").open("x", encoding="utf-8") as handle:
                    json.dump(state, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                _write_publication_marker(temporary, state)
                _write_checkpoint_manifest(temporary, step)
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _publish_checkpoint(temporary, destination)
            _finalize_checkpoint(output_dir, destination, state, keep_checkpoints)
        accelerator.wait_for_everyone()
        return destination
    except BaseException:
        # A uniquely named partial checkpoint is intentionally left for inspection.
        raise


@torch.inference_mode()
def _evaluate(
    accelerator: Accelerator,
    model: Qwen3_5ForCausalLM,
    loader: DataLoader,
    max_batches: int | None,
) -> tuple[float, int]:
    model.eval()
    totals = torch.zeros(2, device=accelerator.device, dtype=torch.float32)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        output = model(**batch, use_cache=False)
        if output.loss is None:
            raise RuntimeError("Qwen model did not return evaluation loss")
        targets = (batch["labels"][:, 1:] != -100).sum()
        totals[0] += output.loss.detach().float() * targets
        totals[1] += targets
    totals = accelerator.reduce(totals, reduction="sum")
    model.train()
    count = int(totals[1].item())
    if count == 0:
        raise ValueError("Evaluation split contained no target tokens")
    return (totals[0] / totals[1]).item(), count


def _complete_report(
    output_dir: Path,
    *,
    config_sha256: str,
    manifest_sha256: str,
    implementation: dict[str, Any],
    mixed_precision: str,
) -> Path | None:
    path = output_dir / "report.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"Invalid report: {path}")
    if (
        report.get("config_sha256") != config_sha256
        or report.get("packed_manifest_sha256") != manifest_sha256
        or report.get("implementation") != implementation
        or report.get("mixed_precision") != mixed_precision
    ):
        raise ValueError(f"Existing report has different provenance: {path}")
    checkpoint_name = report.get("checkpoint")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError(f"Report does not name one exact checkpoint: {path}")
    checkpoint = output_dir / checkpoint_name
    state = _validate_checkpoint(checkpoint, verify_hashes=True)
    if report.get("global_steps") != state.get("global_step"):
        raise ValueError(f"Report and checkpoint step disagree: {path}")
    _validate_state(
        state,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        implementation=implementation,
        mixed_precision=mixed_precision,
    )
    return checkpoint


def validate_generation_checkpoint(
    checkpoint: Path, config_path: Path
) -> dict[str, Any]:
    config = load_qwen_config(config_path)
    state = _validate_checkpoint(checkpoint, verify_hashes=True)
    expected = {
        "config_sha256": qwen_config_sha256(config_path),
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "tokenizer_id": config.data.tokenizer_id,
        "tokenizer_revision": config.data.tokenizer_revision,
        "implementation": _implementation_identity(),
    }
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        raise ValueError(
            "Checkpoint does not match the selected config or implementation: "
            + ", ".join(mismatches)
        )
    return state


def checkpoint_for_generation(config_path: Path) -> Path:
    config = load_qwen_config(config_path)
    output_dir = config.training.output_dir
    report_path = output_dir / "report.json"
    if report_path.is_file():
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if not isinstance(report, dict):
            raise ValueError(f"Invalid report: {report_path}")
        checkpoint_name = report.get("checkpoint")
        if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError(f"Report does not name one exact checkpoint: {report_path}")
        checkpoint = output_dir / checkpoint_name
        state = _validate_checkpoint(checkpoint, verify_hashes=False)
        if report.get("global_steps") != state.get("global_step"):
            raise ValueError(f"Report and checkpoint step disagree: {report_path}")
    else:
        checkpoints = _checkpoint_directories(output_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No complete Qwen checkpoints found in {output_dir}")
        checkpoint = checkpoints[-1]
    return checkpoint


def run_qwen_training(
    config_path: Path,
    *,
    resume: str | Path | None = "auto",
) -> Path:
    config = load_qwen_config(config_path)
    datasets, manifest = load_packed_datasets(config)
    manifest_path = config.data.packed_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    config_sha256 = qwen_config_sha256(config_path)
    implementation = _implementation_identity()
    mixed_precision = _resolved_mixed_precision(config.training.mixed_precision)
    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    complete = _complete_report(
        output_dir,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        implementation=implementation,
        mixed_precision=mixed_precision,
    )
    if complete is not None:
        with _training_lock(output_dir):
            _reconcile_prunes(output_dir)
            print(f"Using completed Qwen run at {output_dir}")
            return complete

    with _training_lock(output_dir):
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        )
        if accelerator.num_processes != 1:
            raise RuntimeError("The Qwen teaching track currently supports one process")
        set_seed(config.training.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            config.data.tokenizer_id,
            revision=config.data.tokenizer_revision,
        )
        model = Qwen3_5ForCausalLM.from_pretrained(
            config.model.model_id,
            revision=config.model.revision,
            dtype=_weights_dtype(config.model.weights_dtype),
            attn_implementation="sdpa",
        )
        if len(tokenizer) > model.config.vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary ({len(tokenizer)}) exceeds the model's padded "
                f"vocabulary ({model.config.vocab_size})"
            )
        model.config.use_cache = False
        if config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        batches_per_epoch = math.ceil(
            len(datasets["train"]) / config.training.micro_batch_size
        )
        updates_per_epoch = math.ceil(
            batches_per_epoch / config.training.gradient_accumulation_steps
        )
        total_steps = config.training.max_steps or config.training.epochs * updates_per_epoch
        warmup_steps = round(total_steps * config.training.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

        resume_path = _resolve_resume(output_dir, resume)
        start_epoch = 0
        skip_batches = 0
        global_step = 0
        trained_tokens = 0
        if resume_path is not None:
            state = _validate_checkpoint(resume_path, verify_hashes=True)
            _validate_state(
                state,
                config_sha256=config_sha256,
                manifest_sha256=manifest_sha256,
                implementation=implementation,
                mixed_precision=mixed_precision,
            )
            _finalize_checkpoint(
                output_dir,
                resume_path,
                state,
                config.training.keep_checkpoints,
                protected=resume_path,
            )
            accelerator.load_state(resume_path)
            _restore_mps_rng_state(
                resume_path,
                accelerator.device,
                saved_device=state.get("device"),
            )
            start_epoch = state["next_epoch"]
            skip_batches = state["next_batch"]
            global_step = state["global_step"]
            trained_tokens = state["trained_tokens"]
            accelerator.print(f"resumed={resume_path} step={global_step}")

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        effective_tokens = (
            config.model.context_length
            * config.training.micro_batch_size
            * config.training.gradient_accumulation_steps
        )
        accelerator.print(
            f"device={accelerator.device} parameters={parameter_count:,} "
            f"blocks={len(datasets['train']):,} context={config.model.context_length} "
            f"effective_tokens={effective_tokens:,}"
            f"{_memory_summary(accelerator.device)}"
        )
        started = time.perf_counter()
        invocation_start_tokens = trained_tokens
        interval_started = started
        interval_tokens = 0
        last_loss = math.nan
        stop = global_step >= total_steps
        final_state: dict[str, Any] | None = None
        last_checkpoint = resume_path

        for epoch in range(start_epoch, config.training.epochs):
            if stop:
                break
            raw_loader = _loader(datasets["train"], config=config, shuffle=True, epoch=epoch)
            train_loader = accelerator.prepare(raw_loader)
            batches_this_epoch = len(train_loader)
            if epoch == start_epoch and skip_batches:
                if skip_batches >= batches_this_epoch:
                    raise ValueError("Checkpoint batch cursor is outside the training epoch")
                train_loader = accelerator.skip_first_batches(train_loader, skip_batches)
            consumed_before = skip_batches if epoch == start_epoch else 0

            model.train()
            for local_batch, batch in enumerate(train_loader):
                batch_in_epoch = consumed_before + local_batch + 1
                with accelerator.accumulate(model):
                    output = model(**batch, use_cache=False)
                    if output.loss is None:
                        raise RuntimeError("Qwen model did not return training loss")
                    loss = output.loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), config.training.gradient_clip
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                batch_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                trained_tokens += batch_tokens
                interval_tokens += batch_tokens
                if not accelerator.sync_gradients:
                    continue

                global_step += 1
                last_loss = loss.detach().float().item()
                next_epoch = epoch
                next_batch = batch_in_epoch
                if next_batch >= batches_this_epoch:
                    next_epoch += 1
                    next_batch = 0
                final_state = {
                    "format_version": TRAINER_FORMAT_VERSION,
                    "global_step": global_step,
                    "next_epoch": next_epoch,
                    "next_batch": next_batch,
                    "trained_tokens": trained_tokens,
                    "last_loss": last_loss,
                    "config_sha256": config_sha256,
                    "packed_manifest_sha256": manifest_sha256,
                    "model_id": config.model.model_id,
                    "model_revision": config.model.revision,
                    "tokenizer_id": config.data.tokenizer_id,
                    "tokenizer_revision": config.data.tokenizer_revision,
                    "implementation": implementation,
                    "mixed_precision": mixed_precision,
                    "device": accelerator.device.type,
                }

                if global_step % config.training.log_interval == 0 or global_step == 1:
                    now = time.perf_counter()
                    tokens_per_second = interval_tokens / max(now - interval_started, 1e-9)
                    accelerator.print(
                        f"step={global_step}/{total_steps} loss={last_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.3g} "
                        f"tokens_per_second={tokens_per_second:,.0f}"
                        f"{_memory_summary(accelerator.device)}"
                    )
                    interval_started = now
                    interval_tokens = 0

                if global_step % config.training.checkpoint_interval == 0:
                    last_checkpoint = _save_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        output_dir,
                        final_state,
                        keep_checkpoints=config.training.keep_checkpoints,
                    )
                if global_step >= total_steps:
                    stop = True
                    break

            skip_batches = 0

        if final_state is None:
            if resume_path is None:
                raise RuntimeError("Training produced no optimizer steps")
            final_state = _validate_checkpoint(resume_path, verify_hashes=True)
        if global_step < total_steps:
            raise RuntimeError(
                f"Training data ended at step {global_step}, before requested step {total_steps}"
            )
        if (
            last_checkpoint is not None
            and last_checkpoint.name == f"checkpoint-{global_step:06d}"
        ):
            final_checkpoint = last_checkpoint
        else:
            final_checkpoint = _save_checkpoint(
                accelerator,
                model,
                tokenizer,
                output_dir,
                final_state,
                keep_checkpoints=config.training.keep_checkpoints,
            )

        validation_loader = accelerator.prepare(
            _loader(datasets["validation"], config=config, shuffle=False)
        )
        test_loader = accelerator.prepare(
            _loader(datasets["test"], config=config, shuffle=False)
        )
        validation_loss, validation_tokens = _evaluate(
            accelerator,
            model,
            validation_loader,
            config.training.max_eval_batches,
        )
        test_loss, test_tokens = _evaluate(
            accelerator,
            model,
            test_loader,
            config.training.max_eval_batches,
        )
        elapsed = time.perf_counter() - started
        report = {
            "format_version": 1,
            "mode": config.mode,
            "device": str(accelerator.device),
            "parameters": parameter_count,
            "global_steps": global_step,
            "trained_tokens": trained_tokens,
            "elapsed_seconds_this_invocation": elapsed,
            "tokens_per_second_this_invocation": (
                trained_tokens - invocation_start_tokens
            )
            / max(elapsed, 1e-9),
            "validation_loss": validation_loss,
            "validation_perplexity": math.exp(validation_loss),
            "validation_tokens": validation_tokens,
            "test_loss": test_loss,
            "test_perplexity": math.exp(test_loss),
            "test_tokens": test_tokens,
            "max_eval_batches": config.training.max_eval_batches,
            "checkpoint": final_checkpoint.name,
            "config_sha256": config_sha256,
            "packed_manifest_sha256": manifest_sha256,
            "implementation": implementation,
            "mixed_precision": mixed_precision,
            "config": _jsonable_config(config),
            "packed_splits": manifest["splits"],
        }
        if accelerator.is_main_process:
            _publish_json(output_dir / "report.json", report)
        accelerator.print(
            f"validation_loss={validation_loss:.4f} test_loss={test_loss:.4f} "
            f"checkpoint={final_checkpoint}"
        )
        return final_checkpoint


@torch.inference_mode()
def generate_qwen(
    checkpoint: Path,
    *,
    config_path: Path,
    year: int,
    samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    if samples <= 0 or max_new_tokens <= 0:
        raise ValueError("samples and max_new_tokens must be positive")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    state = validate_generation_checkpoint(checkpoint, config_path)
    tokenizer_path = checkpoint / "tokenizer"
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Checkpoint tokenizer is missing: {tokenizer_path}")

    accelerator = Accelerator()
    set_seed(seed)
    if accelerator.device.type == "mps":
        inference_dtype = torch.bfloat16
    elif accelerator.device.type == "cuda":
        inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        inference_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint,
        dtype=inference_dtype,
        attn_implementation="sdpa",
    ).to(accelerator.device)
    model.config.use_cache = True
    model.eval()
    prompt = f"<ENTRY>\nYEAR: {year}\nWORD:"
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
        accelerator.device
    )
    output = model.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=samples,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_tokens = encoded["input_ids"].shape[1]
    results = tokenizer.batch_decode(output[:, prompt_tokens:], skip_special_tokens=True)
    return [text.rstrip() for text in results], state
