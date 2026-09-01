from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from safetensors.torch import load_file


def create_accelerator(mixed_precision: str = "no") -> Accelerator:
    """Let Accelerate choose MPS, CUDA, or CPU without platform branches here."""
    accelerator = Accelerator(mixed_precision=mixed_precision)
    try:
        launched_processes = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise ValueError("WORLD_SIZE must be an integer") from error
    if accelerator.num_processes != 1 or launched_processes != 1:
        raise RuntimeError("Slangchain is intentionally a single-process teaching demo")
    return accelerator


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    directory: Path,
    metadata: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    accelerator.save_model(model, directory, safe_serialization=True)
    if accelerator.is_main_process:
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")


def load_checkpoint(
    model: torch.nn.Module, directory: Path, device: torch.device
) -> dict[str, Any]:
    metadata_path = directory / "metadata.json"
    weights_path = directory / "model.safetensors"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint is incomplete: {directory}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    model.load_state_dict(load_file(weights_path, device="cpu"))
    model.to(device)
    return metadata


def read_checkpoint_metadata(directory: Path) -> dict[str, Any]:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata is missing: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata is invalid: {metadata_path}")
    return value
