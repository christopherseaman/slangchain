from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_config
from .data import EntryDataset, load_entries, split_entries
from .model import PAD_ID, TemporalSlangModel
from .runtime import create_accelerator, load_checkpoint, save_checkpoint


def _loader(
    dataset: EntryDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(
    accelerator: Accelerator,
    model: TemporalSlangModel,
    loader: DataLoader,
    max_batches: int | None,
) -> float:
    model.eval()
    totals = torch.zeros(2, device=accelerator.device, dtype=torch.float32)
    for batch_index, (tokens, targets, years) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        _, loss = model(tokens, years, targets)
        if loss is None:
            raise RuntimeError("Model did not return an evaluation loss")
        target_count = (targets != PAD_ID).sum()
        totals[0] += loss.detach().float() * target_count
        totals[1] += target_count
    totals = accelerator.reduce(totals, reduction="sum")
    model.train()
    if totals[1].item() == 0:
        raise ValueError("Evaluation loader contained no target tokens")
    return (totals[0] / totals[1]).item()


def _metadata(
    config: ExperimentConfig,
    model: TemporalSlangModel,
    *,
    step: int,
    validation_loss: float,
    split_sizes: dict[str, int],
    observed_years: tuple[int, int],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "model": model.metadata(),
        "training": {
            "step": step,
            "validation_loss": validation_loss,
            "seed": config.training.seed,
        },
        "data": {
            "train_through": config.data.train_through,
            "validation_through": config.data.validation_through,
            "split_sizes": split_sizes,
            "observed_years": list(observed_years),
        },
    }


def run_training(
    config_path: Path,
    output_dir: Path | None = None,
    data_path: Path | None = None,
) -> Path:
    config = load_config(config_path)
    if data_path is not None:
        config = replace(config, data=replace(config.data, path=data_path))
    if output_dir is not None:
        config = replace(config, training=replace(config.training, output_dir=output_dir))
    destination = config.training.output_dir
    accelerator = create_accelerator(config.training.mixed_precision)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(
            f"Refusing to overwrite training destination: {destination}. Choose a new --output-dir."
        )
    destination.mkdir(parents=True, exist_ok=True)

    set_seed(config.training.seed)
    entries = load_entries(config.data.path)
    train_entries, validation_entries, test_entries = split_entries(
        entries, config.data.train_through, config.data.validation_through
    )
    split_sizes = {
        "train": len(train_entries),
        "validation": len(validation_entries),
        "test": len(test_entries),
    }

    train_loader = _loader(
        EntryDataset(train_entries, config.model.context_length),
        batch_size=config.training.batch_size,
        shuffle=True,
        workers=config.training.num_workers,
        seed=config.training.seed,
    )
    validation_loader = _loader(
        EntryDataset(validation_entries, config.model.context_length),
        batch_size=config.training.batch_size,
        shuffle=False,
        workers=config.training.num_workers,
        seed=config.training.seed,
    )
    test_loader = _loader(
        EntryDataset(test_entries, config.model.context_length),
        batch_size=config.training.batch_size,
        shuffle=False,
        workers=config.training.num_workers,
        seed=config.training.seed,
    )

    model = TemporalSlangModel(config.model)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    model, optimizer, train_loader, validation_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, test_loader
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    accelerator.print(
        f"device={accelerator.device} parameters={parameter_count:,} splits={split_sizes}"
    )

    best_loss = math.inf
    best_directory = destination / "best"
    train_iterator = iter(train_loader)
    for step in range(1, config.training.steps + 1):
        try:
            tokens, targets, years = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            tokens, targets, years = next(train_iterator)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens, years, targets)
        if loss is None:
            raise RuntimeError("Model did not return a training loss")
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()

        if step % config.training.log_interval == 0 or step == 1:
            accelerator.print(f"step={step} train_loss={loss.item():.4f}")

        if step % config.training.eval_interval == 0 or step == config.training.steps:
            validation_loss = evaluate(
                accelerator, model, validation_loader, config.training.eval_batches
            )
            accelerator.print(
                f"step={step} validation_loss={validation_loss:.4f} "
                f"perplexity={math.exp(validation_loss):.2f}"
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(
                    accelerator,
                    unwrapped,
                    best_directory,
                    _metadata(
                        config,
                        unwrapped,
                        step=step,
                        validation_loss=validation_loss,
                        split_sizes=split_sizes,
                        observed_years=(
                            min(entry.year for entry in entries),
                            max(entry.year for entry in entries),
                        ),
                    ),
                )

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    load_checkpoint(unwrapped, best_directory, accelerator.device)
    test_loss = evaluate(accelerator, model, test_loader, max_batches=None)
    report = {
        "device": str(accelerator.device),
        "parameters": parameter_count,
        "best_validation_loss": best_loss,
        "best_validation_perplexity": math.exp(best_loss),
        "test_loss": test_loss,
        "test_perplexity": math.exp(test_loss),
        "splits": split_sizes,
        "config": {
            "data": {**asdict(config.data), "path": str(config.data.path)},
            "model": asdict(config.model),
            "training": {
                **asdict(config.training),
                "output_dir": str(config.training.output_dir),
            },
        },
    }
    if accelerator.is_main_process:
        with (destination / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    accelerator.print(
        f"test_loss={test_loss:.4f} perplexity={math.exp(test_loss):.2f} "
        f"checkpoint={best_directory}"
    )
    return best_directory
