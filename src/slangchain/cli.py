from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

from accelerate.utils import set_seed

from .config import ModelConfig
from .data import prepare_dataset
from .model import TemporalSlangModel, sample
from .runtime import create_accelerator, load_checkpoint, read_checkpoint_metadata
from .train import run_training


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slangchain",
        description="Train scratch and pretrained time-conditioned slang language models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Normalize a dated JSONL, CSV, or SQLite corpus"
    )
    prepare.add_argument("input", type=_path)
    prepare.add_argument("--output", type=_path, required=True)
    prepare.add_argument("--max-per-year", type=int)
    prepare.add_argument("--max-chars", type=int, default=1_000)
    prepare.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="Train and evaluate from a TOML experiment")
    train.add_argument("--config", type=_path, default=Path("configs/tiny.toml"))
    train.add_argument("--data", type=_path)
    train.add_argument("--output-dir", type=_path)

    generate = subparsers.add_parser("generate", help="Sample entries from a checkpoint")
    generate.add_argument("--checkpoint", type=_path, required=True)
    generate.add_argument("--year", type=int, required=True)
    generate.add_argument("--samples", type=int, default=5)
    generate.add_argument("--prompt", default="WORD: ")
    generate.add_argument("--max-new-tokens", type=int, default=220)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--seed", type=int, default=42)

    qwen_pack = subparsers.add_parser(
        "qwen-pack", help="Tokenize and pack the chronological corpus for Qwen3.5"
    )
    qwen_pack.add_argument("--config", type=_path, default=Path("configs/qwen35-0.8b.toml"))

    qwen_train = subparsers.add_parser(
        "qwen-train", help="Continue pretraining Qwen3.5 on the packed corpus"
    )
    qwen_train.add_argument("--config", type=_path, default=Path("configs/qwen35-0.8b.toml"))
    qwen_train.add_argument(
        "--resume",
        default="auto",
        help="auto, none, latest, or a checkpoint directory (default: auto)",
    )

    qwen_generate = subparsers.add_parser(
        "qwen-generate", help="Generate slang with a continued-pretraining checkpoint"
    )
    qwen_generate.add_argument(
        "--config", type=_path, default=Path("configs/qwen35-0.8b.toml")
    )
    qwen_generate.add_argument("--checkpoint", type=_path)
    qwen_generate.add_argument("--year", type=int, required=True)
    qwen_generate.add_argument("--samples", type=int, default=5)
    qwen_generate.add_argument("--max-new-tokens", type=int, default=220)
    qwen_generate.add_argument("--temperature", type=float, default=0.9)
    qwen_generate.add_argument("--top-p", type=float, default=0.95)
    qwen_generate.add_argument("--seed", type=int, default=42)
    return parser


def _run_generate(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    metadata = read_checkpoint_metadata(args.checkpoint)
    model_data = metadata.get("model")
    if not isinstance(model_data, dict):
        raise ValueError("Checkpoint does not contain a model configuration")
    config = ModelConfig(**model_data)
    accelerator = create_accelerator()
    set_seed(args.seed)
    random.seed(args.seed)
    model = TemporalSlangModel(config)
    load_checkpoint(model, args.checkpoint, accelerator.device)
    accelerator.print(f"device={accelerator.device} year={args.year}")
    for index in range(args.samples):
        generated = sample(
            model,
            prompt=args.prompt,
            year=args.year,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=accelerator.device,
        )
        accelerator.print(f"\n--- sample {index + 1} ---\n{generated.rstrip()}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        stats = prepare_dataset(
            args.input,
            args.output,
            max_per_year=args.max_per_year,
            max_chars=args.max_chars,
            seed=args.seed,
        )
        print(json.dumps({**stats.__dict__, "by_year": stats.by_year}, indent=2))
    elif args.command == "train":
        run_training(args.config, args.output_dir, args.data)
    elif args.command == "generate":
        _run_generate(args)
    elif args.command == "qwen-pack":
        from .qwen_config import load_qwen_config
        from .qwen_data import pack_corpus

        stats = pack_corpus(load_qwen_config(args.config))
        print(
            json.dumps(
                {
                    "seen": stats.seen,
                    "rejected": stats.rejected,
                    "splits": {
                        name: split_stats.__dict__
                        for name, split_stats in stats.splits.items()
                    },
                },
                indent=2,
            )
        )
    elif args.command == "qwen-train":
        from .qwen_train import run_qwen_training

        run_qwen_training(args.config, resume=args.resume)
    elif args.command == "qwen-generate":
        from .qwen_train import checkpoint_for_generation, generate_qwen

        checkpoint = args.checkpoint or checkpoint_for_generation(args.config)
        generated, checkpoint_state = generate_qwen(
            checkpoint,
            config_path=args.config,
            year=args.year,
            samples=args.samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(f"model={checkpoint_state['model_id']} year={args.year}")
        for index, text in enumerate(generated, start=1):
            print(f"\n--- sample {index} ---\n{text}")
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
