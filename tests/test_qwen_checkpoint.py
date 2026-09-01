from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.data import Dataset

import slangchain.qwen_train as qwen_train
from slangchain.qwen_config import (
    QwenDataConfig,
    QwenExperimentConfig,
    QwenModelConfig,
    QwenTrainingConfig,
)
from slangchain.qwen_train import (
    _complete_report,
    _publish_json,
    _resolved_mixed_precision,
    _retain_checkpoints,
    _save_checkpoint,
    _validate_checkpoint,
    _write_checkpoint_manifest,
    _write_checkpoint_stats,
)


class _TinyConfig:
    def save_pretrained(self, path: Path) -> None:
        (path / "config.json").write_text("{}", encoding="utf-8")


class _TinyModel(torch.nn.Linear):
    config = _TinyConfig()


class _TinyTokenizer:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir()
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    def __len__(self) -> int:
        return 13


class _TrainingDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = (torch.arange(128) + index) % 13
        return {"input_ids": tokens, "labels": tokens.clone()}


class _TrainingConfig:
    vocab_size = 13
    use_cache = False

    def save_pretrained(self, path: Path) -> None:
        (path / "config.json").write_text(
            json.dumps({"vocab_size": self.vocab_size}), encoding="utf-8"
        )


class _TrainingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _TrainingConfig()
        self.embedding = torch.nn.Embedding(13, 8)
        self.dropout = torch.nn.Dropout(0.2)
        self.output = torch.nn.Linear(8, 13)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        use_cache: bool,
    ) -> SimpleNamespace:
        del use_cache
        logits = self.output(self.dropout(self.embedding(input_ids)))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1)
        )
        return SimpleNamespace(loss=loss, logits=logits)


def _checkpoint(
    path: Path,
    *,
    step: int = 3,
    config_sha256: str = "config",
    manifest_sha256: str = "data",
    implementation: dict[str, object] | None = None,
) -> dict[str, object]:
    implementation = implementation or {"source_sha256": "source"}
    files = {
        "config.json": b"{}",
        "model.safetensors": b"model",
        "optimizer.bin": b"optimizer",
        "publication.json": b"{}",
        "random_states_0.pkl": b"random",
        qwen_train.MPS_RNG_STATE_FILENAME: b"mps-random",
        "scheduler.bin": b"scheduler",
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
    }
    path.mkdir(parents=True)
    for relative, content in files.items():
        file_path = path / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
    state = {
        "format_version": 2,
        "global_step": step,
        "config_sha256": config_sha256,
        "packed_manifest_sha256": manifest_sha256,
        "implementation": implementation,
        "mixed_precision": "bf16",
    }
    (path / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    _write_checkpoint_manifest(path, step)
    return state


def test_checkpoint_manifest_detects_changed_optimizer_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-000003"
    _checkpoint(checkpoint)
    (checkpoint / "optimizer.bin").write_bytes(b"changed!!")

    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_checkpoint(checkpoint, verify_hashes=True)


def test_completed_report_requires_its_exact_checkpoint(tmp_path: Path) -> None:
    implementation = {"source_sha256": "source"}
    _checkpoint(tmp_path / "checkpoint-000003", implementation=implementation)
    report = {
        "checkpoint": "checkpoint-000004",
        "config_sha256": "config",
        "packed_manifest_sha256": "data",
        "implementation": implementation,
        "mixed_precision": "bf16",
    }
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        _complete_report(
            tmp_path,
            config_sha256="config",
            manifest_sha256="data",
            implementation=implementation,
            mixed_precision="bf16",
        )


def test_auto_precision_falls_back_on_cuda_without_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: False)

    assert _resolved_mixed_precision("auto") == "fp16"


def test_checkpoint_save_is_complete_and_refuses_an_existing_target(tmp_path: Path) -> None:
    accelerator = Accelerator(cpu=True)
    model = _TinyModel(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    state = {
        "format_version": 2,
        "global_step": 3,
        "config_sha256": "config",
        "packed_manifest_sha256": "data",
        "implementation": {"source_sha256": "source"},
        "mixed_precision": "no",
    }

    checkpoint = _save_checkpoint(
        accelerator, model, _TinyTokenizer(), tmp_path, state
    )

    assert checkpoint.is_symlink()
    assert _validate_checkpoint(checkpoint, verify_hashes=True) == state
    occupied = tmp_path / "checkpoint-000004"
    occupied.mkdir()
    (occupied / "user.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        _save_checkpoint(
            accelerator,
            model,
            _TinyTokenizer(),
            tmp_path,
            {**state, "global_step": 4},
        )
    assert (occupied / "user.txt").read_text(encoding="utf-8") == "keep"


def test_checkpoint_save_retries_an_interrupted_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accelerator = Accelerator(cpu=True)
    model = _TinyModel(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    state = {
        "format_version": 2,
        "global_step": 7,
        "config_sha256": "config",
        "packed_manifest_sha256": "data",
        "implementation": {"source_sha256": "source"},
        "mixed_precision": "no",
    }
    real_symlink = qwen_train.os.symlink
    calls = 0

    def interrupt_once(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected publication interruption")
        real_symlink(*args, **kwargs)

    monkeypatch.setattr(qwen_train.os, "symlink", interrupt_once)
    with pytest.raises(OSError, match="injected publication interruption"):
        _save_checkpoint(accelerator, model, _TinyTokenizer(), tmp_path, state)

    partials = list(tmp_path.glob(".checkpoint-000007.*"))
    assert len(partials) == 1
    assert _validate_checkpoint(partials[0], verify_hashes=True) == state
    assert not (tmp_path / "checkpoint-000007").exists()

    checkpoint = _save_checkpoint(
        accelerator, model, _TinyTokenizer(), tmp_path, state
    )
    assert calls == 2
    assert checkpoint.is_symlink()
    assert checkpoint.resolve() == partials[0].resolve()
    assert _validate_checkpoint(checkpoint, verify_hashes=True) == state


def test_json_publication_is_atomic_and_never_replaces_existing_data(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    _publish_json(report, {"complete": True})

    assert json.loads(report.read_text(encoding="utf-8")) == {"complete": True}
    with pytest.raises(FileExistsError):
        _publish_json(report, {"complete": False})
    assert json.loads(report.read_text(encoding="utf-8")) == {"complete": True}
    assert list(tmp_path.glob(".report.json.*")) == []


def test_checkpoint_retention_keeps_newest_three_and_stats(tmp_path: Path) -> None:
    for step in range(1, 6):
        target = tmp_path / f".checkpoint-{step:06d}.complete"
        state = _checkpoint(target, step=step)
        os.symlink(target.name, tmp_path / f"checkpoint-{step:06d}")
        (tmp_path / "checkpoint-stats").mkdir(exist_ok=True)
        _write_checkpoint_stats(tmp_path, tmp_path / f"checkpoint-{step:06d}", state)
    unknown = tmp_path / "checkpoint-unknown"
    unknown.write_text("keep", encoding="utf-8")
    unknown_target = tmp_path / "outside-target"
    unknown_target.write_text("also keep", encoding="utf-8")
    os.symlink(unknown_target.name, tmp_path / "checkpoint-000002-extra")

    _retain_checkpoints(tmp_path, 3)
    _retain_checkpoints(tmp_path, 3)

    assert not (tmp_path / "checkpoint-000001").exists()
    assert not (tmp_path / ".checkpoint-000001.complete").exists()
    assert sorted(
        p.name
        for p in tmp_path.glob("checkpoint-[0-9]*")
        if p.is_symlink() and p.name != "checkpoint-000002-extra"
    ) == ["checkpoint-000003", "checkpoint-000004", "checkpoint-000005"]
    assert all(
        (tmp_path / "checkpoint-stats" / f"checkpoint-{step:06d}.json").is_file()
        for step in range(1, 6)
    )
    assert unknown.read_text(encoding="utf-8") == "keep"
    assert (tmp_path / "checkpoint-000002-extra").is_symlink()
    assert unknown_target.read_text(encoding="utf-8") == "also keep"


def test_checkpoint_retention_recovers_after_delete_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = []
    for step in range(1, 5):
        target = tmp_path / f".checkpoint-{step:06d}.complete"
        states.append(_checkpoint(target, step=step))
        os.symlink(target.name, tmp_path / f"checkpoint-{step:06d}")
    oldest_target = tmp_path / ".checkpoint-000001.complete"
    real_rmtree = qwen_train.shutil.rmtree

    def fail_delete(path: Path) -> None:
        assert Path(path) == oldest_target
        (oldest_target / "checkpoint_manifest.json").unlink()
        raise OSError("injected checkpoint deletion failure")

    monkeypatch.setattr(qwen_train.shutil, "rmtree", fail_delete)
    with pytest.raises(OSError, match="injected checkpoint deletion failure"):
        _retain_checkpoints(tmp_path, 3, expected_state=states[-1])

    assert not os.path.lexists(tmp_path / "checkpoint-000001")
    assert oldest_target.is_dir()
    assert (tmp_path / "checkpoint-stats/checkpoint-000001.json").is_file()
    assert (tmp_path / ".checkpoint-prunes/checkpoint-000001.json").is_file()

    monkeypatch.setattr(qwen_train.shutil, "rmtree", real_rmtree)
    _retain_checkpoints(tmp_path, 3, expected_state=states[-1])

    assert not oldest_target.exists()
    assert not (tmp_path / ".checkpoint-prunes/checkpoint-000001.json").exists()
    assert [path.name for path in qwen_train._checkpoint_directories(tmp_path)] == [
        "checkpoint-000002",
        "checkpoint-000003",
        "checkpoint-000004",
    ]


def test_checkpoint_retention_recovers_after_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = []
    for step in range(1, 5):
        target = tmp_path / f".checkpoint-{step:06d}.complete"
        states.append(_checkpoint(target, step=step))
        os.symlink(target.name, tmp_path / f"checkpoint-{step:06d}")
    oldest = tmp_path / "checkpoint-000001"
    real_unlink = Path.unlink

    def fail_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == oldest:
            raise OSError("injected checkpoint unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="injected checkpoint unlink failure"):
        _retain_checkpoints(tmp_path, 3, expected_state=states[-1])

    assert oldest.is_symlink()
    assert (tmp_path / ".checkpoint-000001.complete").is_dir()
    assert (tmp_path / ".checkpoint-prunes/checkpoint-000001.json").is_file()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    _retain_checkpoints(tmp_path, 3, expected_state=states[-1])
    assert not os.path.lexists(oldest)
    assert not (tmp_path / ".checkpoint-000001.complete").exists()


def test_checkpoint_retention_ignores_foreign_provenance(tmp_path: Path) -> None:
    foreign_target = tmp_path / ".checkpoint-000001.complete"
    _checkpoint(foreign_target, step=1, config_sha256="foreign")
    os.symlink(foreign_target.name, tmp_path / "checkpoint-000001")
    current_state = None
    for step in range(2, 6):
        target = tmp_path / f".checkpoint-{step:06d}.complete"
        current_state = _checkpoint(target, step=step)
        os.symlink(target.name, tmp_path / f"checkpoint-{step:06d}")
    assert current_state is not None

    _retain_checkpoints(tmp_path, 3, expected_state=current_state)

    assert (tmp_path / "checkpoint-000001").is_symlink()
    assert foreign_target.is_dir()
    assert not os.path.lexists(tmp_path / "checkpoint-000002")
    assert [path.name for path in qwen_train._checkpoint_directories(tmp_path)] == [
        "checkpoint-000001",
        "checkpoint-000003",
        "checkpoint-000004",
        "checkpoint-000005",
    ]


def test_checkpoint_retention_rejects_an_external_journal_directory(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep"
    marker.write_text("user data", encoding="utf-8")
    os.symlink(external.name, tmp_path / ".checkpoint-prunes")

    with pytest.raises(ValueError, match="not a private directory"):
        _retain_checkpoints(tmp_path, 3)
    assert marker.read_text(encoding="utf-8") == "user data"


def test_fabricated_prune_journal_cannot_delete_an_unrelated_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / ".checkpoint-000001.user"
    target.mkdir()
    marker = target / "keep"
    marker.write_text("user data", encoding="utf-8")
    journal = tmp_path / ".checkpoint-prunes"
    journal.mkdir()
    _publish_json(
        journal / "checkpoint-000001.json",
        {
            "format_version": qwen_train.CHECKPOINT_MANIFEST_VERSION,
            "checkpoint": "checkpoint-000001",
            "target": target.name,
            "stats_sha256": "0" * 64,
            "manifest_sha256": "0" * 64,
            "state_identity": {},
        },
    )

    _retain_checkpoints(tmp_path, 3)

    assert marker.read_text(encoding="utf-8") == "user data"
    assert (journal / "checkpoint-000001.json").is_file()


def test_mismatched_prune_journal_cannot_delete_a_foreign_checkpoint(
    tmp_path: Path,
) -> None:
    target = tmp_path / ".checkpoint-000001.foreign"
    foreign_state = _checkpoint(target, step=1, config_sha256="foreign")
    checkpoint = tmp_path / "checkpoint-000001"
    os.symlink(target.name, checkpoint)
    _write_checkpoint_stats(tmp_path, checkpoint, foreign_state)
    stats = tmp_path / "checkpoint-stats/checkpoint-000001.json"
    journal = tmp_path / ".checkpoint-prunes"
    journal.mkdir()
    mismatched_identity = {
        **qwen_train._checkpoint_identity(foreign_state),
        "config_sha256": "current",
    }
    _publish_json(
        journal / "checkpoint-000001.json",
        {
            "format_version": qwen_train.CHECKPOINT_MANIFEST_VERSION,
            "checkpoint": checkpoint.name,
            "target": target.name,
            "stats_sha256": qwen_train.sha256_file(stats),
            "manifest_sha256": qwen_train.sha256_file(
                target / "checkpoint_manifest.json"
            ),
            "state_identity": mismatched_identity,
        },
    )

    _retain_checkpoints(tmp_path, 3)

    assert checkpoint.is_symlink()
    assert target.is_dir()
    assert (journal / "checkpoint-000001.json").is_file()


def test_altered_checkpoint_stats_prevent_authorized_deletion(tmp_path: Path) -> None:
    target = tmp_path / ".checkpoint-000001.complete"
    state = _checkpoint(target, step=1)
    checkpoint = tmp_path / "checkpoint-000001"
    os.symlink(target.name, checkpoint)
    _write_checkpoint_stats(tmp_path, checkpoint, state)
    stats = tmp_path / "checkpoint-stats/checkpoint-000001.json"
    stats_sha256 = qwen_train.sha256_file(stats)
    journal = tmp_path / ".checkpoint-prunes"
    journal.mkdir()
    _publish_json(
        journal / "checkpoint-000001.json",
        {
            "format_version": qwen_train.CHECKPOINT_MANIFEST_VERSION,
            "checkpoint": checkpoint.name,
            "target": target.name,
            "stats_sha256": stats_sha256,
            "manifest_sha256": qwen_train.sha256_file(
                target / "checkpoint_manifest.json"
            ),
            "state_identity": qwen_train._checkpoint_identity(state),
        },
    )
    stats.write_text('{"altered": true}\n', encoding="utf-8")

    _retain_checkpoints(tmp_path, 3)

    assert checkpoint.is_symlink()
    assert target.is_dir()
    assert (journal / "checkpoint-000001.json").is_file()


def test_legitimate_stats_cannot_authorize_a_substituted_target(tmp_path: Path) -> None:
    legitimate_target = tmp_path / ".checkpoint-000001.complete"
    state = _checkpoint(legitimate_target, step=1)
    checkpoint = tmp_path / "checkpoint-000001"
    os.symlink(legitimate_target.name, checkpoint)
    stats_record = _write_checkpoint_stats(tmp_path, checkpoint, state)
    stats_path = tmp_path / "checkpoint-stats/checkpoint-000001.json"
    checkpoint.unlink()

    unrelated_target = tmp_path / ".checkpoint-000001.user"
    unrelated_target.mkdir()
    marker = unrelated_target / "keep"
    marker.write_text("user data", encoding="utf-8")
    journal = tmp_path / ".checkpoint-prunes"
    journal.mkdir()
    _publish_json(
        journal / "checkpoint-000001.json",
        {
            "format_version": qwen_train.CHECKPOINT_MANIFEST_VERSION,
            "checkpoint": "checkpoint-000001",
            "target": unrelated_target.name,
            "stats_sha256": qwen_train.sha256_file(stats_path),
            "manifest_sha256": stats_record["manifest_sha256"],
            "state_identity": stats_record["state_identity"],
        },
    )

    _retain_checkpoints(tmp_path, 3)

    assert marker.read_text(encoding="utf-8") == "user data"
    assert legitimate_target.is_dir()
    assert (journal / "checkpoint-000001.json").is_file()


def test_json_publication_cleans_temp_through_a_symlinked_parent(tmp_path: Path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    alias = tmp_path / "alias"
    os.symlink(real_directory.name, alias)

    _publish_json(alias / "report.json", {"complete": True})

    assert json.loads((real_directory / "report.json").read_text(encoding="utf-8")) == {
        "complete": True
    }
    assert list(real_directory.glob(".report.json.*")) == []


def _training_experiment(tmp_path: Path, name: str) -> tuple[Path, QwenExperimentConfig]:
    packed_dir = tmp_path / f"{name}-packed"
    packed_dir.mkdir()
    (packed_dir / "manifest.json").write_text('{"tiny": true}', encoding="utf-8")
    config_path = tmp_path / f"{name}.toml"
    config_path.write_text(f"name = {name!r}\n", encoding="utf-8")
    revision = "a" * 40
    config = QwenExperimentConfig(
        mode="benchmark",
        data=QwenDataConfig(
            source=tmp_path / "unused.db",
            packed_dir=packed_dir,
            train_through=2023,
            validation_through=2024,
            tokenizer_id="tiny-tokenizer",
            tokenizer_revision=revision,
            tokenizer_batch_size=2,
            max_records_per_split=4,
        ),
        model=QwenModelConfig(
            model_id="tiny-model",
            revision=revision,
            context_length=128,
            weights_dtype="float32",
            gradient_checkpointing=False,
        ),
        training=QwenTrainingConfig(
            output_dir=tmp_path / name,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            epochs=2,
            max_steps=4,
            max_eval_batches=1,
            learning_rate=0.01,
            weight_decay=0.0,
            gradient_clip=1.0,
            warmup_ratio=0.0,
            checkpoint_interval=2,
            log_interval=100,
            seed=1234,
            num_workers=0,
            mixed_precision="no",
        ),
    )
    return config_path, config


def _assert_nested_equal(
    left: object, right: object, *, rtol: float = 0, atol: float = 0
) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key], rtol=rtol, atol=atol)
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item, rtol=rtol, atol=atol)
    else:
        assert left == right


def test_interrupted_training_matches_uninterrupted_from_mid_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interrupted_path, interrupted_config = _training_experiment(tmp_path, "resumed")
    full_path, full_config = _training_experiment(tmp_path, "uninterrupted")
    configs = {
        interrupted_path: interrupted_config,
        full_path: full_config,
    }
    dataset = _TrainingDataset()
    manifest = {
        "splits": {
            name: {"blocks": len(dataset)}
            for name in ("train", "validation", "test")
        }
    }

    monkeypatch.setattr(qwen_train, "load_qwen_config", configs.__getitem__)
    monkeypatch.setattr(
        qwen_train,
        "load_packed_datasets",
        lambda _config: (
            {"train": dataset, "validation": dataset, "test": dataset},
            manifest,
        ),
    )
    monkeypatch.setattr(
        qwen_train, "_implementation_identity", lambda: {"test": "implementation"}
    )
    monkeypatch.setattr(
        qwen_train.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _TinyTokenizer(),
    )
    monkeypatch.setattr(
        qwen_train.Qwen3_5ForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: _TrainingModel(),
    )

    def reset_accelerate_state() -> None:
        # Accelerator caches process/device state globally. Reset it so each
        # run below is an independently initialized MPS (or CPU) experiment.
        AcceleratorState._reset_state(reset_partial_state=True)

    real_save = qwen_train._save_checkpoint

    def interrupt_mid_epoch(*args: object, **kwargs: object) -> Path:
        checkpoint = real_save(*args, **kwargs)
        state = args[4]
        if isinstance(state, dict) and state["global_step"] == 2:
            assert state["next_epoch"] == 0
            assert state["next_batch"] == 4
            raise KeyboardInterrupt("injected mid-epoch interruption")
        return checkpoint

    monkeypatch.setattr(qwen_train, "_save_checkpoint", interrupt_mid_epoch)
    reset_accelerate_state()
    with pytest.raises(KeyboardInterrupt, match="injected mid-epoch interruption"):
        qwen_train.run_qwen_training(interrupted_path, resume="auto")

    reset_accelerate_state()
    monkeypatch.setattr(qwen_train, "_save_checkpoint", real_save)
    resumed_checkpoint = qwen_train.run_qwen_training(interrupted_path, resume="auto")
    reset_accelerate_state()
    full_checkpoint = qwen_train.run_qwen_training(full_path, resume="none")

    resumed_state = qwen_train._state(resumed_checkpoint)
    full_state = qwen_train._state(full_checkpoint)
    for key in (
        "global_step",
        "next_epoch",
        "next_batch",
        "trained_tokens",
        "last_loss",
    ):
        assert resumed_state[key] == full_state[key]

    resumed_weights = load_file(resumed_checkpoint / "model.safetensors")
    full_weights = load_file(full_checkpoint / "model.safetensors")
    tolerance = 1e-6 if torch.backends.mps.is_available() else 0
    _assert_nested_equal(
        resumed_weights, full_weights, rtol=tolerance, atol=tolerance
    )
    for filename in ("optimizer.bin", "scheduler.bin"):
        resumed_value = torch.load(
            resumed_checkpoint / filename, map_location="cpu", weights_only=True
        )
        full_value = torch.load(
            full_checkpoint / filename, map_location="cpu", weights_only=True
        )
        _assert_nested_equal(
            resumed_value, full_value, rtol=tolerance, atol=tolerance
        )

    probe = torch.arange(128).remainder(13).unsqueeze(0)
    resumed_model = _TrainingModel().eval()
    resumed_model.load_state_dict(resumed_weights)
    full_model = _TrainingModel().eval()
    full_model.load_state_dict(full_weights)
    with torch.inference_mode():
        resumed_output = resumed_model(probe, probe, use_cache=False).logits
        full_output = full_model(probe, probe, use_cache=False).logits
    torch.testing.assert_close(
        resumed_output, full_output, rtol=tolerance, atol=tolerance
    )

    output_dir = interrupted_config.training.output_dir
    checkpoint_name = "checkpoint-000002"
    checkpoint = output_dir / checkpoint_name
    target = output_dir / os.readlink(checkpoint)
    checkpoint.unlink()
    state = qwen_train._state(target)
    journal = output_dir / ".checkpoint-prunes"
    journal.mkdir()
    _publish_json(
        journal / f"{checkpoint_name}.json",
        {
            "format_version": qwen_train.CHECKPOINT_MANIFEST_VERSION,
            "checkpoint": checkpoint_name,
            "target": target.name,
            "stats_sha256": qwen_train.sha256_file(
                output_dir / "checkpoint-stats" / f"{checkpoint_name}.json"
            ),
            "manifest_sha256": qwen_train.sha256_file(
                target / "checkpoint_manifest.json"
            ),
            "state_identity": qwen_train._checkpoint_identity(state),
        },
    )

    reused_checkpoint = qwen_train.run_qwen_training(interrupted_path, resume="auto")
    assert reused_checkpoint == resumed_checkpoint
    assert not target.exists()
    assert not journal.exists()
