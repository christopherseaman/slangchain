from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig

BYTE_VALUES = 256
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.embed_dim // config.heads
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embed_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TemporalSlangModel(nn.Module):
    """A byte-level causal Transformer with one numeric year condition."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.year_projection = nn.Linear(1, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output = nn.Linear(config.embed_dim, VOCAB_SIZE, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        years: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = tokens.shape
        if length > self.config.context_length:
            raise ValueError(
                f"Sequence length {length} exceeds context {self.config.context_length}"
            )

        positions = torch.arange(length, device=tokens.device)
        scaled_years = (years.float() - self.config.year_origin) / self.config.year_scale
        year_features = self.year_projection(scaled_years[:, None])[:, None, :]
        x = self.token_embedding(tokens) + self.position_embedding(positions) + year_features
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=PAD_ID,
            )
        return logits, loss

    def metadata(self) -> dict[str, object]:
        return asdict(self.config)


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode_tokens(tokens: list[int]) -> str:
    raw = bytes(token for token in tokens if 0 <= token < BYTE_VALUES)
    return raw.decode("utf-8", errors="replace")


@torch.inference_mode()
def sample(
    model: TemporalSlangModel,
    prompt: str,
    year: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    if max_new_tokens <= 0 or temperature <= 0 or top_k <= 0:
        raise ValueError("max_new_tokens, temperature, and top_k must be positive")

    generated = [BOS_ID, *encode_text(prompt)]
    year_tensor = torch.tensor([year], dtype=torch.float32, device=device)
    model.eval()

    for _ in range(max_new_tokens):
        context = generated[-model.config.context_length :]
        tokens = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(tokens, year_tensor)
        next_logits = logits[0, -1] / temperature
        next_logits[PAD_ID] = -torch.inf
        next_logits[BOS_ID] = -torch.inf
        choices = min(top_k, next_logits.numel())
        values, indices = torch.topk(next_logits, choices)
        probabilities = torch.softmax(values, dim=-1)
        next_token = indices[torch.multinomial(probabilities, 1)].item()
        if next_token == EOS_ID:
            break
        generated.append(next_token)

    return decode_tokens(generated[1:])
