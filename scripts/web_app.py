#!/usr/bin/env python3
"""Serve the preserved Qwen checkpoint with a small local Gradio UI."""

from __future__ import annotations

import argparse
import html
import re
import secrets
from pathlib import Path

import gradio as gr
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from slangchain.qwen_train import validate_generation_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/qwen35-0.8b.toml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts/qwen35-0.8b-final/checkpoint-034798"
LOADING_HTML = (
    '<div class="slang-loading" role="status" aria-live="polite">'
    '<span class="slang-spinner" aria-hidden="true"></span>Generating...'
    "</div>"
)
CARD_CSS = """
.slang-results { display: grid; gap: 1rem; }
.slang-card {
    border: 1px solid var(--block-border-color);
    border-radius: 0.85rem;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 8px rgb(0 0 0 / 6%);
}
.slang-card__title { margin: 0; }
.slang-card__section { margin: 1rem 0 0 1rem; }
.slang-card__section h4 { margin: 0 0 0.35rem; }
.slang-card__content { margin: 0 0 0 1rem; white-space: pre-wrap; }
.slang-card__examples { margin: 0 0 0 1.5rem; padding-left: 1rem; }
.slang-year-form {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    width: 24rem !important;
    max-width: calc(100% - 2rem);
    margin: 0 auto 1rem !important;
    align-items: center;
    gap: 0.75rem;
}
.slang-year-label, .slang-year-label p {
    flex: 0 0 5rem !important;
    margin: 0;
    font-size: 2rem !important;
    font-weight: 700;
    text-align: right;
}
.slang-year { flex: 1 1 auto !important; min-width: 0 !important; }
.slang-year-form input,
.slang-year input,
.slang-year-form input[data-testid="number-input"] {
    font-size: 2rem !important;
    line-height: 1.2;
}
.slang-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    min-height: 160px;
}
.slang-spinner {
    width: 1.25rem;
    height: 1.25rem;
    border: 0.2rem solid currentColor;
    border-right-color: transparent;
    border-radius: 50%;
    animation: slang-spin 0.8s linear infinite;
}
@keyframes slang-spin { to { transform: rotate(360deg); } }
.slang-generate {
    width: 24rem !important;
    max-width: calc(100% - 2rem);
    margin: 0 auto 1.5rem !important;
}
"""


class Generator:
    def __init__(self, checkpoint: Path, config: Path) -> None:
        validate_generation_checkpoint(checkpoint, config)
        self.accelerator = Accelerator()
        device = self.accelerator.device
        dtype = (
            torch.bfloat16
            if device.type == "mps"
            else torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
            if device.type == "cuda"
            else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            checkpoint, dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        self.model.config.use_cache = True
        self.model.eval()

    def __call__(self, year: float | int) -> str:
        try:
            value = int(year)
        except (TypeError, ValueError, OverflowError):
            value = 0
        if year is None or value != year or not 1900 <= value <= 2100:
            raise gr.Error("Year must be a whole number from 1900 through 2100.")
        set_seed(secrets.randbits(32))
        encoded = self.tokenizer(
            f"<ENTRY>\nYEAR: {value}\nWORD:",
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.accelerator.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=160,
                num_return_sequences=5,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_tokens = encoded["input_ids"].shape[1]
        generated = output[:, prompt_tokens:]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        entries = [
            _format_entry(
                index,
                text,
                truncated=generated[index - 1].shape[0] >= 160
                and not bool((generated[index - 1] == self.tokenizer.eos_token_id).any().item()),
            )
            for index, text in enumerate(texts, 1)
        ]
        return '<div class="slang-results">' + "".join(entries) + "</div>"


def _clean_text(text: str) -> str:
    text = re.sub(r"\[([^\[\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\[\]]+)\]", r"\1", text)
    return re.sub(r"\s*</?ENTRY>\s*", "\n", text).strip()


def _split_examples(text: str) -> list[str]:
    markers = list(re.finditer(r"(?m)(?:^|\s)\d{1,2}\s*[:.)-]\s+", text))
    if len(markers) < 2:
        return [text.strip()] if text.strip() else []
    return [
        text[
            marker.end() : markers[index + 1].start() if index + 1 < len(markers) else None
        ].strip()
        for index, marker in enumerate(markers)
    ]


def _format_entry(index: int, text: str, *, truncated: bool) -> str:
    text = _clean_text(text)
    definition_match = re.search(r"(?im)^\s*DEFINITION:\s*", text)
    example_match = re.search(r"(?im)^\s*EXAMPLE:\s*", text)
    first_marker = min(
        (match.start() for match in (definition_match, example_match) if match),
        default=len(text),
    )
    word = text[:first_marker].strip().removeprefix("WORD:").strip() or "(unnamed)"
    definition = ""
    examples: list[str] = []
    if definition_match:
        definition_end = example_match.start() if example_match else len(text)
        definition = text[definition_match.end() : definition_end].strip()
    if example_match:
        examples = _split_examples(text[example_match.end() :].strip())
    if truncated:
        if examples:
            examples[-1] += "..."
        elif definition:
            definition += "..."
        else:
            text += "..."

    sections = [
        '<article class="slang-card">',
        f'<h3 class="slang-card__title">{index}. {html.escape(word)}</h3>',
    ]
    if definition:
        sections.extend(
            [
                '<section class="slang-card__section">',
                "<h4>Definition</h4>",
                f'<p class="slang-card__content">{html.escape(definition)}</p>',
                "</section>",
            ]
        )
    if len(examples) > 1:
        sections.extend(
            [
                '<section class="slang-card__section">',
                "<h4>Examples</h4>",
                '<ol class="slang-card__examples">'
                + "".join(f"<li>{html.escape(example)}</li>" for example in examples)
                + "</ol>",
                "</section>",
            ]
        )
    elif examples:
        sections.extend(
            [
                '<section class="slang-card__section">',
                "<h4>Example</h4>",
                f'<p class="slang-card__content">{html.escape(examples[0])}</p>',
                "</section>",
            ]
        )
    elif not definition:
        sections.append(
            f'<p class="slang-card__content">{html.escape(text or "(no text generated)")}</p>'
        )
    sections.append("</article>")
    return "".join(sections)


def _show_loading() -> str:
    return LOADING_HTML


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio URL")
    args = parser.parse_args()
    generator = Generator(args.checkpoint, args.config)
    with gr.Blocks(title="Future Slang Dictionary") as demo:
        gr.Markdown(
            "# Future Slang Dictionary\n"
            "Generate plausible-looking slang entries for a year. "
            "The 752M-parameter Qwen3.5-0.8B model was continued-pretrained on "
            "dated Urban Dictionary entries from 1999 through 2026."
        )
        with gr.Row(elem_classes="slang-year-form", variant="compact"):
            gr.Markdown("Year", elem_classes="slang-year-label", scale=0, min_width=0)
            year = gr.Number(
                show_label=False,
                value=2030,
                precision=0,
                minimum=1900,
                maximum=2100,
                container=False,
                elem_id="slang-year",
                elem_classes="slang-year",
                scale=1,
        )
        generate = gr.Button("Generate", variant="primary", elem_classes="slang-generate")
        output = gr.HTML(min_height=160)
        generate.click(
            fn=_show_loading,
            outputs=output,
            queue=False,
            show_progress="hidden",
        ).then(
            fn=generator,
            inputs=year,
            outputs=output,
            api_name="Generator",
            show_progress="hidden",
        )
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        css=CARD_CSS,
    )


if __name__ == "__main__":
    main()
