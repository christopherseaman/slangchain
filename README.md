# Slangchain

Slangchain is a teaching project that generates *plausible-looking* slang for a requested year.
It has two deliberately different tracks:

- a 5M-parameter byte-level Transformer written in plain PyTorch, useful for seeing every part;
- continued pretraining of the modern 752M-parameter Qwen3.5-0.8B base model, useful for a much
  more capable result and for discussing what a pretrained model changes scientifically.

The scratch model projects a numeric year beside its byte and position embeddings:

```text
UTF-8 bytes + positions + scaled year
                  │
          causal Transformer
                  │
             next byte
```

This is speculative generation, not a forecast of which words people will adopt. The honest
experiment is whether a model trained through one year transfers to later held-out years.

## Quick start with clean synthetic data

The included entries are invented, classroom-safe examples. They are enough to exercise the
whole pipeline, not enough to learn good slang.

```bash
uv sync
uv run slangchain prepare examples/toy_entries.jsonl \
  --output data/processed/toy.jsonl
uv run slangchain train --config configs/smoke.toml
uv run slangchain generate \
  --checkpoint artifacts/smoke/best \
  --year 2030 \
  --samples 5
```

Each command refuses to replace an existing prepared file or non-empty training directory. Pick
a new output path when repeating an experiment. The training log states the actual device;
Hugging Face Accelerate selects MPS, CUDA, or CPU without platform-specific branches in the
model or training loop. The default is full precision because it is the least surprising option
across backends.

## Qwen3.5-0.8B: the main extended demo

The Qwen track uses the model's native tokenizer and represents each entry as ordinary text:

```text
<ENTRY>
YEAR: 2023
WORD: pocketpause
DEFINITION: The moment of calm after putting a distracting phone out of reach.
EXAMPLE: A pocketpause fixed my homework session.
</ENTRY>
```

Newlines and repeated whitespace inside source fields are canonicalized so they cannot be
mistaken for field boundaries. The year remains visible in the data and in class, even if an
ablation later shows that it adds little predictive value.

The full configuration reads the SQLite database directly, keeps every valid record, partitions
chronologically, tokenizes in batches, and concatenates records with EOS markers into disk-backed
2048-token blocks. It does not impose a per-year or total-record cap. This avoids loading 5.7
million Python objects into memory and avoids the padding waste of putting one short dictionary
entry in every 2048-token sequence.

Run the checksum-pinned download, extraction, packing, training, resume, and generation path:

```bash
./scripts/qwen_demo.sh 2030 10
```

The script downloads or extracts only when the final archive/database is absent. Packing is
provenance-checked and reusable. Training checkpoints contain the model, optimizer, schedule,
random state, exact model/tokenizer revisions, config hash, packed-corpus hash, and data cursor;
an interrupted run resumes from the newest complete checkpoint. Generated files are never
silently replaced.

The model and tokenizer are pinned to Qwen's immutable revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Transformers is isolated in the Qwen modules; the
scratch model remains plain PyTorch. The same Qwen path uses MPS, CUDA, or CPU through Accelerate
and does not require CUDA-only FlashAttention, bitsandbytes, or custom DeltaNet kernels.

For a local dictionary-style UI, run Gradio ephemerally and keep the model resident in one process:

```bash
./web_app.sh
```

Open `http://<your-machine>:7860`, enter a year, and generate five entries. The root launcher
binds to `0.0.0.0` for access from another machine and does not create a public share link. Use
`WEB_HOST=127.0.0.1 ./web_app.sh` for localhost-only access. The app defaults to the preserved
`artifacts/qwen35-0.8b-final/checkpoint-034798` checkpoint; pass `--checkpoint` to use another one.
For a temporary public Gradio URL when network port access is unavailable, run `./web_app.sh --share`.

This is a continued-pretraining and fluency demonstration, not a clean forecast evaluation.
Qwen3.5 was released in 2026 and its original web training may already include Urban Dictionary
or text from the nominal 2024–2026 holdout. Use the scratch model for the uncontaminated cutoff
experiment; use the Qwen track to show how a strong prior changes quality, speed, and scientific
claims.

Before an extended run, the benchmark config deliberately caps records and steps. It uses the
representative `data/processed/entries.jsonl` produced by the scratch demo:

```bash
uv run slangchain qwen-pack --config configs/qwen35-benchmark.toml
uv run slangchain qwen-train --config configs/qwen35-benchmark.toml
uv run slangchain qwen-generate \
  --config configs/qwen35-benchmark.toml \
  --year 2030 \
  --samples 3
```

On the M3 Ultra used for this project, the real 100-update, context-2048 gate sustained about 590
training tokens/second. It used about 11.4 GB of live tensors and 44.3 GB including Metal's driver
cache; activation checkpointing was slower, and batch size two was both slower and much larger.
The uncapped training split contains 5,454,835 entries and 570,126,125 Qwen tokens in 278,382
blocks. One epoch is therefore roughly 11.2 days at benchmark speed, despite fitting easily in 96
GB. Full mode rejects benchmark record, step, and evaluation caps, making it difficult to
accidentally present a short benchmark as the all-corpus run.

The default writes an immutable, fully resumable checkpoint every 1,000 optimizer updates
(roughly eight hours at benchmark speed). Each FP32-master checkpoint is about 8.4 GB; the
config-driven retention window keeps the newest three (about 25 GB steady state), with transient
storage reaching about 34 GB while a new checkpoint is published. A tiny per-checkpoint stats/state
record is retained permanently for auditability; old model checkpoints are pruned only after a
complete publication.

Qwen3.5 is a hybrid model: 18 of its 24 language layers use Gated DeltaNet linear attention and
six use full attention. On Apple Silicon, Transformers uses portable PyTorch fallbacks rather
than the optional CUDA kernels. That is why this project measures the real MPS path instead of
inferring speed from parameter count. Moving to Qwen3.5-2B later requires a config/model revision
change and a fresh hardware benchmark; the packing and training code stays the same if the exact
tokenizer is retained and validated.

## Use a real dated corpus

As of August 28, 2026, the technically useful source is the SQLite snapshot linked in a recent
comment on the original Reddit dataset thread. Its archive metadata says that it contains API
properties, including `written_on`, through April 1, 2026. The scratch demo script downloads the
pinned archive from [data.badmath.org](https://data.badmath.org/UrbanDictionary_04-2026.7z),
verifies its SHA-256, extracts it, prepares a balanced subset, trains, and generates ten samples:

```bash
./scripts/demo.sh
```

Pass a future year and sample count as positional arguments. On a fresh run, the per-year cap can
be changed with an environment variable:

```bash
MAX_PER_YEAR=30000 ./scripts/demo.sh 2035 20
```

Completed stages are safely reusable: an existing extracted database skips download and
extraction, and an existing archive skips download. Provenance manifests bind prepared data to
the archive and preparation settings, then bind the checkpoint to that prepared-data hash and the
training-config hash. A changed `MAX_PER_YEAR`, changed data, or changed config stops with an
explanation instead of silently reusing a stale result. Move the named generated outputs aside
before deliberately starting a different experiment.

The script holds an atomic single-run lock and atomically installs downloads, databases, and
manifests without replacing existing paths. It uses `bsdtar` (included with macOS) or falls back
to `7zz`/`7z` on other systems. If a process is killed without a chance to clean up, it can leave
`.demo.lock`, a hidden download/extraction temporary path, a preparation `.partial`, or an
incomplete `artifacts/tiny` directory. Confirm that no demo is still running, move the reported
stale path aside, and rerun; the script never guesses that an incomplete artifact is safe.

The checked-in lock currently resolves on Apple Silicon with macOS 14+ and on glibc 2.28+ Linux
for x86_64 or arm64. Other operating systems and older Intel Macs may need a regenerated lock or a
compatible PyTorch constraint. Runtime device selection remains portable across MPS, single-GPU
CUDA, and CPU, but only MPS and CPU have been exercised for this project.

To run the stages individually, point the preparer at an extracted `.db`, `.sqlite`, or `.sqlite3`
file:

```bash
uv run slangchain prepare data/raw/path-to-extracted.sqlite \
  --output data/processed/entries.jsonl \
  --max-per-year 20000
uv run slangchain train --config configs/tiny.toml
uv run slangchain generate \
  --checkpoint artifacts/tiny/best \
  --year 2030 \
  --samples 10
```

`--max-per-year` uses seeded reservoir sampling, so early years are not crowded out by recent
high-volume years. Remove the option to retain every valid row. JSONL and CSV inputs also work
when they have aliases for a word, definition, and date column. Rows without a real date are
rejected because inventing dates would invalidate the temporal experiment.

The source options have different limitations:

| Source | What it has | Temporal-demo fit |
|---|---|---|
| [GitHub word list](https://github.com/mattbierner/urban-dictionary-word-list) | About 2.7M names; April 2022 snapshot | No definitions or per-entry dates |
| [Kaggle CSV](https://www.kaggle.com/datasets/therohk/urban-dictionary-words-dataset) | 2,580,925 words, definitions, and votes through May 2016 | No creation-date column |
| [Original Reddit corpus](https://www.reddit.com/r/datasets/comments/63spoc/19gb_of_urban_dictionary_definitions_1999_may_2016/) | 1999–May 2016 JSON records | The documented record has no creation date |
| [2026 Internet Archive snapshot](https://archive.org/details/urban-dictionary-04-2026.7z) | Dated SQLite records through March 2026 | Best technical fit; about 763 MB compressed |

The selected SQLite table also has `defid`, `author`, `current_vote`, `permalink`,
`thumbs_down`, `thumbs_up`, and `lowercase_word`. In this snapshot every thumbs count is zero and
every `current_vote` is empty, so the model does not pretend those columns contain popularity or
quality labels. Only `written_on` and the text fields enter the current record representation.

Do not treat an uploader's repository or dataset license as proof that third-party definitions
can be redistributed. Urban Dictionary content can include slurs, sexual material, harassment,
personal information, and false claims. The [Urban Dictionary terms](https://urbandictionary.help/tos/)
also restrict automated access. Keep raw data out of Git, review generated material before showing
it in class, follow institutional policy, and do not live-scrape the site for this demo. The MIT
license in this repository covers this project's code and synthetic examples, not downloaded data.

## Experiment design

The extracted archive member is named `UrbanDictionary_04-2026.db`. `configs/tiny.toml` trains on
entries through 2023, selects checkpoints using a bounded 2024 evaluation sample, and reports a
final score over every prepared 2025–2026 record. Change those cutoffs to match the actual
snapshot. Random splitting would leak later language into training and defeat the lesson.

The model uses UTF-8 bytes rather than an external tokenizer. Its vocabulary is the 256 byte
values plus padding, beginning, and end markers. That makes misspellings, emoji, and invented
spellings representable while keeping tokenization visible in a few lines of code. During scratch
training, each record looks like:

```text
WORD: pocketpause
DEFINITION: The moment of calm after putting a distracting phone out of reach.
EXAMPLE: A pocketpause fixed my homework session.
```

The numeric year is supplied separately. Because it is a number rather than a lookup-table entry,
the model can accept 2030 even when training ends earlier. That extrapolation is an art control,
not calibrated evidence about 2030.

Useful classroom comparisons include:

- Generate with the same random seed for 2010, 2020, and 2030.
- Compare chronological validation/test loss with a deliberately random split and discuss leakage.
- Search generated terms against the training file to expose memorization.
- Change `year_scale` or remove the year projection and test whether the condition matters.

Validation and test loss are next-byte cross-entropy; the reported perplexity is `exp(loss)`.
Those metrics measure continuation quality, not social adoption, originality, safety, or humor.

## Code map

- `src/slangchain/data.py` validates raw records, samples by year, makes chronological splits, and
  turns entries into byte sequences.
- `src/slangchain/model.py` contains the complete Transformer and sampling loop.
- `src/slangchain/train.py` contains optimization and chronological evaluation.
- `src/slangchain/runtime.py` is the small portability boundary around Accelerate and safe tensor
  checkpoints.
- `src/slangchain/qwen_data.py` streams, normalizes, tokenizes, and packs the all-corpus Qwen data.
- `src/slangchain/qwen_train.py` contains portable continued pretraining, evaluation, resumable
  checkpoints, and native generation.
- `src/slangchain/qwen_config.py` keeps the larger track's full-versus-benchmark invariants
  separate from the scratch configuration.
- `configs/` holds the genuinely changeable experiment settings.

Run the checks with:

```bash
uv run pytest
uv run ruff check .
```

PyTorch documents its [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html), and
Accelerate documents both [automatic MPS use](https://huggingface.co/docs/accelerate/usage_guides/mps)
and its [portable training interface](https://huggingface.co/docs/accelerate/quicktour). MPS is a
single-GPU backend; CUDA-only optimizations are intentionally absent from the portable paths.
