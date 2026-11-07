# Next experiments

The immediate goal is to generate coherent, clearly formatted slang entries and learn
which training choices help. Keep the existing 0.8B checkpoint as a baseline.

- [ ] Create a fixed evaluation set from held-out entries. Use consistent prompts and
  random seeds for comparisons, and select a smaller set for blinded human review.
- [ ] Clean training records: remove Urban Dictionary bracket markup, preserve word,
  definition, example, and year fields, and filter empty or obviously broken entries.
- [ ] Select a smaller pretrained causal language model for faster iteration. Benchmark
  training and generation on Apple Silicon before committing to a full run.
- [ ] Train it to complete a structured entry from a year prompt, applying loss to the
  completion. Try Hugging Face SFTTrainer if it works cleanly on MPS; otherwise use the
  existing PyTorch approach. Begin with a representative subset to validate the pipeline,
  then expand to the full training split.
- [ ] Compare uniform sampling with a separate run that favors stronger vote signals.
  Retain some of the long tail: votes are an imperfect proxy for quality and may reflect
  entry age and visibility.
- [ ] Compare against the existing model using held-out loss and generated examples.
  Review formatting, coherence, novelty, and slang-like quality, and test whether changing
  or removing the year meaningfully changes results.

Treat outputs as speculative inventions. Pretraining data may overlap the held-out years,
so improved generation alone does not establish an ability to forecast slang.
