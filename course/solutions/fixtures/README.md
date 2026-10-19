# Fixtures

Small data files that the solution tests read. Tests find them through the
`TINYLLM_FIXTURES` environment variable, so run them with
`TINYLLM_FIXTURES=course/solutions/fixtures` (relative to the repository root).

| File | What it is | Used by |
|---|---|---|
| `craft.05/oracles.npz` | Reference forward values and autograd gradients from torch for the craft.05 kernels (`linear`, `linear_sq`, `rmsnorm`, `swiglu`, `attn`, `attn_causal`), plus a `__meta__` entry. The generator script is not included. | `primers/craft.05/test_oracles.py` |
| `craft.22/tinylm.npz` | Weights of the two-layer Llama-style tiny model trained on synthetic stories: embeddings, per-layer attention and SwiGLU weights, norm gains, and the character BPE (`chars`, `merges`). The training script is not included. | `primers/craft.22/tinymodel.py` |
| `craft.22/val.txt` | Held-out synthetic story text for the bits-per-byte eval. | `primers/craft.22/evals.py`, `test_model_evals.py` |
| `craft.22/words.txt` | The story vocabulary, one word per line, for the sample-validity eval. | `primers/craft.22/evals.py`, `test_model_evals.py` |
| `ethics.04/lexicon.json` | A small hand-written toxicity lexicon: phrase to weight in [0, 1], combined by noisy-OR. | `tinyllm.eval.safety` |
| `ethics.04/labelled.jsonl` | Texts labelled toxic (1) or not (0), for measuring the lexicon scorer itself. | `tinyllm.eval.safety` |
| `ethics.04/prompts.jsonl` | Prompts with a `should_refuse` flag, for the refusal eval. | `tinyllm.eval.safety` |
| `ethics.04/bias.json` | Bias probes: templates with a `{group}` slot, term pairs per axis, and CrowS-Pairs style sentence pairs. | `tinyllm.eval.bias` |
| `L1.5/golden.jsonl` | Golden encodings for the fast Rust BPE property tests. | `learner-tests/L1.5` |
| `L7.9/` | Tiny Llama, Mistral and Qwen2-style checkpoints with reference logits and greedy outputs, for the model-family loading tests. | `learner-tests/L7.9` |
| `tok-gpt2/`, `tok-smollm2/` | The public GPT-2 and SmolLM2 `tokenizer.json` files (with their configs), used to check byte-level BPE against real vocabularies. | `learner-tests/data.07`, `learner-tests/L1.5` |
| `craft.13/` | The worked v1 and v2 KV-format files used by the migration tests. | `rust/crates/tl-kv-format/tests` |
