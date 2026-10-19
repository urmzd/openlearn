# Third-party model card: SmolLM2-135M-Instruct

We did not train or fine-tune this model. It is third-party weights,
fetched at a pinned revision and served by our engine for the Pass 10 agent.
This card records where it comes from and what we have measured ourselves.

## Provenance

- **Upstream:** Hugging Face model `HuggingFaceTB/SmolLM2-135M-Instruct`,
  revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, by the Hugging Face
  SmolLM team; an instruction-tuned member of the SmolLM2 family.
- **Weights from:** the Hugging Face repository
  `HuggingFaceTB/SmolLM2-135M-Instruct` at the revision above, each file
  checked against a pinned hash after download; `model.safetensors` sha256
  5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c.
- **Architecture (from its config.json):** `LlamaForCausalLM`, 30 layers,
  hidden size 576, 9 attention heads with 3 KV heads, MLP width 1536,
  vocabulary 49,152, context 8,192, tied embeddings, bfloat16 weights;
  134,515,008 parameters by M05.1's param_count.

## License

- Apache-2.0, as declared by the upstream repository. We redistribute
  nothing: each deployment fetches the weights itself.

## Intended use

- The agent model behind our gateway in Pass 10: short tool-calling turns
  over the course documentation, with the usage policy (ethics.05) in front.
- Not for unsupervised users, medical, legal, or financial advice, or any
  decision about a person.

## Evaluation

- Not yet evaluated by this system. The ethics.04 safety and bias suites and
  the agent evals (ag.09 to ag.12) run against it in Pass 10; their
  rows will be added here with intervals. Upstream evaluations exist but
  were not reproduced by us, so we do not quote them.

## Limitations

- A 135M-parameter model: it states wrong facts confidently, loses track of
  long conversations, and follows instructions unreliably.
- Trained mostly on English web text by its authors; we have not audited its
  training data, so its biases are unknown to us until ethics.04 measures them.
