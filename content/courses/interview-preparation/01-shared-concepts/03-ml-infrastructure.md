# ML infrastructure patterns

Training a good model once is a research result. Training it again next week on fresh data, proving it is better than the one in production, rolling it out without an outage, noticing when it starts to degrade, and doing all of that at a cost the business can afford is infrastructure. Interviews at AI companies probe this second kind of work constantly, because it is where most of the engineering effort in production ML actually goes.

This section collects the production ML infrastructure patterns that recur across those interviews. It begins with a maturity model that tells you what level of automation to aim for in a design, then works through the pieces in lifecycle order: feature stores and point-in-time correctness, the model registry and promotion pipeline, training infrastructure from a single GPU to distributed data parallel, serving infrastructure, experimentation, monitoring and drift, retraining, LLM-specific infrastructure, and cost. It builds directly on [Full-stack AI engineering](02-full-stack-ai.md), which introduces the stack layer by layer, and on [Distributed systems fundamentals](01-distributed-systems.md).

> **Learning objectives**
> - Place a design on the MLOps maturity scale and explain what moving up a level requires.
> - Explain point-in-time correctness and design a model promotion pipeline with validation gates.
> - Write a DDP training loop, name the NCCL collectives and their uses, and checkpoint correctly.
> - Detect drift with the KS test and PSI, and choose retraining triggers.
> - Choose between prompting, RAG, and fine-tuning, and compute cost per prediction for alternative GPUs.

## MLOps maturity

A useful way to frame any ML infrastructure design is by how much of the lifecycle is automated:

| Level | Description | Characteristics |
|-------|-------------|----------------|
| 0 | Manual | Jupyter notebooks, manual deployment, no monitoring |
| 1 | ML Pipeline | Automated training, basic serving, manual triggers |
| 2 | CI/CD for ML | Automated retraining, A/B testing, model validation gates |
| 3 | Full MLOps | Automated everything, drift detection, self-healing |

Most companies are between levels 1 and 2. In an interview, aim for level 2: automated retraining with validation gates and experimentation, which is ambitious but credible. The rest of this section describes the components that level requires.

## Feature stores

Feature stores exist because, without one, four problems appear. Training and serving use different code paths, which produces training-serving skew. Teams compute the same features redundantly. Point-in-time correctness is hard to get right. And nobody can discover whether a feature already exists. A feature store addresses all four with a shared registry and one set of pipelines writing to both an online and an offline store:

```
[Data Sources] --> [Feature Pipelines (Batch/Stream)]
                            |
                   [Feature Registry]
                            |
              +-------------+-------------+
              |                           |
       [Online Store]              [Offline Store]
       (Redis, DynamoDB)           (S3, BigQuery)
              |                           |
       [Serving API]              [Training SDK]
       (get_features())           (get_training_data())
```

### Point-in-time correctness

Point-in-time correctness is the most critical concept for feature stores. Every training example describes a moment, and its features must reflect only what was known at that moment:

```
Training Example:
  Label: "User purchased item X on March 15"
  Features must be: "What we knew about the user BEFORE March 15"

WRONG: Using features computed on March 20 (data leakage)
RIGHT: Using features as of March 14 23:59:59
```

Using the March 20 features leaks the future into training. The model looks excellent offline, because it has effectively seen the answer, and then disappoints in production, where that information is not available. The fix is to request features as of each event's timestamp:

```python
def get_training_features(events, feature_store):
    """Generate training data with point-in-time correctness."""
    training_data = []
    for event in events:
        features = feature_store.get_features(
            entity_id=event.user_id,
            feature_names=["purchase_count_30d", "avg_session_time"],
            as_of=event.timestamp  # Features as of BEFORE the event
        )
        features["label"] = event.label
        training_data.append(features)
    return training_data
```

The common tools are Feast (open source), Tecton (a managed feature platform), Hopsworks (a feature store plus ML platform), Vertex AI Feature Store (managed, on Google Cloud), and SageMaker Feature Store (managed, on AWS).

## The model registry

Once models are trained, they need a home that records not just the weights but everything needed to reproduce, serve, and audit them. A registry entry holds the artifact, the metadata of the run that produced it, its serving configuration, its lifecycle stage, and its lineage:

```
Model Registry Entry:
├── Model artifact (weights, binaries)
├── Model metadata
│   ├── Training dataset version
│   ├── Hyperparameters
│   ├── Training metrics
│   ├── Git commit hash
│   └── Framework version
├── Serving configuration
│   ├── Input/output schema
│   ├── Resource requirements (GPU type, memory)
│   └── Preprocessing/postprocessing code
├── Lifecycle stage
│   ├── Development
│   ├── Staging
│   ├── Production
│   └── Archived
└── Lineage
    ├── Parent model (if fine-tuned)
    ├── Training pipeline run ID
    └── Data source versions
```

The lifecycle stages define a promotion pipeline. A new model enters as Development. Automated validation checks that accuracy exceeds a threshold, latency is within the SLO, there are no safety regressions, and bias checks pass; a model that passes moves to Staging. There it is shadow-tested or canaried against the production model and monitored for 24-48 hours. Only then is it promoted to Production, and even then traffic moves gradually.

```
[Training] --> [Model Registry: "Development"]
                        |
                [Automated Validation]
                - Accuracy > threshold
                - Latency < SLO
                - No safety regressions
                - Bias checks pass
                        |
                [Model Registry: "Staging"]
                        |
                [Shadow Testing / Canary]
                - Compare against production model
                - Monitor for 24-48 hours
                        |
                [Model Registry: "Production"]
                        |
                [Gradual Rollout: 1% -> 10% -> 50% -> 100%]
```

## Training infrastructure

Training starts from the standard PyTorch loop on a single device: move a batch to the device, zero the gradients, run the forward pass, compute the loss, backpropagate, and step the optimiser.

```python
# Standard PyTorch training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Distributed data parallel

When one GPU is too slow, Distributed Data Parallel (DDP) runs one process per GPU, each holding a full copy of the model. A `DistributedSampler` gives each rank a disjoint shard of the data, and DDP all-reduces the gradients during the backward pass so that every replica takes the same optimiser step. The training loop itself is unchanged. Calling `sampler.set_epoch(epoch)` matters: without it, every epoch shuffles the shards identically.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_ddp(rank, world_size, model, dataset):
    setup(rank, world_size)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            # Standard training loop -- DDP handles gradient sync
            pass
```

The listing is a sketch: it assumes `torch`, `DataLoader`, and `num_epochs` are in scope, and the per-batch body is the single-device loop above.

### Communication primitives

DDP's gradient sync is one use of the collective operations that NCCL provides. Knowing which collective each form of parallelism needs is a common interview question:

| Operation | Description | Use Case |
|-----------|-------------|----------|
| AllReduce | Sum/avg across all GPUs, result on all | Gradient synchronization |
| Broadcast | One GPU sends to all | Model initialization |
| AllGather | Each GPU contributes, all get full data | Gather predictions |
| ReduceScatter | Reduce + scatter result | ZeRO optimizer |
| All-to-All | Each GPU sends different data to each GPU | Expert parallelism (MoE) |

### Checkpointing

Long training runs fail, so they checkpoint. A checkpoint must include the optimiser state as well as the model weights; resuming with a fresh optimiser loses Adam's moment estimates and destabilises training.

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

Four practices make checkpointing robust. Checkpoint every $N$ steps, not just every epoch, so a failure loses little work. Checkpoint asynchronously, so training is not blocked while the file is written. Store checkpoints on distributed storage such as S3 or GCS rather than local disk, which disappears with the machine. And keep the last $K$ checkpoints, pruning older ones.

## Serving infrastructure

On the serving side, requests pass through a load balancer and API gateway to a model router that decides which model version or variant handles each request, and then to a pool of inference servers. One pool can serve several models:

```
[Load Balancer]
       |
[API Gateway]
       |
[Model Router] --> Which model version/variant?
       |
[Inference Server Pool]
   |        |        |
[GPU 0]  [GPU 1]  [GPU 2]
   |        |        |
[Model A] [Model A] [Model B]  (multi-model serving)
```

Three serving concepts come up repeatedly. The first is dynamic batching, which collects individual requests into batches for GPU efficiency. On this timeline, three requests arriving over five time units are sent to the GPU as one batch when the timeout fires, and their results are dispatched back to the individual clients:

```
Time -->
  t=0: Request A arrives, buffer
  t=1: Request B arrives, buffer
  t=3: Request C arrives, buffer
  t=5: Batch timeout -> send [A, B, C] to GPU as one batch
  t=6: Results returned, dispatch to individual clients
```

The [Full-stack AI engineering](02-full-stack-ai.md) section implements this as an asyncio server.

The second is warm-up. Before an instance takes traffic, it loads the model weights to the GPU, runs $N$ dummy inferences so that CUDA kernels are compiled and caches populated, and only then marks itself ready in service discovery. Skipping this makes the first real requests very slow.

The third is multi-model serving on a single GPU, which comes in three forms: time-multiplexing, swapping models in and out of GPU memory; memory-multiplexing, loading several small models at once; and MIG (Multi-Instance GPU), hardware partitioning on NVIDIA A100 and H100.

### Inference optimisation

Quantisation is the most effective single optimisation. Each halving of precision halves memory and roughly doubles speed, at some cost in quality:

```
FP32:  32 bits per weight -> Baseline
FP16:  16 bits per weight -> 2x memory reduction, ~2x speed
INT8:   8 bits per weight -> 4x memory reduction, ~4x speed
INT4:   4 bits per weight -> 8x memory reduction, ~6x speed

Quality impact: FP32 ≈ FP16 > INT8 >> INT4
```

Compilers such as TensorRT combine several optimisations into one pipeline, taking an ONNX model through layer fusion, precision calibration, and kernel auto-tuning to produce an optimised engine:

```
[ONNX Model] --> [Layer Fusion] --> [Precision Calibration] --> [Kernel Auto-Tuning]
                                                                        |
                                                                [Optimized Engine]
```

## Experimentation

A new model is not better until an experiment says so. Every experiment follows the same lifecycle:

```
1. Hypothesis: "Model B will improve click-through rate by 5%"
2. Design: Sample size, duration, metrics, guardrails
3. Implementation: Feature flag, traffic splitting
4. Analysis: Statistical significance, practical significance
5. Decision: Ship, iterate, or kill
```

The statistics need care. Calculate the sample size required for the statistical power you want before starting. When testing several metrics, correct for multiple testing, for example with a Bonferroni correction. Sequential testing lets you stop early when results are clear, which saves time, but it must be designed in rather than done by peeking. And watch for novelty and primacy effects, since users often behave differently at first.

Every experiment also carries guardrail metrics that must not degrade, whatever the primary metric does:

| Metric | Type | Threshold |
|--------|------|-----------|
| Latency p99 | Performance | < 500ms |
| Error rate | Reliability | < 0.1% |
| Revenue per user | Business | No decrease > 1% |
| Safety violations | Safety | No increase |

If any guardrail is breached, the experiment stops automatically.

## Monitoring and drift

Once in production, a model is monitored at both ends of inference: inputs are validated and their feature distributions checked, and the distribution of outputs is checked after the model runs.

```
[Request] --> [Input Validation] --> [Feature Distribution Check]
                                            |
                                     [Model Inference]
                                            |
                                     [Output Distribution Check]
                                            |
                                     [Response]
```

What those checks look for is drift, which comes in four types:

| Type | What Changes | Detection |
|------|-------------|-----------|
| Data drift | Input feature distributions | KS test, PSI, KL divergence |
| Concept drift | Relationship between features and target | Monitor prediction accuracy |
| Model drift | Model performance over time | Track production metrics vs. baseline |
| Feature drift | Feature computation changes | Compare feature distributions |

Two detectors cover most needs. The two-sample Kolmogorov-Smirnov test compares a feature's current values with its reference values and reports the largest gap between their empirical CDFs; a p-value below the threshold (0.05 here) flags drift. The Population Stability Index (PSI) bins predictions and sums $(c_i - r_i)\ln(c_i / r_i)$ over the bins, where $r_i$ and $c_i$ are the reference and current proportions; by convention a PSI above 0.1 is a moderate shift and above 0.25 a major one.

```python
from scipy import stats
import numpy as np

class ProductionMonitor:
    def __init__(self, reference_data: dict[str, np.ndarray]):
        self.reference = reference_data
        self.alerts = []

    def check_feature_drift(self, feature_name: str, current_values: np.ndarray,
                             threshold: float = 0.05) -> dict:
        """Kolmogorov-Smirnov test for feature drift."""
        ref_values = self.reference[feature_name]
        statistic, p_value = stats.ks_2samp(ref_values, current_values)

        drifted = p_value < threshold
        result = {
            "feature": feature_name,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drifted": drifted
        }

        if drifted:
            self.alerts.append(result)
        return result

    def check_prediction_distribution(self, predictions: np.ndarray,
                                        reference_predictions: np.ndarray) -> dict:
        """Population Stability Index for prediction drift."""
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        ref_hist = np.histogram(reference_predictions, bins=bins, density=True)[0] + 1e-10
        cur_hist = np.histogram(predictions, bins=bins, density=True)[0] + 1e-10

        # PSI
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

        return {
            "psi": psi,
            "drifted": psi > 0.2,  # PSI > 0.2 indicates significant shift
            "severity": "high" if psi > 0.25 else "medium" if psi > 0.1 else "low"
        }
```

> **Warning:** PSI's conventional thresholds assume proportions that sum to one. With `density=True` and ten bins of width 0.1, each histogram value is ten times the bin's proportion, so this code reports ten times the standard PSI and its 0.1 / 0.2 / 0.25 thresholds fire far too easily. Dividing the raw counts by their total fixes it. Also note that with very large samples the KS test flags tiny, harmless differences as significant; teams often threshold the KS statistic itself rather than the p-value.

## Retraining

Drift is one reason to retrain. The common triggers are:

| Trigger | Description | Example |
|---------|-------------|---------|
| Scheduled | Fixed interval | Retrain weekly |
| Drift-based | When drift detected | PSI > 0.2 triggers retrain |
| Performance-based | When accuracy drops | F1 drops below 0.85 |
| Data-based | When new labeled data available | 10K new labels collected |
| Manual | Human-triggered | After fixing a data quality issue |

An automated retraining pipeline reuses the promotion machinery from earlier in the section. The evaluation gate is what keeps automation safe: a retrained model must match or beat the current production model, show no safety regressions, and stay within the latency SLO before it can even reach staging.

```
[Trigger] --> [Data Preparation]
                    |
              [Feature Computation]
                    |
              [Model Training]
                    |
              [Evaluation Gate]
              - Accuracy >= current production model
              - No safety regressions
              - Latency within SLO
                    |
              [Model Registry: Staging]
                    |
              [Shadow/Canary Deployment]
                    |
              [Promotion to Production]
```

## LLM-specific infrastructure

Large language models add infrastructure of their own, starting with retrieval-augmented generation (RAG). A query is embedded, the nearest document chunks are retrieved from a vector index, and they are assembled into the prompt as context:

```
[User Query] --> [Embedding Model] --> [Vector Search]
                                            |
                                    [Top-K Documents]
                                            |
                                    [Context Assembly]
                                            |
                                    [LLM with Context] --> [Response]
```

The key decisions are chunk size, typically 256-1024 tokens, trading specificity against context; overlap between chunks, typically 10-20%, which preserves context at chunk boundaries; the embedding model (All-MiniLM, text-embedding-ada-002, E5-large); and the vector database (Pinecone, Weaviate, Milvus, pgvector, Qdrant).

RAG is one of several ways to adapt an LLM to a task, and choosing among them is a frequent design question. Always try the cheapest first:

| Approach | Data Needed | Cost | When to Use |
|----------|-------------|------|-------------|
| Prompt engineering | 0 | Lowest | First approach, always try first |
| Few-shot examples | 10-50 | Low | Task-specific formatting |
| RAG | Documents | Medium | Knowledge-intensive tasks |
| Fine-tuning (LoRA) | 100-10K | Medium | Domain adaptation, style |
| Full fine-tuning | 10K-1M+ | High | Fundamental capability changes |

### Evaluating LLM outputs

LLM evaluation is harder than classification, because there is rarely one right answer. The evaluator below combines a reference-based metric with reference-free heuristics. ROUGE-L computes the longest common subsequence of words between the response and a reference by dynamic programming and reports the F1 of the resulting precision ($\text{LCS}/m$) and recall ($\text{LCS}/n$). The repetition score is the fraction of word trigrams that are repeats, 0 for none and approaching 1 for degenerate looping output. The coherence heuristic penalises high variance in sentence length.

```python
class LLMEvaluator:
    """Evaluate LLM outputs across multiple dimensions."""

    def evaluate(self, prompt: str, response: str, reference: str = None) -> dict:
        scores = {}

        # Automated metrics
        if reference:
            scores["rouge_l"] = self._rouge_l(response, reference)
            scores["exact_match"] = response.strip() == reference.strip()

        # Quality heuristics
        scores["length"] = len(response.split())
        scores["repetition"] = self._repetition_score(response)
        scores["coherence"] = self._coherence_heuristic(response)

        return scores

    def _rouge_l(self, hypothesis: str, reference: str) -> float:
        """Longest Common Subsequence based metric."""
        h_words = hypothesis.split()
        r_words = reference.split()

        # LCS length
        m, n = len(h_words), len(r_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if h_words[i-1] == r_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs = dp[m][n]
        precision = lcs / m if m > 0 else 0
        recall = lcs / n if n > 0 else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _repetition_score(self, text: str) -> float:
        """Detect repetitive content (lower is better)."""
        words = text.split()
        if len(words) < 10:
            return 0.0
        # Check for repeated n-grams
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1.0
        return 1.0 - unique_ratio  # 0 = no repetition, 1 = all repeated

    def _coherence_heuristic(self, text: str) -> float:
        """Simple coherence check based on sentence structure."""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 1.0
        # Check for sentences that are too short or too long
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        # Normalize: low variance = more coherent
        return max(0, 1.0 - variance / 100)
```

Be candid in an interview about what these numbers can and cannot tell you. ROUGE-L rewards word overlap, not correctness, and the coherence heuristic measures sentence-length regularity, not meaning. They are cheap regression signals; real LLM evaluation adds task-specific test sets, model-graded rubrics, and human review.

## Cost

GPUs are the dominant cost of ML infrastructure, and there are many ways to reduce it:

| Strategy | Savings | Effort |
|----------|---------|--------|
| Spot/preemptible instances for training | 60-90% | Medium (need checkpointing) |
| Right-size GPU selection | 20-50% | Low |
| Quantization for inference | 50-75% | Medium |
| Dynamic batching | 30-50% throughput gain | Medium |
| Model distillation | 80%+ (smaller model) | High |
| Autoscaling (scale to zero off-peak) | 30-60% | Medium |
| Multi-model serving (share GPUs) | 40-60% | High |

The number that ties these together is cost per prediction, the GPU's hourly cost divided by the predictions it serves per hour. A worked comparison shows why the most expensive GPU is not always the cheapest choice:

```
Cost per prediction = GPU cost per hour / predictions per hour

Example:
  H100: $8/hour
  Throughput: 1000 predictions/hour (with batching)
  Cost: $0.008 per prediction

  With INT8 quantization (2x throughput):
  Cost: $0.004 per prediction

  With A10G instead of H100 ($1/hour, 300 predictions/hour):
  Cost: $0.003 per prediction
```

The A10G serves less than a third of the H100's throughput, but at an eighth of the price it is cheaper per prediction ($1/300 \approx \$0.0033$). So always ask whether you need the most expensive GPU, or whether a cheaper one can meet your latency SLO.

## Where this leads

These patterns appear in the system design rounds of nearly every AI company in this book. [Google DeepMind](../02-big-tech-and-ai-labs/03-deepmind/README.md) and [NVIDIA](../02-big-tech-and-ai-labs/07-nvidia/README.md) push deepest on training infrastructure, [Netflix](../03-infrastructure-and-data-platforms/01-netflix/README.md) and [Meta](../02-big-tech-and-ai-labs/05-meta/README.md) on experimentation, and [OpenAI](../02-big-tech-and-ai-labs/04-openai/README.md) and [Anthropic](../02-big-tech-and-ai-labs/01-anthropic/README.md) on serving and LLM-specific infrastructure. For the systems vocabulary underneath, return to [Distributed systems fundamentals](01-distributed-systems.md).

### Further reading

- Google Cloud, "MLOps: Continuous delivery and automation pipelines in machine learning", the source of the maturity-level framing.
- Ron Kohavi, Diane Tang, and Ya Xu, *Trustworthy Online Controlled Experiments*, for the statistics of A/B testing.
- The PyTorch documentation on `DistributedDataParallel` and the NCCL collectives.

## Key terms

- **MLOps maturity**: how much of the ML lifecycle, from training to monitoring and retraining, is automated.
- **Model registry**: a versioned store of model artifacts with their metadata, serving configuration, lifecycle stage, and lineage.
- **Distributed Data Parallel (DDP)**: one model replica per GPU, each on a data shard, with gradients all-reduced every step.
- **AllReduce**: a collective that combines a value across all GPUs and leaves the result on every GPU.
- **Guardrail metric**: a metric an experiment must not degrade, whose breach stops the experiment.
- **Population Stability Index (PSI)**: a binned measure of how far one distribution has shifted from a reference.
- **Cost per prediction**: GPU cost per hour divided by predictions served per hour.

## Summary

- Aim designs at MLOps level 2: automated retraining with validation gates and experimentation.
- Point-in-time correct features and a registry with lineage make training reproducible and leakage-free.
- DDP plus NCCL AllReduce is the default for scaling training; checkpoint model and optimiser state often and asynchronously.
- Promote models through validation, shadow or canary, and gradual rollout, with guardrails that stop bad experiments.
- Monitor for drift with KS and PSI (on proportions), retrain on clear triggers, and judge hardware by cost per prediction.

## Review questions

1. A model scores far better offline than it does in production. What feature store failure would you suspect first, and why?

   <details><summary>Answer</summary>

   A point-in-time correctness failure. If training examples were built with feature values computed after the events they describe, the model learned from future information that is unavailable at inference time, which inflates offline metrics.

   </details>

2. Why must `sampler.set_epoch(epoch)` be called in a DDP training loop?

   <details><summary>Answer</summary>

   `DistributedSampler` seeds its shuffle from the epoch number. Without updating it, every epoch uses the same permutation, so each rank sees its data in the same order every epoch, which weakens the benefit of shuffling.

   </details>

3. Which NCCL collective does ZeRO rely on for gradients, and which does mixture-of-experts parallelism need?

   <details><summary>Answer</summary>

   ZeRO uses ReduceScatter, so each rank ends up with only the reduced slice of the gradients it owns. Mixture-of-experts needs All-to-All, because each GPU sends different tokens to the GPUs that hold their experts.

   </details>

4. An A10G at $1 per hour serves 300 predictions per hour; an H100 at $8 per hour serves 1000. Which is cheaper per prediction, and what else must you check before choosing it?

   <details><summary>Answer</summary>

   The A10G, at about $0.0033 per prediction against the H100's $0.008. You must also check that the A10G meets the latency SLO and that you can provision enough of them for peak throughput.

   </details>

5. When would you choose RAG over LoRA fine-tuning?

   <details><summary>Answer</summary>

   When the task is knowledge-intensive and the knowledge changes or must be cited: RAG retrieves current documents at query time without retraining. Fine-tuning suits changing the model's style or domain behaviour, which retrieval cannot provide.

   </details>
