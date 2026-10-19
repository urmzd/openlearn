# ML Infrastructure Patterns

Production ML infrastructure patterns that appear across top-tier AI company interviews.

## MLOps Maturity Levels

| Level | Description | Characteristics |
|-------|-------------|----------------|
| 0 | Manual | Jupyter notebooks, manual deployment, no monitoring |
| 1 | ML Pipeline | Automated training, basic serving, manual triggers |
| 2 | CI/CD for ML | Automated retraining, A/B testing, model validation gates |
| 3 | Full MLOps | Automated everything, drift detection, self-healing |

Most companies are between Level 1 and 2. Aim for Level 2 in your designs.

## Feature Stores

### Why Feature Stores Exist

Without a feature store:
- Training and serving use different code paths -> training/serving skew
- Features computed redundantly across teams
- Point-in-time correctness is hard to get right
- No discovery: "Does anyone already compute this feature?"

### Architecture

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

### Point-in-Time Correctness

The most critical concept for ML feature stores:

```
Training Example:
  Label: "User purchased item X on March 15"
  Features must be: "What we knew about the user BEFORE March 15"

WRONG: Using features computed on March 20 (data leakage)
RIGHT: Using features as of March 14 23:59:59
```

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

### Tools
- **Feast**: Open-source feature store
- **Tecton**: Managed feature platform
- **Hopsworks**: Feature store + ML platform
- **Vertex AI Feature Store**: Google Cloud managed
- **SageMaker Feature Store**: AWS managed

## Model Registry

### What It Stores

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

### Model Promotion Pipeline

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

## Training Infrastructure

### Single-Node Training

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

### Distributed Data Parallel (DDP)

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

### Communication Primitives (NCCL)

| Operation | Description | Use Case |
|-----------|-------------|----------|
| AllReduce | Sum/avg across all GPUs, result on all | Gradient synchronization |
| Broadcast | One GPU sends to all | Model initialization |
| AllGather | Each GPU contributes, all get full data | Gather predictions |
| ReduceScatter | Reduce + scatter result | ZeRO optimizer |
| All-to-All | Each GPU sends different data to each GPU | Expert parallelism (MoE) |

### Checkpointing

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

Best practices:
- Checkpoint every N steps (not just every epoch)
- Use async checkpointing (don't block training)
- Store to distributed storage (S3, GCS) not local disk
- Keep last K checkpoints, prune old ones

## Model Serving Infrastructure

### Serving Topology

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

### Key Serving Concepts

#### Dynamic Batching
Collect individual requests into batches for GPU efficiency:

```
Time -->
  t=0: Request A arrives, buffer
  t=1: Request B arrives, buffer
  t=3: Request C arrives, buffer
  t=5: Batch timeout -> send [A, B, C] to GPU as one batch
  t=6: Results returned, dispatch to individual clients
```

#### Model Warm-Up
Before serving traffic:
1. Load model weights to GPU
2. Run N dummy inferences to warm up CUDA kernels
3. Mark instance as "ready" in service discovery
4. Start receiving traffic

#### Multi-Model Serving
On a single GPU:
- **Time-multiplexing**: Swap models in/out of GPU memory
- **Memory-multiplexing**: Multiple small models loaded simultaneously
- **MIG (Multi-Instance GPU)**: Hardware partitioning (NVIDIA A100/H100)

### Inference Optimization

#### Quantization

```
FP32:  32 bits per weight -> Baseline
FP16:  16 bits per weight -> 2x memory reduction, ~2x speed
INT8:   8 bits per weight -> 4x memory reduction, ~4x speed
INT4:   4 bits per weight -> 8x memory reduction, ~6x speed

Quality impact: FP32 ≈ FP16 > INT8 >> INT4
```

#### TensorRT Optimization Pipeline

```
[ONNX Model] --> [Layer Fusion] --> [Precision Calibration] --> [Kernel Auto-Tuning]
                                                                        |
                                                                [Optimized Engine]
```

## Experiment Tracking & A/B Testing

### Experiment Lifecycle

```
1. Hypothesis: "Model B will improve click-through rate by 5%"
2. Design: Sample size, duration, metrics, guardrails
3. Implementation: Feature flag, traffic splitting
4. Analysis: Statistical significance, practical significance
5. Decision: Ship, iterate, or kill
```

### Statistical Considerations

- **Sample size**: Calculate required sample for desired statistical power
- **Multiple testing**: Bonferroni correction when testing multiple metrics
- **Sequential testing**: Stop early if results are clear (saves time)
- **Novelty/primacy effects**: Users may behave differently initially

### Guardrail Metrics

Metrics that must NOT degrade, regardless of the experiment:

| Metric | Type | Threshold |
|--------|------|-----------|
| Latency p99 | Performance | < 500ms |
| Error rate | Reliability | < 0.1% |
| Revenue per user | Business | No decrease > 1% |
| Safety violations | Safety | No increase |

If any guardrail is breached, the experiment is automatically stopped.

## Model Monitoring

### What to Monitor

```
[Request] --> [Input Validation] --> [Feature Distribution Check]
                                            |
                                     [Model Inference]
                                            |
                                     [Output Distribution Check]
                                            |
                                     [Response]
```

### Types of Drift

| Type | What Changes | Detection |
|------|-------------|-----------|
| Data drift | Input feature distributions | KS test, PSI, KL divergence |
| Concept drift | Relationship between features and target | Monitor prediction accuracy |
| Model drift | Model performance over time | Track production metrics vs. baseline |
| Feature drift | Feature computation changes | Compare feature distributions |

### Monitoring Implementation

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

## Retraining Pipelines

### Triggers for Retraining

| Trigger | Description | Example |
|---------|-------------|---------|
| Scheduled | Fixed interval | Retrain weekly |
| Drift-based | When drift detected | PSI > 0.2 triggers retrain |
| Performance-based | When accuracy drops | F1 drops below 0.85 |
| Data-based | When new labeled data available | 10K new labels collected |
| Manual | Human-triggered | After fixing a data quality issue |

### Automated Retraining Pipeline

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

## LLM-Specific Infrastructure

### RAG (Retrieval-Augmented Generation)

```
[User Query] --> [Embedding Model] --> [Vector Search]
                                            |
                                    [Top-K Documents]
                                            |
                                    [Context Assembly]
                                            |
                                    [LLM with Context] --> [Response]
```

Key decisions:
- **Chunk size**: 256-1024 tokens per chunk (trade-off: specificity vs. context)
- **Overlap**: 10-20% overlap between chunks (preserves context at boundaries)
- **Embedding model**: All-MiniLM, text-embedding-ada-002, E5-large
- **Vector DB**: Pinecone, Weaviate, Milvus, pgvector, Qdrant

### Fine-Tuning Infrastructure

| Approach | Data Needed | Cost | When to Use |
|----------|-------------|------|-------------|
| Prompt engineering | 0 | Lowest | First approach, always try first |
| Few-shot examples | 10-50 | Low | Task-specific formatting |
| RAG | Documents | Medium | Knowledge-intensive tasks |
| Fine-tuning (LoRA) | 100-10K | Medium | Domain adaptation, style |
| Full fine-tuning | 10K-1M+ | High | Fundamental capability changes |

### LLM Evaluation

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

## Cost Optimization

### GPU Cost Reduction Strategies

| Strategy | Savings | Effort |
|----------|---------|--------|
| Spot/preemptible instances for training | 60-90% | Medium (need checkpointing) |
| Right-size GPU selection | 20-50% | Low |
| Quantization for inference | 50-75% | Medium |
| Dynamic batching | 30-50% throughput gain | Medium |
| Model distillation | 80%+ (smaller model) | High |
| Autoscaling (scale to zero off-peak) | 30-60% | Medium |
| Multi-model serving (share GPUs) | 40-60% | High |

### Cost Per Prediction

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

Always evaluate: Do you need the most expensive GPU, or can a cheaper one meet your latency SLO?
