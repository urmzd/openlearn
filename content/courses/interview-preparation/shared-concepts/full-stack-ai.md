# Full-stack AI engineering

A model is a small part of an AI system. Around it sit the pipelines that collect and validate data, the feature store that serves inputs consistently to training and inference, the training and evaluation machinery, the servers that batch requests onto GPUs, the application and API that users touch, and the monitoring that tells you when the world has drifted away from what the model learned. A full-stack AI engineer can work at every one of those layers, and interviews for AI roles increasingly probe all of them in a single design conversation.

This section walks the stack from bottom to top, in the order data flows through it:

```
[Data Collection] --> [Data Processing] --> [Feature Engineering]
                                                    |
                                            [Model Training]
                                                    |
                                            [Evaluation & Validation]
                                                    |
                                            [Model Serving]
                                                    |
                                            [Monitoring & Feedback]
                                                    |
                              [User-Facing Application (Web/Mobile/API)]
```

For each layer it gives the patterns and trade-offs interviewers expect, with code for the pieces you may be asked to build: a data validator, an online feature store, an evaluator, a dynamically batching model server, and a drift detector. It assumes the general background in [Distributed systems fundamentals](distributed-systems.md); the training side is covered in more depth in [ML infrastructure patterns](ml-infrastructure.md).

> **Learning objectives**
> - Choose a data pipeline pattern and validation checks for a given ML workload.
> - Explain online and offline feature stores and why point-in-time correctness matters for training.
> - Pick distributed training and hyperparameter search strategies, and offline and online evaluation methods.
> - Implement dynamic batching for a model server and explain its latency-throughput trade-off.
> - Design an ML API and a monitoring plan, including drift detection.

## The data layer

Everything starts with data, and the first decision is how it moves. The common pipeline patterns differ in latency:

| Pattern | Use Case | Tools |
|---------|----------|-------|
| Batch ETL | Daily model retraining | Spark, Airflow, dbt |
| Stream processing | Real-time features | Kafka, Flink, Kinesis |
| Change Data Capture | Database -> feature store sync | Debezium, DynamoDB Streams |
| Lambda architecture | Combined batch + real-time | Batch layer + speed layer |

Whatever the pipeline, bad data silently produces bad models, so validate it before training. A small framework is enough: register named checks, each a function that returns whether it passed and a human-readable detail, with a severity so that warnings can be distinguished from blocking errors. The check constructors below are closures over a column name, which keeps each check reusable.

```python
class DataValidator:
    """Schema and quality validation for ML datasets."""

    def __init__(self):
        self.checks = []

    def add_check(self, name: str, fn: callable, severity: str = "error"):
        self.checks.append((name, fn, severity))

    def validate(self, df) -> list[dict]:
        results = []
        for name, fn, severity in self.checks:
            passed, details = fn(df)
            results.append({
                "check": name,
                "passed": passed,
                "severity": severity,
                "details": details
            })
        return results

# Common checks
def no_nulls(column):
    def check(df):
        null_count = df[column].isnull().sum()
        return null_count == 0, f"{null_count} nulls in {column}"
    return check

def value_range(column, min_val, max_val):
    def check(df):
        out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
        return out_of_range == 0, f"{out_of_range} values out of [{min_val}, {max_val}]"
    return check

def no_duplicates(columns):
    def check(df):
        dup_count = df.duplicated(subset=columns).sum()
        return dup_count == 0, f"{dup_count} duplicate rows on {columns}"
    return check
```

Data also needs versioning, because a training run is only reproducible if its data version is pinned. DVC (Data Version Control) provides Git-like versioning for datasets, and table formats such as Delta Lake and Apache Iceberg provide time travel and versioning at the table level.

## Feature engineering

Raw data becomes model inputs through features, and the central piece of infrastructure is the feature store. A feature pipeline writes each feature to two places: an online store (Redis or DynamoDB) that serves the latest values with low latency for real-time inference, and an offline store (S3 or BigQuery) that keeps history for building training sets.

```
[Raw Data] --> [Feature Pipeline] --> [Feature Store]
                                          |
                            +-------------+-------------+
                            |                           |
                     [Online Store]              [Offline Store]
                     (Redis/DynamoDB)            (S3/BigQuery)
                            |                           |
                     [Real-time serving]         [Training data]
```

Computing both from one pipeline is what prevents training-serving skew, where a feature is computed one way for training and another way in production. Features differ in how fresh they must be:

| Type | Example | Latency | Update Frequency |
|------|---------|---------|-----------------|
| Static | User age, location | N/A | Rarely |
| Batch | 30-day purchase count | Hours | Daily |
| Near-real-time | Last 5 items viewed | Seconds | On event |
| Real-time | Current session duration | Milliseconds | Continuous |

A simplified online feature store fits in a page. Each entity maps feature names to a value with a timestamp and version. `get_features` serves inference, returning `None` for missing features. `get_training_data` builds training rows and takes an `as_of` time, so that a training example never sees a feature value recorded after the moment it describes.

```python
from dataclasses import dataclass
from typing import Any
import time

@dataclass
class Feature:
    name: str
    value: Any
    timestamp: float
    version: int

class FeatureStore:
    """Simplified online feature store."""

    def __init__(self):
        self.store: dict[str, dict[str, Feature]] = {}  # entity_id -> {feature_name: Feature}

    def set_feature(self, entity_id: str, name: str, value: Any, version: int = 1):
        if entity_id not in self.store:
            self.store[entity_id] = {}
        self.store[entity_id][name] = Feature(name, value, time.time(), version)

    def get_features(self, entity_id: str, feature_names: list[str]) -> dict[str, Any]:
        """Get multiple features for an entity. Returns dict of name -> value."""
        entity = self.store.get(entity_id, {})
        return {
            name: entity[name].value if name in entity else None
            for name in feature_names
        }

    def get_training_data(self, entity_ids: list[str], feature_names: list[str],
                          as_of: float = None) -> list[dict]:
        """Point-in-time correct feature retrieval for training."""
        rows = []
        for eid in entity_ids:
            entity = self.store.get(eid, {})
            row = {"entity_id": eid}
            for name in feature_names:
                if name in entity:
                    feat = entity[name]
                    if as_of is None or feat.timestamp <= as_of:
                        row[name] = feat.value
                    else:
                        row[name] = None
                else:
                    row[name] = None
            rows.append(row)
        return rows
```

> **Warning:** Because this store keeps only the latest value of each feature, `get_training_data` can only avoid leakage, not achieve full point-in-time correctness: if a feature was updated after `as_of`, it returns `None` instead of the value that was current at `as_of`. A real offline store keeps the full history and does an "as of" join, picking for each example the latest value with a timestamp at or before the example's time. Being able to explain that difference is the point of the question.

## Model training

With features in hand, training follows a pipeline from data loading and preprocessing through training, hyperparameter search, and evaluation, ending in a model registry that versions the result:

```
[Data Loader] --> [Preprocessing] --> [Model Training]
                                          |
                                    [Hyperparameter Search]
                                          |
                                    [Evaluation]
                                          |
                                    [Model Registry]
```

Every training run should be tracked, recording its hyperparameters, dataset version, metrics (loss, accuracy, and custom metrics), model artifacts, code version (the git hash), and environment (library versions). MLflow, Weights & Biases, Neptune, and Comet are the common tools.

When a model or dataset outgrows one GPU, training is distributed in one of four ways:

| Pattern | What's Distributed | Use When |
|---------|-------------------|----------|
| Data Parallel | Training data across GPUs | Model fits on one GPU |
| Model Parallel | Model layers across GPUs | Model too large for one GPU |
| Pipeline Parallel | Model stages as pipeline | Very deep models |
| ZeRO | Optimizer state, gradients, parameters | Memory-efficient data parallel |

Hyperparameter search has its own trade-offs. Random search is often better than grid search when there are many parameters, because it does not waste trials on unimportant dimensions:

| Strategy | Description | When to Use |
|----------|-------------|------------|
| Grid Search | Exhaustive search over combinations | Few parameters, small search space |
| Random Search | Random sampling | Many parameters (often better than grid) |
| Bayesian Optimization | Model the objective, sample promising points | Expensive evaluations |
| Population-Based Training | Evolutionary approach across workers | Very large scale |

## Model evaluation

Before a model ships it is evaluated offline, on held-out data. The evaluator below computes weighted precision, recall, and F1 for classification (weighted by class support, so imbalanced classes are accounted for), plus a one-vs-rest AUC when probabilities are available, and MSE, MAE, and $R^2$ for regression. $R^2 = 1 - \sum (y - \hat y)^2 / \sum (y - \bar y)^2$ measures how much better the model does than always predicting the mean.

```python
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_classification(self, y_true, y_pred, y_prob=None):
        self.metrics = {
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted"),
        }
        if y_prob is not None:
            self.metrics["auc_roc"] = self._auc(y_true, y_prob)
        return self.metrics

    def evaluate_regression(self, y_true, y_pred):
        self.metrics = {
            "mse": np.mean((y_true - y_pred) ** 2),
            "mae": np.mean(np.abs(y_true - y_pred)),
            "r2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        }
        return self.metrics

    def _auc(self, y_true, y_prob):
        # Simplified AUC calculation
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
```

Offline metrics only predict online behaviour, so the model is then evaluated on real traffic, carefully. In shadow mode the new model runs alongside production and its outputs are compared without being served. A canary deployment serves 1-5% of traffic with the new model. Interleaving mixes results from the old and new models in the same response, which is a sensitive test for ranking. And throughout, guardrail metrics for safety and quality are monitored and must not degrade.

## Model serving

Serving patterns are defined by their latency budget:

| Pattern | Latency | Throughput | Use Case |
|---------|---------|------------|----------|
| Real-time (online) | < 100ms | Per-request | API calls, search ranking |
| Near-real-time | < 1s | Micro-batch | Content moderation |
| Batch | Minutes-hours | Very high | Recommendations, reports |
| Streaming | Continuous | High | Anomaly detection, monitoring |

For real-time serving on GPUs, the key technique is dynamic batching. A GPU processes a batch of 32 inputs in little more time than one, so a server that groups concurrent requests gets far more throughput. The listing below shows how. Each call to `predict` enqueues its input with a future and awaits the future. A background loop waits for at least one request, then keeps collecting until it has `max_batch_size` requests or `max_wait_ms` has elapsed since the first, runs one batched inference, and resolves each caller's future with its own result, or with the exception if the batch failed.

```python
from abc import ABC, abstractmethod
from typing import Any
import asyncio

class ModelServer(ABC):
    """Abstract model serving interface."""

    @abstractmethod
    async def predict(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    async def health(self) -> dict:
        pass

class BatchingModelServer(ModelServer):
    """Server with dynamic batching for GPU efficiency."""

    def __init__(self, model, max_batch_size: int = 32, max_wait_ms: float = 10.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def predict(self, input_data: Any) -> Any:
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((input_data, future))
        return await future

    async def _batch_loop(self):
        while self._running:
            batch_inputs = []
            batch_futures = []

            # Wait for at least one item
            input_data, future = await self.queue.get()
            batch_inputs.append(input_data)
            batch_futures.append(future)

            # Collect more items up to batch size or timeout
            deadline = asyncio.get_event_loop().time() + self.max_wait_ms / 1000
            while len(batch_inputs) < self.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    input_data, future = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining
                    )
                    batch_inputs.append(input_data)
                    batch_futures.append(future)
                except asyncio.TimeoutError:
                    break

            # Run batch inference
            try:
                results = self.model.predict_batch(batch_inputs)
                for future, result in zip(batch_futures, results):
                    future.set_result(result)
            except Exception as e:
                for future in batch_futures:
                    future.set_exception(e)

    async def start(self):
        self._running = True
        asyncio.create_task(self._batch_loop())

    async def health(self) -> dict:
        return {"status": "healthy", "queue_size": self.queue.qsize()}
```

`max_wait_ms` is the knob that trades latency for throughput: under light load a request waits at most 10 ms for company, and under heavy load batches fill instantly and no one waits. Two production details to mention: `model.predict_batch` is called synchronously, so it blocks the event loop while the GPU works (run it in an executor), and inside coroutines `asyncio.get_running_loop()` is the modern replacement for `get_event_loop()`. LLM servers go a step further with continuous batching, adding and removing sequences at every decoding step.

Batching is one of several ways to make serving cheaper. Roughly what each buys, and costs:

| Technique | Speed Improvement | Quality Impact |
|-----------|-------------------|----------------|
| Quantization (FP16 -> INT8) | 2-4x | Minor (<1% accuracy loss) |
| Knowledge Distillation | 5-10x (smaller model) | Moderate (1-5% loss) |
| Pruning | 2-5x | Minor-moderate |
| ONNX Runtime | 1.5-3x | None |
| TensorRT | 2-5x | None-minor |
| Batching | 3-10x throughput | None |

## The application layer

At the top of the stack, AI products have developed their own UI patterns:

| Pattern | Description | Example |
|---------|-------------|---------|
| Streaming response | Show results as they generate | ChatGPT, Claude |
| Progressive disclosure | Show confidence levels | Search suggestions |
| Human-in-the-loop | Require approval for actions | GitHub Copilot suggestions |
| Feedback collection | Thumbs up/down on outputs | Training data for improvement |
| Explanation | Show reasoning or sources | RAG with citations |

The API behind them should follow five principles. Make it asynchronous by default, because inference can be slow. Include request IDs for tracing. Return confidence scores, not just labels. Support streaming for long-running inference. And version your models in the API, so that clients choose when to move. A request and response for a text classifier look like this:

```python
# Good ML API design principles

# 1. Async by default (inference can be slow)
# 2. Include request IDs for tracing
# 3. Return confidence scores
# 4. Support streaming for long-running inference
# 5. Version your models in the API

# POST /v1/predict
{
    "model": "text-classifier-v2",
    "input": {"text": "This movie was great!"},
    "parameters": {
        "temperature": 0.0,
        "top_k": 5
    }
}

# Response
{
    "request_id": "req_abc123",
    "model": "text-classifier-v2",
    "output": {
        "label": "positive",
        "confidence": 0.94,
        "alternatives": [
            {"label": "neutral", "confidence": 0.04},
            {"label": "negative", "confidence": 0.02}
        ]
    },
    "usage": {
        "input_tokens": 12,
        "processing_time_ms": 45
    }
}
```

## Monitoring and drift

The loop closes with monitoring. An ML system needs the usual service metrics plus two kinds that ordinary services lack, model quality and data quality:

| Category | Metrics | Alerting Threshold |
|----------|---------|-------------------|
| Latency | p50, p95, p99 inference time | p99 > 2x baseline |
| Throughput | Requests/sec, tokens/sec | Drops below expected |
| Errors | Error rate by type | > 0.1% |
| Model quality | Prediction distribution shift | KL divergence > threshold |
| Data quality | Feature null rates, value distributions | Out of expected range |
| Resources | GPU utilization, memory, CPU | > 90% sustained |

Model quality is the hard one, because labels usually arrive late or never. A practical proxy is drift in the distribution of predictions. The detector below records a reference histogram from validation predictions, keeps a sliding window of recent predictions, and once the window is half full compares the two with the KL divergence $D_{KL}(P_\text{ref} \,\|\, P_\text{cur}) = \sum_i p_i \log (p_i / q_i)$, flagging drift above a threshold. The small constant added to each bin avoids dividing by or taking the log of zero.

```python
import numpy as np
from collections import deque

class DriftDetector:
    """Detect distribution shift in model predictions."""

    def __init__(self, window_size: int = 1000, threshold: float = 0.1):
        self.reference_distribution = None
        self.window = deque(maxlen=window_size)
        self.threshold = threshold

    def set_reference(self, predictions: list[float]):
        """Set the reference distribution from validation data."""
        self.reference_distribution = np.histogram(predictions, bins=50, density=True)

    def observe(self, prediction: float) -> bool:
        """Observe a new prediction. Returns True if drift detected."""
        self.window.append(prediction)

        if len(self.window) < self.window.maxlen // 2:
            return False  # Not enough data

        current_hist = np.histogram(list(self.window),
                                     bins=self.reference_distribution[1],
                                     density=True)

        # KL divergence
        ref = self.reference_distribution[0] + 1e-10
        cur = current_hist[0] + 1e-10
        kl_div = np.sum(ref * np.log(ref / cur))

        return kl_div > self.threshold
```

Two refinements are worth raising. With `density=True` the histogram values are densities, not probabilities, so the sum is scaled by the inverse bin width; normalising the counts to sum to one makes the threshold independent of the bins. And because the current histogram reuses the reference bin edges, predictions outside the reference range are silently dropped, which is exactly the kind of drift you most want to catch. Recomputing the histogram on every observation is also costly; checking every few hundred observations is usually enough. The [ML infrastructure patterns](ml-infrastructure.md) section covers PSI and the Kolmogorov-Smirnov test as alternatives.

## Numbers for system design

Latency estimates for an ML system draw on a few component figures:

| Component | Latency | Notes |
|-----------|---------|-------|
| Feature store lookup (online) | 1-10ms | Redis/DynamoDB |
| Embedding search (ANN) | 5-50ms | Depends on index size |
| Classification model inference | 5-50ms | Depends on model size |
| LLM inference (first token) | 100-500ms | Depends on input length |
| LLM inference (per token) | 10-30ms | After first token |
| Image model inference | 50-200ms | ResNet/ViT class |
| Batch training job | Minutes-days | Depends on data/model size |

## Where this leads

This material comes up across the companies in this book, each with its own emphasis. At [Google](../google/README.md) it is recommendation system design and ML serving platforms; at [OpenAI](../openai/README.md) and [Anthropic](../anthropic/README.md), LLM inference and training infrastructure; at [Netflix](../netflix/README.md), recommendation ML and A/B testing platforms; at [Amazon](../amazon/README.md), SageMaker-like platform design and product ranking; at [Apple](../apple/README.md), on-device and privacy-preserving ML; at [NVIDIA](../nvidia/README.md), GPU-optimised training and inference; at [Palantir](../palantir/README.md), enterprise ML platforms and data pipelines; and at [Jane Street](../jane-street/README.md), signal generation and online learning. The next section, [ML infrastructure patterns](ml-infrastructure.md), goes deeper into training infrastructure, experiment tracking, and drift detection.

### Further reading

- Chip Huyen, *Designing Machine Learning Systems*, for the end-to-end view this section summarises.
- D. Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015), on why the model is the smallest part of the system.

## Key terms

- **Feature store**: a system that computes features once and serves them both online, for inference, and offline, for training.
- **Training-serving skew**: a mismatch between how features are computed for training and for production.
- **Point-in-time correctness**: building each training example only from feature values that existed at the example's timestamp.
- **Dynamic batching**: grouping concurrent inference requests into one batch, bounded by a size and a maximum wait.
- **Shadow mode**: running a candidate model on live traffic without serving its outputs.
- **Drift**: a change in the distribution of inputs or predictions relative to what the model was trained on.

## Summary

- An AI system is a pipeline from data to monitoring; the model is one stage of it.
- Validate and version data, and serve features from one pipeline to avoid training-serving skew and leakage.
- Evaluate offline with metrics suited to the task, then online with shadow mode, canaries, interleaving, and guardrails.
- Dynamic batching trades a bounded wait for large GPU throughput gains; quantisation, distillation, and compilers cut cost further.
- Monitor model and data quality as well as service health, using prediction drift as a proxy when labels are late.

## Review questions

1. Why does a feature store have both an online and an offline store, rather than one?

   <details><summary>Answer</summary>

   Inference needs the latest value of a few features for one entity within milliseconds, which suits a key-value store such as Redis. Training needs historical values of many features for millions of entities, which suits a warehouse or object store. Writing both from the same pipeline keeps the two consistent and prevents training-serving skew.

   </details>

2. What is label leakage through features, and how does an `as_of` timestamp prevent it?

   <details><summary>Answer</summary>

   Leakage happens when a training example includes a feature value recorded after the event it is meant to predict, so the model learns from information it will not have at inference time. Restricting each example to values with timestamps at or before its `as_of` time prevents it; a full implementation picks the latest such value from the feature's history.

   </details>

3. In the batching server, what happens to latency and throughput as `max_wait_ms` increases?

   <details><summary>Answer</summary>

   Under light load, larger waits let more requests join each batch, raising throughput but adding up to `max_wait_ms` of latency to each request. Under heavy load batches fill to `max_batch_size` before the deadline, so the wait rarely matters. The setting is a cap on the latency you are willing to pay for batching.

   </details>

4. Why monitor the distribution of predictions rather than accuracy directly?

   <details><summary>Answer</summary>

   Accuracy needs ground-truth labels, which often arrive days later or never. The distribution of predictions is available immediately, and a significant shift in it is an early signal that inputs have drifted or the model is behaving differently.

   </details>
