# Full-Stack AI Engineering

End-to-end knowledge for building, deploying, and scaling AI systems -- from data to production.

## The Full-Stack AI Engineer's Domain

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

A full-stack AI engineer can work at every layer. This document covers what you need to know across all of them.

## Data Layer

### Data Pipelines

| Pattern | Use Case | Tools |
|---------|----------|-------|
| Batch ETL | Daily model retraining | Spark, Airflow, dbt |
| Stream processing | Real-time features | Kafka, Flink, Kinesis |
| Change Data Capture | Database -> feature store sync | Debezium, DynamoDB Streams |
| Lambda architecture | Combined batch + real-time | Batch layer + speed layer |

### Data Quality

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

### Data Versioning

- **DVC (Data Version Control)**: Git-like versioning for datasets
- **Delta Lake / Apache Iceberg**: Table format with time-travel and versioning
- **Importance**: Reproducibility of training runs requires pinned data versions

## Feature Engineering

### Feature Store Architecture

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

### Feature Types

| Type | Example | Latency | Update Frequency |
|------|---------|---------|-----------------|
| Static | User age, location | N/A | Rarely |
| Batch | 30-day purchase count | Hours | Daily |
| Near-real-time | Last 5 items viewed | Seconds | On event |
| Real-time | Current session duration | Milliseconds | Continuous |

### Feature Implementation

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

## Model Training

### Training Pipeline

```
[Data Loader] --> [Preprocessing] --> [Model Training]
                                          |
                                    [Hyperparameter Search]
                                          |
                                    [Evaluation]
                                          |
                                    [Model Registry]
```

### Experiment Tracking

Track every training run with:
- Hyperparameters
- Dataset version
- Metrics (loss, accuracy, custom metrics)
- Model artifacts
- Code version (git hash)
- Environment (library versions)

Tools: MLflow, Weights & Biases, Neptune, Comet

### Distributed Training Patterns

| Pattern | What's Distributed | Use When |
|---------|-------------------|----------|
| Data Parallel | Training data across GPUs | Model fits on one GPU |
| Model Parallel | Model layers across GPUs | Model too large for one GPU |
| Pipeline Parallel | Model stages as pipeline | Very deep models |
| ZeRO | Optimizer state, gradients, parameters | Memory-efficient data parallel |

### Hyperparameter Optimization

| Strategy | Description | When to Use |
|----------|-------------|------------|
| Grid Search | Exhaustive search over combinations | Few parameters, small search space |
| Random Search | Random sampling | Many parameters (often better than grid) |
| Bayesian Optimization | Model the objective, sample promising points | Expensive evaluations |
| Population-Based Training | Evolutionary approach across workers | Very large scale |

## Model Evaluation

### Offline Evaluation

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

### Online Evaluation (A/B Testing)

- **Shadow mode**: Run new model alongside production, compare outputs without serving
- **Canary deployment**: Serve 1-5% of traffic with new model
- **Interleaving**: Mix results from old and new model in the same response
- **Guardrail metrics**: Monitor safety/quality metrics that must not degrade

## Model Serving

### Serving Patterns

| Pattern | Latency | Throughput | Use Case |
|---------|---------|------------|----------|
| Real-time (online) | < 100ms | Per-request | API calls, search ranking |
| Near-real-time | < 1s | Micro-batch | Content moderation |
| Batch | Minutes-hours | Very high | Recommendations, reports |
| Streaming | Continuous | High | Anomaly detection, monitoring |

### Model Serving Architecture

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

### Model Optimization for Serving

| Technique | Speed Improvement | Quality Impact |
|-----------|-------------------|----------------|
| Quantization (FP16 -> INT8) | 2-4x | Minor (<1% accuracy loss) |
| Knowledge Distillation | 5-10x (smaller model) | Moderate (1-5% loss) |
| Pruning | 2-5x | Minor-moderate |
| ONNX Runtime | 1.5-3x | None |
| TensorRT | 2-5x | None-minor |
| Batching | 3-10x throughput | None |

## Frontend / Application Layer

### AI-Powered UI Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| Streaming response | Show results as they generate | ChatGPT, Claude |
| Progressive disclosure | Show confidence levels | Search suggestions |
| Human-in-the-loop | Require approval for actions | GitHub Copilot suggestions |
| Feedback collection | Thumbs up/down on outputs | Training data for improvement |
| Explanation | Show reasoning or sources | RAG with citations |

### API Design for ML Services

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

## Monitoring & Observability

### What to Monitor

| Category | Metrics | Alerting Threshold |
|----------|---------|-------------------|
| Latency | p50, p95, p99 inference time | p99 > 2x baseline |
| Throughput | Requests/sec, tokens/sec | Drops below expected |
| Errors | Error rate by type | > 0.1% |
| Model quality | Prediction distribution shift | KL divergence > threshold |
| Data quality | Feature null rates, value distributions | Out of expected range |
| Resources | GPU utilization, memory, CPU | > 90% sustained |

### Model Drift Detection

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

## Key Numbers for System Design

| Component | Latency | Notes |
|-----------|---------|-------|
| Feature store lookup (online) | 1-10ms | Redis/DynamoDB |
| Embedding search (ANN) | 5-50ms | Depends on index size |
| Classification model inference | 5-50ms | Depends on model size |
| LLM inference (first token) | 100-500ms | Depends on input length |
| LLM inference (per token) | 10-30ms | After first token |
| Image model inference | 50-200ms | ResNet/ViT class |
| Batch training job | Minutes-days | Depends on data/model size |

## Interview Application

This knowledge maps to interviews at:
- **Google**: Recommendation system design, ML serving platform
- **OpenAI/Anthropic**: LLM inference, training infrastructure
- **Netflix**: Recommendation ML, A/B testing platform
- **Amazon**: SageMaker-like platform design, product ranking
- **Apple**: On-device ML, privacy-preserving ML
- **NVIDIA**: GPU-optimized training/inference
- **Palantir**: Enterprise ML platform, data pipeline
- **Jane Street**: Signal generation, online learning
