# Weight staging and deployment

This part of [Deploying LLMs on Kubernetes: GPUs, Replicas, and Rollouts](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Trace immutable model artifacts into a ready serving replica.

## Getting weights from Hugging Face onto the node

A replica cannot serve until its weights are in GPU memory, and on Kubernetes the dominant term is usually getting them onto the node. [Model Loading](../03-model-loading/README.md) gives cold start as roughly the checkpoint size divided by the slowest link, plus engine initialization. On a cluster there is a second cost: a scale-out event loads the same bytes into several pods at once, through a link they share.

The strategies, from worst to best for production:

| Strategy | How | Cold start (70 GB, FP8 70B) | Fails how |
|----------|-----|-----------------------------|-----------|
| Download at pod start | `hf download` in an init container, into an `emptyDir` | Minutes, bounded by the Hub and egress | Hub rate limits and outages become your outages; every replica pays it |
| Bake into the image | `COPY` weights into the container image | Image pull: minutes for a 70 GB layer, and the registry must store it | Every weight change is a 70 GB image push |
| Shared volume | A Job downloads once into a `ReadWriteMany` PVC (NFS, Filestore, EFS, CephFS); pods mount it read-only | Volume bandwidth, shared by every pod loading at once | A scale-out storm divides the bandwidth |
| Object-store mirror plus streaming | Mirror the pinned revision to S3 or GCS; the engine streams tensors directly (`--load-format runai_streamer`) | 30 to 90 s at several GB/s | Needs a fast bucket in the same region |
| Node-local cache | A DaemonSet or an init container copies to local NVMe once per node; pods mount it with `hostPath` or a local PV | About 10 s at 7 GB/s | Node replacement re-pays the copy |

Work the scale-out storm for the shared volume. Five replicas start at once; the NFS server sustains 1.5 GB/s in aggregate; each must read 70 GB. The total is 350 GB, so the last replica is ready after $350 / 1.5 \approx 233$ s, plus a minute of engine initialization, compared with about 10 s from local NVMe. If the autoscaler fires because the queue is already growing, that difference is four minutes of SLO breach.

Whichever strategy you pick, four rules hold:

1. **Pin the revision.** Download a 40-character commit sha, not `main`. A replica that starts tomorrow must load the bytes you load-tested today.
2. **Never contact the Hub from a serving pod.** Set `HF_HUB_OFFLINE=1` so that a missing file fails loudly at startup instead of triggering a download during a scale-out.
3. **Keep the token in a Secret.** Gated models such as Llama need an `HF_TOKEN` with read access, mounted into the download Job only, not into the serving pods.
4. **Download only what you serve.** A repo may hold the same weights in several formats; `--include` patterns keep `*.safetensors`, the JSON files, and the tokenizer, and skip `original/` and `*.pth`.

Here is the download Job for the shared-volume strategy. It runs once per model revision:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: fetch-llama-3-1-70b-instruct
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: fetch
          image: python:3.12-slim
          command: ["bash", "-c"]
          args:
            - |
              pip install --no-cache-dir "huggingface_hub[hf_xet]" &&
              hf download meta-llama/Llama-3.1-70B-Instruct \
                --revision "$REVISION" \
                --include "*.safetensors" "*.json" "tokenizer*" \
                --local-dir "/models/llama-3.1-70b-instruct/$REVISION"
          env:
            - name: REVISION
              value: "<pinned 40-character commit sha>"
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef: { name: hf-read-token, key: token }
          volumeMounts:
            - { name: models, mountPath: /models }
      volumes:
        - name: models
          persistentVolumeClaim: { claimName: model-store }
```

The path includes the revision, so two revisions can coexist on the volume during a rollout and a rollback is a path change, not a re-download.

## The Deployment

With weights on a volume, the Deployment is short, and almost every field is there for a reason specific to model servers:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-70b
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
  selector: { matchLabels: { app: llama-70b } }
  template:
    metadata:
      labels: { app: llama-70b }
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
      tolerations:
        - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
      terminationGracePeriodSeconds: 120
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.31.0
          args:
            - --model=/models/llama-3.1-70b-instruct/<sha>
            - --served-model-name=llama-70b
            - --tensor-parallel-size=2
            - --quantization=fp8
            - --kv-cache-dtype=fp8
            - --max-model-len=8192
            - --max-num-seqs=160
            - --max-num-batched-tokens=4096
            - --enable-prefix-caching
            - --gpu-memory-utilization=0.90
            - --port=8000
          env:
            - { name: HF_HUB_OFFLINE, value: "1" }
          ports:
            - { name: http, containerPort: 8000 }
          resources:
            requests: { nvidia.com/gpu: 2, cpu: "12", memory: 128Gi }
            limits:   { nvidia.com/gpu: 2, memory: 128Gi }
          startupProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 10
            failureThreshold: 60
          readinessProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 15
            failureThreshold: 4
          lifecycle:
            preStop:
              exec: { command: ["sleep", "15"] }
          volumeMounts:
            - { name: models, mountPath: /models, readOnly: true }
            - { name: dshm, mountPath: /dev/shm }
      volumes:
        - name: models
          persistentVolumeClaim: { claimName: model-store, readOnly: true }
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 16Gi }
```

Read it field by field, as the reasons are the lesson.

- **The engine flags are the runbook's output.** Nothing here is chosen in the manifest; TP, precision, `max-model-len`, and `max-num-seqs` all came from the memory budget and Little's law.
- **The startup probe covers the cold start.** Ten-second periods with a failure threshold of 60 give the pod ten minutes to load weights and capture CUDA graphs before Kubernetes kills it. Without a startup probe, the liveness probe would start immediately, fail during loading, and restart the pod forever, never getting past the load. Set the budget from the cold-start math, about twice the measured worst case.
- **Readiness gates traffic.** `/health` returns success only once the engine is serving. Until then the pod receives no requests from the Service.
- **The preStop sleep and the grace period drain streams.** When a pod is deleted, it is removed from the Service's endpoints at roughly the same time it receives SIGTERM. The 15-second sleep lets load balancers stop sending new requests first; the 120-second grace lets in-flight generations finish rather than be cut off mid-stream. That covers outputs of about 2,000 tokens; [A request's path through the cluster](03-rollouts-autoscaling-and-routing.md#a-requests-path-through-the-cluster) sizes it for longer ones.
- **`memory` has a limit; `cpu` does not.** A memory limit protects the node. A CPU limit throttles the tokenizer and scheduler under CFS quotas, which shows up as unexplained TTFT spikes.
- **`/dev/shm` is memory-backed.** This is the shared-memory fix from the pod-shape table. Its size counts against the pod's memory.

The Service in front of it is an ordinary `ClusterIP` on port 8000. Its round-robin load balancing is acceptable for one model without a large shared prefix; the routing section below says when it is not.

## Review questions

1. Why should readiness wait beyond the point where the process starts?

   <details><summary>Answer</summary>

   Weights, device buffers, kernel warmup, and any required graph capture must be ready to serve the declared workload.

   </details>

[Contents](README.md) · [Next: Rollouts, autoscaling, and routing](03-rollouts-autoscaling-and-routing.md)
