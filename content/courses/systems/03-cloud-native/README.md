# Cloud Native

## Overview

- **Primary reference**: *Cloud Native DevOps with Kubernetes* 2nd ed. (recommended)
- **Supplementary**: [Kubernetes docs](https://kubernetes.io/docs/) (free), [The Twelve-Factor App](https://12factor.net/) (free)
- **Prerequisites**: Linux basics, containers concept
- **Estimated time**: 3-4 weeks at 8-10 hrs/week

## Key Takeaways

- Cloud native is about designing for failure, automation, and horizontal scaling from the start
- Kubernetes is the orchestration standard, but understanding the concepts matters more than memorizing YAML
- The Twelve-Factor App principles apply whether you use K8s, ECS, or bare metal
- GitOps (declarative infrastructure in git) is the modern deployment standard

---

# Concepts & Techniques

## Core Insight

Cloud native applications are designed to exploit cloud infrastructure: they're containerized, dynamically orchestrated, and microservices-oriented. The key shift is from "how do I keep this server running" to "how do I make the system resilient when servers fail."

## 1. Containers & Docker

**Key ideas**: images (immutable layers), containers (running instances), Dockerfile (build instructions), multi-stage builds (smaller images), security scanning. **12-factor principle**: strict separation between build and run stages.

**Best practices**: one process per container, pin base image versions, don't run as root, use .dockerignore, layer ordering matters for cache hits.

## 2. Kubernetes Fundamentals

**Key ideas**: Pods (smallest deployable unit), Deployments (declarative desired state, rolling updates), Services (stable networking, ClusterIP/NodePort/LoadBalancer), ConfigMaps & Secrets (externalized config), Namespaces (isolation).

**Mental model**: you declare desired state (YAML), the control plane continuously reconciles actual state toward desired state.

## 3. Advanced Kubernetes

**Key ideas**: StatefulSets (stable network identities for databases), DaemonSets (one pod per node for agents), Jobs/CronJobs (batch and scheduled work), RBAC (who can do what), Network Policies (pod-to-pod firewall), CRDs + Operators (extend K8s with custom resources and controllers).

## 4. Helm & GitOps

**Key ideas**: Helm charts (templated K8s manifests, versioned releases, values.yaml for config). GitOps: ArgoCD or Flux watches a git repo and syncs cluster state. Git becomes the source of truth for infrastructure.

**Why GitOps**: audit trail (git log), rollback (git revert), review (pull requests for infrastructure changes).

## 5. CI/CD Pipelines

**Key ideas**: GitHub Actions / GitLab CI / Jenkins. Build → test → scan → push image → deploy. Canary deployments (route small % of traffic to new version), blue-green (switch all traffic at once), feature flags (decouple deploy from release).

## 6. Service Mesh & Networking

**Key ideas**: Istio / Linkerd -- sidecar proxies handle mTLS, traffic management, observability. No code changes needed. Service-to-service encryption, circuit breaking, traffic splitting, distributed tracing.

**When NOT to use**: small clusters, when the operational overhead exceeds the benefit.

## 7. Cloud Patterns

| Pattern | Problem | Solution |
|---------|---------|----------|
| Circuit breaker | Cascading failures | Stop calling a failing service; fail fast |
| Bulkhead | One component failure takes down everything | Isolate resources per component |
| Retry with backoff | Transient failures | Retry with exponential backoff + jitter |
| Sidecar | Cross-cutting concerns (logging, auth) | Attach helper container to each pod |
| Ambassador | Service discovery complexity | Local proxy handles routing |
| Strangler fig | Migrating from monolith | Gradually route traffic to new service |

---

## Company Relevance

| Company | Infrastructure | Focus |
|---------|---------------|-------|
| Google | GKE, Borg (predecessor to K8s) | K8s was literally built here |
| Amazon | EKS, ECS, Lambda | Multi-orchestration, serverless |
| Netflix | Spinnaker, Titus (custom on EC2) | Deployment tooling, resilience |
| Anthropic | K8s + GPU orchestration | Inference serving at scale |
| Stripe | K8s for backend services | Reliable deployments, zero downtime |
