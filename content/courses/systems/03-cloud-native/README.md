# Cloud Native

Cloud-native applications are designed to exploit cloud infrastructure: they are containerized, dynamically orchestrated, and oriented around services. The key shift is in the question you ask. Instead of "how do I keep this server running?", you ask "how do I make the system resilient when servers fail?" That means designing for failure, automation, and horizontal scaling from the start rather than bolting them on later.

This chapter builds the vocabulary from the bottom up. It starts with containers and images, moves to the Kubernetes objects that run them and the reconciliation model behind those objects, then to packaging and deployment with Helm, GitOps, and CI/CD pipelines, and finally to the networking and resilience patterns that keep a distributed system standing. Kubernetes is the orchestration standard, but understanding the concepts matters far more than memorizing YAML, and the Twelve-Factor App principles apply whether you run on Kubernetes, ECS, or bare metal.

> **Learning objectives**
> - Build small, secure container images and explain the build/run separation.
> - Describe the core and advanced Kubernetes objects and the desired-state reconciliation model.
> - Explain how Helm and GitOps make deployments auditable, reviewable, and reversible.
> - Compare canary, blue-green, and feature-flag release strategies.
> - Apply circuit breakers, bulkheads, retries with backoff, sidecars, ambassadors, and the strangler fig to concrete failure problems.

## Before you begin

You need Linux basics and the concept of a container. If containers are new to you, read [Containers and Kubernetes](../../programming-languages/02-language-and-tool-primers/07-containers-and-kubernetes.md) in *Programming Languages* first. For a hands-on treatment that deploys the system built across these books, see [Containers, Kubernetes, and Workloads](../../infrastructure/01-containers-kubernetes/README.md) in *Infrastructure*.

## Containers

A container **image** is a stack of immutable layers; a **container** is a running instance of an image; a **Dockerfile** holds the build instructions. Multi-stage builds keep the final image small by leaving build tools behind, and security scanning checks the result. Underneath this is a Twelve-Factor principle: strict separation between the build stage and the run stage.

A few practices pay for themselves every time. Run one process per container. Pin base image versions. Do not run as root. Use `.dockerignore`. And order layers deliberately, because layer ordering decides how often the build cache hits.

## Kubernetes fundamentals

A **Pod** is the smallest deployable unit. A **Deployment** declares a desired state for a set of Pods and performs rolling updates toward it. A **Service** gives Pods stable networking, as ClusterIP, NodePort, or LoadBalancer. **ConfigMaps** and **Secrets** externalize configuration, and **Namespaces** provide isolation.

The mental model that ties these together is reconciliation: you declare the desired state in YAML, and the control plane continuously moves the actual state toward it. Every object above is a statement of what should be true, not a command to do something once.

## Advanced Kubernetes

The same model extends to harder workloads. **StatefulSets** give Pods stable network identities, which databases need. **DaemonSets** run one Pod per node, for agents. **Jobs** and **CronJobs** handle batch and scheduled work. **RBAC** controls who can do what, and **Network Policies** act as a pod-to-pod firewall. Finally, **CRDs** and **Operators** extend Kubernetes itself with custom resources and the controllers that reconcile them.

## Helm and GitOps

**Helm charts** are templated Kubernetes manifests with versioned releases, configured through `values.yaml`. **GitOps** goes one step further: a tool such as ArgoCD or Flux watches a git repository and syncs the cluster to it, so git becomes the source of truth for infrastructure. The payoff is that infrastructure inherits git's workflow: an audit trail (`git log`), rollback (`git revert`), and review (pull requests for infrastructure changes). GitOps is the modern deployment standard for exactly this reason.

## CI/CD pipelines

A pipeline, whether in GitHub Actions, GitLab CI, or Jenkins, runs the same sequence: build, test, scan, push the image, deploy. How the new version reaches users is a separate choice. A **canary** deployment routes a small percentage of traffic to the new version first. A **blue-green** deployment switches all traffic at once between two complete environments. **Feature flags** decouple deploying code from releasing behavior, so a deploy can ship dark and be turned on later.

## Service mesh and networking

A service mesh such as Istio or Linkerd puts a sidecar proxy next to each service. The proxies handle mTLS, traffic management, and observability with no code changes: service-to-service encryption, circuit breaking, traffic splitting, and distributed tracing. The cost is operational. On small clusters, or wherever the overhead exceeds the benefit, do not use one.

## Cloud patterns

Most resilience problems in a distributed system have a named pattern. The following table pairs each problem with its pattern and the solution it applies.

| Pattern | Problem | Solution |
|---|---|---|
| Circuit breaker | Cascading failures | Stop calling a failing service; fail fast |
| Bulkhead | One component failure takes down everything | Isolate resources per component |
| Retry with backoff | Transient failures | Retry with exponential backoff + jitter |
| Sidecar | Cross-cutting concerns (logging, auth) | Attach helper container to each pod |
| Ambassador | Service discovery complexity | Local proxy handles routing |
| Strangler fig | Migrating from monolith | Gradually route traffic to new service |

How large companies run their infrastructure shows these ideas at scale. The following table summarizes a few.

| Company | Infrastructure | Focus |
|---|---|---|
| Google | GKE, Borg (predecessor to Kubernetes) | Kubernetes was built here |
| Amazon | EKS, ECS, Lambda | Multi-orchestration, serverless |
| Netflix | Spinnaker, Titus (custom on EC2) | Deployment tooling, resilience |
| Anthropic | Kubernetes + GPU orchestration | Inference serving at scale |
| Stripe | Kubernetes for backend services | Reliable deployments, zero downtime |

## Summary

- Cloud native means designing for failure, automation, and horizontal scaling from the start.
- Kubernetes objects declare desired state; the control plane reconciles actual state toward it.
- Helm packages manifests; GitOps makes git the source of truth, giving deployments an audit trail, rollback, and review.
- Canaries, blue-green switches, and feature flags are different answers to how a new version reaches users.
- Named cloud patterns (circuit breaker, bulkhead, retry with backoff, sidecar, ambassador, strangler fig) address specific failure problems.

## Further reading

- *Cloud Native DevOps with Kubernetes*, 2nd edition.
- The [Kubernetes documentation](https://kubernetes.io/docs/) (free).
- [The Twelve-Factor App](https://12factor.net/) (free).
