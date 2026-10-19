# Study plan

This plan paces *Systems and Architecture* over about fifteen weeks at eight to ten hours a week. It is a suggestion, not a schedule to keep: if a topic is already familiar, move faster; if a design problem takes a weekend to think through, let it. What matters is that each chapter's ideas are in place before the next one builds on them.

## Weeks 1 to 5: System Design

Spend the first five weeks on [System Design](01-system-design/README.md), with ByteByteGo as your companion reference. Begin with scalability, load balancing, caching, and CDNs, the tools for handling more traffic than one machine can. In the second week turn to data: database design, SQL versus NoSQL, the CAP theorem, and replication, reading *Designing Data-Intensive Applications* alongside. The third week is the hardest and deserves the most patience: distributed-systems primitives such as consensus and distributed transactions. Week four covers messaging, event systems, and API design. In week five, put it all together on the classic problems (a URL shortener, a chat system, a news feed) and then read the chapter's design reviews of the real system, checking your own back-of-envelope arithmetic against theirs.

## Weeks 6 to 8: Software Architecture

Then read [Software Architecture](02-software-architecture/README.md) over three weeks, with *Software Architecture Patterns* as the reference. Take the layered, event-driven, and microkernel patterns first, then microservices and space-based architecture. Use the third week for the habits that make architecture defensible: writing architecture decision records, naming quality attributes, and doing explicit trade-off analysis. Try writing an ADR for a decision in a system you know; it is the fastest way to find out whether you can defend it.

## Weeks 9 to 12: Cloud Native

Give [Cloud Native](03-cloud-native/README.md) four weeks, with *Cloud Native DevOps with Kubernetes* as the reference. Start with containers, Docker, and multi-stage builds. Move to Kubernetes fundamentals (pods, deployments, services) and, in the following week, the advanced objects: StatefulSets, CRDs, operators, and RBAC. Finish with Helm, GitOps, CI/CD, and service meshes. If you can, run everything on a local kind cluster as you read; the objects make much more sense once you have watched one reconcile.

## Weeks 13 to 15: Observability, then incidents

Spend the last three weeks on [Observability](04-observability/README.md), with *Observability Engineering* and the Google SRE Book as references. Begin with the difference between observability and monitoring, instrumentation, and OpenTelemetry. Then work through SLOs, error budgets, and debugging with observability, taking time over the burn-rate arithmetic. In the final week, read about production excellence, incident response, and chaos engineering, and start the drills of [Incident Response and Chaos](05-incident-response-and-chaos/README.md).

The drills do not need to wait for the end, though. Each one breaks a component built elsewhere in these books, so the natural time to read a drill is right after you have built or read about the piece it breaks: the first crashloop drill once the tracer is deployed, the decode-worker drill once burn-rate alerts exist, the durable-engine drills once the control plane runs. Before you read each drill's solution, predict the diagnosis and its output, and after each one read the matching chapter of the SRE book and compare its advice with the model response.
