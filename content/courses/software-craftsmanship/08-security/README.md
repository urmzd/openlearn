# Security

Security is a property of the whole system, so it has to be reviewed against the whole system's diagram. By this chapter the system is complete: a public gateway, a serving engine, stores for keys, usage, and models, an operator control plane, and an agent that reaches external tools. Each piece was built with its own tests, but no test of a single component tells you where an attacker can act on the whole. This chapter turns the architecture into a list of threats, ties each threat to a control and a test, and makes the shipped artifacts accountable through supply-chain records.

The chapter is organized around the four questions every threat model asks: what are we building, what can go wrong, what are we going to do about it, and did we do a good job? The first section answers the first two by drawing a data-flow diagram from the system's C4 container view and applying STRIDE (spoofing, tampering, repudiation, information disclosure, denial of service, elevation of privilege) to every trust boundary, including the path where retrieved text reaches a tool call. The second and third sections take rows of that model and answer the last two questions in depth. One treats the supply chain as part of the system: an SBOM per artifact, plus provenance for builds, records what was shipped so the next CVE can be traced to the artifacts it affects. The other reviews where secrets live (`TL_API_KEY`, the HMAC pepper) and how scopes and tenants are checked in the gateway.

Every section produces a document rather than code, and each has a model answer in the solutions tree; the threat model is [`docs/THREAT_MODEL.md`](../../../../course/solutions/docs/THREAT_MODEL.md).

> **Learning objectives**
> - Build a data-flow diagram with trust boundaries from a C4 container view and threat-model it with STRIDE.
> - Write threat rows that name a control, evidence, residual risk, and an owner, and say when the model must be revisited.
> - Read and validate an SPDX SBOM and compare it against a dependency policy during a CVE response.
> - Review an authorization design as a matrix of principals, actions, and resources, and inventory every secret with its logging and rotation rules.
> - Define the findings that block a release and the evidence a release must attach.

## Before you begin

The threat model is drawn from the container view in [Diagramming with the C4 Model](../../diagramming/01-diagramming-c4/README.md) in *Diagramming and Documentation*, so read that chapter first if you have not. The chapter also assumes you know the system it reviews end to end, in particular the gateway's API keys, scopes, and tenants from [Authorization and Access Control](../../ai-platform-engineering/08-authorization-and-access-control/README.md) in *AI Platform Engineering*. A good warm-up is to read the OWASP Threat Modeling Cheat Sheet, then sketch your own data-flow diagram of the system before reading the first section, and compare it with the model's afterwards.

## In this chapter

1. [Threat modeling with STRIDE](01-threat-model-stride.md): draw the data-flow diagram, find its four trust boundaries, and build a STRIDE table in which every threat has a control, evidence, residual risk, and owner.
2. [SBOM and supply chain](02-sbom-and-supply-chain.md): read SPDX 2.3 inventories for the engine and gateway, see why they must be generated from the image digest, and compare them against the dependency policy.
3. [Secrets and authorization review](03-secrets-and-authorization-review.md): build the authorization matrix and secret inventory, check them against the gateway's `auth.go`, and set the release-blocking findings.

## Related chapters

Two chapters of *Responsible AI* pick up where this one stops. [Usage Policy](../../responsible-ai/05-usage-policy/README.md) covers abuse cases and their enforcement, and [Professional Responsibility](../../responsible-ai/06-professional-responsibility/README.md) covers disclosure when a threat becomes a vulnerability.

## Further reading

- Adam Shostack, *Threat Modeling: Designing for Security* (Wiley).
- OWASP, [Threat Modeling Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Threat_Modeling_Cheat_Sheet.html).
- [SLSA](https://slsa.dev/), for build provenance.
- The [CycloneDX](https://cyclonedx.org/) and [SPDX](https://spdx.dev/) SBOM specifications.
- [OWASP ASVS](https://owasp.org/www-project-application-security-verification-standard/).
- OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/).
