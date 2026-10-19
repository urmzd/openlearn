# Security

## Overview

- **Primary references**: Shostack, *Threat Modeling: Designing for Security* (Wiley); OWASP, [Threat Modeling Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Threat_Modeling_Cheat_Sheet.html) (free)
- **Supplementary**: [SLSA](https://slsa.dev/) (free); [CycloneDX](https://cyclonedx.org/) and [SPDX](https://spdx.dev/) SBOM specs (free); [OWASP ASVS](https://owasp.org/www-project-application-security-verification-standard/) (free); OWASP, [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) (free)
- **Prerequisites**: [Diagramming & the C4 Model](../05-diagramming-c4/) (the data-flow diagram the threat model is built on), the full system from course Passes 1 to 10
- **Estimated time**: 1 to 2 weeks in course Pass 11

## Key Takeaways

- **Threat modeling asks four questions**: what are we building, what can go wrong, what are we going to do about it, and did we do a good job.
- **STRIDE per element of a data-flow diagram** (spoofing, tampering, repudiation, information disclosure, denial of service, elevation of privilege) makes "what can go wrong" systematic.
- **Trust boundaries are where checks belong**: every boundary in the system's interface matrix, including retrieved text reaching a tool call, gets a named control and a test.
- **Supply chain is part of the system**: an SBOM per artifact and provenance for builds record what was shipped, so the next CVE can be traced to the artifacts it affects.

## How to Study

Read the OWASP cheat sheet, then Shostack part I. Sketch the system's data-flow diagram from its C4 container view before reading further, then compare it with the solution's. In the solution, `craft.17` is the threat model [`docs/THREAT_MODEL.md`](../../../../course/solutions/docs/THREAT_MODEL.md) with every trust boundary of the system, `craft.18` adds SBOMs and build provenance, and `craft.19` reviews secrets handling and authorization paths.

---

# Concepts & Techniques

## Core Insight

Security is a property of the whole system, so it is reviewed against the whole system's diagram. The threat model turns the architecture into a list of threats, each tied to a control and a test, and the supply-chain records make the shipped artifacts accountable.

## 1. Threat modeling the system

**Key ideas**:
- **DFD**: processes, data stores, external entities, data flows, trust boundaries; built from the system's C4 container view.
- **STRIDE per element** with a control and an owner per threat, including the RAG-to-tools path (`craft.17`).

## 2. Supply chain, secrets, and authorization

**Key ideas**:
- **SBOM and provenance** for every image and binary (`craft.18`).
- **Secrets and authz review** (`craft.19`): where keys live (`TL_API_KEY`, HMAC peppers), how scopes and tenants are checked in the gateway (`gw.02`).

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `craft.17` | Threat model (STRIDE on the system's DFD) | practice | 11 |
| `craft.18` | SBOM and supply chain | practice | 11 |
| `craft.19` | Secrets and authz review | practice | 11 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `craft.17` | [Threat model (STRIDE on your DFD)](01-threat-model-stride.md) | practice | 11 |
| 2 | `craft.18` | [SBOM and supply chain](02-sbom-and-supply-chain.md) | practice | 11 |
| 3 | `craft.19` | [Secrets and authorization review](03-secrets-and-authorization-review.md) | practice | 11 |

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Diagramming & the C4 Model](../05-diagramming-c4/) | the diagrams the threat model starts from |
| [Authorization & Access Control](../../ai-platform-engineering/08-authorization-and-access-control/) | API keys, scopes, and tenants |
| [Usage Policy](../../responsible-ai/05-usage-policy/) | abuse cases and their enforcement |
| [Professional Responsibility](../../responsible-ai/06-professional-responsibility/) | disclosure when a threat becomes a vulnerability |
