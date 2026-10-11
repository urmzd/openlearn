# Authorization & Access Control

## Overview

- **Primary references**: [NIST RBAC model](https://csrc.nist.gov/projects/role-based-access-control) (free), [NIST ABAC — SP 800-162](https://csrc.nist.gov/pubs/sp/800/162/final) (free), [NIST NGAC (Next Generation Access Control)](https://csrc.nist.gov/projects/policy-machine) (free), [Google Zanzibar paper](https://research.google/pubs/pub48190/) (free)
- **Supplementary**: [Open Policy Agent (OPA/Rego)](https://www.openpolicyagent.org/docs/latest/) (free), [OpenFGA](https://openfga.dev/docs) (free), [Casbin](https://casbin.org/docs/overview), [Oso](https://www.osohq.com/docs), [SpiceDB](https://authzed.com/docs); a formal-languages refresher (Chomsky hierarchy, pushdown automata) — [Sipser](https://math.mit.edu/~sipser/book.html) or the [Discrete Math 2](../../math/06-discrete-math-2/) track
- **Prerequisites**: [Retrieval & RAG](../07-retrieval-and-rag/) (why retrieval must be permission-aware), [System Design](../../systems/01-system-design/), graph reachability ([Algorithms — Graphs](../../algorithms/06-graphs/))
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- **Authorization answers one question: "can *subject* do *action* on *resource* (in *context*)?"** Every model — RBAC, ABAC, ReBAC, NGAC — is a different way to compute that boolean.
- **The models sit on a complexity ladder.** Flat RBAC is nearly a lookup table; nested groups and resource hierarchies need *recursion*; rich conditions need a *policy language*. Match the model to the structure of your permissions, not to fashion.
- **Recursion is why naive authz breaks** — "members of groups that own folders containing this doc." Evaluating nested/transitive relationships is a graph-reachability (stack-driven, pushdown-style) computation, not a finite-state check. Pulling that idea out is the deepest pattern here.
- **Separate policy from enforcement.** A **Policy Decision Point** (PDP) decides; a **Policy Enforcement Point** (PEP) enforces. Externalizing authz (OPA, Zanzibar-style services) keeps it consistent, auditable, and testable.
- **Authorization is part of retrieval, not a wrapper around it.** In RAG you must filter to what the user may see *during* retrieval — "secure RAG." Post-hoc filtering leaks existence and ranking signal.

## How to Study

- Model the same app three ways: RBAC (roles→permissions), ABAC (attribute rules), and ReBAC (a relationship graph). Note which questions each makes easy and which it makes painful.
- Implement a tiny relationship-graph checker (à la Zanzibar): `check(user, relation, object)` via graph traversal. Add nested groups and watch it become a reachability problem.
- Write one policy in OPA/Rego and the *same* policy as ReBAC tuples — feel the declarative vs relational trade-off.
- Wire permission filters into the [RAG retriever](../07-retrieval-and-rag/) so a query can only return authorized chunks.

---

# Concepts & Techniques

## Core Insight

Authorization looks like a boolean function but is really a *language-recognition* problem: given a request, decide whether it's in the set of permitted requests. How hard that is depends entirely on the **structure** of your permissions. If permissions are flat (this role → these actions), recognizing them is a finite lookup — a regular language, a DFA. The moment permissions become *recursive* — groups contain groups, folders contain folders, roles inherit roles, delegation chains — you can no longer decide membership with finite state; you need a stack to track how deep you've descended. That is the jump from a finite automaton to a **pushdown automaton**, and it's why real authorization systems are graph-traversal engines, not permission tables. Recognizing this saves you from the classic mistake: bolting recursion onto a model that can't express it.

## 1. The Vocabulary

- **Authentication** ("authn") — *who are you* (identity). **Authorization** ("authz") — *what may you do*. This topic is authz; they're constantly confused.
- **Subject / principal** (user, service), **action** (read, write), **resource / object** (doc, row, endpoint), **context** (time, IP, device).
- **PEP / PDP / PAP / PIP**: Enforcement Point (intercepts and enforces), Decision Point (evaluates policy), Administration Point (manages policy), Information Point (supplies attributes). The standard XACML split — **separate deciding from enforcing**.
- **Principle of least privilege** and **deny-by-default**: grant the minimum; absence of a grant is a denial.

## 2. RBAC — Role-Based Access Control

**Permissions through roles**

- **Model**: users are assigned **roles**; roles carry **permissions**. `user → role → permission`. Add **role hierarchies** (senior inherits junior) and you have NIST RBAC.
- **Strength**: simple, auditable, matches org charts; the industry default.
- **Limits**: **role explosion** when you need per-resource or conditional access ("editors, but only for their own team's docs in business hours") — you end up minting a role per combination.
- **Complexity lens**: *flat* RBAC is essentially a finite lookup (regular). *Hierarchical* RBAC introduces inheritance — recursion up the role tree — and quietly steps up the ladder.

## 3. ABAC — Attribute-Based Access Control

**Permissions through rules over attributes**

- **Model**: decisions are **policies evaluated over attributes** of subject, resource, action, and context: `allow if subject.dept == resource.dept and context.time in hours and resource.clearance <= subject.clearance`.
- **Strength**: expressive, fine-grained, context-aware; no role explosion — one rule covers many cases.
- **Limits**: harder to *audit* ("who can access X?" requires evaluating rules over all attributes), and you need a trustworthy attribute source (PIP).
- **Instance**: XACML (the classic), and modern policy engines like **OPA/Rego**, Cedar, Casbin — a policy *language* is essentially how you express an ABAC decision function.

## 4. ReBAC & Zanzibar — Relationship-Based Access Control

**Permissions through a graph of relationships**

- **Model**: store **relationship tuples** — `(object, relation, subject)`, e.g. `(doc:1, viewer, group:eng#member)` — and answer `check(user, relation, object)` by **traversing the relationship graph**. This is Google **Zanzibar** (the model behind Drive/YouTube/Cloud), and OSS implementations OpenFGA / SpiceDB.
- **Strength**: naturally expresses **nesting and inheritance** — groups of groups, folders of folders, "viewer of the parent folder is a viewer of the child." Exactly the recursive structure RBAC/ABAC struggle with.
- **The pushdown insight made concrete**: `check()` is **graph reachability** — a traversal that descends through nested groups and resource hierarchies, using a stack/frontier to remember where it is. You're recognizing membership in a recursively-defined set: a context-free, pushdown-style computation, not a finite table lookup. (Connects directly to [graph BFS/DFS](../../algorithms/06-graphs/).)
- **At scale**: Zanzibar adds consistency tokens ("zookies") and caching so a globally-distributed authz graph stays correct and fast — itself a [distributed data + caching](../04-distributed-data-and-caching/) problem.

## 5. NGAC — Next Generation Access Control

**Permissions through a policy graph (NIST's unifying model)**

- **Model**: NIST NGAC (the "Policy Machine") represents users, objects, and attributes as nodes in a **directed policy graph** with assignment and association edges; a decision is whether a privilege is **derivable by traversing** that graph under the policy.
- **Why it matters**: it's a *general* model — RBAC and ABAC are special cases of NGAC. It unifies attributes (like ABAC) and relationship/graph evaluation (like ReBAC) under one formalism, and like ReBAC its evaluation is graph reachability over a recursively-structured policy — again, pushdown-style, not finite-state.
- **Use**: when you need both attribute conditions *and* relationship inheritance with formal, analyzable semantics.

## 6. The Complexity Ladder (the unifying pattern)

| Model | Permission structure | Computation | Recognizes |
|-------|----------------------|-------------|------------|
| **Flat RBAC / ACL** | role/user → permission | Lookup | Regular (DFA) |
| **Hierarchical RBAC** | + role inheritance | Bounded tree walk | (Steps up the ladder) |
| **ABAC** | rules over attributes | Evaluate a decision function | Depends on the language |
| **ReBAC / NGAC** | recursive relationship/policy graph | **Graph reachability (stack-driven)** | Context-free / **pushdown** |

**The lesson**: pick the model whose native shape matches your permissions. If you're adding "nested" or "inherited" or "transitive" to an RBAC system, you've outgrown finite state — move to a relationship/graph model instead of nesting hacks. This is the same "match the structure to the computation" instinct as choosing a [data engine by access pattern](../04-distributed-data-and-caching/).

## 7. Separating Policy from Enforcement

- **Externalized authz**: run a **PDP** as a library or service (OPA sidecar, OpenFGA/SpiceDB service); apps call it at the **PEP**. One source of truth, testable policies, central audit — instead of `if user.role == "admin"` scattered across the codebase.
- **Policy as code**: version, review, and *test* policies like any other code (a fixed point with [Software Craftsmanship](../../software-craftsmanship/) and docs-as-code).
- **Decision logging**: log every allow/deny with its reason for audit and debugging — the authz analogue of [observability](../../systems/04-observability/).

## 8. Secure RAG — Authorization Applied to Retrieval

**Where this topic meets [Retrieval & RAG](../07-retrieval-and-rag/)**

- **The requirement**: a user must only retrieve and be grounded on content they're authorized to see. An LLM that quotes a doc the user can't access is a data breach, not a hallucination.
- **The pattern — filter at retrieval time**:
  - Attach **ACLs / permission labels** to every chunk's metadata at ingestion ([chunking](../07-retrieval-and-rag/)).
  - At query time, resolve the user's permissions (RBAC/ABAC/ReBAC check) and **pre-filter or constrain** the vector/BM25 search to authorized chunks.
  - Prefer **pre-filtering** (or a permission-aware index): post-filtering after ranking leaks *existence* and skews results, and naive post-filtering can starve the top-k.
- **Multi-tenancy**: tenant id is the coarsest permission filter — partition or namespace the index so one tenant can never retrieve another's data (ties to [sharding](../04-distributed-data-and-caching/)).
- **Tools/agents**: an agent's tools must enforce authz too — the agent acting "on behalf of" a user inherits that user's permissions, not the service's.

---

## Decision Cheat Sheet

| Situation | Reach for |
|-----------|-----------|
| Simple, org-chart-shaped permissions | RBAC |
| Context/attribute conditions, no role explosion | ABAC (OPA/Rego, Cedar) |
| Nested groups, folder hierarchies, sharing | ReBAC (Zanzibar / OpenFGA / SpiceDB) |
| Need attributes *and* graph inheritance, formal | NGAC |
| Scattered `if role ==` checks | Externalize to a PDP |
| RAG over private/multi-tenant data | Permission-filter at retrieval (secure RAG) |
| "Why can this user see X?" | Decision logging + an auditable model |

## Patterns Worth Internalizing

- **Authz is recognizing whether a request is in the permitted set** — and the *structure* of permissions sets the complexity. Flat = lookup; recursive = graph reachability (pushdown).
- **Don't nest hacks onto a finite model** — when permissions become recursive (groups-of-groups, hierarchies), move to a relationship/policy graph.
- **Separate the decision from the enforcement** — one PDP, many PEPs; policy as code, deny by default, least privilege.
- **Authorize *inside* retrieval, not around it** — filter to authorized data during the search, or you leak.
- **Tenant id is your first and coarsest filter** — isolate before you rank.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Pushdown automata, Chomsky hierarchy, regular vs context-free | [Discrete Math 2](../../math/06-discrete-math-2/) | Why recursion needs a stack |
| Graph reachability, BFS/DFS | [Algorithms — Graphs](../../algorithms/06-graphs/) | How ReBAC/NGAC checks evaluate |
| Permission-filtered retrieval | [Retrieval & RAG](../07-retrieval-and-rag/) | Secure RAG |
| Distributed graph, consistency tokens, caching | [Distributed Data & Caching](../04-distributed-data-and-caching/) | Zanzibar at scale |
| Multi-tenancy, partitioning | [Distributed Data & Caching](../04-distributed-data-and-caching/) | Tenant isolation |
| Policy as code, testing, review | [Software Craftsmanship](../../software-craftsmanship/) | Versioned, tested policies |

## How Companies Apply These Patterns

| Company | The pattern they lean on | Instance |
|---------|--------------------------|----------|
| Google | ReBAC at planetary scale | Zanzibar (Drive/YouTube/Cloud IAM) |
| Auth0 / Okta | RBAC + fine-grained ReBAC | FGA, OpenFGA |
| AWS | ABAC policy evaluation | IAM policies, Cedar / Verified Permissions |
| Airbnb / Carta | Policy-as-code authorization | OPA/Rego, Oso |
| Glean / enterprise AI | Permission-aware retrieval | Secure RAG over corporate data |
| NIST (standard) | Unifying policy-graph model | NGAC / Policy Machine |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `gw.02` | [AuthN (API keys) and AuthZ (scopes, model allowlist, tenants)](01-api-keys-and-scopes.md) | build | 7 |
| 2 | `gw.03` | [Rate limiting: RPM token bucket + TPM reserve/settle](02-rate-limiting.md) | build | 7 |
<!-- /ol:chapters -->
