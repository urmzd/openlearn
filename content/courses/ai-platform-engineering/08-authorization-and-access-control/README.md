# Authorization and Access Control

Authorization answers one question: **can *subject* do *action* on *resource*, in *context*?** Every model you will meet in this chapter (RBAC, ABAC, ReBAC, NGAC) is a different way to compute that boolean. The question looks simple, but it is really a *language-recognition* problem: given a request, decide whether it is in the set of permitted requests. How hard that is depends entirely on the **structure** of your permissions.

If permissions are flat (this role, these actions), recognizing them is a finite lookup: a regular language, decidable by a DFA. The moment permissions become *recursive* (groups contain groups, folders contain folders, roles inherit roles, delegation chains), you can no longer decide membership with finite state. You need a stack to track how deep you have descended, which is the jump from a finite automaton to a **pushdown automaton**. That is why real authorization systems are graph-traversal engines, not permission tables, and recognizing it saves you from the classic mistake of bolting recursion onto a model that cannot express it.

The chapter has two halves. The first surveys the models on that complexity ladder, the architecture that separates deciding from enforcing, and the place where authorization meets retrieval: secure RAG. The second builds the gateway's own access control in Go: per-client API keys with scopes, model allowlists, and tenants, and then per-key rate limits that keep one tenant from starving the rest.

> **Learning objectives**
> - Distinguish authentication from authorization and name the PEP, PDP, PAP, and PIP roles.
> - Compare RBAC, ABAC, ReBAC, and NGAC, and place each on the complexity ladder from lookup to graph reachability.
> - Explain why recursive permissions need a relationship graph and how a Zanzibar-style `check` works.
> - Design permission-aware retrieval that filters during the search, not after it.
> - Build API-key authentication, scope and allowlist authorization, and RPM/TPM rate limiting for an LLM gateway.

## Before you begin

The secure-RAG discussion assumes [Retrieval & RAG](../07-retrieval-and-rag/), which explains why retrieval must be permission-aware. The architecture assumes some [System Design](../../systems/01-system-design/) from *Systems and Architecture*, and the ReBAC material assumes graph reachability from [Graphs](../../algorithms/06-graphs/) in *Algorithms*. For the formal-languages side (the Chomsky hierarchy and pushdown automata), see [Discrete Math 2](../../mathematics-for-computing/07-discrete-math-2/) in *Mathematics for Computing* or [Sipser](https://math.mit.edu/~sipser/book.html). The two build sections plug into the gateway, so the [server skeleton](../12-gateway/02-server-skeleton.md) is a prerequisite for them.

## The vocabulary

Start by separating two words that are constantly confused. **Authentication** ("authn") is *who are you*: identity. **Authorization** ("authz") is *what may you do*. This chapter is about authz, though its first build section does both, and the difference shows up directly as the difference between an HTTP 401 and a 403.

A request has a **subject** or **principal** (a user or a service), an **action** (read, write), a **resource** or **object** (a doc, a row, an endpoint), and a **context** (time, IP, device). The standard XACML architecture splits the machinery into four roles: the **Policy Enforcement Point** (PEP) intercepts requests and enforces decisions; the **Policy Decision Point** (PDP) evaluates policy; the **Policy Administration Point** (PAP) manages policy; and the **Policy Information Point** (PIP) supplies attributes. The key idea is to **separate deciding from enforcing**. Two principles apply everywhere: **least privilege** (grant the minimum) and **deny by default** (the absence of a grant is a denial).

## RBAC: permissions through roles

In **role-based access control**, users are assigned **roles** and roles carry **permissions**: user, then role, then permission. Add **role hierarchies**, where a senior role inherits a junior one, and you have NIST RBAC. It is simple, auditable, and matches org charts, which is why it is the industry default.

Its limit is **role explosion**. When you need per-resource or conditional access ("editors, but only for their own team's docs in business hours"), you end up minting a role per combination. Through the complexity lens, *flat* RBAC is essentially a finite lookup, a regular language. *Hierarchical* RBAC introduces inheritance, recursion up the role tree, and quietly steps up the ladder.

## ABAC: permissions through rules over attributes

**Attribute-based access control** evaluates **policies over attributes** of the subject, resource, action, and context, for example `allow if subject.dept == resource.dept and context.time in hours and resource.clearance <= subject.clearance`. It is expressive, fine-grained, and context-aware, and it avoids role explosion because one rule covers many cases.

The price is auditability. Answering "who can access X?" requires evaluating the rules over all attributes, and you need a trustworthy attribute source, the PIP. XACML is the classic instance; modern policy engines such as **OPA/Rego**, Cedar, and Casbin are the current ones. A policy *language* is essentially how you express an ABAC decision function.

## ReBAC and Zanzibar: permissions through a graph

**Relationship-based access control** stores **relationship tuples** of the form `(object, relation, subject)`, such as `(doc:1, viewer, group:eng#member)`, and answers `check(user, relation, object)` by **traversing the relationship graph**. This is Google's **Zanzibar**, the model behind Drive, YouTube, and Cloud, with open-source implementations in OpenFGA and SpiceDB. It naturally expresses **nesting and inheritance**: groups of groups, folders of folders, "a viewer of the parent folder is a viewer of the child." That is exactly the recursive structure RBAC and ABAC struggle with.

Here the pushdown insight becomes concrete. `check()` is **graph reachability**: a traversal that descends through nested groups and resource hierarchies, using a stack or frontier to remember where it is. You are recognizing membership in a recursively defined set, a context-free, pushdown-style computation, not a finite table lookup. It connects directly to [graph BFS and DFS](../../algorithms/06-graphs/). At scale, Zanzibar adds consistency tokens ("zookies") and caching so that a globally distributed authorization graph stays correct and fast, which is itself a [distributed data and caching](../04-distributed-data-and-caching/) problem.

## NGAC: NIST's unifying policy graph

NIST's **Next Generation Access Control** (the "Policy Machine") represents users, objects, and attributes as nodes in a **directed policy graph** with assignment and association edges. A decision asks whether a privilege is **derivable by traversing** that graph under the policy. It matters because it is general: RBAC and ABAC are special cases of NGAC. It unifies attributes (like ABAC) and relationship evaluation (like ReBAC) under one formalism, and like ReBAC its evaluation is graph reachability over a recursively structured policy, pushdown-style again. Reach for it when you need both attribute conditions *and* relationship inheritance, with formal, analyzable semantics.

## The complexity ladder

The four models line up by the structure of the permissions they handle:

| Model | Permission structure | Computation | Recognizes |
|-------|----------------------|-------------|------------|
| **Flat RBAC / ACL** | role/user → permission | Lookup | Regular (DFA) |
| **Hierarchical RBAC** | + role inheritance | Bounded tree walk | (Steps up the ladder) |
| **ABAC** | rules over attributes | Evaluate a decision function | Depends on the language |
| **ReBAC / NGAC** | recursive relationship/policy graph | **Graph reachability (stack-driven)** | Context-free / **pushdown** |

The lesson is to pick the model whose native shape matches your permissions, not the one in fashion. If you find yourself adding "nested", "inherited", or "transitive" to an RBAC system, you have outgrown finite state; move to a relationship or graph model instead of piling up nesting hacks. It is the same "match the structure to the computation" instinct as choosing a [data engine by access pattern](../04-distributed-data-and-caching/).

## Separating policy from enforcement

Whatever the model, keep the decision in one place. **Externalized authorization** runs a PDP as a library or service (an OPA sidecar, an OpenFGA or SpiceDB service), and applications call it at their PEPs. That gives one source of truth, testable policies, and a central audit, instead of `if user.role == "admin"` scattered across the codebase. **Policy as code** means policies are versioned, reviewed, and *tested* like any other code, in the spirit of [Software Craftsmanship](../../software-craftsmanship/). And **decision logging** records every allow and deny with its reason, for audit and debugging: the authorization analogue of [observability](../../systems/04-observability/).

## Secure RAG: authorization applied to retrieval

This is where the chapter meets [Retrieval & RAG](../07-retrieval-and-rag/). The requirement is that a user may retrieve, and be grounded on, only content they are authorized to see. An LLM that quotes a document the user cannot access is a data breach, not a hallucination. Authorization is therefore part of retrieval, not a wrapper around it.

The pattern is to filter at retrieval time. Attach **ACLs or permission labels** to every chunk's metadata at ingestion. At query time, resolve the user's permissions with an RBAC, ABAC, or ReBAC check and **pre-filter or constrain** the vector and BM25 searches to authorized chunks. Prefer pre-filtering, or a permission-aware index: post-filtering after ranking leaks *existence* and ranking signal and skews results, and naive post-filtering can starve the top $k$, the same "filter before the cut" rule the [retrieval](../07-retrieval-and-rag/02-retrieval.md) section enforces with its `keep` predicate.

**Multi-tenancy** is the coarsest permission filter: partition or namespace the index so one tenant can never retrieve another's data (compare [sharding](../04-distributed-data-and-caching/)). Isolate by tenant before you rank. And **tools and agents** must enforce authorization too: an agent acting on behalf of a user inherits that user's permissions, not the service's.

## Choosing a model

The survey reduces to a lookup from situation to approach:

| Situation | Reach for |
|-----------|-----------|
| Simple, org-chart-shaped permissions | RBAC |
| Context/attribute conditions, no role explosion | ABAC (OPA/Rego, Cedar) |
| Nested groups, folder hierarchies, sharing | ReBAC (Zanzibar / OpenFGA / SpiceDB) |
| Need attributes *and* graph inheritance, formal | NGAC |
| Scattered `if role ==` checks | Externalize to a PDP |
| RAG over private/multi-tenant data | Permission-filter at retrieval (secure RAG) |
| "Why can this user see X?" | Decision logging + an auditable model |

Industry practice follows the same map. Google runs ReBAC at planetary scale with Zanzibar (Drive, YouTube, Cloud IAM). Auth0 and Okta combine RBAC with fine-grained ReBAC in FGA and OpenFGA. AWS evaluates ABAC policies in IAM and in Cedar and Verified Permissions. Airbnb and Carta use policy-as-code authorization with OPA/Rego and Oso. Glean and other enterprise AI products depend on permission-aware retrieval for secure RAG over corporate data. And NIST's NGAC Policy Machine is the standard's unifying policy-graph model.

To try the ideas yourself, model one application three ways (RBAC roles to permissions, ABAC attribute rules, ReBAC relationship graph) and note which questions each makes easy and which it makes painful. Implement a tiny Zanzibar-style `check(user, relation, object)` by graph traversal, then add nested groups and watch it become a reachability problem. Write one policy in OPA/Rego and the same policy as ReBAC tuples to feel the declarative versus relational trade-off. And wire permission filters into the [RAG retriever](../07-retrieval-and-rag/02-retrieval.md) so a query can return only authorized chunks.

## In this chapter

The build sections apply the simplest rung of the ladder, done carefully, to the LLM gateway. The route-to-scope table and per-key model allowlist are flat lookups, which is exactly right for their structure, and the tenant is the coarse isolation boundary that later stages, such as the response cache, rely on.

1. [API Keys, Scopes, Model Allowlists, and Tenants](01-api-keys-and-scopes.md) replaces the tracer's shared static key with per-client `tl_<id>_<secret>` keys stored as peppered HMACs, a short-TTL lookup cache that still honors revocation, and a pure authorization table that answers 403 for scope and allowlist failures.
2. [Rate Limiting: RPM Token Buckets and TPM Reserve/Settle](02-rate-limiting.md) gives each key a requests-per-minute and a tokens-per-minute budget, reserving an estimated token cost before a request runs and settling the actual usage afterwards, with exact `Retry-After` and `x-ratelimit-*` headers.

## Further reading

- NIST, [Role-Based Access Control](https://csrc.nist.gov/projects/role-based-access-control) (free).
- NIST, [SP 800-162, Attribute Based Access Control](https://csrc.nist.gov/pubs/sp/800/162/final) (free).
- NIST, [Next Generation Access Control (Policy Machine)](https://csrc.nist.gov/projects/policy-machine) (free).
- Pang et al., [Zanzibar: Google's Consistent, Global Authorization System](https://research.google/pubs/pub48190/) (free).
- [Open Policy Agent (OPA/Rego)](https://www.openpolicyagent.org/docs/latest/) (free) and [OpenFGA](https://openfga.dev/docs) (free).
- [Casbin](https://casbin.org/docs/overview), [Oso](https://www.osohq.com/docs), and [SpiceDB](https://authzed.com/docs).
