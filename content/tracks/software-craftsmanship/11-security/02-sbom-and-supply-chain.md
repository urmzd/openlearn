# SBOM and supply chain

## Overview

| | |
|---|---|
| **Module** | `craft.18` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | Model SBOMs: [`engine.spdx.json`](../../../../course/solutions/docs/sbom/engine.spdx.json) and [`gateway.spdx.json`](../../../../course/solutions/docs/sbom/gateway.spdx.json), SPDX 2.3 documents with packages and dependency relationships |
| **Contract** | [`course/contracts/allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml), the dependency policy each inventory is compared against |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact. It expands the model-files and release-pipeline row of the [threat model](01-threat-model-stride.md). |

## Key Takeaways

- An SBOM describes one built artifact, identified by its image digest, and is generated from that artifact's resolved dependency graph.
- Each package has a unique SPDX id, and every dependency relationship points at a package in the same document.
- The dependency policy (`allowed-deps.toml`) is the source of truth for what may be used; the SBOM is the record of what was used, and the review compares the two.

## How to read this chapter

Sections 1 to 3 explain why each image needs an inventory and how it is produced. Section 4 lists what a valid SBOM must satisfy. Section 5 walks through the model SBOMs field by field, explains which fields the model leaves as `NOASSERTION` and why a release SBOM must fill them, and shows the comparison against the policy. Try to list the fields you would need during a CVE response before reading the model.

---

## 1. Why now

A deployment is built from many direct and transitive packages. An SBOM and dependency policy help responders identify exposure and prove what was shipped.

## 2. Principles

Generate inventories from the exact image digest. Pin base images and dependencies, record licenses and provenance, and define a review path for exceptions. [`course/contracts/allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml) is the source policy; it is not a generated artifact. Regenerate an SPDX document for each released image and retain the image digest with the CI run.

## 3. Worked example

Build each service image, capture its digest, generate an SPDX or CycloneDX SBOM, and compare every dependency with the allowlist. Investigate unpinned transitive dependencies and known vulnerabilities before signing off.

A CVE response, by hand: an advisory names `google.golang.org/grpc` below some fixed version. The responder opens the gateway SBOM for the running digest, finds the package row with `SPDXRef-Package-Grpc`, reads its `versionInfo`, and follows the `DEPENDS_ON` relationship from `SPDXRef-Package-Gateway` to confirm the gateway pulls it in. The engine SBOM has no gRPC package row, so the engine is not exposed through this package. The answer comes from two files in minutes, without rebuilding anything.

## 4. The artifact

The artifacts are `docs/sbom/engine.spdx.json` and `docs/sbom/gateway.spdx.json`.

### What a valid SBOM must satisfy

| Case | Checks | Why it matters downstream |
|---|---|---|
| it is SPDX 2.3 JSON | `spdxVersion` is `SPDX-2.3` and the document id is `SPDXRef-DOCUMENT`. | Standard tools (scanners, license checkers) can read it. |
| it has packages and relationships | Both arrays are non-empty. | An inventory with no edges cannot answer "who depends on this?". |
| package ids are unique | No two packages share an `SPDXID`. | Relationships resolve to exactly one package. |
| relationships are connected | Every non-`DESCRIBES` relationship points at a package in the document. | No dangling edges; the graph is walkable from the root. |
| the policy has a section per component | `allowed-deps.toml` has `[rust.tl-engine]` and `[go.tinyllm]`. | Each inventory has a policy to be compared against. |

## 5. The solution, step by step

### 5.1 From first principles

1. When a vulnerability is announced, the question is "which running artifacts contain the affected version?". Source manifests cannot answer it, because they state requirements; only the resolved graph of the built artifact can. So an SBOM is per artifact, tied to its digest.
2. "Contains" is a graph question: a package can arrive transitively. So the SBOM records packages **and** the edges between them, rooted at the artifact.
3. Edges must resolve. A relationship to an id that is not in the document is a hole in the graph exactly where a responder needs to walk. So ids are unique and every edge target exists.
4. Not every fact is known when an SBOM is written (a license not yet reviewed, a download location the build does not record). Writing a guess is worse than writing nothing, and writing nothing is ambiguous; SPDX's `NOASSERTION` says explicitly "this document makes no claim".
5. Inventory and policy are different things. The policy says what **may** be used and is maintained by people; the SBOM says what **was** used and is generated by the build. Comparing them is the review.

### 5.2 The model SBOMs, annotated

**Document header** (engine):

```json
{
  "spdxVersion": "SPDX-2.3",
  "dataLicense": "CC0-1.0",
  "SPDXID": "SPDXRef-DOCUMENT",
  "name": "tinyllm-engine-reference-sbom",
  "documentNamespace": "https://example.invalid/spdx/tinyllm-engine-reference-2026-10-09",
  "creationInfo": {
    "creators": ["Tool: supersource-course-reference"],
    "created": "2026-10-09T00:00:00Z",
    "comment": "Reference example only. Release automation must replace this inventory with the exact image digest and resolved Cargo.lock packages."
  },
```

`dataLicense` must be `CC0-1.0` in SPDX 2.3 (the SBOM's own metadata is public domain). `documentNamespace` is a unique URI per document so two SBOMs never collide in a database; the model uses the reserved `.invalid` domain because it is an example. `creators` and `created` say who generated it and when, which is how a responder knows whether it matches the deployed build. The comment is step 1: the real document must come from the image digest and the lockfile.

**The root package and a dependency** (engine):

```json
    {
      "name": "tinyllm-engine",
      "SPDXID": "SPDXRef-Package-Engine",
      "downloadLocation": "NOASSERTION",
      "filesAnalyzed": false,
      "licenseConcluded": "NOASSERTION",
      "licenseDeclared": "NOASSERTION",
      "copyrightText": "NOASSERTION",
      "primaryPackagePurpose": "APPLICATION"
    },
    {
      "name": "anyhow",
      "SPDXID": "SPDXRef-Package-Anyhow",
      // ... downloadLocation, license, copyright: NOASSERTION
      "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl",
                        "referenceLocator": "pkg:cargo/anyhow"}]
    },
```

The root is marked `primaryPackagePurpose: APPLICATION`. Each dependency carries a **purl** (package URL), the identifier vulnerability databases match on. `filesAnalyzed: false` says the document does not list individual files, which is normal for a dependency inventory. The license fields are `NOASSERTION` (step 4): the model makes no license claim it has not checked.

**A versioned dependency** (gateway):

```json
    {
      "name": "github.com/BurntSushi/toml",
      "SPDXID": "SPDXRef-Package-Toml",
      "versionInfo": "1.6.0",
      // ... NOASSERTION fields
      "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl",
                        "referenceLocator": "pkg:golang/github.com/%21burnt%21sushi/toml@1.6.0"}]
    },
```

This is the complete form: `versionInfo` and a versioned purl. The odd-looking `%21burnt%21sushi` is the Go module proxy's case encoding (`!b` for `B`) percent-encoded inside the purl. The engine model's package rows omit `versionInfo` and use unversioned purls (`pkg:cargo/anyhow`): that is the gap its comment points at. Without versions, section 3's CVE question cannot be answered, so a release SBOM generated from `Cargo.lock` must include them.

**The relationships** (engine):

```json
  "relationships": [
    {"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": "SPDXRef-Package-Engine"},
    {"spdxElementId": "SPDXRef-Package-Engine", "relationshipType": "DEPENDS_ON", "relatedSpdxElement": "SPDXRef-Package-Anyhow"},
    {"spdxElementId": "SPDXRef-Package-Engine", "relationshipType": "DEPENDS_ON", "relatedSpdxElement": "SPDXRef-Package-SerdeJson"}
  ]
```

`DESCRIBES` names the root: the document is about the engine. `DEPENDS_ON` edges go from the root to each dependency (step 2). Every target id is one of the three package ids (step 3). A full SBOM would also have edges between dependencies (serde_json depends on serde, and so on), which a generator produces from the lockfile.

**The comparison with the policy** (step 5). The engine's direct dependencies `anyhow` and `serde_json` both appear under `[rust.tl-engine]` in [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml) (`anyhow = "1"`, `serde_json = "1"`). The gateway's `github.com/BurntSushi/toml` and `google.golang.org/grpc` appear under `[go.tinyllm]` with major `v1`, and the SBOM's `1.6.0` satisfies it. A package in an SBOM that has no row in its component's policy section is a finding: either the policy is extended through review, or the dependency is removed.

Full model answers: [`engine.spdx.json`](../../../../course/solutions/docs/sbom/engine.spdx.json), [`gateway.spdx.json`](../../../../course/solutions/docs/sbom/gateway.spdx.json).

### 5.3 Why this approach

- **Generated from the built artifact** rather than written from the manifest: the manifest is a requirement; transitive resolution, vendored code, and the base image only show up in what was built.
- **SPDX** rather than an ad-hoc JSON list: scanners, license tools, and registries already read it, and its relationship model answers graph questions. CycloneDX would serve equally well; the point is a standard format.
- **`NOASSERTION` over guesses:** a wrong license or location in an SBOM is worse than an honest gap, because downstream tools will trust it.
- **Tradeoff accepted:** an SBOM is only as fresh as its last build. It must be regenerated and stored with every released digest, which is a release-pipeline step to maintain.

### 5.4 Explain it back

**Why is the SBOM tied to an image digest and not to a git commit?**

<details><summary>Answer</summary>

The same commit can build different images (a different base image, a different resolution of a loose requirement, a different toolchain). The digest identifies exactly what runs, and the SBOM must describe exactly that.

</details>

**Why does a dangling relationship matter?**

<details><summary>Answer</summary>

A responder walks edges from the root to find whether a vulnerable package is reachable. An edge to an id that is not in the document breaks that walk, and the answer becomes "unknown" exactly where it is needed.

</details>

**The engine model has no `versionInfo`. Which question can it not answer?**

<details><summary>Answer</summary>

"Is the running engine affected by an advisory for versions below X?" It shows that `anyhow` is present but not which version, so the responder would have to rebuild or inspect the image. A release SBOM generated from `Cargo.lock` includes versions and versioned purls.

</details>

**Why is `allowed-deps.toml` not generated from the SBOM?**

<details><summary>Answer</summary>

Then the policy would allow whatever happened to be used, and the comparison would always pass. The policy is a human decision about what may be used; the SBOM is a machine record of what was used. Keeping them separate is what makes the comparison a review.

</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The SBOM is written from the manifest, not from the artifact | Transitive and base-image packages are missing; a CVE response gives the wrong answer. | The creation comment requires regeneration from the image digest and resolved lockfile. |
| 2 | Package ids collide or relationships dangle | The graph cannot be walked from the root. | Each package has a unique `SPDXRef-Package-*` id and every `DEPENDS_ON` target is in `packages`. |
| 3 | Guessed metadata | License or location fields that downstream tools trust are wrong. | Unknown fields are `NOASSERTION`. |
| 4 | Missing versions | Advisories cannot be matched. | The gateway model shows the complete form (`versionInfo` and a versioned purl); the engine model flags the gap for release automation. |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | [`craft.17`](01-threat-model-stride.md) | The threat model's release-pipeline row: provenance and digests are controls this SBOM provides evidence for. |
| Related | [`craft.15`](../10-maintenance/03-dependency-upgrades.md) | An upgrade changes the resolved graph, so it produces a new SBOM to review. |

## Going further

| Resource | What to inspect |
|---|---|
| [SPDX 2.3 specification](https://spdx.github.io/spdx-spec/v2.3/) | Required document and package fields, and the relationship types. |
| [SLSA](https://slsa.dev/) | Build provenance: who built the artifact, from what, and how to verify it. |
| [package-url specification](https://github.com/package-url/purl-spec) | How purls identify packages across ecosystems. |
