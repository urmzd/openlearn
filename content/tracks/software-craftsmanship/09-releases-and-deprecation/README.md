# Releases and Deprecation

## Overview

- **Primary references**: [Semantic Versioning 2.0.0](https://semver.org/) (free); *Software Engineering at Google*, ch. 15 [Deprecation](https://abseil.io/resources/swe-book/html/ch15.html) (free)
- **Supplementary**: [Conventional Commits](https://www.conventionalcommits.org/) (free); [Keep a Changelog](https://keepachangelog.com/) (free); [Hyrum's Law](https://www.hyrumslaw.com/) (free); the [Kubernetes deprecation policy](https://kubernetes.io/docs/reference/using-api/deprecation-policy/) (free)
- **Prerequisites**: [Code Review and CI](../08-code-review-and-ci/) (your CI gate and conventional commits from Pass 0)
- **Estimated time**: 1 week in course Pass 11

## Key Takeaways

- **A version number is a promise about compatibility**: semver says a major bump may break callers and a minor bump may not.
- **Release from history, not from memory.** Conventional commits make the next version and the changelog a function of the commits since the last tag.
- **With enough users, every observable behavior is depended on** (Hyrum's Law), so removing anything needs a policy: announce, warn, measure usage, migrate, then remove.
- **Deprecation is a feature with a budget**: it has an owner, a timeline, and telemetry that shows when the last caller is gone.

## How to Study

Read the semver spec and the SWE at Google deprecation chapter, then the Kubernetes policy as a worked example of rules per API maturity level. In the course, `craft.11` tags `v1.0.0` of your system from a release workflow that writes the changelog, and `craft.12` writes the deprecation policy that the API v1 to v2 migration (`craft.14`) follows.

---

# Concepts & Techniques

## Core Insight

Shipping is a contract with everyone downstream. Versions, changelogs, and deprecation windows are how that contract is stated, and automation from commit history is how it stays true.

## 1. Versions and releases

**Key ideas**:
- **semver**: MAJOR.MINOR.PATCH, pre-release and build metadata, and what counts as the public API (for your system: the HTTP surface, the C ABI, the file formats).
- **Release workflow**: tag, changelog, built artifacts, all from CI, never from a laptop (`craft.11`, adding to the CI of `dep.05`).

## 2. Deprecation

**Key ideas**:
- **Policy**: notice period per surface, warnings in responses and logs (`Deprecation` and `Sunset` HTTP headers), usage metrics, removal criteria (`craft.12`).
- **Applied**: the `openai-subset` v1 to v2 migration in [Maintenance](../10-maintenance/).

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `craft.11` | Commits, semver, releases (`v1.0.0`) | practice | 11 |
| `craft.12` | Deprecation policy | practice | 11 |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `craft.11` | [Commits, semver, releases (v1.0.0)](01-commits-semver-and-releases.md) | practice | 11 |
| 2 | `craft.12` | [Deprecation policy](02-deprecation-policy.md) | practice | 11 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Code Review and CI](../08-code-review-and-ci/) | the CI gate and commit lint the release builds on |
| [Maintenance](../10-maintenance/) | migrations that follow the deprecation policy |
| [Software Engineering at Google](../02-swe-at-google/) | Hyrum's Law and deprecation at scale |
