# Releases and Deprecation

Shipping is a contract with everyone downstream. Versions, changelogs, and deprecation windows are how that contract is stated, and automation from commit history is how it stays true. This chapter covers both ends of an interface's life: the release that first promises it, and the policy that eventually retires it.

A version number is a promise about compatibility. Semantic versioning says a major bump may break callers and a minor bump may not, and for this system the public API is the HTTP surface, the C ABI, and the file formats. The first section releases `v1.0.0` from history rather than from memory: conventional commits make the next version and the changelog a function of the commits since the last tag, and a release workflow running in CI, never on a laptop, checks that the tag, the tested commit, and the changelog agree.

The second section deals with what happens after. With enough users, every observable behavior is depended on (Hyrum's Law), so removing anything needs a policy: announce, warn, measure usage, migrate, then remove. Deprecation is a feature with a budget. It has an owner, a timeline, warnings in responses and logs (the `Deprecation` and `Sunset` HTTP headers), and telemetry that shows when the last caller is gone. The policy written here is the one the migrations in [Maintenance](../07-maintenance/) follow.

> **Learning objectives**
> - Apply semver to the system's public interfaces and explain what 1.0.0 promises.
> - Release from an annotated, immutable tag with a workflow that verifies the tag and changelog agree.
> - Write a deprecation policy whose notices have an owner, a window, a meter, an exit condition, and a rollback.
> - Explain why format changes roll out readers first, then writers.

## Before you begin

You need the CI gate and conventional commits from [Code Review and CI](../05-code-review-and-ci/), which the release builds on. Hyrum's Law and deprecation at scale are discussed in [Software Engineering at Google](../02-swe-at-google/), which is useful background.

## In this chapter

1. [Commits, semver, and releases](01-commits-semver-and-releases.md): semver as a promise, an annotated `v1.0.0` tag, a changelog grouped by user effect, and a tag-triggered workflow that checks the tested commit is the tagged one.
2. [Deprecation policy](02-deprecation-policy.md): what every notice must name, measured exit conditions, readers-first rollout for formats, and the review that must happen before anything is removed.

## Further reading

- [Semantic Versioning 2.0.0](https://semver.org/) (free).
- *Software Engineering at Google*, ch. 15, [Deprecation](https://abseil.io/resources/swe-book/html/ch15.html) (free).
- [Conventional Commits](https://www.conventionalcommits.org/) and [Keep a Changelog](https://keepachangelog.com/) (free).
- [Hyrum's Law](https://www.hyrumslaw.com/) (free).
- The [Kubernetes deprecation policy](https://kubernetes.io/docs/reference/using-api/deprecation-policy/) (free), a worked example of rules per API maturity level.
