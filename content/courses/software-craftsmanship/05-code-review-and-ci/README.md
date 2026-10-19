# Code Review and CI

Every change to the system passes through two gates before it merges. The first is mechanical: a CI workflow that runs on every push and pull request, and commit messages a machine can read. The second is human: a reviewer who reads the change against the system it modifies and finds the defects no test anticipated. This chapter builds both.

It starts with the mechanical gate, because it should exist from the very first commit. CI is a function from a commit to pass or fail, and branch protection turns that function into a gate. The rule it enforces lives in one place: the hook you run locally is the same file CI runs, so the two cannot disagree. Conventional commit headers make the history machine-readable, which is what releases are later computed from. Keep the gate strict from the start: loosening it later is easy, while tightening it means rewriting history.

The chapter then turns to the human gate. Once the system bills tenants and pages on-call engineers, a green test suite is not enough, because tests catch only what someone thought to test. The second section puts you in the reviewer's seat for a pull request that passes its own test and still ships five bugs.

> **Learning objectives**
> - Build a CI workflow and a tracked commit-msg hook that enforce Conventional Commits on every new commit.
> - Explain how branch protection, exit codes, and full-history checkouts make CI a reliable gate.
> - Review a pull request with the reviewer's loop and write findings that carry evidence and a severity tied to consequence.
> - Recognize data races and other state-dependent defects in Go code, and avoid false alarms on correct code.

## Before you begin

You should be comfortable with processes, exit codes, and git, as covered in [Shell, git, make, processes](../../programming-languages/02-language-and-tool-primers/02-shell-git-make.md) in *Programming Languages*. The code review section also draws on [Go](../../programming-languages/02-language-and-tool-primers/06-go.md) for goroutines and the race detector, and on [Mutation testing in depth](../03-testing-mentality/05-mutation-testing-in-depth.md) from earlier in this book.

## In this chapter

1. [The repository and its CI gate](01-your-repo-and-ci-gate.md): a Conventional Commits hook and a GitHub Actions workflow that lints every new commit and runs the tests, with the shallow-clone and first-push edge cases handled.
2. [Code review](02-code-review.md): the reviewer's loop, the shape and severity of a finding, and a worked review of a rate-limiter pull request that finds five seeded defects without raising false alarms.

## Further reading

- *Software Engineering at Google*, ch. 9, "Code Review", and ch. 23, "Continuous Integration", abseil.io/resources/swe-book (free).
- Conventional Commits 1.0.0, conventionalcommits.org (free).
