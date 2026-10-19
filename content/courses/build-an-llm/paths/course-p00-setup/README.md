# Pass 0: Setup

Before there is any model, there is a repository, and every later pass rests on it. Each pass adds code in four languages, and the only way to know that a change did not break an earlier layer is a check that runs on every commit. So the course starts with the toolchain and the habits that make such checks possible: the Python and numpy needed to count bytes fast and hand arrays to C, the shell, git, and make that every later build uses, and a CI setup that lints every commit message and runs the native tests on every push.

This pass is short and has no machine learning in it. What you get from it is a repository whose history and CI you can trust, which is what lets you move quickly later.

> **Learning objectives**
> - Install and verify the tools Pass 0 and Pass 1 need.
> - Predict the shape of a numpy broadcast and vectorize a count.
> - Explain exit codes, signals, and how make decides what to rebuild.
> - Explain how the repository's CI gate lints commit messages and runs native tests on every push.

## Before you begin

You need only a machine you can install software on and some familiarity with a terminal. Read [the course opening](../course/README.md) and [The system the solutions build](../course/SYSTEM.md) first, so you know what the repository you are about to set up will eventually hold.

## The toolchain

Pass 0 needs Python, `uv`, a C compiler, git, and make. Pass 1 adds cargo, go, docker, kubectl, kind, and helm, and `tilt` joins in Pass 7.

On macOS, run `xcode-select --install`, then `brew install uv rustup go kubectl kind helm`, and install Docker Desktop. On Linux, use your package manager and the upstream installers. Check each tool with its version flag: `python3 --version`, `uv --version`, `cc --version`, `cargo --version`, `go version`, `kubectl version --client`, `kind version`, and `helm version`. You are done with this step when you can say which tools Pass 0 needs and which Pass 1 adds, and every version command prints a version.

## In this pass

With the tools installed, read these sections in order:

2. [Python and numpy](../../../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md). Afterwards you can predict a broadcast's shape and explain how a count is vectorized.
3. [Shell, git, make, processes](../../../programming-languages/02-language-and-tool-primers/02-shell-git-make.md). Afterwards you can explain why a script exits 143 on SIGTERM and why make rebuilds only what changed.
4. [Your repo and CI gate](../../../software-craftsmanship/05-code-review-and-ci/01-your-repo-and-ci-gate.md). Afterwards you can explain how the repo's CI lints commit messages, runs native tests, and runs the test suite on every push.
5. [What the system does after Pass 0](milestone.md). Afterwards you can explain why every later pass depends on a green CI and conventional commits.

The pass ends with [what the system does after Pass 0](milestone.md): every commit is a conventional commit and CI is green on it. The stages and their "done when" statements are also in this pass's manifest, [`path.tsv`](path.tsv). The next pass, [the tracer](../course-p01-tracer/README.md), puts the first code through that gate.
