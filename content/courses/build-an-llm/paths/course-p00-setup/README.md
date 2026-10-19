# Course Pass 0: Setup

Before any model, the toolchain and the habits every later build rests on. This pass covers the Python and numpy needed to count bytes fast and hand arrays to C, the shell, git, and make that every later build uses, and a CI setup that lints every commit message and runs the native tests on every push.

**Part of**: [the course](../course/). About 1.5 weeks at 10 to 12 hours a week.

**After this pass**: [what the system does](milestone.md): every commit is a conventional commit and CI is green on it.

## Toolchain (stage 1)

Pass 0 needs Python, `uv`, a C compiler, git, and make. Pass 1 adds cargo, go, docker, kubectl, kind, and helm (`tilt` joins in Pass 7).

On macOS: `xcode-select --install`, then `brew install uv rustup go kubectl kind helm` and Docker Desktop. On Linux use your package manager and the upstream installers. Check each with its version flag (`python3 --version`, `uv --version`, `cc --version`, `cargo --version`, `go version`, `kubectl version --client`, `kind version`, `helm version`).

The stages, their chapters, and what "done" means for each are in [`path.tsv`](path.tsv).
