# Course Pass 0: Setup

Before any model, a repo that grades itself. You install the toolchain, learn the Python and numpy you need to count bytes fast and hand arrays to C, learn the shell, git, and make that every later build rests on, and set up your repo's CI so every push lints its commit message, runs your native tests, and runs `ol check --all --ci`.

**Part of**: [the course](../course/). About 1.5 weeks at 10 to 12 hours a week.

**Gate**: [MS-P0](milestone.md), your first conventional commit green in CI.

## Toolchain (stage 1)

```bash
practice/bin/ol doctor              # what pass 0 needs: python, uv, cc, git, make
practice/bin/ol doctor --pass 1     # what pass 1 adds: cargo, go, docker, kubectl, kind, helm
```

On macOS: `xcode-select --install`, then `brew install uv rustup go kubectl kind helm` and Docker Desktop (`tilt` joins in Pass 7). On Linux use your package manager and the upstream installers. Then create your repo:

```bash
practice/bin/ol course init --name <system>   # a git repo with contracts/ vendored and system.toml
```

The chapters write plain `ol`. Put openlearn's `practice/bin` on your `PATH` once, so `ol` works from any directory:

```bash
export PATH="$PWD/practice/bin:$PATH"   # run in your openlearn clone; add the line to ~/.zshrc or ~/.bashrc with the full path
which ol                                # prints .../practice/bin/ol
```

The stages, their chapters, and what "done" means for each are in [`path.tsv`](path.tsv).

```bash
practice/bin/ol learn course-p00-setup         # stages and progress
practice/bin/ol learn course-p00-setup next    # read the next unfinished stage
```
