# MS-P0: Your Repo's CI Is Green

**Spec**: [`course/milestones/MS-P0.toml`](../../course/milestones/MS-P0.toml). **Requires**: `lang.01`, `lang.02`, and `craft.01` each have a fresh pass of your own (`ol check <ID>`); otherwise the milestone is blocked (exit 3).

## What it runs

| Step | Smoke | Passes when |
|---|---|---|
| HEAD is a conventional commit | yes | `git log -1` matches Conventional Commits (`feat: ...`, `fix(scope): ...`) |
| your CI is green | yes | with a GitHub remote: the latest run of your CI on the default branch concluded `success` (`gh run list`); without one: every `[ci].local` command in your `system.toml` exits 0 |

Both steps are smoke steps, so every later pass gate reruns them: a red CI or a non-conventional HEAD breaks MS-P1 and every gate after it.

## Run it

```bash
practice/bin/ol milestone MS-P0          # logs in .ol/milestones/MS-P0/<timestamp>/
```

Without a GitHub remote, declare your local CI in `system.toml`:

```toml
[ci]
local = [["bash", "scripts/ci-local.sh"]]   # OL_BIN points it at practice/bin/ol
```

`ol course ci` prints the recipe your hosted CI runs.
