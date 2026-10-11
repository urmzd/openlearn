"""What milestones, `ol conform`, and drills share: the learner repo, its
system.toml, the course tree its contracts/VERSION names, the run environment,
and the placeholders that do not depend on a step (DESIGN 5.7).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import (
    HarnessError,
    conform,
    ctx,
    learner,
    ledger,
    placeholders,
    registry,
    system,
    tree,
)


@dataclass
class Run:
    learner: Path | None
    system: system.System | None
    tree: tree.CourseTree
    reg: registry.Registry

    @property
    def course(self) -> Path:
        return self.tree.path

    # -- environment -----------------------------------------------------------

    def env(self, seed: int = 0) -> dict:
        env = ctx.base_env()
        env.pop(
            "VIRTUAL_ENV", None
        )  # the learner's `uv run` must pick its own project env
        env.update(
            {
                "OL_SEED": str(seed),
                "TINYLLM_FIXTURES": str(self.course / "fixtures"),
                "TINYLLM_CACHE": str(ctx.cache_dir()),
                # [ci].local commands and learner scripts reach the harness through
                # this (base_env drops OL_ROOT so learner processes never see it).
                "OL_BIN": str(ctx.root() / "practice" / "bin" / "ol"),
            }
        )
        src_epoch = self._source_date_epoch()
        if src_epoch:
            env["SOURCE_DATE_EPOCH"] = src_epoch
        return env

    def _source_date_epoch(self) -> str | None:
        repo = ctx.git_toplevel(self.course)
        if repo is None:
            return None
        rc, out = ctx.git(["log", "-1", "--format=%ct"], repo)
        return out.strip() if rc == 0 and out.strip().isdigit() else None

    # -- placeholders ----------------------------------------------------------

    def lookup(self, name: str):
        if name.startswith("fixture:"):
            p = self.course / "fixtures" / name[len("fixture:") :]
            if not p.exists():
                raise HarnessError(f"{{{name}}}: no committed fixture at {p}")
            return str(p)
        if name.startswith("asset:"):
            rest = name[len("asset:") :]
            asset = rest.split("/", 1)[0]
            p = ctx.cache_dir() / "assets" / rest
            if not p.exists():
                raise HarnessError(
                    f"{{{name}}}: asset {asset!r} is not in {ctx.cache_dir() / 'assets'} "
                    f"(fetch it with `ol fetch {asset}`)"
                )
            return str(p)
        if name == "models":
            return str(ctx.cache_dir() / "models")
        if name == "date":
            return time.strftime("%Y-%m-%d", time.gmtime())
        if name == "system":
            return self.system.name if self.system else None
        if self.system is not None:
            if name.startswith("deploy."):
                return placeholders.from_dict(self.system.deploy, "")(
                    name[len("deploy.") :]
                )
            if name == "endpoints.model":
                # The model id clients send ([endpoints].model; conformance's default).
                return str(self.system.endpoints.get("model") or "tracer")
            if name.startswith("endpoints."):
                return placeholders.from_dict(self.system.endpoints, "")(
                    name[len("endpoints.") :]
                )
            entries = self.system.raw.get("entry", {})
            if name in entries:
                return list(entries[name])
        return None

    def course_file(self, rel: str) -> Path:
        """`course/fixtures/...` paths in specs are relative to the course tree's parent."""
        p = Path(rel)
        if p.is_absolute():
            return p
        return (
            self.course.parent / rel if rel.startswith("course/") else self.course / rel
        )

    # -- verdict state -----------------------------------------------------------

    def module_passed(self, mid: str, smoke_ok: bool = False) -> bool:
        """A learner-sourced fresh pass (or a frozen superseded pass). A
        practice pass recorded under OL_SMOKE counts only when `smoke_ok`
        (a `--smoke` milestone): its cluster tier never ran."""
        if self.learner is None:
            return False
        if mid.startswith("MS-") or mid.startswith("conform:"):
            v = ledger.latest(self.learner, mid, full_only=True)
            return bool(v and v.get("result") == "pass")
        if mid not in self.reg.modules:
            return False
        st = learner.state(self.learner, self.course, self.reg, mid)
        # "self": a self-graded proof (solve sets, reviews) the learner signed off.
        return st.status in ("pass", "self") or (smoke_ok and st.status == "smoke")

    def spec_path(self, version: str) -> Path:
        base = self.learner / "contracts" if self.learner else self.course / "contracts"
        return base / "openapi" / f"openai-subset.{version}.yaml"

    def api_key(self) -> str | None:
        envname = self.system.api_key_env if self.system else "TL_API_KEY"
        return os.environ.get(envname)

    def conform_client(
        self,
        suite: conform.Suite,
        base: str,
        health_base: str | None,
        model: str | None,
    ) -> conform.Client:
        spec = conform.load_spec(self.spec_path(suite.version))
        m = (
            model
            or (self.system.endpoints.get("model") if self.system else None)
            or "tracer"
        )
        return conform.Client(
            base=base,
            tier=suite.tier or "engine",
            spec=spec,
            model=m,
            api_key=self.api_key(),
            health_base=health_base,
            count_tokens=self._token_counter(),
            api_version="2" if suite.version == "v2" else None,
        )

    def _token_counter(self):
        """chat.usage compares prompt_tokens with YOUR tokenizer's count of the
        templated prompt when your CLI answers `{tinyllm} tokenize --chat-json
        <messages>` with a last line {"count": N}; otherwise it checks only the
        usage arithmetic."""
        if self.system is None or "tinyllm" not in self.system.raw.get("entry", {}):
            return None

        def count(messages: list[dict]) -> int | None:
            import json as _json

            try:
                argv = self.system.entry("tinyllm", "chat.usage") + [
                    "tokenize",
                    "--chat-json",
                    _json.dumps(messages),
                ]
            except HarnessError:
                return None
            rc, out = ctx.run(argv, cwd=self.learner, env=self.env(), timeout=120)
            if rc != 0:
                return None
            for line in reversed(out.strip().splitlines()):
                try:
                    v = _json.loads(line)
                except ValueError:
                    continue
                n = v.get("count") if isinstance(v, dict) else None
                return n if isinstance(n, int) else None
            return None

        return count


def open_run(need_learner: bool = True) -> Run:
    h = ctx.learner_home()
    if (h / "system.toml").is_file():
        ct = tree.resolve(h)
        return Run(h, system.load(h), ct, registry.load(ct.path))
    if need_learner:
        learner.require()
    ct = tree.resolve(None)
    return Run(None, None, ct, registry.load(ct.path))
