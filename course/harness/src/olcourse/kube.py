"""kubectl, scoped to the learner's cluster (DESIGN 5.10, 2.16 [deploy]).

The drill safety gate refuses unless `kubectl config current-context` equals
[deploy].kube_context, that context starts with `kind-` or `k3d-`, and
[deploy].namespace exists. Every command then carries `--context` and `-n`,
so no action leaves that namespace. There is no override flag.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass

from . import HarnessError, ctx

LOCAL_PREFIXES = ("kind-", "k3d-")


class GateRefused(HarnessError):
    def __init__(self, msg: str):
        super().__init__(msg, code=1)


@dataclass
class Kube:
    context: str
    namespace: str

    def run(self, args: list[str], timeout: float = 60) -> tuple[int, str]:
        return ctx.run(
            ["kubectl", "--context", self.context, "-n", self.namespace, *args],
            timeout=timeout,
        )

    def must(self, args: list[str], timeout: float = 60) -> str:
        rc, out = self.run(args, timeout)
        if rc != 0:
            raise HarnessError(
                f"kubectl {' '.join(args)} failed (exit {rc}):\n{ctx.indent(out.strip())}"
            )
        return out

    def json(self, args: list[str]) -> dict:
        out = self.must([*args, "-o", "json"])
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            raise HarnessError(
                f"kubectl {' '.join(args)} -o json printed no JSON: {out[:200]}"
            ) from None


def _need_kubectl() -> None:
    if shutil.which("kubectl") is None:
        raise HarnessError("`kubectl` is not on PATH (see `ol doctor`)")


def current_context() -> str:
    _need_kubectl()
    rc, out = ctx.run(["kubectl", "config", "current-context"], timeout=20)
    return out.strip() if rc == 0 else ""


def safety_gate(deploy: dict) -> Kube:
    want = str(deploy.get("kube_context", ""))
    ns = str(deploy.get("namespace", ""))
    if not want or not ns:
        raise GateRefused(
            "drills need [deploy].kube_context and [deploy].namespace in system.toml"
        )
    if not want.startswith(LOCAL_PREFIXES):
        raise GateRefused(
            f"refusing: [deploy].kube_context {want!r} is not a local kind- or k3d- cluster"
        )
    cur = current_context()
    if cur != want:
        raise GateRefused(
            f"refusing: kubectl's current context is {cur or '(none)'!r}, not [deploy].kube_context {want!r}"
            f" (kubectl config use-context {want})"
        )
    k = Kube(want, ns)
    rc, out = ctx.run(
        ["kubectl", "--context", want, "get", "namespace", ns], timeout=20
    )
    if rc != 0:
        raise GateRefused(
            f"refusing: namespace {ns!r} does not exist in {want}:\n{ctx.indent(out.strip())}"
        )
    return k


def cluster_ready(deploy: dict) -> tuple[bool, str]:
    """For milestone kind steps: is the [deploy] cluster reachable? Never raises."""
    want = str(deploy.get("kube_context", ""))
    ns = str(deploy.get("namespace", ""))
    if not want or not ns:
        return False, "system.toml has no [deploy].kube_context and namespace"
    if shutil.which("kubectl") is None:
        return False, "kubectl is not installed"
    if want.startswith("kind-") and shutil.which("kind") is None:
        return (
            False,
            f"kind is not installed, so context {want} cannot exist (brew install kind; see `ol doctor`)",
        )
    rc, out = ctx.run(
        ["kubectl", "--context", want, "get", "namespace", ns], timeout=20
    )
    if rc != 0:
        return (
            False,
            f"cannot reach namespace {ns} in context {want}: {out.strip()[:200]}",
        )
    return True, ""
