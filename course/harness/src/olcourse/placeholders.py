"""Placeholder expansion for milestone steps, service configs, and drills
(DESIGN 5.7 placeholder table).

A placeholder is `{name}` where name is made of letters, digits, and
`_ . : / + -`, so JSON braces in argv (`{"a": 1}`) are left alone. A lookup
returns a string, or a list for an entry role: an argv element that is exactly
`{tinyllm}` splices the role's argv; inside a longer string the list is
shell-joined. An unknown name is an error that names the placeholder.
"""

from __future__ import annotations

import re
import shlex
from typing import Callable

from . import HarnessError

PH = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.:/+\-]*)\}")

Lookup = Callable[[str], "str | list[str] | None"]


def names(s: str) -> list[str]:
    return PH.findall(s)


def expand(s: str, lookup: Lookup, where: str = "") -> str:
    def sub(m: re.Match) -> str:
        v = lookup(m.group(1))
        if v is None:
            raise HarnessError(
                f"unknown placeholder {{{m.group(1)}}}"
                + (f" in {where}" if where else "")
            )
        return shlex.join(v) if isinstance(v, list) else str(v)

    return PH.sub(sub, s)


def expand_argv(argv: list[str], lookup: Lookup, where: str = "") -> list[str]:
    out: list[str] = []
    for a in argv:
        m = PH.fullmatch(a)
        if m:
            v = lookup(m.group(1))
            if v is None:
                raise HarnessError(
                    f"unknown placeholder {a}" + (f" in {where}" if where else "")
                )
            if isinstance(v, list):
                out.extend(v)
                continue
            out.append(str(v))
            continue
        out.append(expand(a, lookup, where))
    return out


def expand_obj(obj, lookup: Lookup, where: str = ""):
    """Expand every string inside a JSON-like value (dicts, lists, scalars)."""
    if isinstance(obj, str):
        return expand(obj, lookup, where)
    if isinstance(obj, list):
        return [expand_obj(x, lookup, where) for x in obj]
    if isinstance(obj, dict):
        return {k: expand_obj(v, lookup, where) for k, v in obj.items()}
    return obj


def chain(*lookups: Lookup) -> Lookup:
    def f(name: str):
        for lk in lookups:
            v = lk(name)
            if v is not None:
                return v
        return None

    return f


def from_dict(d: dict, prefix: str = "") -> Lookup:
    """`{deploy.services.decode}` style lookups into a nested table."""

    def f(name: str):
        if prefix:
            if not name.startswith(prefix + "."):
                return None
            name = name[len(prefix) + 1 :]
        node = d
        for part in name.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return None if isinstance(node, dict) else node

    return f
