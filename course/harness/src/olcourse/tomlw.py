"""A minimal TOML writer for the manifests the harness generates (farm
Cargo.toml files). tomllib reads; the stdlib has no writer."""

from __future__ import annotations

import json


def _key(k: str) -> str:
    return k if k.replace("-", "").replace("_", "").isalnum() else json.dumps(k)


def _val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_val(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _is_table(v) -> bool:
    return isinstance(v, dict)


def _is_table_array(v) -> bool:
    return isinstance(v, list) and v != [] and all(isinstance(x, dict) for x in v)


def dumps(doc: dict, _prefix: str = "") -> str:
    lines: list[str] = []
    for k, v in doc.items():
        if not (_is_table(v) or _is_table_array(v)):
            lines.append(f"{_key(k)} = {_val(v)}")
    for k, v in doc.items():
        name = f"{_prefix}.{_key(k)}" if _prefix else _key(k)
        if _is_table(v):
            scalars = [
                x for x in v.values() if not (_is_table(x) or _is_table_array(x))
            ]
            if scalars or not v:
                lines.append(f"\n[{name}]")
            body = dumps(v, name)
            if body.strip():
                lines.append(body.rstrip("\n"))
        elif _is_table_array(v):
            for item in v:
                lines.append(f"\n[[{name}]]")
                body = dumps(item, name)
                if body.strip():
                    lines.append(body.rstrip("\n"))
    return "\n".join(lines).lstrip("\n") + "\n"
