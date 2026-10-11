"""A small JSON Schema validator (stdlib only) for milestone `json-schema`
matchers, `file-produced` sub-checks, and OpenAPI response bodies.

Covers the subset the course contracts use: `$ref` (local `#/...` pointers
into the root document), `type` (string or list), OpenAPI 3.0 `nullable`,
`enum`, `const`, `required`, `properties`, `additionalProperties`, `items`,
`minItems`, `maxItems`, `minimum`, `maximum`, `exclusiveMinimum`,
`exclusiveMaximum`, `minLength`, `maxLength`, `pattern`, `allOf`, `anyOf`,
`oneOf`, `not`. Unknown keywords are ignored, as JSON Schema says.

    errors = validate(instance, schema, root=document)   # [] when valid
"""

from __future__ import annotations

import re
from typing import Any

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and v.is_integer())
    ),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


def resolve_ref(ref: str, root: Any) -> Any:
    if not ref.startswith("#"):
        raise ValueError(f"only local $ref pointers are supported, got {ref!r}")
    node = root
    for part in [p for p in ref[1:].split("/") if p]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise ValueError(f"$ref {ref!r} does not resolve")
    return node


def validate(
    inst: Any, schema: Any, root: Any = None, path: str = "$", _depth: int = 0
) -> list[str]:
    if root is None:
        root = schema
    if schema is True or schema is None or schema == {}:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    if _depth > 64:
        return [f"{path}: schema nesting too deep (recursive $ref?)"]
    if "$ref" in schema:
        try:
            target = resolve_ref(schema["$ref"], root)
        except (ValueError, IndexError) as e:
            return [f"{path}: {e}"]
        errs = validate(inst, target, root, path, _depth + 1)
        rest = {k: v for k, v in schema.items() if k != "$ref"}
        return errs + (validate(inst, rest, root, path, _depth + 1) if rest else [])

    errs: list[str] = []
    if inst is None and schema.get("nullable") is True:
        return []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPES.get(x, lambda v: True)(inst) for x in types):
            return [f"{path}: expected {' or '.join(types)}, got {_kind(inst)}"]
    if "enum" in schema and inst not in schema["enum"]:
        errs.append(f"{path}: {inst!r} is not one of {schema['enum']}")
    if "const" in schema and inst != schema["const"]:
        errs.append(f"{path}: expected {schema['const']!r}, got {inst!r}")

    if isinstance(inst, dict):
        for k in schema.get("required", []):
            if k not in inst:
                errs.append(f"{path}: missing required property {k!r}")
        props = schema.get("properties", {})
        for k, v in inst.items():
            if k in props:
                errs += validate(v, props[k], root, f"{path}.{k}", _depth + 1)
            elif "additionalProperties" in schema:
                ap = schema["additionalProperties"]
                if ap is False:
                    errs.append(f"{path}: unexpected property {k!r}")
                elif isinstance(ap, dict):
                    errs += validate(v, ap, root, f"{path}.{k}", _depth + 1)
    if isinstance(inst, list):
        if "items" in schema and isinstance(schema["items"], (dict, bool)):
            for i, v in enumerate(inst):
                errs += validate(v, schema["items"], root, f"{path}[{i}]", _depth + 1)
        if len(inst) < schema.get("minItems", 0):
            errs.append(
                f"{path}: {len(inst)} items, want at least {schema['minItems']}"
            )
        if "maxItems" in schema and len(inst) > schema["maxItems"]:
            errs.append(f"{path}: {len(inst)} items, want at most {schema['maxItems']}")
    if _TYPES["number"](inst):
        if "minimum" in schema and inst < schema["minimum"]:
            errs.append(f"{path}: {inst} < minimum {schema['minimum']}")
        if "maximum" in schema and inst > schema["maximum"]:
            errs.append(f"{path}: {inst} > maximum {schema['maximum']}")
        em, eM = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
        if isinstance(em, (int, float)) and not isinstance(em, bool) and inst <= em:
            errs.append(f"{path}: {inst} <= exclusiveMinimum {em}")
        if isinstance(eM, (int, float)) and not isinstance(eM, bool) and inst >= eM:
            errs.append(f"{path}: {inst} >= exclusiveMaximum {eM}")
    if isinstance(inst, str):
        if len(inst) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(inst) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], inst):
            errs.append(f"{path}: {inst!r} does not match {schema['pattern']!r}")

    for sub in schema.get("allOf", []):
        errs += validate(inst, sub, root, path, _depth + 1)
    if "anyOf" in schema and not any(
        not validate(inst, s, root, path, _depth + 1) for s in schema["anyOf"]
    ):
        errs.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema:
        n = sum(
            1 for s in schema["oneOf"] if not validate(inst, s, root, path, _depth + 1)
        )
        if n != 1:
            errs.append(f"{path}: matches {n} of oneOf, want exactly 1")
    if "not" in schema and not validate(inst, schema["not"], root, path, _depth + 1):
        errs.append(f"{path}: matches a schema it must not")
    return errs


def _kind(v: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _TYPES[name](v):
            return name
    return type(v).__name__
