"""Licensing ledger verification and the datasheet (data.08).

The ledger answers, for every source, where it came from, under which
license, and what the pipeline kept. Verification joins it with the shards:
every row of training text must trace to a source whose license permits
training. A failure is a data error (exit 65), never retried, because no
retry will change a license.

Contract: contracts/py/corpus/ledger.pyi. Files: formats/ledger.schema.json,
contracts/templates/DATASHEET.md.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from corpus.shard import read_shards

EX_DATAERR = 65
_BOTH = frozenset({"train", "eval"})
_EVAL = frozenset({"eval"})
ALLOWLIST: dict[str, frozenset[str]] = {
    "CC0-1.0": _BOTH,
    "PDDL-1.0": _BOTH,
    "LicenseRef-PublicDomain": _BOTH,
    "CC-BY-3.0": _BOTH,
    "CC-BY-4.0": _BOTH,
    "CC-BY-SA-3.0": _BOTH,
    "CC-BY-SA-4.0": _BOTH,
    "CDLA-Permissive-1.0": _BOTH,
    "CDLA-Permissive-2.0": _BOTH,
    "CDLA-Sharing-1.0": _BOTH,
    "ODC-By-1.0": _BOTH,
    "ODbL-1.0": _BOTH,
    "MIT": _BOTH,
    "Apache-2.0": _BOTH,
    "BSD-2-Clause": _BOTH,
    "BSD-3-Clause": _BOTH,
    "CC-BY-NC-4.0": _EVAL,
    "CC-BY-NC-SA-4.0": _EVAL,
    "CC-BY-ND-4.0": _EVAL,
    "CC-BY-NC-ND-4.0": _EVAL,
}
_REQUIRED = (
    "source_id",
    "url",
    "license_spdx",
    "retrieved_at",
    "sha256",
    "n_docs",
    "allowed_uses",
    "pii_policy",
    "filters_applied",
    "kept",
    "dropped",
    "notes",
)
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIME = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$"
)


class LedgerError(Exception):
    """A ledger that does not license the corpus: exit code 65."""

    exit_code = EX_DATAERR

    def __init__(self, problems: list[str]) -> None:
        # SOLUTION-BEGIN data.08
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))
        # SOLUTION-END


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Every row in file order; blank lines skipped."""
    # SOLUTION-BEGIN data.08
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: not JSON: {e}") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{i}: a ledger line must be a JSON object")
        rows.append(row)
    return rows
    # SOLUTION-END


def latest(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """source_id -> its last row."""
    # SOLUTION-BEGIN data.08
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r["source_id"]] = dict(r)
    return out
    # SOLUTION-END


def _is_count(v: Any) -> bool:
    """A JSON integer >= 0 (bool is not an integer here)."""
    # SOLUTION-BEGIN data.08
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0
    # SOLUTION-END


def row_errors(row: Mapping[str, Any]) -> list[str]:
    """Violations of formats/ledger.schema.json by one row."""
    # SOLUTION-BEGIN data.08
    errs = [f"missing key {k!r}" for k in _REQUIRED if k not in row]
    errs += [f"unknown key {k!r}" for k in row if k not in _REQUIRED and k not in ("revoked", "modality", "biometric", "consent", "attribution", "prohibited_uses")]

    def check(key: str, ok: bool, what: str) -> None:
        if key in row and not ok:
            errs.append(f"{key} must be {what}")

    sid = row.get("source_id")
    check(
        "source_id",
        isinstance(sid, str) and bool(_SOURCE_ID.match(sid)),
        "a lowercase id",
    )
    check(
        "url",
        isinstance(row.get("url"), str) and row.get("url") != "",
        "a non-empty string",
    )
    lic = row.get("license_spdx")
    check("license_spdx", isinstance(lic, str) and lic != "", "a non-empty string")
    t = row.get("retrieved_at")
    check(
        "retrieved_at",
        isinstance(t, str) and bool(_TIME.match(t)),
        "an RFC 3339 UTC time",
    )
    h = row.get("sha256")
    check(
        "sha256",
        isinstance(h, str) and bool(_SHA256.match(h)),
        "64 lowercase hex digits",
    )
    for k in ("n_docs", "kept", "dropped"):
        check(k, _is_count(row.get(k)), "an integer >= 0")
    uses = row.get("allowed_uses")
    check(
        "allowed_uses",
        isinstance(uses, list) and all(u in ("train", "eval") for u in uses),
        "a list of train and eval",
    )
    check(
        "pii_policy",
        row.get("pii_policy") in ("scrub", "drop", "none"),
        "scrub, drop, or none",
    )
    fa = row.get("filters_applied")
    check(
        "filters_applied",
        isinstance(fa, list) and all(isinstance(x, str) for x in fa),
        "a list of strings",
    )
    check("notes", isinstance(row.get("notes"), str), "a string")
    check("revoked", isinstance(row.get("revoked"), bool), "true or false")
    check("modality", row.get("modality") in ("text", "image", "audio"), "text, image, or audio")
    check("biometric", isinstance(row.get("biometric"), bool), "true or false")
    check("consent", row.get("consent") in ("none", "public-domain", "cc-licensed", "explicit-donation"), "a consent category")
    check("attribution", isinstance(row.get("attribution"), str), "a string")
    uses = row.get("prohibited_uses")
    check("prohibited_uses", isinstance(uses, list) and all(isinstance(x, str) for x in uses), "a list of strings")
    return errs
    # SOLUTION-END


def permitted_uses(
    license_spdx: str, allowlist: Mapping[str, Iterable[str]] = ALLOWLIST
) -> Optional[frozenset[str]]:
    """The uses an SPDX expression permits (OR: union, AND: intersection,
    AND binding tighter), or None when any part is unknown."""
    # SOLUTION-BEGIN data.08
    tokens = license_spdx.split()
    if not tokens:
        return None
    result: frozenset[str] = frozenset()
    for alternative in " ".join(tokens).split(" OR "):
        ids = alternative.split(" AND ")
        term: Optional[frozenset[str]] = None
        for lid in ids:
            if lid not in allowlist:
                return None
            uses = frozenset(allowlist[lid])
            term = uses if term is None else term & uses
        result |= term or frozenset()
    return result
    # SOLUTION-END


def _shard_counts(corpus: Path) -> tuple[Counter, dict[str, set[str]]]:
    """Rows per source_id and the license strings seen per source."""
    # SOLUTION-BEGIN data.08
    counts: Counter = Counter()
    licenses: dict[str, set[str]] = {}
    for d in read_shards(corpus):
        counts[d.source_id] += 1
        licenses.setdefault(d.source_id, set()).add(d.meta["license_spdx"])
    return counts, licenses
    # SOLUTION-END


def verify(
    ledger: Path,
    corpus: Path,
    *,
    use: str = "train",
    allowlist: Mapping[str, Iterable[str]] = ALLOWLIST,
) -> list[str]:
    """Every problem; [] when the corpus may be used for `use`."""
    # SOLUTION-BEGIN data.08
    problems: list[str] = []
    good = []
    for i, row in enumerate(read_ledger(ledger), 1):
        errs = row_errors(row)
        if errs:
            problems.append(f"ledger row {i}: " + "; ".join(errs))
        else:
            good.append(row)
    cur = latest(good)
    counts, licenses = _shard_counts(corpus)
    for sid in sorted(cur):
        row = cur[sid]
        lic = permitted_uses(row["license_spdx"], allowlist)
        if lic is None:
            problems.append(
                f"source {sid}: license {row['license_spdx']!r} is unknown (not in the allowlist)"
            )
            continue
        extra = sorted(set(row["allowed_uses"]) - lic)
        if extra:
            problems.append(
                f"source {sid}: allowed_uses claims {extra}, which {row['license_spdx']} does not permit"
            )
        if counts[sid]:
            if row.get("revoked", False):
                problems.append(
                    f"source {sid}: revoked, but {counts[sid]} shard rows come from it"
                )
            if use not in row["allowed_uses"]:
                problems.append(
                    f"source {sid}: not allowed for {use}, but {counts[sid]} shard rows come from it"
                )
            bad = sorted(licenses[sid] - {row["license_spdx"]})
            if bad:
                problems.append(
                    f"source {sid}: shard rows say license {bad}, the ledger says {row['license_spdx']!r}"
                )
        if row["kept"] != counts[sid]:
            problems.append(
                f"source {sid}: ledger kept {row['kept']}, the shards hold {counts[sid]} rows"
            )
    for sid in sorted(set(counts) - set(cur)):
        problems.append(
            f"source {sid}: {counts[sid]} shard rows, but no valid ledger row"
        )
    return problems
    # SOLUTION-END


def check(
    ledger: Path,
    corpus: Path,
    *,
    use: str = "train",
    allowlist: Mapping[str, Iterable[str]] = ALLOWLIST,
) -> None:
    """verify(), raising LedgerError when there is any problem."""
    # SOLUTION-BEGIN data.08
    problems = verify(ledger, corpus, use=use, allowlist=allowlist)
    if problems:
        raise LedgerError(problems)
    # SOLUTION-END


def reconcile(
    ledger: Path, corpus: Path, *, filters_applied: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Append a corrected copy of every source's latest row."""
    # SOLUTION-BEGIN data.08
    cur = latest(read_ledger(ledger))
    counts, _ = _shard_counts(corpus)
    rows = []
    for sid in sorted(cur):
        row = dict(cur[sid])
        row["kept"] = counts[sid]
        row["dropped"] = max(0, int(row["n_docs"]) - counts[sid])
        row["filters_applied"] = list(filters_applied)
        rows.append(row)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows
    # SOLUTION-END


def datasheet(ledger: Path, corpus: Path, tokens: Optional[Path] = None) -> str:
    """contracts/templates/DATASHEET.md with the counts filled in."""
    # SOLUTION-BEGIN data.08
    corpus = Path(corpus)
    m = json.loads((corpus / "_MANIFEST.json").read_text())
    cur = latest(read_ledger(ledger))
    splits: Counter = Counter()
    langs: Counter = Counter()
    for d in read_shards(corpus):
        splits[d.meta["split"]] += 1
        langs[d.meta["lang"]] += 1
    dd = m["dedup"]
    out = [f"# Datasheet: {m['dataset']} {m['version']}", ""]
    out += [
        "## Motivation",
        "",
        "- <Why the dataset was built, by whom, for which model and task.>",
        "",
    ]
    out += ["## Composition", ""]
    out.append(
        f"- {m['n_docs']} documents in {m['n_shards']} Parquet shards: {splits['train']} train, {splits['val']} val (document-hash split)."
    )
    out.append(
        "- Languages: " + ", ".join(f"{k} {v}" for k, v in sorted(langs.items())) + "."
    )
    out.append(
        "- PII placeholders inserted: "
        + ", ".join(f"{k} {n}" for k, n in m["pii"].items())
        + ". Detection is pattern-based: names and addresses are not scrubbed."
    )
    out += [
        "",
        "## Collection",
        "",
        "| source_id | url | license_spdx | retrieved_at | sha256 |",
        "|---|---|---|---|---|",
    ]
    for sid in sorted(cur):
        r = cur[sid]
        out.append(
            f"| {sid} | {r['url']} | {r['license_spdx']} | {r['retrieved_at']} | {r['sha256']} |"
        )
    out += ["", "## Preprocessing", ""]
    filt = ", ".join(f"{k} {v}" for k, v in m["filters"].items()) or "none"
    out.append(f"- Quality filters (documents dropped): {filt}.")
    out.append(f"- Exact dedup dropped {dd['exact_dropped']}.")
    out.append(
        f"- Near dedup dropped {dd['near_dropped']} (MinHash, jaccard_threshold {dd['jaccard_threshold']}, "
        f"num_perm {dd['num_perm']}, bands {dd['bands']})."
    )
    out.append(
        f"- Decontamination dropped {dd['decontaminated']} documents sharing a {dd['ngram']}-gram with a protected set."
    )
    if tokens is not None:
        t = json.loads((Path(tokens) / "_MANIFEST.json").read_text())
        per = Counter()
        for f in t["files"]:
            per[f["split"]] += f["n_tokens"]
        out.append(
            f"- Tokenizer {t['tokenizer_id']}: {per['train']} train tokens, {per['val']} val tokens."
        )
    out += ["", "## Uses", ""]
    for sid in sorted(cur):
        out.append(
            f"- {sid}: allowed for {', '.join(cur[sid]['allowed_uses']) or 'nothing'} ({cur[sid]['license_spdx']})."
        )
    out += ["", "## Distribution and maintenance", ""]
    out.append(f"- Config sha256 {m['config_sha256']}; ledger {m['ledger_ref']}.")
    out.append(
        "- A revoked source (ledger `revoked: true`) fails `ledger verify` until its shards are rebuilt without it."
    )
    return "\n".join(out) + "\n"
    # SOLUTION-END
