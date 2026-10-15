"""ethics.02 artifact checks: your privacy and PII policy.

Run by `ol check ethics.02` (through ./check) from the root of your repo,
which `check` passes as OL_REPO. Annotated exemplars (DESIGN 5.12).

The formats are the chapter's (content/tracks/responsible-ai/02-privacy-and-pii/
01-privacy-and-pii-policy.md, section 4):

    docs/data/PII_POLICY.md       the policy, with fixed `##` sections
    docs/data/pii-policy.toml     categories, actions, placeholders, audit, logs
    docs/data/pii-review.toml     your answers to the twelve review cases

The categories are the five kinds data.05 detects (contracts/py/corpus/
pii.pyi KINDS) and gw.08 redacts from logs: email, phone, card (Luhn-valid
numbers only), ip (IPv4 and IPv6), key (API-key shapes), each replaced by
its data.05 placeholder.
"""

from __future__ import annotations

import datetime
import os
import re
import tomllib
from pathlib import Path

REPO = Path(os.environ.get("OL_REPO", "."))
POLICY = REPO / "docs" / "data" / "PII_POLICY.md"
TOML = REPO / "docs" / "data" / "pii-policy.toml"
REVIEW = REPO / "docs" / "data" / "pii-review.toml"

SECTIONS = {
    "Scope": 20,
    "Data flows": 30,
    "Categories and actions": 20,
    "Placeholders": 15,
    "Audit": 15,
    "Logs and retention": 15,
    "Erasure requests": 20,
    "Owner and review": 8,
}
# data.05's KINDS and PLACEHOLDERS (contracts/py/corpus/pii.pyi): the policy
# must name what the pipeline actually writes.
PLACEHOLDERS = {
    "email": "<EMAIL>",
    "phone": "<PHONE>",
    "card": "<CARD>",
    "ip": "<IP>",
    "key": "<KEY>",
}
CATEGORIES = tuple(PLACEHOLDERS)
SOURCE_POLICIES = {
    "scrub",
    "drop",
}  # a new source's ledger pii_policy; "none" only per named source
AUDIT_FIELDS = {"doc_id", "source_id", "kind", "start", "end"}
# The twelve review cases of section 4 and the category each one is. "none"
# marks a lookalike that a careless detector flags: a number that fails the
# Luhn check, an ISBN, a version string, a date, a trace id.
CASES = {
    "c01": ("Write to ana.lopez@example.org for the slides.", "email"),
    "c02": ("Call me at +1 (415) 555-0132 after six.", "phone"),
    "c03": ("Card on file: 4539 1488 0343 6467", "card"),
    "c04": ("Order number 4539 1488 0343 6468", "none"),
    "c05": ("ISBN 978-0-306-40615-7", "none"),
    "c06": ("Upgraded the engine to v2.10.3 last night.", "none"),
    "c07": ("The request came from 203.0.113.42 at noon.", "ip"),
    "c08": ("Peer 2001:db8:85a3::8a2e:370:7334 dropped the stream.", "ip"),
    "c09": ("export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "key"),
    "c10": ("Meeting moved to 2026-10-09 at 14:30.", "none"),
    "c11": ("trace_id=4bf92f3577b34da6a3ce929d0e0e4736", "none"),
    "c12": ("Authorization: Bearer tl_k7f3_9s8d7f6g5h4j3k2l1m0nq8w7", "key"),
}
PLACEHOLDER_TEXT = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b|<[a-z][^<>\n]*>")


def sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur, buf, fence = None, [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and re.match(r"^## \S", line):
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur, buf = line[3:].strip(), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out


def words(body: str) -> int:
    body = re.sub(r"^```.*?^```", "", body, flags=re.S | re.M)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", body))


def load(p: Path) -> dict:
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise AssertionError(f"{p.relative_to(REPO)} is not valid TOML: {e}") from None


def categories() -> dict[str, dict]:
    cats = load(TOML).get("category", [])
    assert isinstance(cats, list), (
        "`category` must be an array of tables ([[category]])"
    )
    return {c.get("name"): c for c in cats}


# -- the policy document ---------------------------------------------------------


def test_policy_has_every_section():
    # WHY: a PII policy answers fixed questions: what it covers, where
    #      personal data enters and leaves the system, what is done to each
    #      kind, what replaces it, what the audit trail keeps, what logs keep
    #      and for how long, how a person gets their data removed, and who
    #      owns the answers.
    # KIND: unit
    sec = sections(POLICY.read_text(encoding="utf-8"))
    missing = [h for h in SECTIONS if h not in sec]
    assert not missing, (
        f"PII_POLICY.md has no `## {missing[0]}` section (needs: {', '.join(SECTIONS)})"
    )
    thin = [
        f"{h} ({words(sec[h])} of {n} words)"
        for h, n in SECTIONS.items()
        if words(sec[h]) < n
    ]
    assert not thin, f"sections too short to say anything: {', '.join(thin)}"


def test_policy_has_no_placeholders():
    # WHY: a template with "<retention>" or TODO left in it promises nothing.
    # KIND: unit
    text = re.sub(r"`[^`\n]*`", "", POLICY.read_text(encoding="utf-8"))
    text = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    hits = [m.group(0) for m in PLACEHOLDER_TEXT.finditer(text)]
    assert not hits, f"PII_POLICY.md still has template placeholders: {hits[:5]}"


def test_data_flows_name_the_ways_in_and_out():
    # WHY: section 2.1: personal data enters through the corpus and through
    #      requests, and leaves through generations, logs, and traces. A
    #      flow the policy does not name is a flow nobody controls.
    # KIND: unit
    body = sections(POLICY.read_text(encoding="utf-8")).get("Data flows", "").lower()
    for word in ("corpus", "prompt", "log", "trace"):
        assert word in body, f"the Data flows section never mentions {word!r}s"


# -- the machine-readable policy ------------------------------------------------------


def test_policy_toml_has_an_owner_and_a_review_date():
    # WHY: as for the license allowlist: someone owns it, and the review
    #      date says how stale it is (a TOML date, not in the future).
    # KIND: unit
    doc = load(TOML)
    assert doc.get("version") == 1, "set `version = 1`"
    assert isinstance(doc.get("owner"), str) and doc["owner"].strip(), "set `owner`"
    rev = doc.get("reviewed")
    assert isinstance(rev, datetime.date) and not isinstance(rev, datetime.datetime), (
        "`reviewed` must be a TOML date"
    )
    assert rev <= datetime.date.today(), "`reviewed` is in the future"
    extra = sorted(
        set(doc)
        - {
            "version",
            "owner",
            "reviewed",
            "default_source_policy",
            "category",
            "audit",
            "logs",
        }
    )
    assert not extra, f"unknown top-level keys {extra}"


def test_every_category_is_named_once_with_its_placeholder():
    # WHY: the five kinds data.05 detects are each listed once, with the
    #      placeholder data.05 writes for it (<EMAIL>, <PHONE>, <CARD>,
    #      <IP>, <KEY>) and why it counts as personal data or a secret. A
    #      policy that promises <SECRET> while the shards say <KEY> is a
    #      policy nobody can check against the corpus.
    # KIND: unit
    cats = categories()
    names = [c.get("name") for c in load(TOML).get("category", [])]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, f"categories listed twice: {dup}"
    missing = [c for c in CATEGORIES if c not in cats]
    assert not missing, f"no [[category]] for {missing}"
    extra = sorted(set(cats) - set(CATEGORIES))
    assert not extra, (
        f"unknown categories {extra} (data.05 detects {', '.join(CATEGORIES)})"
    )
    for name, c in cats.items():
        assert c.get("placeholder") == PLACEHOLDERS[name], (
            f"{name}: placeholder {c.get('placeholder')!r}; data.05 writes {PLACEHOLDERS[name]}"
        )
        assert words(str(c.get("why", ""))) >= 6, f"{name}: `why` needs a sentence"
        assert set(c) <= {"name", "placeholder", "why"}, (
            f"{name}: unknown keys {sorted(set(c) - {'name', 'placeholder', 'why'})}"
        )


def test_new_sources_are_scrubbed_or_dropped():
    # WHY: the ledger's pii_policy decides what data.05 does with each
    #      source: scrub (placeholders, keep the document), drop (remove
    #      any document with a span), or none (skip detection). "none" is a
    #      claim that a source holds no personal data, so it may only be
    #      written for a named source after someone checked it; the default
    #      for a new source is scrub or drop.
    # KIND: boundary
    pol = load(TOML).get("default_source_policy")
    assert pol in SOURCE_POLICIES, (
        f"default_source_policy must be one of {sorted(SOURCE_POLICIES)}, not {pol!r}"
    )


def test_audit_never_stores_the_value():
    # WHY: the audit trail records that something was replaced, where, and
    #      as what (the span); storing the value itself would move the PII
    #      from the corpus into the audit log.
    # KIND: boundary
    audit = load(TOML).get("audit", {})
    assert audit.get("store_values") is False, "[audit] store_values must be false"
    fields = set(audit.get("fields", []))
    assert {"kind", "start", "end"} <= fields, (
        "[audit] fields must include kind, start, and end"
    )
    assert fields <= AUDIT_FIELDS, (
        f"[audit] fields {sorted(fields - AUDIT_FIELDS)} could hold the value; allowed: {sorted(AUDIT_FIELDS)}"
    )


def test_logs_redact_every_category_and_expire():
    # WHY: gw.08 applies the same detector list to gateway logs, so a
    #      prompt with an email does not land in a log file. Logs that are
    #      kept forever are a second corpus; retention is a number of days
    #      between 1 and 365.
    # KIND: boundary
    logs = load(TOML).get("logs", {})
    missing = [c for c in CATEGORIES if c not in logs.get("redact", [])]
    assert not missing, f"[logs] redact does not cover {missing}"
    days = logs.get("retain_days")
    assert isinstance(days, int) and not isinstance(days, bool) and 1 <= days <= 365, (
        "[logs] retain_days must be a whole number of days from 1 to 365"
    )


# -- the review cases --------------------------------------------------------------------


def test_review_cases_are_classified():
    # WHY: section 4's twelve strings, each one an email, phone, card, ip,
    #      key, or none. The lookalikes are the point: a 16-digit
    #      number is a card only when it passes the Luhn check (section 3),
    #      and a 32-digit hex trace id is not a secret while a key with a
    #      known prefix (AKIA, tl_) is. data.05 is graded on exactly these
    #      false positives.
    # KIND: unit
    answers = load(REVIEW)
    missing = [k for k in CASES if k not in answers]
    assert not missing, f"no answer for {missing} in pii-review.toml"
    wrong = [
        f"{k} {CASES[k][0]!r}: you said {answers[k]!r}"
        for k in CASES
        if str(answers[k]).strip().lower() != CASES[k][1]
    ]
    assert not wrong, (
        f"{len(wrong)} case(s) misclassified; work them as in section 3: "
        + "; ".join(wrong)
    )
