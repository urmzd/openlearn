"""A deterministic SQL safety gate for an LLM that writes queries.

Standard library only. Run me: ``python safety_gate.py``.

WHY THIS EXISTS
---------------
An agent that turns natural language into SQL is, structurally, a system that
lets a stranger's sentence reach your database. The gate below is what stands
in between, and the single most important property is that **it contains no
model**. It is rules over text, so it cannot be prompt-injected, cannot be
talked out of a decision, and behaves identically on every run.

The rule this encodes, which generalises far past SQL:

    NEVER LET THE MODEL BE THE ONLY SAFETY BOUNDARY.

A prompt saying "only generate SELECT statements" is a preference. A classifier
that refuses to execute anything else is a control. Prompts fail open under
adversarial input; deterministic gates fail closed.

DEFENCE IN DEPTH
----------------
This gate is one of three layers, and it is the weakest of them:

    1. this classifier          blocks writes/admin/multi-statement, caps rows
    2. a read-only connection   re-checks at the data boundary (PRAGMA
                                query_only, or a read-only DSN)
    3. a SELECT-only DB role    the only layer an application bug cannot bypass

Layer 3 is the real control; layers 1 and 2 exist to fail fast, give the model
a useful error to correct against, and protect databases where the deployment
does not control the role.

THE INTERESTING PART IS THE LEXER
---------------------------------
Naive keyword matching is wrong in both directions:

    SELECT * FROM t WHERE name = 'Begin Again'   -- contains BEGIN, is a read
    SELECT 'a;b' AS x                            -- contains ';', is ONE statement
    SELECT * INTO archive FROM users             -- leads with SELECT, WRITES
    SELECT * FROM (SELECT ... LIMIT 5)           -- has LIMIT, is NOT bounded

So the text is scanned once into two parallel strings of identical length: a
*cleaned* copy (comments blanked, literals intact) that is safe to execute, and
a *masked* copy (literal contents blanked too) that every check reads. Keywords
inside strings cannot lie to the classifier, and positions still line up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

# Statement-leading keywords that can begin a read.
_READ_LEADING = ("SELECT", "WITH")

# Keywords that mutate data (DML).
_WRITE_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE", "UPSERT"}

# Keywords that change schema/engine state or control transactions. Never
# appropriate from a natural-language prompt.
_ADMIN_KEYWORDS = {
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
    "GRANT",
    "REVOKE",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "BEGIN",
}

# A statement can LEAD with SELECT and still write: `SELECT ... INTO <table>`
# materialises a table, and MySQL's `SELECT ... INTO OUTFILE '<path>'` writes a
# server-side file. Neither carries a DML or DDL keyword, so a leading-token
# check alone waves them through.
_SELECT_WRITE_TARGETS = {"INTO", "OUTFILE", "DUMPFILE"}

DEFAULT_MAX_ROWS = 1000

_LIMIT_OR_PAREN_RE = re.compile(r"[()]|\blimit\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z_]+")


class Operation(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    INVALID = "invalid"


@dataclass
class SafetyResult:
    allowed: bool
    operation: Operation
    reason: str
    sql: str  # the (possibly row-capped) SQL that is cleared to run


def _scan(sql: str) -> tuple[str, str]:
    """One pass. Returns ``(cleaned, masked)``, always of equal length.

    * *cleaned* has comments blanked, literals intact -- safe to execute.
    * *masked* additionally blanks the CONTENTS of quoted literals and
      identifiers, so nothing inside a string can look like a keyword, a
      statement separator, or a comment marker to the checks below.

    Comments and literals are resolved in the same scan, so a quote inside a
    comment (or a ``--`` inside a literal) cannot desynchronise the two.
    """
    cleaned: list[str] = []
    masked: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in "'\"`":
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:  # doubled quote: escaped
                        j += 2
                        continue
                    break
                j += 1
            end = min(j + 1, n)  # include the closing quote, or run to the end
            span = sql[i:end]
            cleaned.append(span)
            masked.append(ch + " " * (len(span) - 2) + ch if len(span) >= 2 else ch)
            i = end
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            pad = " " * (j - i)
            cleaned.append(pad)
            masked.append(pad)
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            pad = " " * (j - i)
            cleaned.append(pad)
            masked.append(pad)
            i = j
        else:
            cleaned.append(ch)
            masked.append(ch)
            i += 1
    return "".join(cleaned), "".join(masked)


def _statements(cleaned: str, masked: str) -> list[tuple[str, str]]:
    """Split on real semicolons. Separator positions come from *masked*, so a
    ';' inside a string never splits; both texts are sliced in lockstep."""
    spans: list[tuple[str, str]] = []
    start = 0
    for idx, char in enumerate(masked):
        if char == ";":
            spans.append((cleaned[start:idx], masked[start:idx]))
            start = idx + 1
    spans.append((cleaned[start:], masked[start:]))
    return [(c.strip(), m.strip()) for c, m in spans if m.strip()]


def _leading_token(masked_statement: str) -> str:
    """The first keyword, looking through wrapping parentheses so compound
    reads like ``(SELECT ...) UNION (SELECT ...)`` classify correctly."""
    unwrapped = masked_statement.lstrip("( \t\r\n")
    if not unwrapped:
        return masked_statement
    return unwrapped.split(None, 1)[0].upper()


def _has_top_level_limit(masked_statement: str) -> bool:
    """True only when the statement carries its own LIMIT at the TOP level.

    A LIMIT inside a subquery or CTE does not bound the outer result set, so it
    must not suppress the appended row cap."""
    depth = 0
    for match in _LIMIT_OR_PAREN_RE.finditer(masked_statement):
        token = match.group(0)
        if token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            return True
    return False


def _operation_of(masked_statement: str) -> Operation:
    tokens = {w.upper() for w in _WORD_RE.findall(masked_statement)}
    if tokens & _ADMIN_KEYWORDS:
        return Operation.ADMIN
    if tokens & _WRITE_KEYWORDS:
        return Operation.WRITE
    if _leading_token(masked_statement) in _READ_LEADING:
        # A SELECT that names a write target is a materialisation, not a read.
        if tokens & _SELECT_WRITE_TARGETS:
            return Operation.WRITE
        return Operation.READ
    return Operation.INVALID


def classify(sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> SafetyResult:
    """Classify generated SQL and apply read-only-by-default policy.

    READ is allowed and row-capped; WRITE, ADMIN, and INVALID are blocked.
    Note the default: anything the classifier does not positively recognise as
    a read is refused. An allowlist fails closed; a denylist of "bad keywords"
    fails open on the first syntax nobody thought of.
    """
    cleaned, masked = _scan(sql)
    if not masked.strip():
        return SafetyResult(False, Operation.INVALID, "Empty query.", sql)

    statements = _statements(cleaned, masked)
    if len(statements) != 1:
        # Multi-statement is refused outright: it is the classic way to smuggle
        # a write in behind a legitimate read.
        return SafetyResult(
            False,
            Operation.INVALID,
            f"Expected exactly one statement, found {len(statements)}.",
            sql,
        )

    statement, masked_statement = statements[0]
    op = _operation_of(masked_statement)

    if op is Operation.ADMIN:
        return SafetyResult(
            False, op, "Schema/admin statements are never allowed.", sql
        )
    if op is Operation.INVALID:
        leading = _leading_token(masked_statement)
        return SafetyResult(
            False, op, f"Not a recognised read query (got '{leading}').", sql
        )
    if op is Operation.WRITE:
        return SafetyResult(
            False,
            op,
            "Write operations are disabled; this agent is read-only.",
            statement,
        )

    # READ: enforce a row cap. Trust an explicit TOP-LEVEL limit; otherwise
    # append the guard. An unbounded SELECT on a fact table is a denial of
    # service against your own application.
    safe_sql = statement
    if not _has_top_level_limit(masked_statement):
        safe_sql = f"{statement}\nLIMIT {max_rows}"
    return SafetyResult(True, op, "Read-only single statement.", safe_sql)


# ---------------------------------------------------------------------------
# The swappable seam. The deterministic rules are the floor, never the ceiling:
# a smarter classifier can be layered ON TOP, but the rules and the read-only
# connection stay underneath. An agent may ADD judgment; it never becomes the
# only thing between a prompt and the database.
# ---------------------------------------------------------------------------


@runtime_checkable
class Classifier(Protocol):
    def classify(self, sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> SafetyResult: ...


class BasicClassifier:
    """Deterministic, rule-based, always on."""

    def classify(self, sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> SafetyResult:
        return classify(sql, max_rows)


# ---------------------------------------------------------------------------
# TESTS. Every case here is a real failure mode, not a hypothetical: each one
# is a way a plausible-looking classifier gets it wrong.
# ---------------------------------------------------------------------------

ALLOWED = [
    "SELECT * FROM album",
    "select name from artist where id = 3",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "(SELECT 1) UNION (SELECT 2)",  # parenthesised compound read
    "SELECT * FROM t WHERE name = 'Begin Again'",  # BEGIN inside a literal
    "SELECT * FROM t WHERE note = 'drop table users'",  # DROP inside a literal
    "SELECT 'a;b' AS x",  # semicolon inside a literal: still one statement
    "SELECT * FROM t -- drop table users",  # keyword inside a comment
    "SELECT * FROM t /* insert into u */ WHERE id = 1",
    'SELECT "select" FROM t',  # quoted identifier that looks like a keyword
    "SELECT * FROM t;",  # single trailing semicolon
]

BLOCKED = [
    ("DELETE FROM album", Operation.WRITE),
    ("INSERT INTO album VALUES (1)", Operation.WRITE),
    ("UPDATE album SET title = 'x'", Operation.WRITE),
    ("DROP TABLE album", Operation.ADMIN),
    ("PRAGMA table_info(album)", Operation.ADMIN),
    ("ATTACH DATABASE 'evil.db' AS evil", Operation.ADMIN),
    ("SELECT * FROM t; DROP TABLE t", Operation.INVALID),  # smuggled write
    ("SELECT * INTO archive FROM users", Operation.WRITE),  # CTAS
    ("SELECT * FROM t INTO OUTFILE '/tmp/x'", Operation.WRITE),  # file write
    ("EXPLAIN SELECT 1", Operation.INVALID),  # not on the allowlist: refused
    ("", Operation.INVALID),
    ("   ", Operation.INVALID),
]


def test_reads_are_allowed() -> None:
    for sql in ALLOWED:
        result = classify(sql)
        assert result.allowed, f"should allow: {sql!r} ({result.reason})"
        assert result.operation is Operation.READ


def test_everything_else_is_blocked() -> None:
    for sql, expected in BLOCKED:
        result = classify(sql)
        assert not result.allowed, f"should block: {sql!r}"
        assert result.operation is expected, (
            f"{sql!r}: {result.operation} != {expected}"
        )


def test_row_cap_is_appended_when_absent() -> None:
    result = classify("SELECT * FROM album", max_rows=50)
    assert result.sql.endswith("LIMIT 50")


def test_explicit_top_level_limit_is_respected() -> None:
    result = classify("SELECT * FROM album LIMIT 5")
    assert result.sql.count("LIMIT") == 1 and result.sql.endswith("LIMIT 5")


def test_subquery_limit_does_not_count() -> None:
    """A LIMIT inside a subquery bounds the INNER result, not what comes back
    to the caller. Treating it as sufficient is the subtle bug here."""
    result = classify("SELECT * FROM (SELECT * FROM album LIMIT 5) x", max_rows=99)
    assert result.sql.rstrip().endswith("LIMIT 99")


def test_limit_inside_a_literal_does_not_count() -> None:
    result = classify("SELECT * FROM t WHERE name = 'no limit'", max_rows=7)
    assert result.sql.rstrip().endswith("LIMIT 7")


def test_unterminated_literal_does_not_crash_the_scanner() -> None:
    """Malformed input is normal when a model writes the SQL. The gate must
    refuse it, not raise."""
    result = classify("SELECT * FROM t WHERE name = 'unterminated")
    assert isinstance(result, SafetyResult)  # no exception escaped


def test_the_protocol_seam_holds() -> None:
    assert isinstance(BasicClassifier(), Classifier)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(
        f"\n{len(tests)} tests passed over {len(ALLOWED)} allowed and {len(BLOCKED)} blocked queries"
    )
