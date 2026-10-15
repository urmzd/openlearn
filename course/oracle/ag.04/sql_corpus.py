"""SQL corpus for ag.04 with the expected decision of every query.

    python course/oracle/ag.04/sql_corpus.py   # rewrites course/fixtures/ag.04/sql_corpus.json

The oracle is the Python classifier of case study 02
(content/case-studies/02-grounded-sql-agent/safety_gate.py), which the Go gate
ports: every expected (allowed, operation, cleared SQL) below is that
classifier's output, so the Go port is checked against an independent
implementation, not against itself. The corpus is the case study's ALLOWED
and BLOCKED lists plus queries over the usage ledger (formats/usage.v1.sql)
that the agent's query_usage tool will see.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "course" / "fixtures" / "ag.04" / "sql_corpus.json"

spec = importlib.util.spec_from_file_location(
    "safety_gate", ROOT / "case-studies" / "02-grounded-sql-agent" / "safety_gate.py"
)
gate = importlib.util.module_from_spec(spec)
sys.modules["safety_gate"] = gate  # dataclasses look the module up by name
spec.loader.exec_module(gate)

EXTRA = [
    "SELECT COUNT(*) FROM usage WHERE tenant = 'acme' AND ts_ms >= 1760000000000",
    "select tenant, sum(prompt_tokens + completion_tokens) as tokens from usage group by tenant order by tokens desc",
    "SELECT * FROM usage WHERE error_code = 'rate_limited' LIMIT 20",
    "SELECT * FROM (SELECT * FROM usage ORDER BY ts_ms DESC LIMIT 5) ORDER BY ts_ms",
    "WITH t AS (SELECT tenant FROM usage) SELECT DISTINCT tenant FROM t",
    "SELECT * FROM usage WHERE route = '/v1/chat/completions; DROP TABLE usage'",
    "SELECT * FROM usage -- ; DELETE FROM usage",
    "SELECT 'it''s; fine' AS x",
    "SELECT * FROM usage WHERE error_code = 'no limit'",
    "DELETE FROM usage WHERE tenant = 'acme'",
    "UPDATE usage SET status = 200",
    "REPLACE INTO usage (request_id) VALUES ('x')",
    "INSERT INTO usage SELECT * FROM usage",
    "SELECT * FROM usage; DELETE FROM usage",
    "VACUUM",
    "PRAGMA query_only = OFF",
    "ATTACH DATABASE '/tmp/x.db' AS x",
    "CREATE TABLE stolen AS SELECT * FROM usage",
    "BEGIN; DELETE FROM usage; COMMIT",
    "SELECT * FROM usage /* unterminated comment",
    "SELECT * FROM usage WHERE tenant = 'unterminated",
    "explain query plan select * from usage",
    ";;",
]


def main() -> None:
    queries = list(gate.ALLOWED) + [q for q, _ in gate.BLOCKED] + EXTRA
    cases = []
    for q in queries:
        r = gate.classify(q, max_rows=100)
        cases.append(
            {
                "sql": q,
                "allowed": r.allowed,
                "op": r.operation.value,
                "cleared": r.sql if r.allowed else None,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"max_rows": 100, "cases": cases}, indent=1) + "\n")
    print(
        f"wrote {OUT}: {sum(c['allowed'] for c in cases)} allowed, {sum(not c['allowed'] for c in cases)} blocked"
    )


if __name__ == "__main__":
    main()
