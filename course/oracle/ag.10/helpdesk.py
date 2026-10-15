"""The helpdesk-world oracle and mutant suite for ag.10's state grader
(case study 05, content/case-studies/05-agent-eval-harness/, ported).

    uv run python course/oracle/ag.10/helpdesk.py

Writes course/fixtures/ag.10/helpdesk.json: a small world (inventory items
and helpdesk tickets) with the traps real data has (an owner that is null on
one ticket and absent on another), four tasks, and attempts. Each attempt is
a final state plus the tool calls that produced it. Oracles are correct
solutions and must pass. Mutants are deliberate mistakes; each declares the
criteria it was built to fail (`fails`), and this script checks every
declaration with its own implementation of the checker before writing, so a
mutant whose prediction is wrong never reaches the fixture.

Checker (case study 05, section 4): delta = diff(before, after) with
bookkeeping fields (updated_at) ignored, absent kept apart from null;
required_changes: every required key is in delta with its value;
only_requested_changes: every delta key is allowed (exact or "<row>.*");
read_before_write: every write's row id was returned by an earlier read.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "fixtures" / "ag.10" / "helpdesk.json"
ABSENT = {"$absent": True}

WORLD = {
    "items/i1": {"sku": "KB-1", "wh": "berlin", "qty": 3, "updated_at": 100},
    "items/i2": {"sku": "KB-1", "wh": "paris", "qty": 7, "updated_at": 100},
    "items/i3": {"sku": "MS-2", "wh": "berlin", "qty": 0, "updated_at": 100},
    "tickets/t1": {"status": "open", "owner": "ann", "updated_at": 100},
    "tickets/t2": {"status": "open", "owner": None, "updated_at": 100},
    "tickets/t3": {"status": "open", "updated_at": 100},
    "tickets/t4": {"status": "closed", "owner": "bob", "updated_at": 100},
}

TASKS = {
    "restock": {
        "instruction": "Set the Berlin stock of KB-1 to 5.",
        "required": {"items/i1.qty": 5},
        "allowed": ["items/i1.qty"],
        "read_before_write": True,
    },
    "assign-unowned": {
        "instruction": "Assign every open ticket without an owner to ann.",
        "required": {"tickets/t2.owner": "ann", "tickets/t3.owner": "ann"},
        "allowed": ["tickets/t2.owner", "tickets/t3.owner"],
        "read_before_write": False,
    },
    "unassign": {
        "instruction": "Remove the owner from ticket t1.",
        "required": {"tickets/t1.owner": ABSENT},
        "allowed": ["tickets/t1.owner"],
        "read_before_write": False,
    },
    "new-ticket": {
        "instruction": "Open a ticket for the damaged keyboard.",
        "required": {"tickets/t5": "created"},
        "allowed": ["tickets/t5"],
        "read_before_write": False,
    },
}

READS, WRITES = ["get_item", "list_items"], ["update_item"]


def read(name, args, returned):
    return {
        "name": name,
        "args": args,
        "verdict": "allow",
        "is_error": False,
        "result": json.dumps(returned),
    }


def write(item_id, fields):
    return {
        "name": "update_item",
        "args": {"id": item_id, **fields},
        "verdict": "allow",
        "is_error": False,
        "result": '{"ok": true}',
    }


def after(edit):
    w = copy.deepcopy(WORLD)
    edit(w)
    return w


def diff(before, aft):
    delta = {}
    for row in before:
        if row not in aft:
            delta[row] = "deleted"
    for row, a in aft.items():
        if row not in before:
            delta[row] = "created"
            continue
        b = before[row]
        for f in sorted(set(a) | set(b)):
            if f == "updated_at":
                continue
            if f in b and f not in a:
                delta[f"{row}.{f}"] = ABSENT
            elif f not in b or a[f] != b[f]:
                delta[f"{row}.{f}"] = a[f]
    return delta


def ids_in(v, out):
    if isinstance(v, dict):
        for k, x in v.items():
            if k == "id" and isinstance(x, str):
                out.add(x)
            ids_in(x, out)
    elif isinstance(v, list):
        for x in v:
            ids_in(x, out)


def grade(task, aft, calls):
    delta = diff(WORLD, aft)
    failed = set()
    for k, v in task["required"].items():
        if k not in delta or delta[k] != v:
            failed.add("required_changes")
    for k in delta:
        if not any(
            k == a or (a.endswith(".*") and k.startswith(a[:-1]))
            for a in task["allowed"]
        ):
            failed.add("only_requested_changes")
    if task["read_before_write"]:
        seen = set()
        for c in calls:
            if c["verdict"] != "allow" or c["is_error"]:
                continue
            if c["name"] in READS:
                ids_in(json.loads(c["result"]), seen)
            if c["name"] in WRITES and c["args"].get("id") not in seen:
                failed.add("read_before_write")
    return sorted(failed)


def set_(row, field, value, stamp=True):
    def edit(w):
        w[row][field] = value
        if stamp:
            w[row]["updated_at"] = 200

    return edit


def chain(*edits):
    def edit(w):
        for e in edits:
            e(w)

    return edit


def drop(row, field):
    def edit(w):
        del w[row][field]

    return edit


def create(row, fields):
    def edit(w):
        w[row] = fields

    return edit


I1 = {"id": "i1", "sku": "KB-1", "wh": "berlin", "qty": 3}
I2 = {"id": "i2", "sku": "KB-1", "wh": "paris", "qty": 7}

ATTEMPTS = [
    # restock
    (
        "restock",
        "oracle",
        "read-then-write",
        after(set_("items/i1", "qty", 5)),
        [
            read("list_items", {"sku": "KB-1"}, {"items": [I1, I2]}),
            write("i1", {"qty": 5}),
        ],
        [],
    ),
    (
        "restock",
        "oracle",
        "bookkeeping-only-extra",
        after(
            chain(
                set_("items/i1", "qty", 5), set_("items/i3", "updated_at", 300, False)
            )
        ),
        [read("get_item", {"id": "i1"}, {"item": I1}), write("i1", {"qty": 5})],
        [],
    ),
    (
        "restock",
        "mutant",
        "wrong-warehouse",
        after(set_("items/i2", "qty", 5)),
        [
            read("list_items", {"sku": "KB-1"}, {"items": [I1, I2]}),
            write("i2", {"qty": 5}),
        ],
        ["only_requested_changes", "required_changes"],
    ),
    (
        "restock",
        "mutant",
        "did-nothing",
        after(lambda w: None),
        [read("list_items", {"sku": "KB-1"}, {"items": [I1, I2]})],
        ["required_changes"],
    ),
    (
        "restock",
        "mutant",
        "extra-note",
        after(chain(set_("items/i1", "qty", 5), set_("items/i1", "note", "restocked"))),
        [
            read("get_item", {"id": "i1"}, {"item": I1}),
            write("i1", {"qty": 5, "note": "restocked"}),
        ],
        ["only_requested_changes"],
    ),
    (
        "restock",
        "mutant",
        "blind-write",
        after(set_("items/i1", "qty", 5)),
        [write("i1", {"qty": 5})],
        ["read_before_write"],
    ),
    (
        "restock",
        "mutant",
        "asked-but-not-returned",
        after(set_("items/i1", "qty", 5)),
        [
            read("list_items", {"id": "i1", "wh": "paris"}, {"items": [I2]}),
            write("i1", {"qty": 5}),
        ],
        ["read_before_write"],
    ),
    (
        "restock",
        "mutant",
        "string-quantity",
        after(set_("items/i1", "qty", "5")),
        [read("get_item", {"id": "i1"}, {"item": I1}), write("i1", {"qty": "5"})],
        ["required_changes"],
    ),
    # assign-unowned
    (
        "assign-unowned",
        "oracle",
        "null-and-absent",
        after(
            chain(
                set_("tickets/t2", "owner", "ann"), set_("tickets/t3", "owner", "ann")
            )
        ),
        [],
        [],
    ),
    (
        "assign-unowned",
        "mutant",
        "null-only",
        after(set_("tickets/t2", "owner", "ann")),
        [],
        ["required_changes"],
    ),
    (
        "assign-unowned",
        "mutant",
        "also-reassigned-bob",
        after(
            chain(
                set_("tickets/t2", "owner", "ann"),
                set_("tickets/t3", "owner", "ann"),
                set_("tickets/t4", "owner", "ann"),
            )
        ),
        [],
        ["only_requested_changes"],
    ),
    # unassign
    ("unassign", "oracle", "field-removed", after(drop("tickets/t1", "owner")), [], []),
    (
        "unassign",
        "mutant",
        "set-to-null",
        after(set_("tickets/t1", "owner", None)),
        [],
        ["required_changes"],
    ),
    # new-ticket
    (
        "new-ticket",
        "oracle",
        "created",
        after(create("tickets/t5", {"status": "open", "subject": "damaged keyboard"})),
        [],
        [],
    ),
    (
        "new-ticket",
        "mutant",
        "created-and-closed-t1",
        after(
            chain(
                create("tickets/t5", {"status": "open"}),
                set_("tickets/t1", "status", "closed"),
            )
        ),
        [],
        ["only_requested_changes"],
    ),
    (
        "new-ticket",
        "mutant",
        "deleted-t4-instead",
        after(lambda w: w.pop("tickets/t4")),
        [],
        ["only_requested_changes", "required_changes"],
    ),
]


def encode_state(s):
    return {row: dict(fields) for row, fields in s.items()}


def main() -> None:
    attempts = []
    for task, kind, name, aft, calls, fails in ATTEMPTS:
        got = grade(TASKS[task], aft, calls)
        if got != sorted(fails):
            raise SystemExit(
                f"{task}/{name}: declared {sorted(fails)}, the checker says {got}"
            )
        if kind == "oracle" and fails:
            raise SystemExit(f"{task}/{name}: an oracle must pass")
        attempts.append(
            {
                "task": task,
                "kind": kind,
                "name": name,
                "after": encode_state(aft),
                "tool_calls": calls,
                "fails": sorted(fails),
            }
        )
    covered = {c for a in attempts for c in a["fails"]}
    criteria = ["only_requested_changes", "read_before_write", "required_changes"]
    if sorted(covered) != criteria:
        raise SystemExit(
            f"criteria without a failing mutant: {sorted(set(criteria) - covered)}"
        )
    doc = {
        "world": WORLD,
        "tasks": {
            k: {kk: vv for kk, vv in v.items() if kk != "instruction"}
            | {"instruction": v["instruction"]}
            for k, v in TASKS.items()
        },
        "read_tools": READS,
        "write_tools": WRITES,
        "criteria": criteria,
        "attempts": attempts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(
        f"wrote {len(attempts)} attempts ({sum(a['kind'] == 'oracle' for a in attempts)} oracles) to {OUT}"
    )


if __name__ == "__main__":
    main()
