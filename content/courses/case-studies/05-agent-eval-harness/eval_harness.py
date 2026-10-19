"""A deterministic evaluation harness for an agent that changes state.

Standard library only. Run me: ``python eval_harness.py``.

WHY THIS EXISTS
---------------
Grading a model that only writes text is a comparison. Grading an agent that
calls tools is harder, because the answer is a *changed world*: rows updated,
rows created, and an order in which that happened. Three questions have to be
answered separately, and mixing them is the most common way a harness lies:

    1. Did execution end?           (outcome:  completed, error, limit)
    2. Can the evidence be trusted? (replay:   verified, or UNGRADED)
    3. Was the task done correctly? (grade:    every criterion passed)

``completed`` is not ``passed``. An attempt can finish cleanly after editing the
wrong row.

THE GRADE IS A SET RELATION
---------------------------
Let ``delta = diff(start, final)``. Each task declares ``required`` changes and
``allowed`` changes, both derived from the starting state. The state grade is:

    required  is a subset of  delta  is a subset of  allowed

The left half catches work that was not done. The right half catches work that
should not have been done. Final state alone cannot see a read that never
happened, so trace criteria sit beside the state criteria.

THE HARNESS IS CODE, SO IT NEEDS TESTS
--------------------------------------
A checker that passes everything is indistinguishable from a checker that works
until someone hands it a wrong answer. So every task ships with:

    an ORACLE   a scripted driver that solves the task correctly
    MUTANTS     the oracle with one deliberate mistake

The oracle must pass. Every mutant must fail, on the criterion it was built to
break. Every criterion must have at least one mutant that fails it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

MISSING = "<absent>"  # a field that is not present, which is not the same as null
ROW = "*row"  # pseudo-field that stands for "this whole row was created"
EPOCH = 1_000  # the logical clock starts here, not at wall time

Row = dict[str, Any]
Key = tuple[str, str, str]  # (entity, row id, field)


# --------------------------------------------------------------------------
# World: immutable. Layer: a thin private overlay. Nothing is copied up front.
# --------------------------------------------------------------------------


class World:
    """The shared starting state. Never mutated after construction."""

    def __init__(self, rows: dict[str, dict[str, Row]]) -> None:
        self._rows = copy.deepcopy(rows)
        canonical = json.dumps(self._rows, sort_keys=True).encode()
        self.hash = hashlib.sha256(canonical).hexdigest()[:12]

    def get(self, entity: str, row_id: str) -> Row | None:
        row = self._rows.get(entity, {}).get(row_id)
        return copy.deepcopy(row)  # a caller must not reach shared memory

    def ids(self, entity: str) -> list[str]:
        return sorted(self._rows.get(entity, {}))


class Layer:
    """Copy-on-write rows over a parent. A read falls through on a miss."""

    def __init__(self, parent: World) -> None:
        self.parent = parent
        self.rows: dict[tuple[str, str], Row] = {}

    def get(self, entity: str, row_id: str) -> Row | None:
        if (entity, row_id) in self.rows:
            return copy.deepcopy(self.rows[(entity, row_id)])
        return self.parent.get(entity, row_id)

    def put(self, entity: str, row_id: str, row: Row) -> None:
        self.rows[(entity, row_id)] = copy.deepcopy(row)

    def ids(self, entity: str) -> list[str]:
        own = {rid for ent, rid in self.rows if ent == entity}
        return sorted(own | set(self.parent.ids(entity)))


# --------------------------------------------------------------------------
# Session: one attempt's private state, clock, id counter, and ordered trace.
# --------------------------------------------------------------------------


@dataclass
class Call:
    seq: int
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    writes: list[dict[str, Any]] = field(default_factory=list)


class Session:
    PAGE_MAX = 2  # small on purpose, so pagination is exercised by tiny data

    def __init__(self, world: World) -> None:
        self.world = world
        self.layer = Layer(world)
        self.clock = EPOCH
        self.counter = 0
        self.trace: list[Call] = []

    # -- determinism: time and ids are functions of the call sequence ------

    def _tick(self) -> int:
        self.clock += 1
        return self.clock

    def _new_id(self, entity: str) -> str:
        self.counter += 1
        digest = hashlib.sha256(f"{entity}:{self.counter}".encode()).hexdigest()
        return f"N-{digest[:8]}"

    # -- the only door a driver has ----------------------------------------

    def call(self, tool: str, **args: Any) -> dict[str, Any]:
        handler = getattr(self, f"_tool_{tool}", None)
        writes: list[dict[str, Any]] = []
        if handler is None:
            result = {"ok": False, "error": "unknown_tool"}
        else:
            result = handler(writes, **args)
        entry = Call(len(self.trace) + 1, tool, copy.deepcopy(args), result, writes)
        self.trace.append(entry)
        return copy.deepcopy(result)

    def _tool_list_items(
        self,
        writes: list[dict[str, Any]],
        warehouse: str | None = None,
        limit: int = PAGE_MAX,
        offset: int = 0,
    ) -> dict[str, Any]:
        rows = [self.layer.get("items", rid) for rid in self.layer.ids("items")]
        rows = [r for r in rows if r and warehouse in (None, r["warehouse"])]
        limit = max(1, min(limit, self.PAGE_MAX))
        page = rows[offset : offset + limit]
        end = offset + len(page)
        return {
            "ok": True,
            "items": page,
            "total": len(rows),
            "next_offset": end if end < len(rows) else None,
        }

    def _tool_get_item(self, writes: list[dict[str, Any]], id: str) -> dict[str, Any]:
        row = self.layer.get("items", id)
        if row is None:
            return {"ok": False, "error": "not_found"}
        return {"ok": True, "item": row}

    def _tool_update_item(
        self, writes: list[dict[str, Any]], id: str, **fields: Any
    ) -> dict[str, Any]:
        before = self.layer.get("items", id)
        if before is None:
            return {"ok": False, "error": "not_found"}
        after = {**before, **fields}
        if after == before:
            # Refusing a no-op gives the driver feedback and keeps it visible.
            return {"ok": False, "error": "redundant_update"}
        after["updated_at"] = self._tick()
        self.layer.put("items", id, after)
        writes.append({"entity": "items", "id": id, "before": before, "after": after})
        return {"ok": True, "item": after}

    def _tool_add_note(
        self, writes: list[dict[str, Any]], item_id: str, text: str
    ) -> dict[str, Any]:
        if self.layer.get("items", item_id) is None:
            return {"ok": False, "error": "not_found"}
        note_id = self._new_id("notes")
        note = {"id": note_id, "item_id": item_id, "text": text, "at": self._tick()}
        self.layer.put("notes", note_id, note)
        writes.append({"entity": "notes", "id": note_id, "before": None, "after": note})
        return {"ok": True, "note": note}

    def final_rows(self) -> dict[str, Row]:
        return {f"{e}/{i}": copy.deepcopy(r) for (e, i), r in self.layer.rows.items()}


# --------------------------------------------------------------------------
# Record and replay: evidence that cannot be reproduced is not graded.
# --------------------------------------------------------------------------


@dataclass
class Record:
    world_hash: str
    calls: list[Call]
    final: dict[str, Row]
    outcome: str = "completed"


def record_of(session: Session) -> Record:
    return Record(
        session.world.hash, copy.deepcopy(session.trace), session.final_rows()
    )


def replay(record: Record, world: World) -> str | None:
    """Re-execute the recorded calls. Return a reason, or None when consistent.

    This proves the record is internally consistent. It does not prove the
    record came from a real execution.
    """
    if record.world_hash != world.hash:
        return "world hash mismatch"
    session = Session(world)
    for recorded in record.calls:
        result = session.call(recorded.tool, **recorded.args)
        replayed = session.trace[-1]
        if result != recorded.result:
            return f"call {recorded.seq}: observed result differs"
        if replayed.writes != recorded.writes:
            return f"call {recorded.seq}: write evidence differs"
    if session.final_rows() != record.final:
        return "final rows differ"
    return None


# --------------------------------------------------------------------------
# Checker: pure and deterministic. It never calls a model.
# --------------------------------------------------------------------------

Expect = Callable[[World], dict[Key, Any]]
Allow = Callable[[World], set[Key]]


@dataclass
class Task:
    id: str
    instruction: str
    required: Expect  # key -> the value it must end with
    allowed: Allow  # every key that may change
    read_before_write: bool = False


def diff(world: World, final: dict[str, Row]) -> dict[Key, Any]:
    """delta: every field that differs from the start, keyed by row and field."""
    delta: dict[Key, Any] = {}
    for path, after in final.items():
        entity, row_id = path.split("/")
        before = world.get(entity, row_id)
        if before is None:
            delta[(entity, row_id, ROW)] = "created"
            continue
        for name in sorted(set(before) | set(after)):
            if name == "updated_at":
                continue  # bookkeeping, not a task-visible change
            old, new = before.get(name, MISSING), after.get(name, MISSING)
            if old != new:
                delta[(entity, row_id, name)] = new
    return delta


def grade(task: Task, record: Record, world: World) -> dict[str, Any]:
    reason = replay(record, world)
    if reason is not None:
        return {"status": "UNGRADED", "reason": reason, "criteria": {}, "passed": False}

    delta = diff(world, record.final)
    required, allowed = task.required(world), task.allowed(world)
    criteria: dict[str, list[str]] = {}

    missing = [k for k, want in required.items() if delta.get(k, MISSING) != want]
    criteria["required_changes"] = [f"missing {k}" for k in missing]

    extra = [k for k in delta if k not in allowed]
    criteria["only_requested_changes"] = [f"unauthorized {k}" for k in extra]

    if task.read_before_write:
        criteria["read_before_write"] = _unread_writes(record)

    return {
        "status": "GRADED",
        "criteria": criteria,
        "passed": all(not evidence for evidence in criteria.values()),
    }


def _unread_writes(record: Record) -> list[str]:
    """A write counts as informed only if a read RETURNED that row earlier.
    A page that was never fetched is not a read."""
    seen: set[str] = set()
    problems = []
    for call in record.calls:
        result = call.result
        if call.tool == "list_items" and result["ok"]:
            seen |= {row["id"] for row in result["items"]}
        if call.tool == "get_item" and result["ok"]:
            seen.add(result["item"]["id"])
        for write in call.writes:
            if write["entity"] == "items" and write["id"] not in seen:
                problems.append(f"call {call.seq} wrote {write['id']} unread")
    return problems


def diagnose(record: Record) -> list[str]:
    """Advisory observations about WHY. They never change a grade, and a
    passing attempt can carry one."""
    modes = []
    calls = record.calls
    pages = [c for c in calls if c.tool == "list_items" and c.result["ok"]]
    for page in pages:
        nxt = page.result["next_offset"]
        followed = any(c.args.get("offset") == nxt for c in pages)
        if nxt is not None and not followed:
            modes.append("unfollowed_page")
            break
    if not any(c.writes for c in calls):
        modes.append("no_writes")
    if any(not c.result["ok"] for c in calls):
        modes.append("refused_calls")
    if _unread_writes(record):
        modes.append("write_without_read")
    return modes


# --------------------------------------------------------------------------
# A small world with the traps real data has.
# --------------------------------------------------------------------------


def seed() -> World:
    def item(row_id: str, sku: str, wh: str, qty: int, **extra: Any) -> Row:
        return {"id": row_id, "sku": sku, "warehouse": wh, "quantity": qty, **extra}

    return World(
        {
            "items": {
                "I-1": item("I-1", "motor", "columbus", 12, owner="ana"),
                "I-2": item("I-2", "motor", "reno", 7, owner="ben"),
                "I-3": item("I-3", "cable", "reno", 90),  # owner ABSENT
                "I-4": item("I-4", "valve", "reno", 4, owner=None),  # owner NULL
                "I-5": item("I-5", "gasket", "reno", 31),  # owner ABSENT
                "I-6": item("I-6", "valve", "columbus", 9, owner=None),
            },
            "notes": {},
        }
    )


def _reno_unowned(world: World) -> list[str]:
    rows = [world.get("items", rid) for rid in world.ids("items")]
    return [
        r["id"]
        for r in rows
        if r and r["warehouse"] == "reno" and r.get("owner") is None
    ]


TASKS = {
    "correct-count": Task(
        id="correct-count",
        instruction="Set the motor count in the Reno warehouse to 40.",
        required=lambda w: {("items", "I-2", "quantity"): 40},
        allowed=lambda w: {("items", "I-2", "quantity")},
    ),
    "assign-unowned": Task(
        id="assign-unowned",
        instruction="Give every Reno item with no owner to ops.",
        # The target set is DERIVED from the start. Nothing is hardcoded, so
        # the same task grades a seed with three targets or three thousand.
        required=lambda w: {("items", i, "owner"): "ops" for i in _reno_unowned(w)},
        allowed=lambda w: {("items", i, "owner") for i in _reno_unowned(w)},
        read_before_write=True,
    ),
}


# --------------------------------------------------------------------------
# Drivers: the oracle, and the oracle with one deliberate mistake.
# --------------------------------------------------------------------------


def _read_all(session: Session, warehouse: str, pages: int | None = None) -> list[Row]:
    rows, offset, fetched = [], 0, 0
    while offset is not None and (pages is None or fetched < pages):
        page = session.call("list_items", warehouse=warehouse, offset=offset)
        rows += page["items"]
        offset, fetched = page["next_offset"], fetched + 1
    return rows


def drive(task_id: str, session: Session, fault: str | None = None) -> None:
    if task_id == "correct-count":
        target = "I-1" if fault == "wrong-warehouse" else "I-2"
        if fault != "no-writes":
            session.call("update_item", id=target, quantity=40)
        if fault == "extra-note":
            session.call("add_note", item_id="I-2", text="count corrected")
        return

    if fault == "blind-write":  # correct rows, but nothing was read first
        for row_id in ("I-3", "I-4", "I-5"):
            session.call("update_item", id=row_id, owner="ops")
        return

    rows = _read_all(session, "reno", pages=1 if fault == "first-page-only" else None)
    for row in rows:
        unowned = "owner" in row and row["owner"] is None
        if fault != "skip-absent":
            unowned = row.get("owner") is None  # absent AND null both count
        if unowned:
            session.call("update_item", id=row["id"], owner="ops")
    if fault == "extra-write":
        session.call("get_item", id="I-6")
        session.call("update_item", id="I-6", owner="ops")


# fault -> (task, the criteria that MUST fail)
MUTANTS = {
    "wrong-warehouse": (
        "correct-count",
        {"required_changes", "only_requested_changes"},
    ),
    "no-writes": ("correct-count", {"required_changes"}),
    "extra-note": ("correct-count", {"only_requested_changes"}),
    "skip-absent": ("assign-unowned", {"required_changes"}),
    "first-page-only": ("assign-unowned", {"required_changes"}),
    "extra-write": ("assign-unowned", {"only_requested_changes"}),
    "blind-write": ("assign-unowned", {"read_before_write"}),
}


def attempt(task_id: str, fault: str | None = None) -> tuple[Record, dict[str, Any]]:
    world = seed()
    session = Session(world)
    drive(task_id, session, fault)
    record = record_of(session)
    return record, grade(TASKS[task_id], record, world)


def failed(result: dict[str, Any]) -> set[str]:
    return {name for name, evidence in result["criteria"].items() if evidence}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_sessions_are_isolated() -> None:
    world = seed()
    a, b = Session(world), Session(world)
    a.call("update_item", id="I-2", quantity=1)
    assert b.call("get_item", id="I-2")["item"]["quantity"] == 7
    assert world.get("items", "I-2")["quantity"] == 7


def test_a_returned_row_cannot_reach_shared_state() -> None:
    session = Session(seed())
    row = session.call("get_item", id="I-2")["item"]
    row["quantity"] = 999  # the caller scribbles on what it was handed
    assert session.call("get_item", id="I-2")["item"]["quantity"] == 7


def test_absent_and_null_stay_different() -> None:
    """A write must not normalise fields it was not asked to touch."""
    session = Session(seed())
    session.call("update_item", id="I-3", quantity=91)
    assert "owner" not in session.call("get_item", id="I-3")["item"]
    session.call("update_item", id="I-4", quantity=5)
    assert session.call("get_item", id="I-4")["item"]["owner"] is None


def test_ids_and_time_depend_only_on_the_call_sequence() -> None:
    def run() -> dict[str, Any]:
        session = Session(seed())
        session.call("update_item", id="I-2", quantity=40)
        return session.call("add_note", item_id="I-2", text="x")["note"]

    assert run() == run()
    assert run()["at"] == EPOCH + 2  # logical time, never the wall clock


def test_pagination_reports_the_truth() -> None:
    session = Session(seed())
    first = session.call("list_items", warehouse="reno", limit=50)
    assert len(first["items"]) == Session.PAGE_MAX  # a large limit is capped
    assert first["total"] == 4 and first["next_offset"] == 2
    last = session.call("list_items", warehouse="reno", offset=2)
    assert last["next_offset"] is None  # the final page ends exactly


def test_redundant_and_unknown_calls_are_refused_and_recorded() -> None:
    session = Session(seed())
    assert session.call("update_item", id="I-2", quantity=7)["error"] == (
        "redundant_update"
    )
    assert session.call("delete_everything")["error"] == "unknown_tool"
    assert [c.tool for c in session.trace] == ["update_item", "delete_everything"]
    assert session.final_rows() == {}


def test_every_oracle_passes() -> None:
    for task_id in TASKS:
        _, result = attempt(task_id)
        assert result["passed"], f"{task_id}: {result['criteria']}"


def test_every_mutant_fails_the_criteria_it_was_built_to_break() -> None:
    for fault, (task_id, expected) in MUTANTS.items():
        record, result = attempt(task_id, fault)
        assert record.outcome == "completed"  # it ran fine. It is still wrong.
        assert not result["passed"], f"{fault} slipped through"
        assert failed(result) == expected, f"{fault}: {failed(result)} != {expected}"


def test_every_criterion_has_a_mutant_that_fails_it() -> None:
    """A criterion no mutant can fail is a criterion nobody has tested."""
    covered: set[tuple[str, str]] = set()
    for fault, (task_id, expected) in MUTANTS.items():
        covered |= {(task_id, name) for name in expected}
    for task_id in TASKS:
        _, result = attempt(task_id)
        for name in result["criteria"]:
            assert (task_id, name) in covered, f"untested: {task_id}/{name}"


def test_the_checker_is_deterministic() -> None:
    record, first = attempt("assign-unowned", "skip-absent")
    assert all(grade(TASKS["assign-unowned"], record, seed()) == first for _ in "abc")


def test_a_substituted_read_is_rejected() -> None:
    """The bug that motivates replay: swap one recorded observation and a
    failing attempt can look like a passing one."""
    record, _ = attempt("assign-unowned")
    record.calls[0].result["items"][0]["owner"] = None
    result = grade(TASKS["assign-unowned"], record, seed())
    assert result["status"] == "UNGRADED" and not result["passed"]


def test_an_edited_final_state_is_rejected() -> None:
    record, _ = attempt("correct-count", "wrong-warehouse")
    record.final = {"items/I-2": {**seed().get("items", "I-2"), "quantity": 40}}
    result = grade(TASKS["correct-count"], record, seed())
    assert result["status"] == "UNGRADED"


def test_a_record_from_another_world_is_rejected() -> None:
    record, _ = attempt("correct-count")
    record.world_hash = "0" * 12
    assert grade(TASKS["correct-count"], record, seed())["status"] == "UNGRADED"


def test_diagnostics_explain_but_never_grade() -> None:
    record, result = attempt("assign-unowned", "first-page-only")
    assert "unfollowed_page" in diagnose(record)
    assert result == grade(TASKS["assign-unowned"], record, seed())

    record, result = attempt("correct-count", "no-writes")
    assert diagnose(record) == ["no_writes"]

    # A passing attempt can still carry an observation.
    record, result = attempt("correct-count")
    assert result["passed"] and diagnose(record) == ["write_without_read"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")

    print()
    for fault, (task_id, _) in [(None, ("correct-count", None)), *MUTANTS.items()]:
        record, result = attempt(task_id, fault)
        verdict = "PASS" if result["passed"] else "FAIL"
        why = ", ".join(sorted(failed(result))) or "-"
        label = fault or "oracle"
        print(f"{verdict}  {task_id:<15} {label:<16} {record.outcome:<10} {why}")

    print(
        f"\n{len(tests)} tests passed over {len(TASKS)} oracles "
        f"and {len(MUTANTS)} mutants"
    )
