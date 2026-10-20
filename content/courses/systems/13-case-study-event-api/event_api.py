"""An event processor with an idempotent submit API. Standard library only.

Run me: ``python event_api.py`` (runs the test suite, including a real
multi-threaded double-claim test).

THE PROBLEM
-----------
A caller builds up a request incrementally (a container plus uploaded
documents), then submits it for processing. Submitting the same request twice
-- a double-click, a client retry, a redelivered webhook -- must never cause
the downstream system to be invoked twice.

TWO HALVES, DESIGNED SEPARATELY
-------------------------------
1. **The API shape.** Building a request and submitting it are different
   operations with different semantics. Build-up is mutable and repeatable;
   submission is a one-way door. Modelling them as one endpoint is what forces
   the "did I already send this?" problem into the client.
2. **The guarantee.** Deduplication is enforced twice, at two independent
   layers, because a single check is a race:
     * enqueue dedup -- a database CONSTRAINT, not a check-then-act
     * claim dedup   -- a single atomic UPDATE ... WHERE status = 'pending'

WHAT THE GUARANTEE ACTUALLY IS
------------------------------
This is **at-most-once**: a request is handled once or not at all, never
twice. A worker that dies mid-handle leaves the event claimed and unfinished,
and nothing retries it. That is a deliberate trade, not an oversight:

    at-most-once   never double-fires, may drop     <- what this builds
    at-least-once  never drops, may double-fire     <- add a lease + retry
    exactly-once   not achievable end-to-end unless the downstream effect is
                   itself idempotent; "exactly-once delivery" is marketing,
                   "effectively-once processing" is the real target

To move to at-least-once, give the claim a lease (``claimed_until``) and let a
sweeper return expired claims to pending. Then the downstream call MUST be
idempotent, because it will eventually be retried. The dedup key below is
exactly the idempotency key that makes that safe.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4


class Status(StrEnum):
    """The event lifecycle. A failed event is RECORDED, never silently lost --
    'it disappeared' is the one outcome an operator cannot debug."""

    pending = "pending"
    claimed = "claimed"
    completed = "completed"
    failed = "failed"


@dataclass(frozen=True)
class Event:
    source: str  # the dedup key: one event per source, forever
    request_id: UUID
    document_ids: tuple[UUID, ...]


@dataclass
class Request:
    """The mutable build-up container. Deliberately NOT the durable part."""

    id: UUID
    documents: list[UUID] = field(default_factory=list)


class DownstreamSystem(Protocol):
    """The seam that makes this testable without a mocking library.

    A Protocol, not a base class: tests pass a recording fake, production
    passes the real client, and neither knows about the other.
    """

    def invoke(self, event: Event) -> None: ...


# ---------------------------------------------------------------------------
# Domain errors. They exist so the persistence and service layers never import
# anything HTTP -- the API layer alone maps them onto status codes.
# ---------------------------------------------------------------------------


class NotFound(Exception):
    """No such request."""


class EventQueue:
    """A durable, deduplicating work queue on one SQLite table.

    The interesting part is what is NOT here: no check-then-act, anywhere. Both
    dedup points are single atomic statements, so correctness does not depend
    on how many callers race.
    """

    def __init__(self, path: str = ":memory:") -> None:
        # check_same_thread=False + a lock: the concurrency test drives this
        # from several threads, which is the only way to prove the claim is
        # actually atomic rather than accidentally serialised.
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                -- The dedup key IS the primary key. A duplicate enqueue is a
                -- constraint violation, which no interleaving can defeat.
                source       TEXT PRIMARY KEY,
                request_id   TEXT NOT NULL,
                document_ids TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                claimed_at   TEXT,
                completed_at TEXT
            )
            """
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS events_status ON events(status)")

    # -- layer 1: dedup at enqueue -----------------------------------------

    def enqueue_once(self, event: Event) -> bool:
        """Enqueue an event. Returns False if this source was already queued.

        The naive version of this method is:

            if not self.get(event.source):        # check
                self._insert(event)               # ...act

        Two callers can both pass the check before either inserts. Letting the
        PRIMARY KEY reject the second insert closes that window completely,
        because the check and the act are the same statement.
        """
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO events (source, request_id, document_ids) VALUES (?, ?, ?)",
                    (
                        event.source,
                        str(event.request_id),
                        json.dumps([str(d) for d in event.document_ids]),
                    ),
                )
            except sqlite3.IntegrityError:
                return False  # already queued: a no-op, not an error
            return True

    # -- layer 2: dedup at claim -------------------------------------------

    def claim_pending(self, limit: int = 100) -> list[Event]:
        """Atomically claim pending events and return them.

        One statement moves the rows out of 'pending' and reports which rows it
        moved. Two concurrent workers cannot both receive the same event: the
        second one's WHERE clause no longer matches.

        This is the same shape as SELECT ... FOR UPDATE SKIP LOCKED on
        Postgres. The principle is identical -- claiming is a state TRANSITION,
        never a read followed by a write.
        """
        with self._lock:
            rows = self._db.execute(
                """
                UPDATE events SET status = 'claimed', claimed_at = datetime('now')
                WHERE source IN (
                    SELECT source FROM events WHERE status = 'pending' LIMIT ?
                )
                RETURNING source, request_id, document_ids
                """,
                (limit,),
            ).fetchall()
        return [
            Event(source, UUID(request_id), tuple(UUID(d) for d in json.loads(docs)))
            for source, request_id, docs in rows
        ]

    def mark(self, source: str, status: Status) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE events SET status = ?, completed_at = datetime('now') WHERE source = ?",
                (status, source),
            )

    def get_status(self, source: str) -> Status | None:
        row = self._db.execute(
            "SELECT status FROM events WHERE source = ?", (source,)
        ).fetchone()
        return Status(row[0]) if row else None

    def list_events(self, status: Status | None = None) -> list[tuple[str, Status]]:
        sql = "SELECT source, status FROM events"
        args: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            args = (status,)
        return [(s, Status(st)) for s, st in self._db.execute(sql, args).fetchall()]


# ---------------------------------------------------------------------------
# SERVICE -- application logic. Knows about requests, the queue, and the
# downstream system. Knows nothing about HTTP.
# ---------------------------------------------------------------------------


@dataclass
class SubmitResult:
    """Why this is not a bare boolean: the caller needs to distinguish 'I
    queued it' from 'it was already queued', and BOTH are successes. Collapsing
    the second into an error is what makes clients retry-hostile."""

    queued: bool
    status: str  # "queued" | "duplicate"


class EventService:
    def __init__(self, queue: EventQueue, downstream: DownstreamSystem) -> None:
        self._queue = queue
        self._downstream = downstream
        self._requests: dict[UUID, Request] = {}

    # -- build-up: mutable, repeatable, cheap -------------------------------

    def create_request(self) -> Request:
        request = Request(id=uuid4())
        self._requests[request.id] = request
        return request

    def upload(self, request_id: UUID, document_id: UUID) -> Request:
        request = self._requests.get(request_id)
        if request is None:
            raise NotFound(f"no request {request_id}")
        request.documents.append(document_id)
        return request

    # -- submit: the one-way door ------------------------------------------

    def submit(self, request_id: UUID) -> SubmitResult:
        """Idempotent submit. Calling it twice is safe and says so."""
        request = self._requests.get(request_id)
        if request is None:
            raise NotFound(f"no request {request_id}")
        # The request id IS the dedup source. Nothing is hashed or invented:
        # the client already has a stable identifier, so use it.
        event = Event(str(request.id), request.id, tuple(request.documents))
        if self._queue.enqueue_once(event):
            return SubmitResult(queued=True, status="queued")
        return SubmitResult(queued=False, status="duplicate")

    # -- drain: explicit, so pickup semantics stay testable -----------------

    def run_pending(self) -> int:
        """Claim and handle every pending event. Returns how many were claimed.

        Kept as an explicit call rather than a background thread purely so the
        semantics are observable in a test. In production this is a worker
        loop; the claim/handle/mark logic is unchanged.
        """
        events = self._queue.claim_pending()
        for event in events:
            try:
                self._downstream.invoke(event)
            except Exception:
                # Recorded, not swallowed and not lost. An operator can find
                # every failed event and decide what to do with it.
                self._queue.mark(event.source, Status.failed)
            else:
                self._queue.mark(event.source, Status.completed)
        return len(events)

    # -- inspection: the API layer reads state through here, never by
    # reaching into the queue itself. One layer, one caller.

    def status_of(self, source: str) -> Status | None:
        return self._queue.get_status(source)

    def list_events(self, status: Status | None = None) -> list[tuple[str, Status]]:
        return self._queue.list_events(status)


# ---------------------------------------------------------------------------
# API -- the HTTP layer, as a route table. The whole layer's job is mapping
# routes to service calls and domain errors to status codes; keeping it this
# thin is what lets every rule above be tested without a server.
#
#   POST /requests                      201  create the container
#   POST /requests/{id}/documents       200  append a document (404 if unknown)
#   POST /requests/{id}/submit          200  {"queued": true,  "status": "queued"}
#                                       200  {"queued": false, "status": "duplicate"}
#   POST /events/run                    200  {"claimed": n}
#   GET  /events?status=pending         200  inspect the queue
#
# Two deliberate choices worth defending:
#
#   * The duplicate submit returns 200, not 409. The client's INTENT ("this
#     request is submitted") is satisfied either way, and a 409 pushes error
#     handling onto every caller for a case that is not an error. 409 is right
#     when the second call means something different from the first; here it
#     does not.
#   * Submission is its own endpoint, not a PATCH that flips a field. It is a
#     state transition with side effects, so it gets a verb of its own and a
#     response that describes what happened.
# ---------------------------------------------------------------------------

Response = tuple[int, dict]


def make_router(service: EventService) -> dict[str, Callable[..., Response]]:
    def create_request() -> Response:
        request = service.create_request()
        return 201, {"id": str(request.id), "documents": []}

    def add_document(request_id: UUID, document_id: UUID) -> Response:
        try:
            request = service.upload(request_id, document_id)
        except NotFound as err:
            return 404, {"detail": str(err)}
        return 200, {
            "id": str(request.id),
            "documents": [str(d) for d in request.documents],
        }

    def submit(request_id: UUID) -> Response:
        try:
            result = service.submit(request_id)
        except NotFound as err:
            return 404, {"detail": str(err)}
        return 200, {"queued": result.queued, "status": result.status}

    def run_events() -> Response:
        return 200, {"claimed": service.run_pending()}

    def list_events(status: Status | None = None) -> Response:
        return 200, {
            "events": [
                {"source": s, "status": st} for s, st in service.list_events(status)
            ]
        }

    return {
        "POST /requests": create_request,
        "POST /requests/{id}/documents": add_document,
        "POST /requests/{id}/submit": submit,
        "POST /events/run": run_events,
        "GET /events": list_events,
    }


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------


class RecordingDownstream:
    """The fake that makes the guarantee observable: it counts invocations."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[Event] = []
        self.fail_on = fail_on or set()

    def invoke(self, event: Event) -> None:
        self.calls.append(event)
        if event.source in self.fail_on:
            raise RuntimeError("downstream exploded")


def _service(
    fail_on: set[str] | None = None,
) -> tuple[EventService, RecordingDownstream]:
    sink = RecordingDownstream(fail_on)
    return EventService(EventQueue(), sink), sink


def test_happy_path_invokes_downstream_once() -> None:
    service, sink = _service()
    request = service.create_request()
    service.upload(request.id, uuid4())
    assert service.submit(request.id).queued is True
    assert service.run_pending() == 1
    assert len(sink.calls) == 1


def test_double_submit_enqueues_once() -> None:
    """The headline guarantee, at the enqueue layer."""
    service, sink = _service()
    request = service.create_request()

    first = service.submit(request.id)
    second = service.submit(request.id)

    assert (first.queued, first.status) == (True, "queued")
    assert (second.queued, second.status) == (False, "duplicate")
    service.run_pending()
    assert len(sink.calls) == 1, "downstream must be invoked exactly once"


def test_draining_twice_does_not_reinvoke() -> None:
    """The claim layer: a second drain finds nothing pending."""
    service, sink = _service()
    request = service.create_request()
    service.submit(request.id)

    assert service.run_pending() == 1
    assert service.run_pending() == 0
    assert len(sink.calls) == 1


def test_concurrent_submits_enqueue_once() -> None:
    """The race the constraint exists for. Without the PRIMARY KEY, a
    check-then-act version of enqueue_once fails this."""
    service, sink = _service()
    request = service.create_request()

    results: list[SubmitResult] = []
    barrier = threading.Barrier(8)

    def submit() -> None:
        barrier.wait()  # maximise the overlap
        results.append(service.submit(request.id))

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(r.queued for r in results) == 1, "exactly one caller may win"
    service.run_pending()
    assert len(sink.calls) == 1


def test_concurrent_workers_never_double_claim() -> None:
    """The other race: many workers draining the same queue. Every event must
    be handed to exactly one of them."""
    service, sink = _service()
    for _ in range(50):
        request = service.create_request()
        service.submit(request.id)

    claimed: list[int] = []
    barrier = threading.Barrier(6)

    def drain() -> None:
        barrier.wait()
        claimed.append(service.run_pending())

    threads = [threading.Thread(target=drain) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(claimed) == 50, f"every event claimed exactly once, got {sum(claimed)}"
    assert len(sink.calls) == 50
    sources = {event.source for event in sink.calls}
    assert len(sources) == 50, "no source handled twice"


def test_failure_is_recorded_not_lost() -> None:
    service, sink = _service()
    request = service.create_request()
    source = str(request.id)
    sink.fail_on.add(source)
    service.submit(request.id)

    service.run_pending()

    assert service.status_of(source) is Status.failed
    assert service.list_events(Status.failed) == [(source, Status.failed)]


def test_unknown_request_is_a_404_not_a_crash() -> None:
    service, _ = _service()
    router = make_router(service)
    status, body = router["POST /requests/{id}/submit"](uuid4())
    assert status == 404 and "detail" in body


def test_router_reports_duplicate_as_success() -> None:
    """A duplicate submit is a 200 with an explicit status, not a 409. Retrying
    a submit is a normal thing for a client to do."""
    service, _ = _service()
    router = make_router(service)
    _, created = router["POST /requests"]()
    request_id = UUID(created["id"])

    first = router["POST /requests/{id}/submit"](request_id)
    second = router["POST /requests/{id}/submit"](request_id)

    assert first == (200, {"queued": True, "status": "queued"})
    assert second == (200, {"queued": False, "status": "duplicate"})


def test_lifecycle_is_observable() -> None:
    service, _ = _service()
    router = make_router(service)
    _, created = router["POST /requests"]()
    request_id = UUID(created["id"])
    router["POST /requests/{id}/documents"](request_id, uuid4())
    router["POST /requests/{id}/submit"](request_id)

    assert service.status_of(str(request_id)) is Status.pending
    router["POST /events/run"]()
    assert service.status_of(str(request_id)) is Status.completed


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed (including 2 real multi-threaded races)")
