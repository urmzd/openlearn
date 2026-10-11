"""The Python half of the course fault and determinism kit (DESIGN 4.4).

    flakyhttp   a fixture HTTP server: 5xx, truncated bodies, Range, checksum lies
    failpoint   TL_FAILPOINTS named failpoints (the same spec as the Go, Rust, and C kits)
    clock       a fake clock for contracts that take a Clock

`ol check` puts course/testkit/python on PYTHONPATH, so course tests and the
learner's graded tests import `sstestkit.flakyhttp` and friends.
"""
