"""The Python half of the course fault and determinism kit.

    flakyhttp   a fixture HTTP server: 5xx, truncated bodies, Range, checksum lies
    failpoint   TL_FAILPOINTS named failpoints (the same spec as the Go, Rust, and C kits)
    clock       a fake clock for contracts that take a Clock

Put course/solutions/testkit/python on PYTHONPATH to import
`sstestkit.flakyhttp` and friends from the model test suites.
"""
