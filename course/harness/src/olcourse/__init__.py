"""olcourse: the course half of practice/bin/ol.

Bash keeps the practice kinds (predict, build, reattempt) and `ol learn`.
Everything addressed by a course id (course/DESIGN.md 3.5) lands here.

Exit codes are the verdict (DESIGN 5.3):
  0 pass, 1 fail, 2 not started, 3 blocked by deps, 4 contract drift,
  5 harness or toolchain error.
"""

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_NOT_STARTED = 2
EXIT_BLOCKED = 3
EXIT_DRIFT = 4
EXIT_HARNESS = 5


class HarnessError(Exception):
    """A harness or toolchain failure: exit 5 unless `code` says otherwise."""

    def __init__(self, msg: str, code: int = EXIT_HARNESS):
        super().__init__(msg)
        self.code = code
