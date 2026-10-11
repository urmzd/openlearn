"""The Hypothesis profile `ol` as a pytest plugin (DESIGN 5.9).

Course tests get the profile from course/tests/conftest.py. The learner's
graded tests run from their own directory, where that conftest is never
loaded, so `ol mutate`, `ol tdd`, and the mutation grade in `ol check` load
this module with `-p _lib.ol_hypothesis`: property tests are derandomized,
write no example database, and have no deadline, so a mutant is killed or
survives the same way on every run.
"""

from __future__ import annotations

PROFILE = {
    "derandomize": True,
    "database": None,
    "deadline": None,
    "max_examples": 100,
}

try:
    from hypothesis import settings
except ImportError:  # a module without property tests needs no Hypothesis
    settings = None


def register() -> bool:
    if settings is None:
        return False
    settings.register_profile("ol", **PROFILE)
    settings.load_profile("ol")
    return True


register()


def pytest_report_header(config):
    return (
        "hypothesis profile: ss (derandomized, no database, no deadline)"
        if settings
        else None
    )
