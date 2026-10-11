"""ol where (course part)   locations the course harness is using"""

from __future__ import annotations

from .course import main as course_main


def main(argv: list[str]) -> int:
    return course_main(["where"])
