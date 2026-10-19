#!/bin/python3.9
import sys
from typing import Iterator


def greedy_algorithm(h, questions: list[int]) -> Iterator[int]:
    hours_sorted = sorted(questions)
    return map(
        lambda x: x[0] + 1,
        filter(
            lambda i: sum(hours_sorted[: i[0] + 1]) <= h,
            enumerate(hours_sorted),
        ),
    )


if __name__ == "__main__":
    h = int(sys.argv[1])
    questions = list(map(int, sys.argv[2:]))

    for question in greedy_algorithm(h, questions):
        print(f"Answer question #{question}")
