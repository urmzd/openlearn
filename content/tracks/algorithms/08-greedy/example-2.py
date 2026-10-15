#!/bin/python3.9
from typing import Iterator
import sys


def greedy_algorithm(d: int, pitstops: Iterator[int]) -> Iterator[int]:
    """
    @param d: Dr. Gagie's max lung capacity in terms of distance.
    @param pitstops: List of integers indicating distance between two consecutive pitstops.
    @returns: Indices of pitstops that should be used.
    """

    current_distance = 0
    previous_pitstop_index = 0

    for pitstop_index, pitstop in enumerate(pitstops):
        if current_distance + pitstop > d:
            # For every iteration, we pass i+1 pitstops.
            # This occurs because we start at a pitstop.
            yield previous_pitstop_index + 1
            current_distance = 0

        current_distance += pitstop
        previous_pitstop_index = pitstop_index


if __name__ == "__main__":
    d = int(sys.argv[1])
    pitstops = map(int, sys.argv[2:])
    for pitstop in greedy_algorithm(d, pitstops):
        print(f"Stop at pitstop #{pitstop}.")
