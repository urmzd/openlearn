#!/bin/python3.9
from math import ceil
from pprint import pprint
import sys
import numpy as np

Item = tuple[int, int]
Items = list[Item]
CostMatrix = list[list[float]]

"""
    On the assumption that the binary knapsack solution given in lecture
    is correct, we can demonstrate that the following algorithm is also 
    correct by observing the following:
    
    1. We extend the solution. Thus in scenerios where 
        no squashing is required, we arrive to the optimal solution. 
    2. We add another choice into our selection pool, which allows for 
        previously unselectable items to be selected. This allows us to
        gain a total profit that is greater or equal to a 
        profit gained in a non-squashing binary knapsack problem.
"""


def binary_knapsack(I: Items, V_MAX: int, M: CostMatrix, S: Items) -> Items:
    """
    @param I: Items containg a price associated with a volume.
    @param V_MAX: The volume capacity of a bag.
    @param M:
        A matrix containing the max profit given a given weight and price.
    @param S:
        A list of squashed items.
    """
    i = len(M) - 1
    j = (len(M[-1]) if M else 0) - 1

    # If cost matrix is empty, return.
    if i <= 0 or j <= 0:
        return []

    # If the current item adds no profit, continue.
    if M[i][j] == M[i][j - 1]:
        return binary_knapsack(I, V_MAX, M[: i - 1], S)

    # If the current item adds profit, pick it and continue.
    return [I[i - 1]] + binary_knapsack(
        I, V_MAX - I[i - 1][1], np.array(M)[: i - 1, : V_MAX - I[i - 1][1]].tolist(), S
    )


def compute_cost(I: Items, V_MAX: int) -> tuple[CostMatrix, Items]:

    B = [
        [0.0] + [-float("inf")] * V_MAX if not item_no else [0.0] * (V_MAX + 1)
        for item_no in range(len(I) + 1)
    ]

    S = []

    for item_no in range(1, len(B)):
        for max_v in range(1, len(B[item_no])):
            p, v = I[item_no - 1]
            squashed_pick = (
                B[item_no - 1][max_v - ceil(v / 2)] + p / 2
                if max_v - v / 2 >= 0
                else -float("inf")
            )
            unsquashed_pick = (
                B[item_no - 1][max_v - v] + p if max_v - v >= 0 else -float("inf")
            )
            reset_pick = B[item_no - 1][max_v]
            B[item_no][max_v] = max(unsquashed_pick, squashed_pick, reset_pick)

            if B[item_no][max_v] == squashed_pick and squashed_pick > 0:
                S.append((item_no, max_v))

    pprint(B)
    pprint(S)

    return B, S


if __name__ == "__main__":
    P = [int(x) for x in sys.argv[2::2]]
    V = [int(x) for x in sys.argv[3::2]]
    I = list(zip(P, V))

    V_MAX = int(sys.argv[1])
    M, S = compute_cost(I, V_MAX)
    PICK = binary_knapsack(I, V_MAX, *compute_cost(I, V_MAX))

    print("PICK")
    pprint(PICK)

    for s in S:
        if s in PICK:
            print(f"SQUASH {s}")
