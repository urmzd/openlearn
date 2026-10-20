#!/bin/python3.9
import numpy as np
from pprint import pprint
import sys


def global_alignment(S: str, T: str, M: list[list[int]]):
    i_end = len(M) - 1
    j_end = len(M[-1]) - 1

    if i_end - 1 < 0 or j_end - 1 < 0:
        if i_end - 1 < 0 and j_end - 1 < 0:
            return "", ""
        elif i_end - 1 < 0:
            return "-", T[j_end - 1]
        else:
            return S[j_end - 1], "-"

    change = M[i_end - 1][j_end - 1]
    insert = M[i_end - 1][j_end]
    delete = M[i_end][j_end - 1]

    optimal_choice = min(change, insert, delete)

    if optimal_choice == change:
        S_ = S[i_end - 1]
        T_ = T[j_end - 1]
        _S, _T = global_alignment(S, T, np.array(M)[:i_end, :j_end].tolist())
    elif optimal_choice == insert:
        S_ = S[i_end - 1]
        T_ = "-"
        _S, _T = global_alignment(S, T, np.array(M)[:i_end, : j_end + 1].tolist())
    else:
        S_ = "-"
        T_ = T[j_end - 1]
        _S, _T = global_alignment(S, T, np.array(M)[: i_end + 1, :j_end].tolist())

    return _S + S_, _T + T_


def compute_cost(S: str, T: str) -> list[list[int]]:
    len_s, len_t = len(S) + 1, len(T) + 1

    M = [list(range(len_t)) if not i else [i] + [0] * (len_t - 1) for i in range(len_s)]

    for i in range(1, len_s):
        for j in range(1, len_t):
            match_cost = S[i - 1] != T[j - 1]

            change_cost = M[i - 1][j - 1] + match_cost
            insert_cost = M[i - 1][j] + 1
            delete_cost = M[i][j - 1] + 1

            M[i][j] = min(change_cost, insert_cost, delete_cost)

    pprint(M)
    return M


if __name__ == "__main__":
    S, T = sys.argv[1:]

    M = compute_cost(S, T)
    print("The alignments are:")
    print(*global_alignment(S, T, M), sep="\n")
