#!/bin/python3.9
import sys

# Define definitions.
Town = str
Company = str
Buses = list[Company]
Route = dict[Company, set[Town]]
Map = dict[Town, Route]
Pr = dict[str, float]

# Define example.
S: Town = "A"
B: Buses = ["R", "G", "B", "G"]
M: Map = {
    "A": {
        "R": {"B", "C"},
        "G": {
            "C",
        },
        "B": {"B", "D"},
    },
    "B": {
        "R": {
            "B",
        },
        "G": {"C", "D"},
        "B": {
            "A",
        },
    },
    "C": {
        "R": {
            "A",
        },
        "G": {"A", "B"},
        "B": {
            "D",
        },
    },
    "D": {
        "G": {
            "B",
        },
        "B": {"A", "C"},
    },
}


def saving_private_ryan(S: Town = S, B: Buses = B, M: Map = M) -> Pr:
    paths = [{S: M[S][B[0]]}]
    pr = [{S: 1.0}]

    for trip in range(1, len(B)):
        paths.append(
            {
                curr_city: M[curr_city][B[trip]]
                for prev_city in paths[trip - 1]
                for curr_city in paths[trip - 1][prev_city]
            }
        )

    for trip in range(1, len(B) + 1):
        current_probabilities = {}
        for prev_city in paths[trip - 1]:
            for curr_city in paths[trip - 1][prev_city]:
                curr_pr = pr[trip - 1].get(prev_city, 0) / len(
                    paths[trip - 1][prev_city]
                )
                prev_pr = current_probabilities.get(curr_city, 0)
                current_probabilities[curr_city] = curr_pr + prev_pr

        pr.append(current_probabilities)

    return pr[-1]


if __name__ == "__main__":
    print("The probabilities are:")
    if len(sys.argv) == 1:
        print(saving_private_ryan())
    else:
        S_ = sys.argv[1]
        string_input = " ".join(sys.argv[2:])
        B_, *M__ = string_input.split(",")
        B_ = B_.strip().split(" ")
        M__ = [(m.strip().split(" ")) for m in M__]
        M__ = [(*m[:2], set(m[2:])) for m in M__]
        M_ = {}

        for m in M__:
            start, company, destinations = m
            if start not in M_:
                M_[start] = {}

            M_[start][company] = destinations

        print(saving_private_ryan(S_, B_, M_))
