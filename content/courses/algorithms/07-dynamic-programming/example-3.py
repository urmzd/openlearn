#!/bin/python3.9
import sys


"""
    The following algorithm's correctness can be explained by the following
    observations.

    1. We having a running counter which either starts at and includes the
        current index, OR we have a running counter which includes
        all values up to and including the current index.

    Note: In other words, we have a subarray containing all values which
    will produce the CURRENT max.

    2. We then choose the previous subarray if the total sum is greater than
    the current subarray producing the current max.
"""


def maximum_subarray(nums: list[int]) -> int:
    high_max = -(10**5) - 1
    current_max = -(10**5) - 1

    for index in range(len(nums)):
        current_max = max(nums[index], current_max + nums[index])

        high_max = max(high_max, current_max)

    return high_max


if __name__ == "__main__":
    nums = [int(x) for x in sys.argv[1:]]
    print("The maximum sum is:")
    print(maximum_subarray(nums))
