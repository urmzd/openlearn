from typing import List
import sys


class Solution:
    # 1. Shift array right by 1 element and calculate its sum.
    # 2. If the sum is greater than the sum of the array shifted left 1 element,
    #    assign max variable the current sum.
    # 3. Repeat 1 and 2 until the number of right shifts
    #    equal to the length of the array.
    # 4. Output max array.
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        max_sum = -(10**5) - 1

        for i in range(len(nums)):
            rotated_array = nums[-i:] + nums[:-i]
            subarray_sum = self.maxSubArray(rotated_array)

            if subarray_sum > max_sum:
                max_sum = subarray_sum

        return max_sum

    def maxSubArray(self, nums: List[int]) -> int:
        sum_subarray, *rest = self.get_max_subarray(nums, 0, len(nums) - 1)
        mid = (len(nums) - 1) // 2

        return sum_subarray

    def get_max_crossing_subarray(
        self, array: list[int], low: int, mid: int, high: int
    ) -> tuple[int, int, int]:
        total_sum = 0

        left_index = right_index = 0
        left_sum = right_sum = -(10 ** (5)) - 1

        index = mid
        while index >= low:
            value = array[index]
            total_sum += value
            if total_sum > left_sum:
                left_sum = total_sum
                left_index = index
            index -= 1

        total_sum = 0

        right_index = 0
        index = mid + 1

        while index <= high:
            value = array[index]
            total_sum += value
            if total_sum > right_sum:
                right_sum = total_sum
                right_index = index
            index += 1

        return (left_sum + right_sum, left_index, right_index)

    def get_max_subarray(
        self, array: list[int], low: int, high: int
    ) -> tuple[int, int, int]:
        if high == low:
            return (array[low], low, high)

        mid = (low + high) // 2

        lmsa = self.get_max_subarray(array, low, mid)
        rmsa = self.get_max_subarray(array, mid + 1, high)
        cmsa = self.get_max_crossing_subarray(array, low, mid, high)

        return max(lmsa, rmsa, cmsa, key=lambda x: x[0])


# Execute test case
# Run python q5.py %name_of_input_file_here%
if __name__ == "__main__":
    file = open(sys.argv[1], "r")
    array = [int(x) for x in file.read().split(",")]

    sol = Solution()
    print(sol.maxSubArray(list(array)))
    print(sol.maxSubarraySumCircular(list(array)))

    # args = [int(s) for s in sys.argv[1:]]
    # print(sol.maxSubarraySumCircular(args))
