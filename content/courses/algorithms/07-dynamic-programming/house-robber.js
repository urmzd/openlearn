/*
 * @param {number[]} nums
 * @return {number}
 */
var rob = function(nums) {
    return recurse(nums)
};

function recurse(nums, i = 0, prev = 0, prev2 = 0, curr = 0) {
    if (i >= nums.length) {
        return curr;
    }
    const newMax = Math.max(curr, nums[i] + prev2)
    return recurse(nums, i+1, newMax, prev, newMax)
}
