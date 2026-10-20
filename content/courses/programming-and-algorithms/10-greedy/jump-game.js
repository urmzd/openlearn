/**
 * @param {number[]} nums
 * @return {boolean}
 */
var canJump = function(nums) {
    return recurse(nums)
};

function recurse(nums, i = nums.length - 1, lastPosition = nums.length - 1) {
    if (i < 0) {
        return lastPosition === 0
    }

    return recurse(nums, i - 1, nums[i] + i >= lastPosition ? i : lastPosition)
}
