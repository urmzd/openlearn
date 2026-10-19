/**
 * @param {number[]} nums
 * @return {number}
 */
var rob = function(nums) {
    if (nums.length === 1) {
        return nums[0]
    }
    const startAt0 = recurse(nums, 0, nums.length - 2)
    const startAt1 = recurse(nums, 1, nums.length - 1)
    return Math.max(startAt0, startAt1)
};

function recurse(nums, i, j, prev = 0, prev2 = 0, curr = 0) {
    if (i > j) {
        return curr;
    }
    
    const newMax = Math.max(prev, prev2 + nums[i])
    
    return recurse(nums, i + 1, j, newMax, prev, newMax)
}
