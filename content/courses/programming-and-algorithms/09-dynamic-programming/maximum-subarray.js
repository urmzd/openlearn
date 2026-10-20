/**
 * @param {number[]} nums
 * @return {number}
 */
var maxSubArray = function(nums) {
    if (nums.length === 1) {
        return nums[0];
    }
    
    let max_sum = nums[0]
    let current_sum = nums[0]
    
    for (let i = 1; i < nums.length; i++) {
        current_sum = Math.max(current_sum + nums[i], nums[i])
        max_sum = Math.max(current_sum, max_sum)
    }
    
    return max_sum
};
