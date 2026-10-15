/**
 * @param {number[]} nums
 * @return {number}
 */
var missingNumber = function(nums) {
    
    let sum = nums.length
    
    for (let n = 0; n < nums.length; n++) {
        sum += n - nums[n]
    }
    
    return sum
};
