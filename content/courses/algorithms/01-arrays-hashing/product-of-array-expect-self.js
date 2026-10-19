/**
 * @param {number[]} nums
 * @return {number[]}
 */
var productExceptSelf = function(nums) {
    // nums = [1, 2, 3, 4]
    const answers = []
    
    // Values of 'answers': 24, 12, 4, 1
    for (let i = 0; i < nums.length; i++) {
        answers.push((answers?.[i- 1] ?? 1) * (nums?.[i-1] ?? 1))
    }
   
    // Values of R: 24, 12, 4, 1
    let R = 1;
    
    for (let i = nums.length - 1; i >= 0; i--) {
        answers[i] *= R
        R *= nums[i]
    }
    
    return answers
    
};
