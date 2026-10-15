/**
 * @param {number[]} nums
 * @return {number}
 */
function maxProduct(nums) {
    if (nums.length === 1) {
        return nums[0]
    }
    
    let [min_product, max_product, res] = [nums[0],nums[0], nums[0]]
    for (let i = 1; i < nums.length; i++) {
        const n = nums[i]
        compare_arr = [n, min_product * n, max_product * n]
        max_product = Math.max(...compare_arr)
        min_product = Math.min(...compare_arr)
        res = Math.max(res, max_product)
    }
    
    return res
};
