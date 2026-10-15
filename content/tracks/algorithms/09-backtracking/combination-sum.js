/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var combinationSum4 = function(nums, target) { 
    return recurse(nums, target)
};

function recurse(nums, target, cache = {0: 1}) {
    if (cache?.[target] !== undefined) {
        return cache?.[target]
    }
    
    let combinations = 0
    
    for (const num of nums) {
        if (num <= target) {
            combinations += recurse(nums, target - num, cache)
        }
    }
    
    cache[target] = combinations
    
    return combinations;
}
