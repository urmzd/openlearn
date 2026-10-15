/*
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
var twoSum = function(nums, target) {
    
    // Use map to quickly check if the matching pair exists.
    value_index_pairs = {}
    
    // Start at the beginning of the array.
    let i = 0;
    
    // Iterate through the array to find a matching pair in the hashmap.
    for (let i = 0; i < nums.length; i++) {
        const pair_index = value_index_pairs?.[target - nums[i]];
        
        // If found, return.
        if (pair_index !== undefined) {
            return [pair_index, i]
        }
        
        // Otherwise, store the value and continue.
        value_index_pairs[nums[i]] = i
    }
    
    // Return an empty array if pairs not found
    return []
};
