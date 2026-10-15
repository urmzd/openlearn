/**
 * @param {number[]} nums
 * @return {number}
 */
function findMin(nums) {    
    let low = 0;
    let high = nums.length - 1;
    
    if (nums.length === 1) {
        return nums[0]
    }
    
    // If Low == Min && High == Max; Return;
    if (nums[low] < nums[high]) {
        return nums[low]
    }
    
    while (high >= low) {
        const mid = low + Math.round((high - low) / 2)
        
        // Inflection Hit: H -> L
        if (nums[mid] > nums[mid + 1]) {
            return nums[mid + 1]
        }
        
        // Inflection Hit: L -> H
        if (nums[mid - 1] > nums[mid]) {
            return nums[mid]
        }
        
        // Inflection Missed - Value on Right
        if (nums[mid] > nums[0]) {
            low = mid + 1
        } else {
            // Inflection Missed - Value on Left
            high = mid - 1
        }
    }
    
    return -1
};
