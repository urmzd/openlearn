/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var search = function(nums, target) {
    
    let low = 0;
    let high = nums.length - 1;
    
    const indexof = proxy(nums)
    console.log(nums)
    while (low <= high) {
        const mid = low + Math.floor((high-low)/2)
        const index_of_mid = indexof(mid)
        
        if (nums[index_of_mid] === target) {
            return index_of_mid
        }
        
        if (target < nums[index_of_mid]) {
            high = mid - 1
        } 
        
        if (target > nums[index_of_mid]) {
            low = mid + 1
        }
    }
    
    return -1;
};
    
function proxy(nums) {
    let low = 0;
    let high = nums.length - 1;
    
    // Find the "lowest" value index.
    while (low < high) {
        const mid = low + Math.floor((high-low)/2)

        if (nums[mid] > nums[high]) {
            low = mid+1
        } else {
            high = mid
        }
    }
        
    return (index) => (index + low)%(nums.length)
}
