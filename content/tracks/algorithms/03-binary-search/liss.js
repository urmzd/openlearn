/**
 * @param {number[]} nums
 * @return {number}
 */
var lengthOfLIS = function(nums) {
    
    const piles = []
    
    piles[0] = nums[0]
    
    for (let index = 1; index < nums.length; index++) {        
        // returns index if found
        const pileFound = binarySearch(piles, nums[index], 0, piles.length - 1)
        if (pileFound === -1) {
            piles.push(nums[index])
        } else {
            piles[pileFound] = nums[index] 
        }
    }
    
    return piles.length;
};

function binarySearch(nums, target, lo, hi) {
    if (lo >= hi) {
        if (nums[lo] >= target) {
            return lo
        }
        return -1
    } 
    
    const mid = Math.floor((lo + hi) / 2)
    if (nums[mid] === target) {
        return mid;
    }
    
    // Target is on left
    if (nums[mid] > target) {
        return binarySearch(nums, target, lo, mid)
    }
    
    return binarySearch(nums, target, mid+1, hi)
}
