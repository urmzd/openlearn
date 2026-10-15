/**
 * @param {number[]} nums
 * @return {number[][]}
 */
function threeSum(nums) {
    
    if (nums.length < 3) {
        return []      
    }
    
    // Sort integer values in increasing order.
    nums.sort((a, b) => a - b)
    
    // Same numbers
    if (nums[0] === nums[nums.length - 1]) {
        if (nums[0] === 0) {
            return [new Array(3).fill(0)]
        }
        
        return []
    }

        
    // Store the final values.
    const triplets = []
    for (let i = 0; i < nums.length - 1; i ++) {
        if (i > 0 && nums?.[i] === nums?.[i-1]) continue;
        const a = nums[i]
        let j = i + 1;
        let k = nums.length - 1;
        while (j < k) {
            const b = nums[j]
            const c = nums[k]

            if (a + b + c === 0) {
                triplets.push([a, b, c])
                j++
                k--
                // Prevent duplicates.
                while (nums?.[j] === nums?.[j - 1]) j++
                while (nums?.[k] === nums?.[k + 1]) k--
            } else if (a + b + c < 0) {
                j++
            } else {
                k--
            }
        }
    }

    return triplets;
};
