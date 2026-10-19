/**
 * @param {umber} n - a positive integer
 * @return {number}
 */
var hammingWeight = function(n) {
    let [count, itr] = [0, 32]
    while(itr > 0)  {
        count += n & 1
        n = n>>>1
        itr--
    }
    
    return count
};
