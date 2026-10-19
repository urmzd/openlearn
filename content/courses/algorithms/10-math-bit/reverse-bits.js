/**
 * @param {number} n - a positive integer
 * @return {number} - a positive integer
 */
var reverseBits = function(n) {
    
    let out = 0;
    
    for (let i = 0; i < 32; i++) {
        out = (out << 1) ^ (n & 1)
        n >>>= 1
    }
    
    return out >>> 0
};
