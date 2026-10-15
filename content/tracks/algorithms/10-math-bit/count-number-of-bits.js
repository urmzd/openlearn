/*
 * @param {number} n
 * @return {number[]}
 */
var countBits = function(n) {
    if (n === 0) {
        return [0]
    }
    
    if (n === 1) {
        return [0, 1]
    }
    
    const N = [0]
    let lastPowerOf2 = 1
    for (let i = 1; i <= n; i++) {
        if (lastPowerOf2 * 2 === i) {
            N.push(1)
            lastPowerOf2 = i
        } else {
            N.push(N[i - lastPowerOf2] + 1)
        }
    }
    
    return N
};
