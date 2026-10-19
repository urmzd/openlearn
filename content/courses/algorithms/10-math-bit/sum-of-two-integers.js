/*
 * @param {number} a
 * @param {number} b
 * @return {number}
 */
var getSum = function(a, b) {
    let sum = a ^ b;
    let carry = (a & b) << 1;
    
    while (carry !== 0) {
        const newSum = sum ^ carry
        const newCarry = (sum & carry) << 1
        sum = newSum
        carry = newCarry
    }
    
    return sum
};
