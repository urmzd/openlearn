/*
 * @param {number} n
 * @return {number}
 */
var climbStairs = function(n) {
    // All unique combinations of N
    
    switch (n) {
            case 0: 
                return 0;
            case 1: 
                return 1;        
            case 2:
                return 2;
    }
    
    let beforeMinus1 = 2;
    let beforeMinus2 = 1;
    for (let i = 3; i <=n; i++) {
        const oldBefore1 = beforeMinus1;
        beforeMinus1 = beforeMinus2 + beforeMinus1
        beforeMinus2 = oldBefore1;
    }
    
    return beforeMinus1
};
