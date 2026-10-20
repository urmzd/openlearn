/**
 * @param {number[]} coins
 * @param {number} amount
 * @return {number}
 */
var coinChange = function(coins, amount) {

    const MAX = amount + 1
    const dp = [0, ...new Array(amount).fill(MAX)]
    
    for (let subAmount = 1; subAmount <= amount; subAmount++) {
        for (let coin = 0; coin < coins.length; coin++) {
            const coinV = coins[coin]
            if (coins[coin] <= subAmount) {
                dp[subAmount] = Math.min(dp[subAmount], dp[subAmount - coinV] + 1)
            }
        }
    }
    const solution = dp[dp.length - 1]
    return solution >= MAX ? -1 : solution
};
