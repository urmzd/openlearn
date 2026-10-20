/**
 * @param {number[]} prices
 * @return {number}
 */
var maxProfit = function(prices) {
    
    if (prices.length === 1) {
        return 0;
    }
    
    if (prices.length === 2) {
        const [_buy, _sell] = prices;
        const _profit = _sell - _buy
        if (_profit >= 0) {
            return _profit
        }
        
        return 0
    } 
    
    let max_profit = 0;
    let buy_index = 0;
    
    for (let i = 0; i < prices.length; i++) {
        if (prices[i] < prices[buy_index]) {
            buy_index = i
        } 
        
        if (prices[i] - prices[buy_index] > max_profit) {
            max_profit = prices[i] - prices[buy_index]
        }
    }
    
    return max_profit
};
