/**
 * @param {string} s
 * @param {string[]} wordDict
 * @return {boolean}
 */
var wordBreak = function(s, wordDict) {
    
    let maxLen = 0;
    const wordSet = {};
    
    wordDict.forEach(word => {
        maxLen = Math.max(word.length, maxLen)
        wordSet[word] = true
    })
    
    return recurse(s, wordSet, {}, 0, maxLen)
};

function recurse(s, wordSet, cache = {}, startIndex=0, maxLen) {
    
    if (startIndex === s.length) {
        return true;
    }
    
    if (cache?.[startIndex] !== undefined) {
        return cache[startIndex]
    }
    
    for (let endIndex = startIndex + 1; endIndex <= Math.min(startIndex + maxLen, s.length); endIndex++) {
        const subString = s.substring(startIndex, endIndex)
        
        if (!wordSet?.[subString]) {
            continue;
        }
        
        if (recurse(s, wordSet, cache, endIndex, maxLen)) {
            cache[endIndex] = true
            return true;
        }
    } 
    cache[startIndex] = false
    
    return false;
}
