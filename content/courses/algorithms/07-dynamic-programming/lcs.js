/**
 * @param {string} text1
 * @param {string} text2
 * @return {number}
 */
var longestCommonSubsequence = function(text1, text2) {
    
    const ss = new Array(2).fill(0).map(() => {
        return new Array(text1.length + 1).fill(0).map(() => {
            return 0
        })
    })
    
    for (let i = 1; i <= text2.length; i++) {
        for (let j = 1; j <= text1.length; j++) {
            const proxiedIndex = i % 2 
            const top = ss[proxiedIndex === 0 ? 1 : 0][j]
            const left = ss[proxiedIndex][j-1]
            const topLeft = ss[proxiedIndex === 0 ? 1 : 0][j - 1]
            
            const max = Math.max(top, left)
            
            matched = text2[i - 1] === text1[j - 1]
            ss[proxiedIndex][j] = (matched ? topLeft + 1: max)
        }
    }

    return ss[text2.length % 2][text1.length]
};
