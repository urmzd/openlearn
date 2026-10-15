/*
 * // Definition for a Node.
 * function Node(val, neighbors) {
 *    this.val = val === undefined ? 0 : val;
 *    this.neighbors = neighbors === undefined ? [] : neighbors;
 * };
 */

/**
 * @param {Node} node
 * @return {Node}
 */
var cloneGraph = function(node) {
    
    const M = {}
    
    if (!node) {
        return
    }
    
    function dfs(n) {
        if (n.val in M) {
            return M[n.val]
        }
        
        M[n.val] = new Node(n.val)
        
        for (const nbour of n.neighbors) {
            M[n.val].neighbors.push(dfs(nbour))
        }
        
        return M[n.val]
    }
    
    return dfs(node)
};
