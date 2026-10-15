function numIslands(grid: string[][]): number {
    const m = grid.length;
    const n = grid[0].length;
    let count = 0;
        
    function dfs(i, j) {       
        // If already visited or water.
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] === "0") {
            return;
        }
                
        grid[i][j] = "0"
        
        i - 1 >= 0 && dfs(i - 1, j)
        i + 1 < m && dfs(i + 1, j)
        j - 1 >= 0 && dfs(i, j - 1)
        j + 1 < n && dfs(i, j + 1)
    }
    
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            if (grid[i][j] !== "0") {
                count++
                dfs(i, j)
            }
        }
    }
    
    return count;
};
