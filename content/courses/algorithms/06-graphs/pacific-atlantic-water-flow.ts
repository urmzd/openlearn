type PointMap = Map<`${number},${number}`, number[]>
type Height = number[][]

function pacificAtlantic(heights: Height): Height {
   
    const M = heights.length;
    const N = heights?.[0]?.length;
    
    const pOcean: Set<string> = new Set()
    const aOcean: Set<string>= new Set()
    const answers: Set<string> = new Set()
   
    const dfs = (x: number, y: number, ocean: Set<string>) => {
        
        const key = `${y},${x}` as const;
        
        if (ocean.has(key)) {
            return;     
        }
        
        // Mark as visited
        ocean.add(key)
        
        if (pOcean.has(key) && aOcean.has(key)) {
            answers.add(key)
        }
        
        const goLeft = x - 1 >= 0 && heights[y][x] <= heights[y][x-1] 
        const goDown = y - 1 >= 0 && heights[y][x] <= heights[y - 1][x] 
        const goRight = x + 1 < N && heights[y][x] <= heights[y][x+1] 
        const goUp = y + 1 < M && heights[y][x] <= heights[y + 1][x] 
        
        // Left
        goLeft && dfs(x - 1, y, ocean)
        // Right
        goRight && dfs(x + 1, y, ocean)
        // Top
        goUp && dfs(x, y + 1, ocean)
        // Bottom
        goDown && dfs(x, y - 1, ocean)
    }
    
    for (let n = 0; n < N; n++) {
        // Start from TOP
        dfs(n, 0, pOcean)
        
        // Start from BOTTOM
        dfs(n, M - 1, aOcean)
    }
    
    for (let m = 0; m < M; m++) {
        // Start from LEFT
        dfs(0, m, pOcean)
        
        // Start from RIGHT
        dfs(N - 1, m, aOcean)
    }
    
    return [...answers].map(cell => cell.split(",").map(c => Number(c)))
};
