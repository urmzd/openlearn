/**
 * @param {number} numCourses
 * @param {number[][]} prerequisites
 * @return {boolean}
 */
var canFinish = function(numCourses, prerequisites) {
    return topoSort(...getHelperStructure(numCourses, prerequisites), numCourses)
};

function getHelperStructure(nCourses, preReqs) {
    const degreeOfNodes = []
    
    const map = new Map()
    
    for (const [courseAfter, courseBefore] of preReqs) {
        if (map.has(courseBefore)) {
            map.get(courseBefore).push(courseAfter)
        } else {
            map.set(courseBefore, [courseAfter])
        }
        
        degreeOfNodes[courseAfter] = (degreeOfNodes[courseAfter] ?? 0) + 1
    }
    
    return [map, degreeOfNodes]
}

function topoSort(map, degreeOfNodes, nCourses) {
    if (!degreeOfNodes.length) {
        return true
    }
    
    const Q = []
    
    for (let i = 0; i < nCourses; i++) {
        if (!degreeOfNodes?.[i]) {
            Q.push(i)
        }
    }
    
    let count = 0;
    while (Q.length) {
        const head = Q.shift()
        count++

        if (map.has(head)) {
            for (const neighbour of map.get(head)) {
                degreeOfNodes[neighbour] -= 1
                if (!degreeOfNodes[neighbour]) {
                    Q.push(neighbour)
                }
            }
        }

    }
    
    console.log(map, degreeOfNodes)
        
    return count === degreeOfNodes.length
}
