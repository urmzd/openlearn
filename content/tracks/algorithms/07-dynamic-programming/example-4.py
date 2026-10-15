from functools import reduce


def topsort(G):
    """
    @return: A list containing the topological sorted vertices in G.
    """
    return sorted(G)


def dfs(node, color, visited, dp):
    """
    Uses DFS to determine the length of longest monochromatic path (LMP).
    The length of the LMP of the current path combined with
    max length found from the children of the neighbouring nodes.
    """
    visited[node] = True

    for neighbour in node.neighbours:
        if not visited[neighbour]:
            dfs(neighbour, color, visited, dp)

        dp[node] = max(
            dp[node], 1 + dp[neighbour] if neighbour.color == color else dp[neighbour]
        )


def dp_algorithm(L):
    """
    Sub-Problem: The length of the LMP containing node N.
    @return: The LMP.
    """
    dp = -float("inf") * L.length
    visited = [False] * len(L)

    for node in L:
        if not visited[node]:
            dfs(node, node.color, visited, dp)

    # Get the max value of all sub MP's traversed.
    return reduce(lambda x, y: max(x, y), dp)
