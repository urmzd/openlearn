def max2(*S):
    """
    @return: Two subtrees in S with the largest lengths.
    """
    return []


def conquer(R, *S):
    """
    Joins the subtrees *S with the root R.
    @return: A LinkedList containing *S's nodes and R.
    """
    # By selecting only two, we prevent revisiting nodes.
    largest_subtrees = max2(S)
    return LinkedList(R, largest_subtrees)


def LLNode(node):
    """
    @return: Node containing data and NULL pointer to next.
    """
    return {}


def LinkedList(node, *linkedlist):
    """
    @return: LinkedList containing node and nodes in LinkedList.
    """
    return []


def divide(root, lp=None):
    """
    Determines the longest monochromatic path in a rooted tree.
    @return: The longest path as a iterable LinkedList.
    """
    if root == None:
        return lp

    # Restart longest path parent colour not found.
    if lp == None or root.colour != lp.tail.colour:
        _lp = LinkedList(LLNode(root))
    else:
        _lp = LinkedList(LLNode(root), lp)

    # Find longest paths in subtrees.
    L = [divide(S, _lp) for S in root]

    # Merge paths.
    return conquer(root, *L)
