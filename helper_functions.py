"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence
        self.group = None


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """

    leaf_nodes = []

    for label in labels:
        leaf_nodes.append(Node(label, None, 0, None, 0))
        
    nodes = len(labels)

    while nodes > 2:

        leaf_sums = np.sum(distances, axis=0)
        q_matrix = np.empty((nodes,nodes))

        # calculate q-value matrix
        for i in range(nodes):
            for j in range(nodes):
                if i != j:
                    q_matrix[i,j] = (nodes - 2)*distances[i,j]-leaf_sums[i]-leaf_sums[j]
        row, col = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)

        # calculate new distances and labels for the joint node
        new_label = f"{labels[row]}{labels[col]}"
        left_branch_dist = (distances[row,col]/2) + ((leaf_sums[row]-leaf_sums[col])/(2*(nodes-2)))
        right_branch_dist = (distances[row,col]) - left_branch_dist
        joint_node = Node(new_label, leaf_nodes[row], left_branch_dist, leaf_nodes[col], right_branch_dist)

        # remove the nodes that were combined and update distance array 
        remaining_nodes = []
        for r in range(nodes):
            if r!=row and r!=col: 
                remaining_nodes.append(r)

        update_distance = []
        for n in remaining_nodes:
            new_d = (distances[row,n] + distances[col,n] - distances[row,col])/2
            update_distance.append(new_d)

        # add combined node distances to new distances 
        horizontal_removed = np.delete(distances, [row, col], axis=0)
        remaining = np.delete(horizontal_removed, [row, col], axis=1)

        remaining = np.column_stack((remaining, update_distance))
        remaining = np.vstack((remaining, np.append(update_distance, 0)))

        # update labels, distances and leaf_node - to include joint nodes and remove the combined
        distances = remaining
        labels.pop(max(row,col))
        labels.pop(min(row,col))
        labels.append(new_label)

        leaf_nodes.pop(max(row,col))
        leaf_nodes.pop(min(row,col))
        leaf_nodes.append(joint_node)

        nodes = len(labels)

    # get root node - combining last two nodes left
    root = Node(f"{labels[0]}{labels[1]}", 
    leaf_nodes[0], distances[0, 1] / 2,
    leaf_nodes[1], distances[0, 1] / 2
    )   

    return root


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")
    """

    def draw(node, x,y):
        # stop when its a leaf
        if node.left == None and node.right == None:
            color = virus_colours.get(node.group)
            ax.text(x+0.1, y, node.name, va="center", color=color)
            return y, y + 1

        # calculate distance between node and its children
        left_horiz = x + node.left_distance   
        right_horiz = x + node.right_distance 

        # go through all children on the right + left
        left_vertical, y = draw(node.left, left_horiz, y)
        right_vertical, y = draw(node.right, right_horiz, y)

        # plot children + vertical line connecting them to parent node
        ax.plot([x, left_horiz], [left_vertical, left_vertical], color="black")
        ax.plot([x, right_horiz], [right_vertical, right_vertical], color="black")
        ax.plot([x, x], [left_vertical, right_vertical], color="black")  

        parent_vertical = (left_vertical + right_vertical)/2
        return parent_vertical, y

    # draw line for the root
    root_line, y = draw(tree, 0,0)
    ax.plot([-0.2, 0], [root_line, root_line], color="black")

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    if tree.left is None and tree.right is None:
        return

    sort_children_by_leaves(tree.left)
    sort_children_by_leaves(tree.right)


    def leaf_count (node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        return leaf_count(node.left) + leaf_count(node.right)

    left_child_leaves = leaf_count(tree.left)
    right_child_leaves = leaf_count(tree.right)

    #swap logic
    if left_child_leaves > right_child_leaves:
        tree.left, tree.right = tree.right, tree.left
        tree.left_distance, tree.right_distance = tree.right_distance, tree.left_distance


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()

# Makes empty grid of row+1 x col+1 dimensions
def initalise_grid(rows, cols):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    return grid 

# Fills in grid based on match, mismatch or gap of both sequences
def score_grid (grid, rows, cols, scoring_function, seq2, seq1,  gap_pen, local):
    for i in range(1, rows):
        for j in range(1,cols):
            
            diagonal_score = grid[i-1][j-1] + scoring_function(seq2[i-1], seq1[j-1])
    
            vertical_score = grid[i][j-1] - gap_pen
    
            horizontal_score = grid[i-1][j] - gap_pen
            
            grid[i][j]= max(diagonal_score, vertical_score, horizontal_score)

            if local:
                if grid[i][j] < 0:
                    grid[i][j] = 0
            
    return grid
    
# Calculates what direction current value is from, updates current position to previous position
def traceback_logic (i, j, scoring_function, seq1, seq2, gap_pen, grid, aligned_seq1, aligned_seq2):
    diagonal_traceback = grid[i-1][j-1] + scoring_function(seq2[i-1], seq1[j-1])
    horizontal_traceback = grid[i][j-1] - gap_pen
    vertical_traceback = grid[i-1][j] - gap_pen

    if diagonal_traceback == grid[i][j]:
        aligned_seq1 += seq1[j-1]
        aligned_seq2 += seq2[i-1]
        i = i - 1
        j = j - 1
    elif horizontal_traceback == grid[i][j]:
        aligned_seq1 += seq1[j-1]
        aligned_seq2 += '-'
        j = j - 1
    elif vertical_traceback == grid[i][j]:
        aligned_seq1 += '-'
        aligned_seq2 += seq2[i-1]
        i = i - 1
    return i, j, aligned_seq1, aligned_seq2

# Traceback logic for global alignment - tracebacks from the bottom right value
def traceback_global (rows, cols, grid, scoring_function, seq2, seq1, gap_pen):
    aligned_seq1 = ""
    aligned_seq2 = ""

    i = rows-1
    j = cols-1
    while i > 0 or j > 0:
        i, j, aligned_seq1, aligned_seq2 = traceback_logic(i, j, scoring_function, seq1, seq2, gap_pen, grid, aligned_seq1, aligned_seq2)
        
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]

    return aligned_seq1, aligned_seq2


# Traceback logic for local alignment - tracebacks from the largest value in grid
def traceback_local (rows, cols, grid, scoring_function, seq2, seq1, gap_pen):
    aligned_seq1 = ""
    aligned_seq2 = ""
    
    # highest score
    high_score = 0
    i = 0
    j = 0

    for x in range(rows):
        for y in range(cols):
            if grid[x][y] > high_score:
                max_score = grid[x][y]
                i = x
                j = y
                
    while grid[i][j] != 0:
        i, j, aligned_seq1, aligned_seq2 = traceback_logic(i, j, scoring_function, seq1, seq2, gap_pen, grid, aligned_seq1, aligned_seq2)
        
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]

    return aligned_seq1, aligned_seq2


# Calculates the score of the alignment, applies gap penalty
def alignment_scoring (aligned_seq1, aligned_seq2, gap_pen):
    total_score = 0
    for i in range(len(aligned_seq1)):
        if aligned_seq1[i] == aligned_seq2[i]:
            total_score += 1.0
        elif aligned_seq1[i] == '-' or aligned_seq2[i] == '-':
            total_score -= gap_pen

        elif aligned_seq1[i] != aligned_seq2[i]:
            total_score -= gap_pen
    return total_score
    
           
def global_alignment(seq1, seq2, scoring_function):
    rows = (len(seq2)+1)
    cols = (len(seq1)+1)
  
    gap_pen = 8
    
    # make empty grid + fill first row and col with gap pen
    grid = initalise_grid(rows, cols)

    for i in range(1,rows):
        grid[i][0] = grid[i-1][0] - gap_pen
        
    for i in range(1,cols):
        grid[0][i] = grid[0][i-1] - gap_pen

    # calculate diagonal, vertical, horizonal scores - take largest - fill entire grid
    grid = score_grid(grid, rows, cols, scoring_function, seq2, seq1, gap_pen, False)
        

    # bottom right value - compare to diagonal, horizonal, vertical scores to traceback - repeat
    aligned_seq1, aligned_seq2 = traceback_global(rows, cols, grid, scoring_function, seq2, seq1, gap_pen)

    # scoring for alignment
    total_score = alignment_scoring (aligned_seq1, aligned_seq2, gap_pen)  
           
    return aligned_seq1, aligned_seq2, total_score


def local_alignment(seq1, seq2, scoring_function):
    rows = (len(seq2)+1)
    cols = (len(seq1)+1)
  
    gap_pen = 8
    
    # make empty grid + all filled with 0s
    grid = initalise_grid(rows, cols)

    # score grid - all negatives turn to 0
    grid = score_grid(grid, rows, cols, scoring_function, seq2, seq1, gap_pen, True)

    aligned_seq1, aligned_seq2 = traceback_local (rows, cols, grid, scoring_function, seq2, seq1, gap_pen)

    
    total_score = alignment_scoring (aligned_seq1, aligned_seq2, gap_pen)  
           
    return aligned_seq1, aligned_seq2, total_score



## This is an example scoring function, you should implement a version which uses a scoring matrix 
def scoring_function_simple(aa_i,aa_j):
    score = [-1, 1][aa_i == aa_j]
    return (score)



virus_colours = {
    "Alphacoronavirus": "pink",
    "Betacoronavirus": "blue",
    "Gammacoronavirus": "green",
    "Deltacoronavirus": "orange",
}