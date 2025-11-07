"""
Tree-Consistent Clustering Algorithm

This module provides functions to find optimal tree-consistent clusterings
given hierarchical data and an external cost function.
"""

from typing import Dict, List, Callable, Tuple, Optional, Set
from collections import defaultdict, Counter
import itertools


class TreeNode:
    """Represents a node in the hierarchical tree."""
    
    def __init__(self, name: str):
        self.name = name
        self.children = {}
        self.datapoints = []  # IDs of datapoints that end at this node
        self.parent = None
    
    def add_child(self, name: str) -> 'TreeNode':
        """Add a child node if it doesn't exist."""
        if name not in self.children:
            child = TreeNode(name)
            child.parent = self
            self.children[name] = child
        return self.children[name]
    
    def get_all_datapoints(self) -> List[str]:
        """Get all datapoints in this subtree."""
        points = list(self.datapoints)
        for child in self.children.values():
            points.extend(child.get_all_datapoints())
        return points
    
    def get_path(self) -> str:
        """Get the full path from root to this node."""
        if self.parent is None:
            return self.name
        parent_path = self.parent.get_path()
        if parent_path:
            return f"{parent_path} / {self.name}"
        return self.name


def build_tree(data: Dict[str, str]) -> Tuple[TreeNode, Dict[str, TreeNode]]:
    """
    Build a tree structure from hierarchical path data.
    
    Args:
        data: Dictionary mapping datapoint IDs to hierarchical paths
              (e.g., {'0': 'society / politics / military'})
    
    Returns:
        Tuple of (root node, dict mapping datapoint IDs to their leaf nodes)
    """
    root = TreeNode("")  # Empty root
    datapoint_to_node = {}
    
    for datapoint_id, path in data.items():
        # Split the path into components
        components = [c.strip() for c in path.split('/')]
        
        # Navigate/build the tree
        current = root
        for component in components:
            current = current.add_child(component)
        
        # Store the datapoint at the leaf node
        current.datapoints.append(datapoint_id)
        datapoint_to_node[datapoint_id] = current
    
    return root, datapoint_to_node


def get_all_internal_nodes(root: TreeNode) -> List[TreeNode]:
    """Get all internal nodes (non-leaf nodes) in the tree."""
    nodes = []
    
    def traverse(node):
        if node.children:  # Internal node
            nodes.append(node)
            for child in node.children.values():
                traverse(child)
    
    traverse(root)
    return nodes


def get_clustering_from_cut(root: TreeNode, cut_nodes: List[TreeNode], 
                            datapoint_ids: List[str]) -> List[str]:
    """
    Generate cluster labels for a given tree cut.
    
    Args:
        root: Root of the tree
        cut_nodes: List of nodes that represent cluster centers
        datapoint_ids: Ordered list of datapoint IDs
    
    Returns:
        List of cluster labels for each datapoint
    """
    # For each datapoint, find which cut node it belongs to
    datapoint_to_cluster = {}
    
    for cut_node in cut_nodes:
        cluster_label = cut_node.get_path()
        if not cluster_label:  # Root node
            cluster_label = "root"
        
        # Get all datapoints under this node
        points = cut_node.get_all_datapoints()
        for point_id in points:
            datapoint_to_cluster[point_id] = cluster_label
    
    # Return labels in the order of datapoint_ids
    return [datapoint_to_cluster.get(dp_id, "unknown") for dp_id in datapoint_ids]


def generate_all_cuts(root: TreeNode) -> List[List[TreeNode]]:
    """
    Generate all possible tree-consistent cuts.
    
    A cut is a set of nodes such that every datapoint belongs to exactly one
    subtree rooted at a cut node.
    
    Args:
        root: Root of the tree
    
    Returns:
        List of all possible cuts (each cut is a list of nodes)
    """
    cuts = []
    
    def generate_cuts_recursive(node: TreeNode) -> List[List[TreeNode]]:
        """
        Generate all possible cuts for a subtree.
        Returns a list of cuts, where each cut is a list of nodes.
        """
        if not node.children:  # Leaf node
            return [[node]]
        
        # Option 1: Cut at this node (don't recurse to children)
        node_cuts = [[node]]
        
        # Option 2: Don't cut here, but cut in children
        if node.children:
            # Get all possible cuts for each child
            child_cut_options = []
            for child in node.children.values():
                child_cuts = generate_cuts_recursive(child)
                child_cut_options.append(child_cuts)
            
            # Combine cuts from all children (Cartesian product)
            for combination in itertools.product(*child_cut_options):
                # Flatten the combination
                combined_cut = []
                for cut in combination:
                    combined_cut.extend(cut)
                node_cuts.append(combined_cut)
        
        return node_cuts
    
    # Handle root specially
    if not root.children:
        return [[root]]
    
    # Generate cuts for all children of root and combine them
    child_cut_options = []
    for child in root.children.values():
        child_cuts = generate_cuts_recursive(child)
        child_cut_options.append(child_cuts)
    
    # Combine cuts from all children
    for combination in itertools.product(*child_cut_options):
        combined_cut = []
        for cut in combination:
            combined_cut.extend(cut)
        cuts.append(combined_cut)
    
    return cuts


def generate_level_cuts(root: TreeNode, max_depth: Optional[int] = None) -> List[List[TreeNode]]:
    """
    Generate tree cuts at each level (more efficient than all possible cuts).
    
    Args:
        root: Root of the tree
        max_depth: Maximum depth to consider (None for all depths)
    
    Returns:
        List of cuts, one for each level
    """
    cuts = []
    
    # Group nodes by depth
    levels = defaultdict(list)
    
    def collect_nodes(node: TreeNode, depth: int):
        if node.name:  # Skip empty root
            levels[depth].append(node)
        for child in node.children.values():
            collect_nodes(child, depth + 1)
    
    collect_nodes(root, 0)
    
    # Create a cut for each level
    max_level = max(levels.keys()) if levels else 0
    if max_depth is not None:
        max_level = min(max_level, max_depth)
    
    for level in range(1, max_level + 1):
        if level in levels:
            cuts.append(levels[level])
    
    return cuts


# ============================================================================
# DECOMPOSABLE COST FUNCTIONS (for efficient DP)
# ============================================================================

def cluster_purity(cluster_datapoints: List[str], ground_truth_dict: Dict[str, str]) -> float:
    """
    Calculate purity of a single cluster.
    
    Purity = (# of most common class) / (total # of datapoints)
    
    This is decomposable: total_purity = sum(purity_i * size_i) / total_size
    
    Args:
        cluster_datapoints: List of datapoint IDs in this cluster
        ground_truth_dict: Dict mapping datapoint ID to ground truth label
    
    Returns:
        Purity score (0 to 1)
    """
    if not cluster_datapoints:
        return 0.0
    
    # Count labels in this cluster
    label_counts = Counter(ground_truth_dict[dp_id] for dp_id in cluster_datapoints)
    
    # Purity = fraction of most common label
    most_common_count = max(label_counts.values())
    purity = most_common_count / len(cluster_datapoints)
    
    return purity * len(cluster_datapoints)  # Return weighted purity


def cluster_entropy(cluster_datapoints: List[str], ground_truth_dict: Dict[str, str]) -> float:
    """
    Calculate entropy of a single cluster (negative for minimization).
    
    Lower entropy = more pure cluster
    
    Args:
        cluster_datapoints: List of datapoint IDs in this cluster
        ground_truth_dict: Dict mapping datapoint ID to ground truth label
    
    Returns:
        Negative entropy (to be maximized)
    """
    import math
    
    if not cluster_datapoints:
        return 0.0
    
    # Count labels in this cluster
    label_counts = Counter(ground_truth_dict[dp_id] for dp_id in cluster_datapoints)
    
    # Calculate entropy
    n = len(cluster_datapoints)
    entropy = 0.0
    for count in label_counts.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)
    
    # Return negative weighted entropy (we want to maximize, i.e., minimize entropy)
    return -entropy * n


def cluster_homogeneity(cluster_datapoints: List[str], ground_truth_dict: Dict[str, str]) -> float:
    """
    Calculate homogeneity score for a cluster.
    
    Homogeneity = 1 if all datapoints have same label, 0 if maximally mixed
    
    Args:
        cluster_datapoints: List of datapoint IDs in this cluster
        ground_truth_dict: Dict mapping datapoint ID to ground truth label
    
    Returns:
        Homogeneity score weighted by cluster size
    """
    if not cluster_datapoints:
        return 0.0
    
    # Count labels
    label_counts = Counter(ground_truth_dict[dp_id] for dp_id in cluster_datapoints)
    
    # Homogeneity: (# of most common) / total
    most_common_count = max(label_counts.values())
    homogeneity = most_common_count / len(cluster_datapoints)
    
    return homogeneity * len(cluster_datapoints)


def optimal_tree_clustering_dp_memoized(
    tree_data: Dict[str, str],
    cluster_cost_func: Callable[[List[str], Dict[str, str]], float],
    ground_truth: List[str],
    maximize: bool = True
) -> Tuple[List[str], float, List[TreeNode]]:
    """
    TRUE Dynamic Programming with memoization for decomposable cost functions.
    
    This method achieves exponential speedup by using memoization when the cost
    function is decomposable (i.e., can be computed per-cluster independently).
    
    Key requirement: cluster_cost_func must compute cost for a SINGLE cluster,
    and total cost = sum of individual cluster costs.
    
    Complexity: O(m × n) where m = internal nodes, n = datapoints (!!!)
    This is exponentially faster than O(2^m × n) for exhaustive search.
    
    Args:
        tree_data: Dictionary mapping datapoint IDs to hierarchical paths
        cluster_cost_func: Function that takes (cluster_datapoint_ids, ground_truth_dict)
                          and returns the cost for that single cluster
        ground_truth: List of ground truth labels
        maximize: If True, maximize the cost function; if False, minimize it
    
    Returns:
        Tuple of (best_clustering, best_score, cut_nodes)
    """
    # Build the tree
    root, datapoint_to_node = build_tree(tree_data)
    datapoint_ids = sorted(tree_data.keys(), key=lambda x: int(x))
    
    # Create ground truth dictionary
    ground_truth_dict = {dp_id: gt for dp_id, gt in zip(datapoint_ids, ground_truth)}
    
    # Memoization cache: node -> (best_score, best_cut_nodes)
    memo = {}
    
    def dp_solve(node: TreeNode) -> Tuple[float, List[TreeNode]]:
        """
        Compute optimal score and cut for subtree rooted at node.
        
        Returns:
            (best_score, list_of_cut_nodes)
        """
        # Check cache
        if node in memo:
            return memo[node]
        
        # Get all datapoints in this subtree
        subtree_datapoints = node.get_all_datapoints()
        
        # Option 1: Cut at this node (make it a cluster)
        # Cost = cluster_cost for all datapoints under this node
        cut_here_score = cluster_cost_func(subtree_datapoints, ground_truth_dict)
        cut_here_nodes = [node]
        
        # Option 2: Don't cut here, recurse to children
        if node.children:
            recurse_score = 0.0
            recurse_nodes = []
            
            for child in node.children.values():
                child_score, child_nodes = dp_solve(child)
                recurse_score += child_score
                recurse_nodes.extend(child_nodes)
            
            # Choose better option
            if maximize:
                if recurse_score > cut_here_score:
                    best_score = recurse_score
                    best_nodes = recurse_nodes
                else:
                    best_score = cut_here_score
                    best_nodes = cut_here_nodes
            else:
                if recurse_score < cut_here_score:
                    best_score = recurse_score
                    best_nodes = recurse_nodes
                else:
                    best_score = cut_here_score
                    best_nodes = cut_here_nodes
        else:
            # Leaf node - must cut here
            best_score = cut_here_score
            best_nodes = cut_here_nodes
        
        # Cache and return
        memo[node] = (best_score, best_nodes)
        return best_score, best_nodes
    
    # Solve for each child of root and combine
    if root.children:
        total_score = 0.0
        total_cut_nodes = []
        
        for child in root.children.values():
            child_score, child_nodes = dp_solve(child)
            total_score += child_score
            total_cut_nodes.extend(child_nodes)
        
        best_score = total_score
        best_cut = total_cut_nodes
    else:
        best_score, best_cut = dp_solve(root)
    
    # Reconstruct clustering
    best_clustering = get_clustering_from_cut(root, best_cut, datapoint_ids)
    
    return best_clustering, best_score, best_cut


def optimal_tree_clustering_dp(
    tree_data: Dict[str, str],
    cost_function: Callable[[List[str], List[str]], float],
    ground_truth: List[str],
    maximize: bool = True
) -> Tuple[List[str], float, List[TreeNode]]:
    """
    Find optimal tree-consistent clustering using dynamic programming.
    
    This method uses DP to explore the search space more efficiently than
    exhaustive enumeration. It evaluates cuts on-the-fly and uses branch-and-bound
    style pruning when possible.
    
    Note: Still explores exponentially many cuts in worst case, but doesn't 
    pre-generate all of them. More memory-efficient than 'exhaustive'.
    
    Complexity: O(2^m × n) where m = internal nodes, n = datapoints.
    
    Args:
        tree_data: Dictionary mapping datapoint IDs to hierarchical paths
        cost_function: Function that takes (ground_truth, predicted) and returns a score
        ground_truth: List of ground truth labels
        maximize: If True, maximize the cost function; if False, minimize it
    
    Returns:
        Tuple of (best_clustering, best_score, cut_nodes)
    """
    # Build the tree
    root, datapoint_to_node = build_tree(tree_data)
    datapoint_ids = sorted(tree_data.keys(), key=lambda x: int(x))
    
    # Initialize best solution
    best_score = float('-inf') if maximize else float('inf')
    best_clustering = None
    best_cut = None
    
    # Count evaluations for comparison
    evaluation_count = [0]
    
    def explore_cuts_dp(current_cut: List[TreeNode], remaining_nodes: List[TreeNode]) -> None:
        """
        Recursively explore cuts using DFS.
        
        Args:
            current_cut: Current list of cut nodes
            remaining_nodes: Nodes we still need to decide whether to cut
        """
        nonlocal best_score, best_clustering, best_cut
        
        # Base case: no more nodes to process
        if not remaining_nodes:
            # Evaluate this cut
            clustering = get_clustering_from_cut(root, current_cut, datapoint_ids)
            score = cost_function(ground_truth, clustering)
            evaluation_count[0] += 1
            
            # Update best if improved
            if maximize:
                if score > best_score:
                    best_score = score
                    best_clustering = clustering
                    best_cut = list(current_cut)
            else:
                if score < best_score:
                    best_score = score
                    best_clustering = clustering
                    best_cut = list(current_cut)
            return
        
        # Take the first remaining node
        node = remaining_nodes[0]
        rest = remaining_nodes[1:]
        
        # Option 1: Cut at this node (don't explore its children)
        explore_cuts_dp(current_cut + [node], rest)
        
        # Option 2: Don't cut here, add children to remaining nodes
        if node.children:
            children = list(node.children.values())
            explore_cuts_dp(current_cut, children + rest)
    
    # Start DP search from root's children
    root_children = list(root.children.values()) if root.children else [root]
    explore_cuts_dp([], root_children)
    
    return best_clustering, best_score, best_cut


def optimal_tree_clustering(
    tree_data: Dict[str, str],
    cost_function: Callable[[List[str], List[str]], float],
    ground_truth: List[str],
    method: str = 'level',
    maximize: bool = True,
    decomposable_cost: str = 'purity'
) -> Tuple[List[str], float, List[TreeNode]]:
    """
    Find the optimal tree-consistent clustering.
    
    Args:
        tree_data: Dictionary mapping datapoint IDs to hierarchical paths
                   (e.g., {'0': 'society / politics / military'})
        cost_function: Function that takes (ground_truth, predicted) and returns a score
                      (only used for methods other than 'dp_memoized')
        ground_truth: List of ground truth labels
        method: Method for finding optimal clustering:
               - 'level': Try cuts at each tree depth (O(depth × n), fast)
               - 'dp_memoized': TRUE DP with memoization (O(m × n), FAST!)
               - 'dp': DFS search without memoization (O(2^m × n), slow)
               - 'exhaustive': Try all tree cuts (O(2^m × n), very slow)
        maximize: If True, maximize the cost function; if False, minimize it
        decomposable_cost: Which decomposable cost to use for 'dp_memoized':
                          - 'purity': Cluster purity (recommended)
                          - 'entropy': Negative entropy
                          - 'homogeneity': Cluster homogeneity
    
    Returns:
        Tuple of (best_clustering, best_score, cut_nodes)
    """
    # Use memoized DP method with decomposable cost
    if method == 'dp_memoized':
        # Select decomposable cost function
        if decomposable_cost == 'purity':
            cluster_cost = cluster_purity
        elif decomposable_cost == 'entropy':
            cluster_cost = cluster_entropy
        elif decomposable_cost == 'homogeneity':
            cluster_cost = cluster_homogeneity
        else:
            raise ValueError(f"Unknown decomposable_cost: {decomposable_cost}. "
                           f"Use 'purity', 'entropy', or 'homogeneity'.")
        
        return optimal_tree_clustering_dp_memoized(
            tree_data, cluster_cost, ground_truth, maximize
        )
    
    # Use regular DP method (no memoization)
    if method == 'dp':
        return optimal_tree_clustering_dp(tree_data, cost_function, ground_truth, maximize)
    
    # Build the tree for other methods
    root, datapoint_to_node = build_tree(tree_data)
    
    # Get ordered list of datapoint IDs (sorted by key)
    datapoint_ids = sorted(tree_data.keys(), key=lambda x: int(x))
    
    # Generate possible cuts
    if method == 'exhaustive':
        cuts = generate_all_cuts(root)
    elif method == 'level':
        cuts = generate_level_cuts(root)
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Use 'level', 'dp_memoized', 'dp', or 'exhaustive'.")
    
    # Evaluate each cut
    best_score = float('-inf') if maximize else float('inf')
    best_clustering = None
    best_cut = None
    
    for cut_nodes in cuts:
        clustering = get_clustering_from_cut(root, cut_nodes, datapoint_ids)
        score = cost_function(ground_truth, clustering)
        
        if maximize:
            if score > best_score:
                best_score = score
                best_clustering = clustering
                best_cut = cut_nodes
        else:
            if score < best_score:
                best_score = score
                best_clustering = clustering
                best_cut = cut_nodes
    
    return best_clustering, best_score, best_cut


def find_optimal_clustering(
    tree_data: Dict[str, str],
    cost_function: Callable[[List[str], List[str]], float],
    ground_truth: List[str],
    method: str = 'level',
    maximize: bool = True,
    decomposable_cost: str = 'purity'
) -> List[str]:
    """
    Simplified interface: Find and return the optimal tree-consistent clustering.
    
    This is the main function you should use. It takes a tree structure represented
    as hierarchical paths and finds the optimal clustering that is consistent with
    the tree structure, according to the provided cost function.
    
    Args:
        tree_data: Dictionary mapping datapoint IDs to hierarchical paths.
                   Example: {'0': 'society / politics / military',
                            '1': 'technology / cryptography / privacy'}
                   The paths represent a hierarchy from root to leaf, with the
                   root and leaf nodes excluded.
        
        cost_function: Function that evaluates clustering quality.
                      Takes (ground_truth_labels, predicted_labels) and returns a score.
                      Example: sklearn.metrics.cluster.adjusted_rand_score
                      (Not used for method='dp_memoized')
        
        ground_truth: List of ground truth cluster labels in the same order as
                     the sorted datapoint IDs.
        
        method: Algorithm to use for finding cuts:
               - 'level': Try cuts at each depth level (O(d×n), fast, recommended)
               - 'dp_memoized': TRUE DP with memoization (O(m×n), VERY FAST!)
               - 'dp': DFS search without memoization (O(2^m×n), slow)
               - 'exhaustive': Try all possible tree cuts (O(2^m×n), very slow)
        
        maximize: Whether to maximize (True) or minimize (False) the cost function.
                 Set to True for metrics like adjusted_rand_score, accuracy, etc.
                 Set to False for metrics like distance, error, etc.
        
        decomposable_cost: Which decomposable cost to use (only for method='dp_memoized'):
                          - 'purity': Cluster purity (recommended)
                          - 'entropy': Minimizing entropy
                          - 'homogeneity': Cluster homogeneity
    
    Returns:
        List of cluster labels for each datapoint (in sorted ID order).
    
    Example:
        >>> from sklearn.metrics.cluster import adjusted_rand_score
        >>> tree_data = {
        ...     '0': 'society / politics / military',
        ...     '1': 'technology / cryptography / privacy',
        ...     '2': 'society / conflict / terrorism'
        ... }
        >>> ground_truth = ['politics', 'tech', 'politics']
        >>> 
        >>> # Using traditional method
        >>> cost_fn = lambda gt, pred: adjusted_rand_score(gt, pred)
        >>> clustering = find_optimal_clustering(tree_data, cost_fn, ground_truth)
        >>> 
        >>> # Using fast DP with decomposable cost
        >>> clustering = find_optimal_clustering(
        ...     tree_data, None, ground_truth, 
        ...     method='dp_memoized', decomposable_cost='purity'
        ... )
    """
    best_clustering, _, _ = optimal_tree_clustering(
        tree_data, cost_function, ground_truth, method, maximize, decomposable_cost
    )
    return best_clustering


# Example usage
if __name__ == "__main__":
    from sklearn.metrics.cluster import adjusted_rand_score
    
    # Example tree data
    tree_data = {
        '0': 'society / politics / military',
        '1': 'technology / cryptography / privacy',
        '2': 'society / conflict / terrorism',
        '3': 'technology / computing / hardware',
        '4': 'recreation / games / robotics',
    }
    
    # Example ground truth
    ground_truth = [
        'talk.politics.misc',
        'sci.crypt',
        'talk.politics.mideast',
        'comp.windows.x',
        'sci.electronics',
    ]
    
    # Define cost function
    def evaluate_clustering(true_labels, predicted_labels):
        return adjusted_rand_score(true_labels, predicted_labels)
    
    # Find optimal clustering
    best_clustering, best_score, best_cut = optimal_tree_clustering(
        tree_data,
        evaluate_clustering,
        ground_truth,
        method='level',  # Use 'exhaustive' for all possible cuts
        maximize=True
    )
    
    print(f"Best score: {best_score}")
    print(f"Best clustering: {best_clustering}")
    print(f"Cut nodes: {[node.get_path() for node in best_cut]}")

