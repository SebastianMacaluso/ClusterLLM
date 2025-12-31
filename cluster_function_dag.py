"""
DAG-Consistent Clustering Algorithm

This module provides functions to find optimal DAG-consistent clusterings
given hierarchical data represented as a Directed Acyclic Graph (DAG) and 
an external cost function.

Key features:
- Each datapoint can have multiple hierarchical paths (creating a true DAG)
- Nodes can have multiple parents (paths can converge)
- Efficient DP algorithms with memoization (O(m×n) complexity)
- Supports both single-path and multi-path data formats

Data format:
  Single path:  {'0': 'Science / CS / AI'}
  Multiple paths: {'0': ['Science / CS / AI', 'Engineering / CS / AI']}
"""

from typing import Dict, List, Callable, Tuple, Optional, Set, Union
from collections import defaultdict, Counter, deque
import itertools


class DAGNode:
    """Represents a node in the hierarchical DAG."""
    
    def __init__(self, name: str):
        self.name = name
        self.children = {}  # name -> DAGNode
        self.parents = set()  # set of parent DAGNodes
        self.datapoints = []  # IDs of datapoints that end at this node
    
    def add_child(self, name: str) -> 'DAGNode':
        """Add or get a child node."""
        if name not in self.children:
            child = DAGNode(name)
            self.children[name] = child
        # Add parent relationship
        self.children[name].parents.add(self)
        return self.children[name]
    
    def get_all_datapoints(self) -> List[str]:
        """Get all datapoints in this subtree (following all descendants)."""
        visited = set()
        points = []
        
        def traverse(node):
            if node in visited:
                return
            visited.add(node)
            points.extend(node.datapoints)
            for child in node.children.values():
                traverse(child)
        
        traverse(self)
        return points
    
    def get_path(self) -> str:
        """
        Get a representative path from root to this node.
        Since DAGs can have multiple paths, we return one arbitrary path.
        """
        if not self.parents:
            return self.name
        # Pick first parent arbitrarily
        parent = next(iter(self.parents))
        parent_path = parent.get_path()
        if parent_path:
            return f"{parent_path} / {self.name}"
        return self.name
    
    def get_all_paths(self, visited=None) -> List[str]:
        """Get all paths from any root to this node."""
        if visited is None:
            visited = set()
        
        if self in visited:
            return []
        
        if not self.parents:
            return [self.name]
        
        visited.add(self)
        all_paths = []
        for parent in self.parents:
            parent_paths = parent.get_all_paths(visited.copy())
            for parent_path in parent_paths:
                if parent_path:
                    all_paths.append(f"{parent_path} / {self.name}")
                else:
                    all_paths.append(self.name)
        
        return all_paths


def build_dag(data: Dict[str, Union[str, List[str]]]) -> Tuple[DAGNode, Dict[str, DAGNode], Dict[str, Set[DAGNode]]]:
    """
    Build a DAG structure from hierarchical path data.
    
    Args:
        data: Dictionary mapping datapoint IDs to hierarchical paths.
              Each datapoint can have a single path (string) or multiple paths (list of strings).
              Examples:
                - Single path: {'0': 'society / politics / military'}
                - Multiple paths: {'0': ['Science / CS / AI', 'Engineering / CS / AI']}
              
              Multiple paths create a true DAG where a datapoint can be reached
              via different hierarchical routes.
    
    Returns:
        Tuple of:
        - root node (empty root that connects all top-level nodes)
        - dict mapping datapoint IDs to their leaf nodes (list if multiple paths)
        - dict mapping node names to sets of DAGNode objects (for nodes that appear in multiple places)
    """
    root = DAGNode("")  # Empty root
    datapoint_to_node = {}
    
    # Track all nodes by their full path context to handle convergence
    # Key: (name, depth) -> DAGNode
    nodes_by_position = {}
    
    for datapoint_id, paths in data.items():
        # Normalize to list format
        if isinstance(paths, str):
            paths = [paths]
        
        # Process each path for this datapoint
        leaf_nodes = []
        for path in paths:
            # Split the path into components
            components = [c.strip() for c in path.split('/')]
            
            # Navigate/build the DAG
            current = root
            for depth, component in enumerate(components):
                # Check if this node already exists at this level with this parent
                if component not in current.children:
                    # Check if we've seen this node at this depth before
                    node_key = (component, depth)
                    if node_key in nodes_by_position:
                        # Reuse existing node but add new parent relationship
                        existing_node = nodes_by_position[node_key]
                        current.children[component] = existing_node
                        existing_node.parents.add(current)
                    else:
                        # Create new node
                        new_node = current.add_child(component)
                        nodes_by_position[node_key] = new_node
                else:
                    # Ensure parent relationship exists
                    current.children[component].parents.add(current)
                
                current = current.children[component]
            
            # Store the datapoint at the leaf node
            if datapoint_id not in current.datapoints:
                current.datapoints.append(datapoint_id)
            leaf_nodes.append(current)
        
        # Store mapping (use first leaf node for backward compatibility)
        datapoint_to_node[datapoint_id] = leaf_nodes[0] if leaf_nodes else None
    
    # Create name -> nodes mapping for reference
    name_to_nodes = defaultdict(set)
    
    def collect_nodes(node):
        if node.name:
            name_to_nodes[node.name].add(node)
        for child in node.children.values():
            collect_nodes(child)
    
    collect_nodes(root)
    
    return root, datapoint_to_node, dict(name_to_nodes)


def get_all_nodes(root: DAGNode) -> List[DAGNode]:
    """Get all nodes in the DAG (using BFS to avoid duplicates)."""
    visited = set()
    nodes = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        
        if node.name:  # Skip empty root
            nodes.append(node)
        
        for child in node.children.values():
            if child not in visited:
                queue.append(child)
    
    return nodes


def get_all_internal_nodes(root: DAGNode) -> List[DAGNode]:
    """Get all internal nodes (non-leaf nodes) in the DAG."""
    all_nodes = get_all_nodes(root)
    return [node for node in all_nodes if node.children]


def get_clustering_from_cut(root: DAGNode, cut_nodes: List[DAGNode], 
                            datapoint_ids: List[str]) -> List[str]:
    """
    Generate cluster labels for a given DAG cut.
    
    Args:
        root: Root of the DAG
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
            # In case of overlap (shouldn't happen with valid cuts), first assignment wins
            if point_id not in datapoint_to_cluster:
                datapoint_to_cluster[point_id] = cluster_label
    
    # Return labels in the order of datapoint_ids
    return [datapoint_to_cluster.get(dp_id, "unknown") for dp_id in datapoint_ids]


def is_valid_cut(cut_nodes: List[DAGNode], root: DAGNode) -> bool:
    """
    Check if a set of nodes forms a valid cut in the DAG.
    A valid cut covers all datapoints exactly once.
    """
    # Get all datapoints from root
    all_datapoints = set(root.get_all_datapoints())
    
    # Get datapoints from cut
    cut_datapoints = set()
    for node in cut_nodes:
        node_points = node.get_all_datapoints()
        # Check for overlap
        if cut_datapoints.intersection(node_points):
            return False
        cut_datapoints.update(node_points)
    
    # Check if all datapoints are covered
    return cut_datapoints == all_datapoints


def generate_all_cuts(root: DAGNode) -> List[List[DAGNode]]:
    """
    Generate all possible DAG-consistent cuts.
    
    A cut is a set of nodes such that every datapoint belongs to exactly one
    subtree rooted at a cut node. This is more complex for DAGs due to 
    multiple paths to nodes.
    
    Args:
        root: Root of the DAG
    
    Returns:
        List of all possible valid cuts (each cut is a list of nodes)
    """
    # Get all nodes
    all_nodes = get_all_nodes(root)
    
    # For efficiency, we'll use a DFS approach similar to the tree version
    # but with validation to ensure no datapoint overlap
    cuts = []
    
    def generate_cuts_recursive(node: DAGNode, visited: Set[DAGNode]) -> List[List[DAGNode]]:
        """
        Generate all possible cuts for a subgraph rooted at node.
        Returns a list of cuts, where each cut is a list of nodes.
        visited tracks nodes we've already processed in this path.
        """
        if node in visited:
            return [[]]  # Already processed, return empty cut
        
        visited = visited | {node}
        
        if not node.children:  # Leaf node
            return [[node]]
        
        # Option 1: Cut at this node (don't recurse to children)
        node_cuts = [[node]]
        
        # Option 2: Don't cut here, but cut in children
        if node.children:
            # Get all possible cuts for each child
            child_cut_options = []
            for child in node.children.values():
                child_cuts = generate_cuts_recursive(child, visited)
                child_cut_options.append(child_cuts)
            
            # Combine cuts from all children (Cartesian product)
            if child_cut_options:
                for combination in itertools.product(*child_cut_options):
                    # Flatten the combination
                    combined_cut = []
                    for cut in combination:
                        combined_cut.extend(cut)
                    # Remove duplicates (can happen in DAGs)
                    combined_cut = list(dict.fromkeys(combined_cut))
                    if combined_cut:  # Only add non-empty cuts
                        node_cuts.append(combined_cut)
        
        return node_cuts
    
    # Handle root specially
    if not root.children:
        return [[root]]
    
    # Generate cuts for all children of root and combine them
    child_cut_options = []
    for child in root.children.values():
        child_cuts = generate_cuts_recursive(child, set())
        child_cut_options.append(child_cuts)
    
    # Combine cuts from all children
    for combination in itertools.product(*child_cut_options):
        combined_cut = []
        for cut in combination:
            combined_cut.extend(cut)
        # Remove duplicates
        combined_cut = list(dict.fromkeys(combined_cut))
        
        # Validate the cut
        if combined_cut and is_valid_cut(combined_cut, root):
            cuts.append(combined_cut)
    
    return cuts


def generate_level_cuts(root: DAGNode, max_depth: Optional[int] = None) -> List[List[DAGNode]]:
    """
    Generate DAG cuts at each level (more efficient than all possible cuts).
    
    Args:
        root: Root of the DAG
        max_depth: Maximum depth to consider (None for all depths)
    
    Returns:
        List of cuts, one for each level
    """
    cuts = []
    
    # Group nodes by depth (minimum depth for nodes reachable via multiple paths)
    levels = defaultdict(set)
    visited = {}
    
    def collect_nodes(node: DAGNode, depth: int):
        # If we've seen this node before at a shallower depth, skip
        if node in visited and visited[node] <= depth:
            return
        
        visited[node] = depth
        
        if node.name:  # Skip empty root
            levels[depth].add(node)
        
        for child in node.children.values():
            collect_nodes(child, depth + 1)
    
    collect_nodes(root, 0)
    
    # Create a cut for each level
    max_level = max(levels.keys()) if levels else 0
    if max_depth is not None:
        max_level = min(max_level, max_depth)
    
    for level in range(1, max_level + 1):
        if level in levels:
            cut = list(levels[level])
            # Validate the cut
            if is_valid_cut(cut, root):
                cuts.append(cut)
    
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


def optimal_dag_clustering_dp_memoized(
    dag_data: Dict[str, Union[str, List[str]]],
    cluster_cost_func: Callable[[List[str], Dict[str, str]], float],
    ground_truth: List[str],
    maximize: bool = True
) -> Tuple[List[str], float, List[DAGNode]]:
    """
    TRUE Dynamic Programming with memoization for decomposable cost functions on DAGs.
    
    This method achieves exponential speedup by using memoization when the cost
    function is decomposable (i.e., can be computed per-cluster independently).
    
    Key requirement: cluster_cost_func must compute cost for a SINGLE cluster,
    and total cost = sum of individual cluster costs.
    
    Note: For DAGs with significant node reuse, this can be even more efficient
    than the tree version due to memoization of shared subgraphs.
    
    Args:
        dag_data: Dictionary mapping datapoint IDs to hierarchical paths (string or list of strings).
                  Example: {'0': ['Science / CS / AI', 'Engineering / CS / AI']}
        cluster_cost_func: Function that takes (cluster_datapoint_ids, ground_truth_dict)
                          and returns the cost for that single cluster
        ground_truth: List of ground truth labels
        maximize: If True, maximize the cost function; if False, minimize it
    
    Returns:
        Tuple of (best_clustering, best_score, cut_nodes)
    """
    # Build the DAG
    root, datapoint_to_node, name_to_nodes = build_dag(dag_data)
    datapoint_ids = sorted(dag_data.keys(), key=lambda x: int(x))
    
    # Create ground truth dictionary
    ground_truth_dict = {dp_id: gt for dp_id, gt in zip(datapoint_ids, ground_truth)}
    
    # Memoization cache: node -> (best_score, best_cut_nodes)
    memo = {}
    
    def dp_solve(node: DAGNode, visited: Set[DAGNode]) -> Tuple[float, List[DAGNode]]:
        """
        Compute optimal score and cut for subgraph rooted at node.
        
        Args:
            node: Current node
            visited: Set of nodes already processed in this path (to handle cycles)
        
        Returns:
            (best_score, list_of_cut_nodes)
        """
        # Check if already visited in this path (cycle detection)
        if node in visited:
            return (0.0, [])
        
        # Check cache
        if node in memo:
            return memo[node]
        
        visited = visited | {node}
        
        # Get all datapoints in this subtree
        subtree_datapoints = node.get_all_datapoints()
        
        # Option 1: Cut at this node (make it a cluster)
        cut_here_score = cluster_cost_func(subtree_datapoints, ground_truth_dict)
        cut_here_nodes = [node]
        
        # Option 2: Don't cut here, recurse to children
        if node.children:
            recurse_score = 0.0
            recurse_nodes = []
            
            for child in node.children.values():
                child_score, child_nodes = dp_solve(child, visited)
                recurse_score += child_score
                recurse_nodes.extend(child_nodes)
            
            # Remove duplicates from recurse_nodes
            recurse_nodes = list(dict.fromkeys(recurse_nodes))
            
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
            child_score, child_nodes = dp_solve(child, set())
            total_score += child_score
            total_cut_nodes.extend(child_nodes)
        
        # Remove duplicates
        total_cut_nodes = list(dict.fromkeys(total_cut_nodes))
        
        best_score = total_score
        best_cut = total_cut_nodes
    else:
        best_score, best_cut = dp_solve(root, set())
    
    # Reconstruct clustering
    best_clustering = get_clustering_from_cut(root, best_cut, datapoint_ids)
    
    return best_clustering, best_score, best_cut


def optimal_dag_clustering_dp(
    dag_data: Dict[str, Union[str, List[str]]],
    cost_function: Callable[[List[str], List[str]], float],
    ground_truth: List[str],
    maximize: bool = True
) -> Tuple[List[str], float, List[DAGNode]]:
    """
    Find optimal DAG-consistent clustering using dynamic programming.
    
    This method uses DP to explore the search space more efficiently than
    exhaustive enumeration.
    
    Args:
        dag_data: Dictionary mapping datapoint IDs to hierarchical paths (string or list of strings).
                  Example: {'0': ['Science / CS / AI', 'Engineering / CS / AI']}
        cost_function: Function that takes (ground_truth, predicted) and returns a score
        ground_truth: List of ground truth labels
        maximize: If True, maximize the cost function; if False, minimize it
    
    Returns:
        Tuple of (best_clustering, best_score, cut_nodes)
    """
    # Build the DAG
    root, datapoint_to_node, name_to_nodes = build_dag(dag_data)
    datapoint_ids = sorted(dag_data.keys(), key=lambda x: int(x))
    
    # Initialize best solution
    best_score = float('-inf') if maximize else float('inf')
    best_clustering = None
    best_cut = None
    
    # Count evaluations for comparison
    evaluation_count = [0]
    
    def explore_cuts_dp(current_cut: List[DAGNode], remaining_nodes: List[DAGNode], 
                        processed: Set[DAGNode]) -> None:
        """
        Recursively explore cuts using DFS.
        
        Args:
            current_cut: Current list of cut nodes
            remaining_nodes: Nodes we still need to decide whether to cut
            processed: Nodes already included in the cut or whose descendants are in the cut
        """
        nonlocal best_score, best_clustering, best_cut
        
        # Base case: no more nodes to process
        if not remaining_nodes:
            # Validate and evaluate this cut
            if is_valid_cut(current_cut, root):
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
        
        # Skip if already processed
        if node in processed:
            explore_cuts_dp(current_cut, rest, processed)
            return
        
        # Option 1: Cut at this node (don't explore its children)
        new_processed = processed | {node}
        explore_cuts_dp(current_cut + [node], rest, new_processed)
        
        # Option 2: Don't cut here, add children to remaining nodes
        if node.children:
            children = [c for c in node.children.values() if c not in processed]
            explore_cuts_dp(current_cut, children + rest, processed)
    
    # Start DP search from root's children
    root_children = list(root.children.values()) if root.children else [root]
    explore_cuts_dp([], root_children, set())
    
    return best_clustering, best_score, best_cut


def optimal_dag_clustering(
    dag_data: Dict[str, Union[str, List[str]]],
    cost_function: Callable[[List[str], List[str]], float],
    ground_truth: List[str],
    method: str = 'level',
    maximize: bool = True,
    decomposable_cost: str = 'purity'
) -> Tuple[List[str], float, List[DAGNode]]:
    """
    Find the optimal DAG-consistent clustering.
    
    Args:
        dag_data: Dictionary mapping datapoint IDs to hierarchical paths (string or list of strings).
                  Single path: {'0': 'society / politics / military'}
                  Multiple paths: {'0': ['Science / CS / AI', 'Engineering / CS / AI']}
        cost_function: Function that takes (ground_truth, predicted) and returns a score
                      (only used for methods other than 'dp_memoized')
        ground_truth: List of ground truth labels
        method: Method for finding optimal clustering:
               - 'level': Try cuts at each DAG depth (O(depth × n), fast)
               - 'dp_memoized': TRUE DP with memoization (O(m × n), FAST!)
               - 'dp': DFS search without memoization (O(2^m × n), slow)
               - 'exhaustive': Try all DAG cuts (O(2^m × n), very slow)
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
        
        return optimal_dag_clustering_dp_memoized(
            dag_data, cluster_cost, ground_truth, maximize
        )
    
    # Use regular DP method (no memoization)
    if method == 'dp':
        return optimal_dag_clustering_dp(dag_data, cost_function, ground_truth, maximize)
    
    # Build the DAG for other methods
    root, datapoint_to_node, name_to_nodes = build_dag(dag_data)
    
    # Get ordered list of datapoint IDs (sorted by key)
    datapoint_ids = sorted(dag_data.keys(), key=lambda x: int(x))
    
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
    dag_data: Dict[str, Union[str, List[str]]],
    cost_function: Callable[[List[str], List[str]], float],
    ground_truth: List[str],
    method: str = 'level',
    maximize: bool = True,
    decomposable_cost: str = 'purity'
) -> List[str]:
    """
    Simplified interface: Find and return the optimal DAG-consistent clustering.
    
    This is the main function you should use. It takes a DAG structure represented
    as hierarchical paths and finds the optimal clustering that is consistent with
    the DAG structure, according to the provided cost function.
    
    Args:
        dag_data: Dictionary mapping datapoint IDs to hierarchical paths (string or list of strings).
                  
                  Single path per datapoint:
                    {'0': 'society / politics / military',
                     '1': 'technology / cryptography / privacy'}
                  
                  Multiple paths per datapoint (true DAG):
                    {'0': ['Science / CS / AI', 'Engineering / CS / AI'],
                     '1': ['Science / Biology / Genetics']}
                  
                  Multiple paths allow a datapoint to belong to different hierarchical
                  contexts simultaneously, creating richer DAG structures.
        
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
               - 'exhaustive': Try all possible DAG cuts (O(2^m×n), very slow)
        
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
        >>> # Example 1: Single path per datapoint
        >>> dag_data = {
        ...     '0': 'society / politics / military',
        ...     '1': 'technology / cryptography / privacy',
        ...     '2': 'technology / computing / privacy'
        ... }
        >>> ground_truth = ['politics', 'tech', 'tech']
        >>> clustering = find_optimal_clustering(
        ...     dag_data, None, ground_truth, 
        ...     method='dp_memoized', decomposable_cost='purity'
        ... )
        >>> 
        >>> # Example 2: Multiple paths per datapoint (true DAG)
        >>> dag_data = {
        ...     '0': ['Science / CS / AI', 'Engineering / CS / AI'],
        ...     '1': ['Science / Biology / Genetics'],
        ...     '2': ['Engineering / CS / Robotics']
        ... }
        >>> ground_truth = ['AI', 'Biology', 'Robotics']
        >>> clustering = find_optimal_clustering(
        ...     dag_data, None, ground_truth,
        ...     method='dp_memoized', decomposable_cost='purity'
        ... )
    """
    best_clustering, _, _ = optimal_dag_clustering(
        dag_data, cost_function, ground_truth, method, maximize, decomposable_cost
    )
    return best_clustering


# Example usage
if __name__ == "__main__":
    # Example DAG data with multiple paths per datapoint
    # This demonstrates the true power of DAG-based clustering
    dag_data = {
        '0': ['Science / Computer Science / AI', 'Science / Computer Science / Databases'],
        '1': ['Science / Biology / Genetics'],
        '2': ['Engineering / Computer Science / AI', 'Engineering / Computer Science / Robotics']
    }
    
    # Example ground truth
    ground_truth = [
        'AI',
        'Biology',
        'Computer Science',
    ]
    
    print("=" * 70)
    print("DAG-Consistent Clustering Example")
    print("=" * 70)
    
    print("\nInput: Datapoints with multiple hierarchical paths")
    for dp_id, paths in sorted(dag_data.items()):
        print(f"\n  Datapoint {dp_id}:")
        if isinstance(paths, list):
            for path in paths:
                print(f"    - {path}")
        else:
            print(f"    - {paths}")
    
    # Build DAG to show structure
    root, dp_to_node, name_to_nodes = build_dag(dag_data)
    print(f"\n\nDAG Structure:")
    print(f"  Total unique nodes: {len(name_to_nodes)}")
    
    # Show nodes with multiple parents
    multi_parent_nodes = []
    for name, nodes in name_to_nodes.items():
        for node in nodes:
            if len(node.parents) > 1:
                parent_names = sorted([p.name for p in node.parents if p.name])
                multi_parent_nodes.append((name, parent_names))
    
    if multi_parent_nodes:
        print(f"  Nodes with multiple parents:")
        for name, parents in multi_parent_nodes:
            print(f"    '{name}' ← {parents}")
    
    # Find optimal clustering using DP memoized (recommended)
    print("\n" + "=" * 70)
    print("Finding Optimal Clustering...")
    print("=" * 70)
    
    best_clustering, best_score, best_cut = optimal_dag_clustering(
        dag_data,
        cost_function=None,  # Not needed for dp_memoized
        ground_truth=ground_truth,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print(f"\nMethod: DP Memoized (Fast, Recommended)")
    print(f"Purity score: {best_score:.4f}")
    print(f"\nClustering results:")
    for dp_id, cluster in enumerate(best_clustering):
        gt = ground_truth[dp_id]
        print(f"  Datapoint {dp_id} [{gt:16}] → {cluster}")
    
    print(f"\n  Number of clusters: {len(set(best_clustering))}")
    print(f"  Cut nodes ({len(best_cut)} total):")
    for node in best_cut:
        datapoints = node.get_all_datapoints()
        print(f"    - {node.get_path()} ({len(datapoints)} datapoints)")
    
    print("\n" + "=" * 70)
    print("✓ Example complete!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  • Multiple paths per datapoint (true DAG structure)")
    print("  • Nodes with multiple parents (path convergence)")
    print("  • Efficient O(m×n) DP algorithm with memoization")
    print("  • Automatic optimal cut selection")
    print("=" * 70)

