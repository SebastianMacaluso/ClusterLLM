# Tree-Consistent Clustering

A Python implementation for finding optimal tree-consistent clusterings given hierarchical data and a custom cost function.

## Overview

This library implements an algorithm that finds the optimal clustering of datapoints that respects a hierarchical tree structure. Given:
- A set of datapoints represented as hierarchical paths (e.g., `"society / politics / military"`)
- A cost function to evaluate clustering quality
- Ground truth labels for optimization

The algorithm finds the best way to "cut" the tree at various levels to create clusters that maximize (or minimize) your cost function.

## Key Concept: Tree-Consistent Clustering

A **tree-consistent clustering** is one where:
1. Each cluster corresponds to a subtree in the hierarchy
2. All datapoints under the same subtree node belong to the same cluster
3. The clustering is formed by selecting certain nodes as "cluster representatives"

For example, if your tree has paths like:
- `technology / computing / hardware`
- `technology / computing / software`
- `technology / cryptography / encryption`

A tree-consistent clustering might group the first two under `technology / computing` and the third under `technology / cryptography`.

## Installation

Requires:
- Python 3.7+
- scikit-learn (for metrics like `adjusted_rand_score`)

```bash
pip install scikit-learn
```

## Quick Start

```python
from cluster_function import find_optimal_clustering
from sklearn.metrics.cluster import adjusted_rand_score

# Define your tree structure
tree_data = {
    '0': 'society / politics / military',
    '1': 'technology / cryptography / privacy',
    '2': 'society / conflict / terrorism',
    '3': 'technology / computing / hardware',
}

# Ground truth labels (in sorted ID order)
ground_truth = ['politics', 'tech', 'politics', 'tech']

# Cost function (higher is better)
cost_fn = lambda true, pred: adjusted_rand_score(true, pred)

# Find optimal clustering
clustering = find_optimal_clustering(
    tree_data=tree_data,
    cost_function=cost_fn,
    ground_truth=ground_truth,
    method='level',      # 'level' or 'exhaustive'
    maximize=True        # True to maximize, False to minimize
)

print(clustering)
# Output: ['society', 'technology', 'society', 'technology']
```

## Main Function

### `find_optimal_clustering()`

```python
find_optimal_clustering(
    tree_data: Dict[str, str],
    cost_function: Callable[[List[str], List[str]], float],
    ground_truth: List[str],
    method: str = 'level',
    maximize: bool = True
) -> List[str]
```

**Parameters:**

- **`tree_data`** (dict): Dictionary mapping datapoint IDs to hierarchical paths.
  - Keys: String IDs (e.g., `'0'`, `'1'`, `'2'`)
  - Values: Hierarchical paths with `/` separator (e.g., `'category / subcategory / item'`)
  - The paths represent intermediate nodes (root and leaf are excluded)

- **`cost_function`** (callable): Function to evaluate clustering quality.
  - Signature: `cost_function(ground_truth, predicted) -> float`
  - Example: `sklearn.metrics.cluster.adjusted_rand_score`

- **`ground_truth`** (list): Ground truth cluster labels.
  - Must be in the same order as sorted datapoint IDs
  - Used by the cost function to evaluate each candidate clustering

- **`method`** (str, default='level'): Algorithm for generating tree cuts.
  - `'level'`: Try cuts at each tree depth level (O(depth), fast)
  - `'exhaustive'`: Try all possible tree cuts (O(2^nodes), thorough but slow)

- **`maximize`** (bool, default=True): Optimization direction.
  - `True`: Maximize the cost function (for metrics like accuracy, ARI)
  - `False`: Minimize the cost function (for metrics like distance, error)

**Returns:**
- List of predicted cluster labels (one per datapoint, in sorted ID order)

## Algorithm Details

### How It Works

1. **Build Tree**: Parse hierarchical paths into a tree structure
2. **Generate Cuts**: Create candidate clusterings by "cutting" the tree at different levels
   - **Level-wise**: Cut at depth 1, depth 2, depth 3, etc.
   - **Exhaustive**: Try all combinations of internal nodes as cut points
3. **Evaluate**: Score each clustering using the provided cost function
4. **Select Best**: Return the clustering with the optimal score

### Complexity

- **Level-wise method**: O(depth × n), where n = number of datapoints
  - Fast and practical for most use cases
  - Guarantees finding the best single-level cut

- **Exhaustive method**: O(2^m × n), where m = number of internal nodes
  - Explores all possible tree-consistent clusterings
  - Can be slow for large, deep trees
  - Guarantees global optimum

## Examples

### Example 1: Basic Usage with Adjusted Rand Index

```python
from cluster_function import find_optimal_clustering
from sklearn.metrics.cluster import adjusted_rand_score

tree_data = {
    '0': 'animals / mammals / cat',
    '1': 'animals / mammals / dog',
    '2': 'animals / birds / sparrow',
    '3': 'animals / birds / eagle',
    '4': 'plants / flowers / rose',
    '5': 'plants / flowers / tulip',
}

ground_truth = ['mammal', 'mammal', 'bird', 'bird', 'plant', 'plant']

clustering = find_optimal_clustering(
    tree_data,
    adjusted_rand_score,
    ground_truth,
    method='level',
    maximize=True
)

# Result might be: ['animals / mammals', 'animals / mammals', 
#                   'animals / birds', 'animals / birds',
#                   'plants / flowers', 'plants / flowers']
```

### Example 2: Custom Cost Function

```python
def purity_score(true_labels, predicted_labels):
    """Calculate cluster purity."""
    from collections import defaultdict
    
    clusters = defaultdict(list)
    for true, pred in zip(true_labels, predicted_labels):
        clusters[pred].append(true)
    
    total_correct = sum(
        max(members.count(label) for label in set(members))
        for members in clusters.values()
    )
    
    return total_correct / len(true_labels)

clustering = find_optimal_clustering(
    tree_data,
    purity_score,
    ground_truth,
    method='level',
    maximize=True
)
```

### Example 3: Large Dataset (200 Datapoints)

See `test_clustering.py` for a complete example with 200 datapoints from the newsgroups dataset.

```bash
python test_clustering.py
```

## API Reference

### Advanced Functions

If you need more control, you can use the lower-level functions:

#### `optimal_tree_clustering()`

Returns additional information including the score and cut nodes:

```python
from cluster_function import optimal_tree_clustering

clustering, score, cut_nodes = optimal_tree_clustering(
    tree_data,
    cost_function,
    ground_truth,
    method='level',
    maximize=True
)

print(f"Best score: {score}")
print(f"Cut at nodes: {[node.get_path() for node in cut_nodes]}")
```

#### `build_tree()`

Build a tree structure from hierarchical paths:

```python
from cluster_function import build_tree

root, datapoint_to_node = build_tree(tree_data)
```

## Files

- **`cluster_function.py`**: Main implementation
- **`example_usage.py`**: Simple examples demonstrating usage
- **`test_clustering.py`**: Full test with 200-datapoint example
- **`README.md`**: This file

## Use Cases

This algorithm is useful when:

1. **You have hierarchical data**: Datapoints naturally form a tree structure (e.g., product categories, topic hierarchies, taxonomies)

2. **You want tree-consistent clusters**: Clusters should respect the hierarchy (e.g., all items in "Electronics / Computers" should be grouped together)

3. **You have a custom quality metric**: You want to optimize clustering according to a specific cost function

4. **You need explainable clustering**: Tree-based clusters are easy to interpret and explain

## Limitations

- **Tree structure required**: Datapoints must be representable as hierarchical paths
- **Exhaustive method can be slow**: Use `method='level'` for large trees
- **Requires ground truth**: The optimization needs reference labels to evaluate against

## Tips

1. **Start with `method='level'`**: It's much faster and often finds the optimal solution
2. **Choose cost function carefully**: Make sure it aligns with your clustering goals
3. **Check tree structure**: Ensure paths are consistently formatted with `/` separators
4. **Sort datapoint IDs**: The algorithm sorts IDs numerically, so ground truth order should match

## License

This code is provided as-is for research and educational purposes.

## Citation

If you use this code in your research, please cite appropriately.

