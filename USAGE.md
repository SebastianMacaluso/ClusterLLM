# Quick Usage Guide

## TL;DR

```python
from cluster_function import find_optimal_clustering
from sklearn.metrics.cluster import adjusted_rand_score

# Your tree data: ID -> hierarchical path
tree_data = {
    '0': 'category / subcategory / item',
    '1': 'category / subcategory / item2',
    # ... more datapoints
}

# Ground truth labels (in sorted ID order)
ground_truth = ['label1', 'label2', ...]

# Find optimal clustering
clustering = find_optimal_clustering(
    tree_data,
    adjusted_rand_score,  # or any cost function
    ground_truth,
    method='level',       # 'level' is fast, 'exhaustive' is thorough
    maximize=True         # True for ARI, accuracy; False for distance, error
)

# clustering is a list of predicted cluster labels
print(clustering)
```

## What Does This Do?

Given:
- **Tree structure**: Datapoints organized as hierarchical paths (e.g., `"animals / mammals / cat"`)
- **Cost function**: A way to measure how good a clustering is (e.g., adjusted Rand index)
- **Ground truth**: Reference labels to optimize against

The algorithm finds the **best way to cut the tree** to create clusters that maximize (or minimize) your cost function, while keeping clusters tree-consistent.

## When to Use This

✅ **Use when:**
- You have hierarchical data (taxonomies, categories, topic hierarchies)
- You want clusters that respect the hierarchy
- You have a specific quality metric to optimize
- You need explainable, interpretable clusters

❌ **Don't use when:**
- Your data isn't hierarchical
- You don't have ground truth labels
- You need non-tree-structured clusters

## Files

| File | Purpose |
|------|---------|
| `cluster_function.py` | Main implementation |
| `example_usage.py` | Simple examples to get started |
| `test_full_example.py` | Full 200-datapoint test case |
| `README.md` | Complete documentation |
| `USAGE.md` | This quick reference |

## Examples

### Run Simple Examples
```bash
python example_usage.py
```

### Run Full Test (200 datapoints)
```bash
python test_full_example.py
```

## Key Parameters

### `method`
- **`'level'`** (recommended): Fast, tries cuts at each tree depth
- **`'exhaustive'`**: Slow, tries all possible tree cuts

### `maximize`
- **`True`**: For metrics where higher = better (accuracy, ARI, F1)
- **`False`**: For metrics where lower = better (error, distance)

## Common Cost Functions

```python
# Adjusted Rand Index (similarity to ground truth)
from sklearn.metrics.cluster import adjusted_rand_score
cost_fn = adjusted_rand_score  # maximize=True

# Normalized Mutual Information
from sklearn.metrics.cluster import normalized_mutual_info_score
cost_fn = normalized_mutual_info_score  # maximize=True

# Fowlkes-Mallows Score
from sklearn.metrics.cluster import fowlkes_mallows_score
cost_fn = fowlkes_mallows_score  # maximize=True

# Custom function
def custom_cost(true_labels, pred_labels):
    # Your logic here
    return score  # Higher or lower is better depending on maximize flag
```

## Return Value

The function returns a **list of cluster labels**, one per datapoint, in the same order as sorted datapoint IDs.

```python
clustering = find_optimal_clustering(...)
# clustering = ['cluster1', 'cluster1', 'cluster2', ...]
```

Each label is a path in the tree (e.g., `'technology / computing'`) representing where the tree was cut for that cluster.

## Tips

1. **Start with `method='level'`** - it's much faster and usually works well
2. **Ensure consistent formatting** - use `' / '` as separator in paths
3. **Match ground truth order** - ground truth should match sorted datapoint IDs
4. **Check your cost function** - make sure it works with string labels

## Need More Details?

- See `example_usage.py` for runnable examples
- See `README.md` for complete documentation
- See `cluster_function.py` for implementation details

## Quick Debugging

**Problem:** "Tree cuts produce poor results"
- Try `method='exhaustive'` (slower but more thorough)
- Check if tree structure makes sense for your data
- Verify ground truth order matches sorted IDs

**Problem:** "Code is too slow"
- Use `method='level'` instead of `'exhaustive'`
- Check tree depth - deep trees are slower
- Consider pruning very deep branches

**Problem:** "Wrong cluster labels returned"
- Verify ground truth is in sorted ID order
- Check that cost function signature is correct
- Ensure maximize flag matches your metric

## Contact

For issues, questions, or contributions, see the main README.md file.

