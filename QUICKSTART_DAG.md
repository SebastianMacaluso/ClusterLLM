# Quick Start: DAG-Based Clustering

Get started with DAG-based clustering in 5 minutes!

## Installation

No additional dependencies needed beyond what's in `requirements.txt`.

```bash
# If you need sklearn for custom cost functions
pip install scikit-learn
```

## Basic Usage (Copy & Paste)

### Single Path per Datapoint

```python
from cluster_function_dag import find_optimal_clustering

# 1. Define your data as paths
dag_data = {
    '0': 'Science / Computer Science / AI',
    '1': 'Engineering / Computer Science / AI',  # Note: paths converge at "AI"
    '2': 'Science / Biology / Genetics',
}

# 2. Provide ground truth labels
ground_truth = ['tech', 'tech', 'bio']

# 3. Find optimal clustering
clustering = find_optimal_clustering(
    dag_data,
    cost_function=None,           # Not needed for dp_memoized
    ground_truth=ground_truth,
    method='dp_memoized',         # Recommended: O(m×n), very fast!
    maximize=True,
    decomposable_cost='purity'    # Options: 'purity', 'entropy', 'homogeneity'
)

# 4. Use the results
print(clustering)
```

### Multiple Paths per Datapoint (Advanced)

```python
from cluster_function_dag import find_optimal_clustering

# 1. Define data with multiple paths per datapoint
dag_data = {
    '0': ['Science / Computer Science / AI',
          'Engineering / Robotics / AI'],        # Datapoint 0 has TWO paths
    '1': ['Science / Biology / Genetics'],       # Datapoint 1 has one path
    '2': ['Engineering / Computer Science / ML',
          'Mathematics / Statistics / ML']       # Datapoint 2 has TWO paths
}

# 2. Provide ground truth labels
ground_truth = ['AI', 'Bio', 'ML']

# 3. Find optimal clustering (same API!)
clustering = find_optimal_clustering(
    dag_data,
    cost_function=None,
    ground_truth=ground_truth,
    method='dp_memoized',
    maximize=True,
    decomposable_cost='purity'
)

# 4. Use the results
print(clustering)
# The algorithm considers all paths for each datapoint
```

## That's It! 🎉

The algorithm automatically:
- ✅ Accepts single paths (string) or multiple paths (list) per datapoint
- ✅ Detects converging paths (e.g., "AI" appears in multiple hierarchies)
- ✅ Merges shared nodes into a DAG structure
- ✅ Handles datapoints with multiple hierarchical identities
- ✅ Finds the optimal cut through the DAG
- ✅ Returns cluster labels for each datapoint

## When to Use DAG vs Tree

### Use DAG (`cluster_function_dag.py`) when:
```python
# Concepts appear in multiple hierarchies
data = {
    '0': 'Technology / Cryptography / Privacy',
    '1': 'Society / Law / Privacy',  # "Privacy" in both!
}
```

### Use Tree (`cluster_function.py`) when:
```python
# Strictly hierarchical, no convergence
data = {
    '0': 'Animals / Mammals / Dogs',
    '1': 'Animals / Birds / Eagles',  # No shared concepts
}
```

## Common Patterns

### Pattern 1: Converging Paths (Node-Level DAG)
```python
# Example: Machine Learning appears in multiple fields
data = {
    '0': 'Computer Science / AI / Machine Learning',
    '1': 'Statistics / Data Science / Machine Learning',
    '2': 'Mathematics / Optimization / Machine Learning',
}
# DAG will recognize all three as the same "Machine Learning" concept
# This is node-level convergence
```

### Pattern 2: Multi-Inheritance Taxonomy (Node-Level DAG)
```python
# Example: Privacy spans Technology and Society
data = {
    '0': 'Technology / Cryptography / Privacy',
    '1': 'Technology / Computing / Privacy',
    '2': 'Society / Law / Privacy',
    '3': 'Society / Ethics / Privacy',
}
# DAG will create one "Privacy" node with multiple parents
```

### Pattern 3: Multiple Paths per Datapoint (Datapoint-Level DAG)
```python
# Example: A research paper that truly spans multiple domains
data = {
    '0': ['Computer Science / AI / Neural Networks',
          'Neuroscience / Computational / Neural Networks'],
    '1': ['Computer Science / Robotics / Control',
          'Engineering / Mechanical / Control'],
}
# Each datapoint belongs to MULTIPLE hierarchical contexts simultaneously
# This creates the richest DAG structure
```

## Methods Comparison

| Method | Speed | When to Use |
|--------|-------|-------------|
| `'dp_memoized'` | ⚡⚡⚡ Very Fast | **Recommended for all cases** |
| `'level'` | ⚡⚡ Fast | Quick approximate solution |
| `'dp'` | 🐌 Slow | Custom cost functions |
| `'exhaustive'` | 🐌🐌 Very Slow | Small datasets only |

**Recommendation:** Always start with `'dp_memoized'` and `'purity'` cost.

## Full API

```python
from cluster_function_dag import optimal_dag_clustering

# Get clustering, score, and cut nodes
clustering, score, cut_nodes = optimal_dag_clustering(
    dag_data=dag_data,
    cost_function=None,           # Only for 'level', 'dp', 'exhaustive'
    ground_truth=ground_truth,
    method='dp_memoized',
    maximize=True,
    decomposable_cost='purity'
)

print(f"Score: {score}")
print(f"Clusters: {len(set(clustering))}")
print(f"Cut points: {[node.get_path() for node in cut_nodes]}")
```

## Examples

Run the included examples:

```bash
# Comprehensive test suite
python test_dag_clustering.py

# Practical examples
python example_dag_usage.py

# Compare tree vs DAG
python compare_tree_vs_dag.py
```

## Troubleshooting

### "TypeError: 'NoneType' object is not callable"
**Problem:** Using `cost_function=None` with methods other than `'dp_memoized'`

**Solution:** Either use `method='dp_memoized'` or provide a cost function:
```python
def my_cost(true_labels, predicted_labels):
    # Your cost function here
    return score

clustering = find_optimal_clustering(
    data, my_cost, ground_truth, method='level'
)
```

### "No converging paths detected"
**Problem:** Your data is actually tree-structured

**Solution:** This is fine! DAG handles trees as a special case. Or use `cluster_function.py` for slightly better performance.

### "Cut validation failed"
**Problem:** Invalid cut (shouldn't happen with built-in methods)

**Solution:** Use built-in methods (`'dp_memoized'`, `'level'`, etc.) rather than manually creating cuts.

## Next Steps

1. **Read the full documentation:** `README_DAG.md`
2. **See practical examples:** `example_dag_usage.py`
3. **Compare approaches:** `compare_tree_vs_dag.py`
4. **Understand the algorithm:** `SUMMARY_DAG_IMPLEMENTATION.md`

## Quick Reference

```python
# Import
from cluster_function_dag import find_optimal_clustering, optimal_dag_clustering

# Minimal example (single path)
clustering = find_optimal_clustering(
    dag_data={'0': 'A/B/C', '1': 'A/B/D'},
    cost_function=None,
    ground_truth=['x', 'y'],
    method='dp_memoized'
)

# Multiple paths per datapoint
clustering = find_optimal_clustering(
    dag_data={
        '0': ['Science/CS/AI', 'Engineering/Robotics/AI'],
        '1': ['Science/Bio/Genetics']
    },
    cost_function=None,
    ground_truth=['AI', 'Bio'],
    method='dp_memoized'
)

# Full example with all options
clustering, score, cuts = optimal_dag_clustering(
    dag_data=data,
    cost_function=None,
    ground_truth=labels,
    method='dp_memoized',        # 'dp_memoized', 'level', 'dp', 'exhaustive'
    maximize=True,               # True to maximize, False to minimize
    decomposable_cost='purity'   # 'purity', 'entropy', 'homogeneity'
)
```

## Performance

- **Small datasets** (< 100 points): All methods work fine
- **Medium datasets** (100-1000 points): Use `'dp_memoized'` or `'level'`
- **Large datasets** (> 1000 points): Use `'dp_memoized'` (O(m×n) complexity)

## Support

- **Documentation:** `README_DAG.md`
- **Examples:** `example_dag_usage.py`
- **Tests:** `test_dag_clustering.py`
- **Comparison:** `compare_tree_vs_dag.py`

---

**You're ready to go! 🚀**

Start with the basic example above and adapt it to your data.

