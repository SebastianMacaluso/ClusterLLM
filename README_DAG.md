# DAG-Based Clustering

This document explains the DAG-based clustering implementation (`cluster_function_dag.py`) and how it differs from the tree-based version (`cluster_function.py`).

## Overview

The DAG-based clustering algorithm extends the tree-based approach to handle **Directed Acyclic Graphs (DAGs)** with two key features:

1. **Nodes can have multiple parents** - hierarchical paths can converge at shared concepts
2. **Datapoints can have multiple paths** - each datapoint can belong to multiple hierarchical contexts simultaneously

This creates richer, more expressive clustering structures for complex data.

## Key Differences from Tree-Based Clustering

### 1. **Node Structure**

**Tree (`TreeNode`):**
- Each node has exactly one parent (except root)
- Single path from root to any node
- Simpler structure

**DAG (`DAGNode`):**
- Nodes can have multiple parents
- Multiple paths from root(s) to any node
- More complex structure to handle convergence

### 2. **Example Data Structures**

**Tree Example:**
```python
tree_data = {
    '0': 'Science / Computer Science / AI',
    '1': 'Science / Biology / Genetics',
    '2': 'Engineering / Robotics / AI',
}
# "AI" appears twice but as separate nodes in different branches
```

**DAG Example (Single Path):**
```python
dag_data = {
    '0': 'Science / Computer Science / AI',
    '1': 'Engineering / Robotics / AI',
    '2': 'Technology / Cryptography / Privacy',
    '3': 'Society / Law / Privacy',
}
# "AI" and "Privacy" can be the SAME node reached via different paths
```

**DAG Example (Multiple Paths per Datapoint):**
```python
dag_data = {
    '0': ['Science / Computer Science / AI', 
          'Engineering / Robotics / AI'],
    '1': ['Technology / Cryptography / Privacy',
          'Society / Law / Privacy'],
    '2': ['Science / Biology / Genetics']
}
# Datapoint 0 belongs to BOTH "Science/CS" AND "Engineering/Robotics" contexts
# Datapoint 1 belongs to BOTH "Technology/Crypto" AND "Society/Law" contexts
# This creates a true DAG where a datapoint can have multiple hierarchical identities
```

### 3. **Path Representation**

In the DAG version:
- Paths can converge at shared nodes
- A node like "Privacy" might have parents "Cryptography", "Computing", and "Law"
- This creates a true DAG structure where concepts can belong to multiple hierarchies

### 4. **Cut Validation**

**Tree:**
- Cuts are always valid if they partition the tree
- No overlap checking needed

**DAG:**
- Cuts must be validated with `is_valid_cut()`
- Ensures no datapoint is covered by multiple cut nodes
- More complex due to potential path convergence

## Usage

### Basic Example (Single Path per Datapoint)

```python
from cluster_function_dag import find_optimal_clustering

# Define DAG structure with converging paths
dag_data = {
    '0': 'Technology / Cryptography / Privacy',
    '1': 'Technology / Computing / Privacy',
    '2': 'Society / Law / Privacy',
    '3': 'Technology / Cryptography / Encryption',
}

ground_truth = ['tech', 'tech', 'society', 'tech']

# Find optimal clustering using DP with memoization
clustering = find_optimal_clustering(
    dag_data,
    cost_function=None,  # Not needed for dp_memoized
    ground_truth=ground_truth,
    method='dp_memoized',
    maximize=True,
    decomposable_cost='purity'
)

print(f"Clustering: {clustering}")
```

### Advanced Example (Multiple Paths per Datapoint)

```python
from cluster_function_dag import find_optimal_clustering

# Each datapoint can have multiple hierarchical contexts
dag_data = {
    '0': ['Science / Computer Science / AI',
          'Engineering / Robotics / AI'],
    '1': ['Science / Biology / Genetics'],
    '2': ['Engineering / Robotics / Control',
          'Mathematics / Optimization / Control']
}

ground_truth = ['AI', 'Biology', 'Control']

# Find optimal clustering
clustering = find_optimal_clustering(
    dag_data,
    cost_function=None,
    ground_truth=ground_truth,
    method='dp_memoized',
    maximize=True,
    decomposable_cost='purity'
)

print(f"Clustering: {clustering}")
# Datapoint 0 is accessible via both Science and Engineering paths
# The algorithm finds the optimal cut considering all paths
```

### Data Format Options

The DAG clustering supports flexible data formats:

1. **Single path (string)**:
   ```python
   {'0': 'Science / CS / AI'}
   ```

2. **Single path (list)**:
   ```python
   {'0': ['Science / CS / AI']}
   ```

3. **Multiple paths (list)**:
   ```python
   {'0': ['Science / CS / AI', 'Engineering / Robotics / AI']}
   ```

4. **Mixed formats**:
   ```python
   {
       '0': 'Science / Biology / Genetics',  # String
       '1': ['Science / CS / AI', 'Engineering / CS / AI']  # List
   }
   ```

All formats are fully supported and can be mixed in the same dataset.

### Available Methods

Same as tree-based version:

1. **`'dp_memoized'`** (Recommended, FAST!)
   - O(m × n) complexity
   - Uses decomposable cost functions (purity, entropy, homogeneity)
   - Memoization provides exponential speedup
   - Even more efficient for DAGs with shared subgraphs

2. **`'level'`** (Fast, approximate)
   - O(depth × n) complexity
   - Tries cuts at each depth level
   - Good for quick results

3. **`'dp'`** (Slow)
   - O(2^m × n) complexity
   - DFS search without memoization
   - Supports custom cost functions

4. **`'exhaustive'`** (Very slow)
   - O(2^m × n) complexity
   - Pre-generates all possible cuts
   - Only for small DAGs

## When to Use DAG vs Tree

### Use **Tree-based** (`cluster_function.py`) when:
- Your hierarchy is strictly tree-structured
- Each concept belongs to exactly one parent category
- Paths never converge
- Example: File system, organizational chart

### Use **DAG-based** (`cluster_function_dag.py`) when:
- Concepts can belong to multiple parent categories
- Paths can converge at shared nodes
- You have a true DAG structure
- Example: 
  - "Privacy" belongs to both "Technology" and "Law"
  - "AI" belongs to both "Computer Science" and "Robotics"
  - Multi-inheritance taxonomies

## API Reference

### Main Functions

#### `find_optimal_clustering()`
Simplified interface for finding optimal DAG-consistent clustering.

**Parameters:**
- `dag_data`: Dict mapping datapoint IDs to hierarchical paths
- `cost_function`: Clustering evaluation function (not used for `dp_memoized`)
- `ground_truth`: List of ground truth labels
- `method`: Algorithm choice (`'level'`, `'dp_memoized'`, `'dp'`, `'exhaustive'`)
- `maximize`: Whether to maximize (True) or minimize (False) the cost
- `decomposable_cost`: Cost type for `dp_memoized` (`'purity'`, `'entropy'`, `'homogeneity'`)

**Returns:**
- List of cluster labels for each datapoint

#### `optimal_dag_clustering()`
Full interface returning clustering, score, and cut nodes.

**Returns:**
- Tuple of `(clustering, score, cut_nodes)`

### Helper Functions

- `build_dag()`: Build DAG structure from path data
- `get_all_nodes()`: Get all nodes in the DAG
- `get_clustering_from_cut()`: Generate labels from a cut
- `is_valid_cut()`: Validate that a cut covers all datapoints exactly once
- `generate_all_cuts()`: Generate all possible DAG cuts
- `generate_level_cuts()`: Generate cuts at each depth level

### Decomposable Cost Functions

- `cluster_purity()`: Fraction of most common class in cluster
- `cluster_entropy()`: Negative entropy (lower = more pure)
- `cluster_homogeneity()`: Homogeneity score

## Implementation Details

### DAG Construction

The `build_dag()` function:
1. Creates nodes for each path component
2. Detects when paths converge at the same node name and depth
3. Maintains multiple parent relationships
4. Returns root, datapoint-to-node mapping, and name-to-nodes mapping

### Cut Generation

For DAGs, cut generation is more complex:
- Must track visited nodes to avoid cycles
- Must validate cuts to ensure no datapoint overlap
- Removes duplicate nodes that appear via multiple paths

### Memoization Benefits

The DP memoized approach is especially powerful for DAGs:
- Shared subgraphs are computed only once
- Memoization cache is reused across multiple parent paths
- Can provide even greater speedup than tree version

## Testing

Run the test suite:

```bash
python test_dag_clustering.py
```

Tests include:
1. Basic DAG construction
2. DP memoized clustering
3. Level-based clustering
4. Converging paths (true DAG property)
5. Trees as special case of DAGs

## Performance Comparison

| Method | Tree Complexity | DAG Complexity | Notes |
|--------|----------------|----------------|-------|
| `dp_memoized` | O(m × n) | O(m × n) | Even faster for DAGs with shared subgraphs |
| `level` | O(d × n) | O(d × n) | Fast approximate solution |
| `dp` | O(2^m × n) | O(2^m × n) | Slow, explores all cuts |
| `exhaustive` | O(2^m × n) | O(2^m × n) | Very slow, pre-generates all cuts |

Where:
- m = number of internal nodes
- n = number of datapoints
- d = depth of hierarchy

## Examples

### Example 1: Technology Taxonomy with Convergence

```python
dag_data = {
    '0': 'Science / Computer Science / AI',
    '1': 'Science / Computer Science / Databases',
    '2': 'Engineering / Computer Science / AI',
    '3': 'Engineering / Computer Science / Robotics',
}

# Note: "Computer Science" and "AI" appear in multiple paths
# They will be represented as the same nodes with multiple parents
```

### Example 2: Multi-domain Concepts

```python
dag_data = {
    '0': 'Technology / Cryptography / Privacy',
    '1': 'Technology / Computing / Privacy',
    '2': 'Society / Law / Privacy',
    '3': 'Society / Ethics / Privacy',
}

# "Privacy" is a shared concept across Technology and Society domains
```

## Limitations

1. **Path Format**: Paths must use `" / "` as separator
2. **Node Convergence**: Nodes are considered the same if they have the same name and appear at the same depth
3. **Cycle Detection**: While the structure is a DAG, the algorithm includes cycle detection for safety
4. **Memory**: For very large DAGs with many converging paths, memory usage can be higher than trees

## Future Enhancements

Potential improvements:
- Support for weighted edges in the DAG
- Hierarchical clustering with soft assignments
- Parallel processing for large DAGs
- Visualization tools for DAG structure and cuts

## References

- Original tree-based implementation: `cluster_function.py`
- Test suite: `test_dag_clustering.py`
- Full example: Run `python cluster_function_dag.py`

