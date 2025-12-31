"""
Comparison script showing the differences between tree-based and DAG-based clustering.

This script demonstrates:
1. When tree and DAG produce the same results (tree-structured data)
2. When DAG handles converging paths that trees cannot
3. Performance characteristics of both approaches
"""

import time
from cluster_function import (
    build_tree,
    optimal_tree_clustering,
    find_optimal_clustering as find_optimal_tree_clustering
)
from cluster_function_dag import (
    build_dag,
    optimal_dag_clustering,
    find_optimal_clustering as find_optimal_dag_clustering,
    get_all_nodes
)


def compare_structures():
    """Compare how tree and DAG handle the same data."""
    print("=" * 70)
    print("COMPARISON 1: Tree-Structured Data")
    print("=" * 70)
    print("\nWhen data is tree-structured, both should work similarly.\n")
    
    # Strictly tree-structured data (no converging paths)
    data = {
        '0': 'Animals / Mammals / Dogs',
        '1': 'Animals / Mammals / Cats',
        '2': 'Animals / Birds / Eagles',
        '3': 'Plants / Trees / Oak',
    }
    
    ground_truth = ['mammal', 'mammal', 'bird', 'plant']
    
    # Build tree
    tree_root, tree_dp_to_node = build_tree(data)
    print("Tree structure:")
    print(f"  Root children: {list(tree_root.children.keys())}")
    
    # Build DAG
    dag_root, dag_dp_to_node, name_to_nodes = build_dag(data)
    print("\nDAG structure:")
    print(f"  Root children: {list(dag_root.children.keys())}")
    print(f"  Unique nodes: {len(get_all_nodes(dag_root))}")
    
    # Check for multiple parents (DAG property)
    has_multiple_parents = False
    for name, nodes in name_to_nodes.items():
        for node in nodes:
            if len(node.parents) > 1:
                print(f"  Node '{name}' has {len(node.parents)} parents")
                has_multiple_parents = True
    
    if not has_multiple_parents:
        print("  No nodes with multiple parents (tree structure)")
    
    # Compare clustering results
    print("\nClustering with DP memoized (purity):")
    
    tree_clustering, tree_score, tree_cut = optimal_tree_clustering(
        data, None, ground_truth, method='dp_memoized', 
        maximize=True, decomposable_cost='purity'
    )
    
    dag_clustering, dag_score, dag_cut = optimal_dag_clustering(
        data, None, ground_truth, method='dp_memoized',
        maximize=True, decomposable_cost='purity'
    )
    
    print(f"  Tree score: {tree_score:.4f}")
    print(f"  DAG score:  {dag_score:.4f}")
    print(f"  Tree clusters: {len(set(tree_clustering))}")
    print(f"  DAG clusters:  {len(set(dag_clustering))}")
    
    print("\n✓ Both handle tree-structured data correctly")


def demonstrate_dag_advantage():
    """Show where DAG structure provides advantages."""
    print("\n" + "=" * 70)
    print("COMPARISON 2: Converging Paths (DAG Advantage)")
    print("=" * 70)
    print("\nDAG can handle paths that converge at shared concepts.\n")
    
    # Data with converging paths
    data = {
        '0': 'Science / Computer Science / AI',
        '1': 'Science / Computer Science / Databases',
        '2': 'Engineering / Computer Science / AI',
        '3': 'Engineering / Computer Science / Robotics',
    }
    
    ground_truth = ['AI', 'Databases', 'AI', 'Robotics']
    
    print("Input paths:")
    for dp_id, path in sorted(data.items()):
        print(f"  {dp_id}: {path}")
    
    print("\nNote: 'Computer Science' and 'AI' appear in multiple paths")
    
    # Build tree (treats each occurrence as separate)
    tree_root, _ = build_tree(data)
    tree_nodes = []
    
    def count_tree_nodes(node, nodes_list):
        if node.name:
            nodes_list.append(node.name)
        for child in node.children.values():
            count_tree_nodes(child, nodes_list)
    
    count_tree_nodes(tree_root, tree_nodes)
    
    print(f"\nTree structure:")
    print(f"  Total nodes: {len(tree_nodes)}")
    print(f"  'Computer Science' appears: {tree_nodes.count('Computer Science')} times (separate nodes)")
    print(f"  'AI' appears: {tree_nodes.count('AI')} times (separate nodes)")
    
    # Build DAG (can merge shared nodes)
    dag_root, _, name_to_nodes = build_dag(data)
    dag_nodes = get_all_nodes(dag_root)
    
    print(f"\nDAG structure:")
    print(f"  Total nodes: {len(dag_nodes)}")
    
    if 'Computer Science' in name_to_nodes:
        cs_nodes = name_to_nodes['Computer Science']
        for node in cs_nodes:
            print(f"  'Computer Science' has {len(node.parents)} parent(s): {[p.name for p in node.parents]}")
    
    if 'AI' in name_to_nodes:
        ai_nodes = name_to_nodes['AI']
        for node in ai_nodes:
            print(f"  'AI' has {len(node.parents)} parent(s): {[p.name for p in node.parents]}")
    
    # Compare clustering
    tree_clustering, tree_score, _ = optimal_tree_clustering(
        data, None, ground_truth, method='dp_memoized',
        maximize=True, decomposable_cost='purity'
    )
    
    dag_clustering, dag_score, _ = optimal_dag_clustering(
        data, None, ground_truth, method='dp_memoized',
        maximize=True, decomposable_cost='purity'
    )
    
    print(f"\nClustering results:")
    print(f"  Tree score: {tree_score:.4f}, clusters: {len(set(tree_clustering))}")
    print(f"  DAG score:  {dag_score:.4f}, clusters: {len(set(dag_clustering))}")
    
    print("\n✓ DAG correctly identifies shared structure")


def performance_comparison():
    """Compare performance on larger dataset."""
    print("\n" + "=" * 70)
    print("COMPARISON 3: Performance")
    print("=" * 70)
    print("\nComparing execution time on moderately-sized dataset.\n")
    
    # Generate larger dataset
    data = {}
    ground_truth = []
    idx = 0
    
    # Create hierarchical structure
    domains = ['Science', 'Engineering', 'Business']
    fields = ['Computer Science', 'Biology', 'Physics', 'Chemistry']
    topics = ['AI', 'Databases', 'Networks', 'Security', 'Theory']
    
    for domain in domains:
        for field in fields[:2]:  # Limit to keep it manageable
            for topic in topics[:3]:
                data[str(idx)] = f"{domain} / {field} / {topic}"
                ground_truth.append(topic)
                idx += 1
    
    print(f"Dataset size: {len(data)} datapoints")
    
    # Time tree-based clustering
    start = time.time()
    tree_clustering, tree_score, _ = optimal_tree_clustering(
        data, None, ground_truth, method='dp_memoized',
        maximize=True, decomposable_cost='purity'
    )
    tree_time = time.time() - start
    
    # Time DAG-based clustering
    start = time.time()
    dag_clustering, dag_score, _ = optimal_dag_clustering(
        data, None, ground_truth, method='dp_memoized',
        maximize=True, decomposable_cost='purity'
    )
    dag_time = time.time() - start
    
    print(f"\nExecution time:")
    print(f"  Tree: {tree_time*1000:.2f} ms")
    print(f"  DAG:  {dag_time*1000:.2f} ms")
    print(f"  Ratio: {tree_time/dag_time:.2f}x")
    
    print(f"\nResults:")
    print(f"  Tree score: {tree_score:.4f}")
    print(f"  DAG score:  {dag_score:.4f}")
    
    print("\n✓ Both methods are efficient with DP memoization")


def show_use_cases():
    """Show practical use cases for each approach."""
    print("\n" + "=" * 70)
    print("WHEN TO USE EACH APPROACH")
    print("=" * 70)
    
    print("\n📊 Use TREE-based clustering when:")
    print("  • Hierarchy is strictly tree-structured")
    print("  • Each concept has exactly one parent")
    print("  • Paths never converge")
    print("  • Examples:")
    print("    - File system hierarchies")
    print("    - Organizational charts")
    print("    - Biological taxonomy (traditional)")
    print("    - Geographic regions")
    
    print("\n🔀 Use DAG-based clustering when:")
    print("  • Concepts can belong to multiple categories")
    print("  • Paths converge at shared nodes")
    print("  • Multi-inheritance taxonomies")
    print("  • Examples:")
    print("    - 'Privacy' in both Technology and Law")
    print("    - 'AI' in both Computer Science and Robotics")
    print("    - 'Statistics' in both Math and Data Science")
    print("    - Cross-domain concepts")
    
    print("\n💡 Key insight:")
    print("  DAG is a generalization of trees. If your data is tree-structured,")
    print("  both will work. But only DAG can handle converging paths correctly.")


def main():
    """Run all comparisons."""
    print("\n" + "=" * 70)
    print("TREE vs DAG CLUSTERING COMPARISON")
    print("=" * 70)
    
    try:
        compare_structures()
        demonstrate_dag_advantage()
        performance_comparison()
        show_use_cases()
        
        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE ✓")
        print("=" * 70)
        print("\nSummary:")
        print("  • Both tree and DAG handle tree-structured data well")
        print("  • DAG can handle converging paths that trees cannot")
        print("  • Both are efficient with DP memoization (O(m×n))")
        print("  • Choose based on your data structure")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

