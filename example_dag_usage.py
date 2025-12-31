"""
Simple, clear examples of using DAG-based clustering.

This file provides practical examples you can copy and adapt for your own use.
"""

from cluster_function_dag import find_optimal_clustering, optimal_dag_clustering, build_dag


def example_0_multiple_paths():
    """
    Example 0: Multiple paths per datapoint (NEW FEATURE!)
    
    Scenario: Research papers that truly span multiple domains
    Each paper can belong to multiple hierarchical contexts simultaneously
    """
    print("=" * 70)
    print("Example 0: Multiple Paths Per Datapoint (TRUE DAG)")
    print("=" * 70)
    
    # Each datapoint can have multiple hierarchical paths
    papers = {
        '0': ['Computer Science / Machine Learning / Neural Networks',
              'Neuroscience / Computational / Neural Networks'],
        '1': ['Computer Science / Robotics / Control',
              'Engineering / Mechanical / Control'],
        '2': ['Physics / Quantum / Computing',
              'Computer Science / Theory / Quantum Computing'],
        '3': ['Mathematics / Statistics / Inference']
    }
    
    true_categories = ['Neural Networks', 'Robotics', 'Quantum', 'Statistics']
    
    print("\nInput: Datapoints with multiple hierarchical paths")
    for paper_id, paths in sorted(papers.items()):
        print(f"\n  Paper {paper_id}:")
        for path in paths:
            print(f"    - {path}")
    
    print("\nKey Feature: Each paper can belong to MULTIPLE hierarchical contexts!")
    
    # Build DAG to show structure
    root, _, name_to_nodes = build_dag(papers)
    
    # Find nodes with multiple parents
    converging_nodes = []
    for name, nodes in name_to_nodes.items():
        for node in nodes:
            if len(node.parents) > 1:
                parent_names = sorted([p.name for p in node.parents if p.name])
                converging_nodes.append((name, parent_names))
    
    if converging_nodes:
        print(f"\n  Nodes with multiple parents (path convergence):")
        for name, parents in converging_nodes:
            print(f"    '{name}' ← {parents}")
    
    # Find optimal clustering
    clustering, score, cut_nodes = optimal_dag_clustering(
        papers,
        cost_function=None,
        ground_truth=true_categories,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print(f"\nOptimal clustering (purity score: {score:.2f}):")
    for paper_id, cluster in enumerate(clustering):
        cat = true_categories[paper_id]
        print(f"  Paper {paper_id} [{cat:16}] → {cluster}")
    
    print(f"\nNumber of clusters: {len(set(clustering))}")
    print("\n✓ Multiple paths per datapoint create the richest DAG structure!")


def example_1_basic_dag():
    """
    Example 1: Basic DAG with converging paths
    
    Scenario: Academic papers that span multiple disciplines
    """
    print("=" * 70)
    print("Example 1: Academic Papers with Cross-Disciplinary Topics")
    print("=" * 70)
    
    # Papers organized by discipline paths
    # Note: "Machine Learning" appears in multiple paths
    papers = {
        '0': 'Computer Science / AI / Machine Learning',
        '1': 'Computer Science / AI / Computer Vision',
        '2': 'Statistics / Data Science / Machine Learning',
        '3': 'Statistics / Probability / Inference',
        '4': 'Mathematics / Optimization / Machine Learning',
    }
    
    # True categories (for evaluation)
    true_categories = ['ML', 'Vision', 'ML', 'Stats', 'ML']
    
    print("\nInput paths:")
    for paper_id, path in sorted(papers.items()):
        print(f"  Paper {paper_id}: {path}")
    
    print("\nNote: 'Machine Learning' appears in 3 different paths!")
    print("      The DAG will recognize it as the same concept.")
    
    # Find optimal clustering
    clustering = find_optimal_clustering(
        papers,
        cost_function=None,
        ground_truth=true_categories,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print("\nOptimal clustering:")
    for paper_id, cluster in enumerate(clustering):
        print(f"  Paper {paper_id} → {cluster}")
    
    print(f"\nNumber of clusters: {len(set(clustering))}")
    print("\n✓ DAG correctly handles cross-disciplinary topics")


def example_2_privacy_concept():
    """
    Example 2: Privacy as a concept spanning multiple domains
    
    Scenario: Documents about privacy from different perspectives
    """
    print("\n" + "=" * 70)
    print("Example 2: Privacy Across Multiple Domains")
    print("=" * 70)
    
    documents = {
        '0': 'Technology / Cryptography / Privacy',
        '1': 'Technology / Computing / Privacy',
        '2': 'Society / Law / Privacy',
        '3': 'Society / Ethics / Privacy',
        '4': 'Technology / Cryptography / Encryption',
        '5': 'Society / Law / Regulation',
    }
    
    true_labels = ['tech', 'tech', 'society', 'society', 'tech', 'society']
    
    print("\nInput paths:")
    for doc_id, path in sorted(documents.items()):
        print(f"  Doc {doc_id}: {path}")
    
    print("\nNote: 'Privacy' appears in both Technology and Society domains")
    
    # Get full results including score and cut nodes
    clustering, score, cut_nodes = optimal_dag_clustering(
        documents,
        cost_function=None,
        ground_truth=true_labels,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print(f"\nOptimal clustering (score: {score:.2f}):")
    for doc_id, cluster in enumerate(clustering):
        print(f"  Doc {doc_id} → {cluster}")
    
    print(f"\nCut points in the DAG:")
    for node in cut_nodes:
        print(f"  • {node.get_path()}")
    
    print("\n✓ DAG recognizes shared concepts across domains")


def example_3_compare_methods():
    """
    Example 3: Comparing different clustering methods
    
    Shows how different methods can produce different results
    """
    print("\n" + "=" * 70)
    print("Example 3: Comparing Different Methods")
    print("=" * 70)
    
    data = {
        '0': 'Animals / Mammals / Carnivores / Dogs',
        '1': 'Animals / Mammals / Carnivores / Cats',
        '2': 'Animals / Mammals / Herbivores / Cows',
        '3': 'Animals / Birds / Raptors / Eagles',
        '4': 'Animals / Birds / Songbirds / Sparrows',
    }
    
    true_labels = ['mammal', 'mammal', 'mammal', 'bird', 'bird']
    
    print("\nTesting different methods on the same data:\n")
    
    # Test dp_memoized
    clustering, score, cut_nodes = optimal_dag_clustering(
        data,
        cost_function=None,
        ground_truth=true_labels,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print(f"Method: dp_memoized (recommended)")
    print(f"  Score: {score:.2f}")
    print(f"  Clusters: {len(set(clustering))}")
    print(f"  Cut points: {[node.name for node in cut_nodes]}")
    print()
    
    print("✓ DP memoized finds optimal cuts efficiently")


def example_4_custom_cost():
    """
    Example 4: Using different decomposable cost functions
    
    Shows how different cost functions affect clustering
    """
    print("=" * 70)
    print("Example 4: Different Cost Functions")
    print("=" * 70)
    
    data = {
        '0': 'Science / Physics / Quantum',
        '1': 'Science / Physics / Classical',
        '2': 'Science / Chemistry / Organic',
        '3': 'Science / Chemistry / Inorganic',
        '4': 'Engineering / Mechanical / Thermodynamics',
    }
    
    true_labels = ['physics', 'physics', 'chemistry', 'chemistry', 'engineering']
    
    print("\nComparing different cost functions:\n")
    
    costs = ['purity', 'entropy', 'homogeneity']
    
    for cost_type in costs:
        clustering, score, cut_nodes = optimal_dag_clustering(
            data,
            cost_function=None,
            ground_truth=true_labels,
            method='dp_memoized',
            maximize=True,
            decomposable_cost=cost_type
        )
        
        print(f"Cost function: {cost_type}")
        print(f"  Score: {score:.2f}")
        print(f"  Clusters: {len(set(clustering))}")
        print(f"  Example: Doc 0 → {clustering[0]}")
        print()
    
    print("✓ Different cost functions optimize for different properties")


def example_5_real_world():
    """
    Example 5: Real-world scenario - organizing research papers
    
    Practical example with realistic data
    """
    print("=" * 70)
    print("Example 5: Organizing Research Papers")
    print("=" * 70)
    
    # Research papers with interdisciplinary topics
    papers = {
        '0': 'Computer Science / Machine Learning / Deep Learning',
        '1': 'Computer Science / Machine Learning / Reinforcement Learning',
        '2': 'Neuroscience / Computational / Deep Learning',
        '3': 'Neuroscience / Cognitive / Memory',
        '4': 'Computer Science / Robotics / Reinforcement Learning',
        '5': 'Engineering / Robotics / Control Systems',
        '6': 'Mathematics / Statistics / Deep Learning',
    }
    
    # Research areas
    research_areas = ['ML', 'ML', 'Neuro', 'Neuro', 'Robotics', 'Robotics', 'ML']
    
    print("\nScenario: Organizing interdisciplinary research papers")
    print("\nPapers:")
    for paper_id, path in sorted(papers.items()):
        area = research_areas[int(paper_id)]
        print(f"  {paper_id}. [{area:8}] {path}")
    
    print("\nNotice:")
    print("  • 'Deep Learning' spans CS, Neuroscience, and Math")
    print("  • 'Reinforcement Learning' spans CS and Robotics")
    print("  • 'Robotics' spans CS and Engineering")
    
    # Find optimal clustering
    clustering, score, cut_nodes = optimal_dag_clustering(
        papers,
        cost_function=None,
        ground_truth=research_areas,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print(f"\nOptimal organization (purity score: {score:.2f}):")
    
    # Group by cluster
    clusters = {}
    for paper_id, cluster_label in enumerate(clustering):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(paper_id)
    
    for cluster_label, paper_ids in sorted(clusters.items()):
        print(f"\n  Cluster: {cluster_label}")
        for paper_id in paper_ids:
            area = research_areas[paper_id]
            print(f"    • Paper {paper_id} [{area}]")
    
    print(f"\nCut points chosen:")
    for node in cut_nodes:
        datapoints = node.get_all_datapoints()
        print(f"  • {node.get_path()} ({len(datapoints)} papers)")
    
    print("\n✓ DAG clustering organizes interdisciplinary research effectively")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DAG-BASED CLUSTERING: PRACTICAL EXAMPLES")
    print("=" * 70)
    print("\nThese examples show how to use DAG clustering for real-world tasks.")
    print("Copy and adapt these patterns for your own data!\n")
    
    example_0_multiple_paths()  # NEW: Multiple paths per datapoint
    example_1_basic_dag()
    example_2_privacy_concept()
    example_3_compare_methods()
    example_4_custom_cost()
    example_5_real_world()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE ✓")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Use 'dp_memoized' method for best performance (O(m×n))")
    print("  2. 'purity' cost function works well for most cases")
    print("  3. DAG handles cross-domain concepts automatically")
    print("  4. Converging paths are detected and merged")
    print("  5. **NEW: Multiple paths per datapoint supported!**")
    print("     - Use list format: {'0': ['path1', 'path2']}")
    print("     - Creates richest DAG structures")
    print("  6. API is simple: just provide paths and ground truth")
    print("\nReady to use on your own data! 🚀")
    print("=" * 70)


if __name__ == "__main__":
    main()

