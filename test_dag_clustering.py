"""
Test script for DAG-based clustering without external dependencies.
"""

from cluster_function_dag import (
    build_dag, 
    find_optimal_clustering,
    optimal_dag_clustering,
    get_all_nodes,
    is_valid_cut
)


def simple_accuracy(true_labels, predicted_labels):
    """Simple accuracy metric without sklearn."""
    if len(true_labels) != len(predicted_labels):
        return 0.0
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    return correct / len(true_labels)


def test_basic_dag():
    """Test basic DAG construction and clustering."""
    print("=" * 70)
    print("Test 1: Basic DAG Construction")
    print("=" * 70)
    
    # Example DAG data - note how "Computer Science" appears in multiple paths
    dag_data = {
        '0': 'Science / Computer Science / AI',
        '1': 'Science / Computer Science / Databases',
        '2': 'Science / Biology / Genetics',
        '3': 'Engineering / Computer Science / AI',
        '4': 'Engineering / Computer Science / Robotics',
    }
    
    # Build DAG
    root, datapoint_to_node, name_to_nodes = build_dag(dag_data)
    
    print(f"\nDatapoints: {len(dag_data)}")
    print(f"Unique node names: {len(name_to_nodes)}")
    print(f"Node names: {list(name_to_nodes.keys())}")
    
    # Check for DAG property (nodes with multiple parents)
    for name, nodes in name_to_nodes.items():
        for node in nodes:
            if len(node.parents) > 1:
                print(f"  Node '{name}' has {len(node.parents)} parents (DAG property)")
    
    # Get all nodes
    all_nodes = get_all_nodes(root)
    print(f"\nTotal nodes in DAG: {len(all_nodes)}")
    
    print("\n✓ DAG construction successful!")
    return dag_data


def test_dp_memoized():
    """Test DP memoized clustering."""
    print("\n" + "=" * 70)
    print("Test 2: DP Memoized Clustering")
    print("=" * 70)
    
    dag_data = {
        '0': 'Science / Computer Science / AI',
        '1': 'Science / Computer Science / Databases',
        '2': 'Science / Biology / Genetics',
        '3': 'Engineering / Computer Science / AI',
        '4': 'Engineering / Computer Science / Robotics',
    }
    
    ground_truth = ['AI', 'Databases', 'Biology', 'AI', 'Robotics']
    
    # Test with purity cost
    best_clustering, best_score, best_cut = optimal_dag_clustering(
        dag_data,
        None,  # Not used for dp_memoized
        ground_truth,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print(f"\nMethod: DP Memoized (purity)")
    print(f"Best score: {best_score:.4f}")
    print(f"Best clustering: {best_clustering}")
    print(f"Number of clusters: {len(set(best_clustering))}")
    print(f"Cut nodes: {[node.get_path() for node in best_cut]}")
    
    print("\n✓ DP memoized clustering successful!")
    return best_clustering


def test_level_clustering():
    """Test level-based clustering."""
    print("\n" + "=" * 70)
    print("Test 3: Level-based Clustering")
    print("=" * 70)
    
    dag_data = {
        '0': 'Science / Computer Science / AI',
        '1': 'Science / Computer Science / Databases',
        '2': 'Science / Biology / Genetics',
        '3': 'Engineering / Computer Science / AI',
        '4': 'Engineering / Computer Science / Robotics',
    }
    
    ground_truth = ['AI', 'Databases', 'Biology', 'AI', 'Robotics']
    
    best_clustering, best_score, best_cut = optimal_dag_clustering(
        dag_data,
        simple_accuracy,
        ground_truth,
        method='level',
        maximize=True
    )
    
    print(f"\nMethod: Level-based")
    print(f"Best score: {best_score:.4f}")
    print(f"Best clustering: {best_clustering}")
    print(f"Number of clusters: {len(set(best_clustering))}")
    print(f"Cut nodes: {[node.get_path() for node in best_cut]}")
    
    print("\n✓ Level-based clustering successful!")


def test_converging_paths():
    """Test DAG with converging paths."""
    print("\n" + "=" * 70)
    print("Test 4: Converging Paths (True DAG Property)")
    print("=" * 70)
    
    # Create a DAG where paths converge at "Privacy"
    dag_data = {
        '0': 'Technology / Cryptography / Privacy',
        '1': 'Technology / Computing / Privacy',
        '2': 'Society / Law / Privacy',
        '3': 'Technology / Cryptography / Encryption',
    }
    
    ground_truth = ['tech', 'tech', 'society', 'tech']
    
    # Build DAG
    root, datapoint_to_node, name_to_nodes = build_dag(dag_data)
    
    print(f"\nDatapoints: {len(dag_data)}")
    
    # Check if Privacy node has multiple parents
    if 'Privacy' in name_to_nodes:
        privacy_nodes = name_to_nodes['Privacy']
        for node in privacy_nodes:
            print(f"  'Privacy' node has {len(node.parents)} parent(s)")
            parent_names = [p.name for p in node.parents]
            print(f"  Parent nodes: {parent_names}")
    
    # Find optimal clustering
    best_clustering, best_score, best_cut = optimal_dag_clustering(
        dag_data,
        None,
        ground_truth,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print(f"\nBest score: {best_score:.4f}")
    print(f"Best clustering: {best_clustering}")
    print(f"Cut nodes: {[node.get_path() for node in best_cut]}")
    
    print("\n✓ Converging paths test successful!")


def test_simple_tree_as_dag():
    """Test that trees work correctly as special case of DAGs."""
    print("\n" + "=" * 70)
    print("Test 5: Tree as Special Case of DAG")
    print("=" * 70)
    
    # This is actually a tree (no converging paths)
    dag_data = {
        '0': 'Animals / Mammals / Dogs',
        '1': 'Animals / Mammals / Cats',
        '2': 'Animals / Birds / Eagles',
        '3': 'Plants / Trees / Oak',
    }
    
    ground_truth = ['mammal', 'mammal', 'bird', 'plant']
    
    best_clustering, best_score, best_cut = optimal_dag_clustering(
        dag_data,
        None,
        ground_truth,
        method='dp_memoized',
        maximize=True,
        decomposable_cost='purity'
    )
    
    print(f"\nBest score: {best_score:.4f}")
    print(f"Best clustering: {best_clustering}")
    print(f"Number of clusters: {len(set(best_clustering))}")
    print(f"Cut nodes: {[node.get_path() for node in best_cut]}")
    
    print("\n✓ Tree as DAG test successful!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DAG-BASED CLUSTERING TEST SUITE")
    print("=" * 70)
    
    try:
        test_basic_dag()
        test_dp_memoized()
        test_level_clustering()
        test_converging_paths()
        test_simple_tree_as_dag()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

