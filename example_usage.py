"""
Simple example demonstrating how to use the tree-consistent clustering function.
"""

from cluster_function import find_optimal_clustering
from sklearn.metrics.cluster import adjusted_rand_score


# Example 1: Small dataset
def small_example():
    """Simple example with a few datapoints."""
    
    print("="*70)
    print("EXAMPLE 1: Small Dataset")
    print("="*70)
    
    # Tree structure: datapoint IDs mapped to hierarchical paths
    tree_data = {
        '0': 'society / politics / military',
        '1': 'technology / cryptography / privacy',
        '2': 'society / conflict / terrorism',
        '3': 'technology / computing / hardware',
        '4': 'recreation / games / robotics',
        '5': 'technology / computing / software',
        '6': 'society / politics / healthcare',
        '7': 'recreation / sports / hockey',
        '8': 'technology / cryptography / encryption',
        '9': 'recreation / sports / baseball',
    }
    
    # Ground truth cluster labels (in sorted ID order)
    ground_truth = [
        'politics',
        'tech',
        'politics',
        'tech',
        'recreation',
        'tech',
        'politics',
        'recreation',
        'tech',
        'recreation',
    ]
    
    # Cost function: adjusted Rand score (higher is better)
    def cost_function(true_labels, predicted_labels):
        return adjusted_rand_score(true_labels, predicted_labels)
    
    # Find optimal clustering
    clustering = find_optimal_clustering(
        tree_data=tree_data,
        cost_function=cost_function,
        ground_truth=ground_truth,
        method='level',  # Use 'level' for faster execution
        maximize=True    # We want to maximize adjusted_rand_score
    )
    
    # Display results
    print(f"\nNumber of datapoints: {len(tree_data)}")
    print(f"Number of unique clusters found: {len(set(clustering))}")
    print(f"\nPredictions vs Ground Truth:")
    print(f"{'ID':<4} {'Predicted Cluster':<30} {'True Label':<20}")
    print("-" * 70)
    for i, (pred, true) in enumerate(zip(clustering, ground_truth)):
        print(f"{i:<4} {pred:<30} {true:<20}")
    
    # Calculate final score
    score = adjusted_rand_score(ground_truth, clustering)
    print(f"\nAdjusted Rand Score: {score:.4f}")


# Example 2: Using your own cost function
def custom_cost_function_example():
    """Example showing how to use a custom cost function."""
    
    print("\n\n")
    print("="*70)
    print("EXAMPLE 2: Custom Cost Function")
    print("="*70)
    
    tree_data = {
        '0': 'category_a / subcategory_1 / item_x',
        '1': 'category_a / subcategory_1 / item_y',
        '2': 'category_a / subcategory_2 / item_z',
        '3': 'category_b / subcategory_3 / item_w',
        '4': 'category_b / subcategory_4 / item_v',
    }
    
    ground_truth = ['A', 'A', 'A', 'B', 'B']
    
    # Custom cost function: penalize clusters that mix categories
    def custom_cost(true_labels, predicted_labels):
        """Simple purity-based cost function."""
        from collections import defaultdict
        
        # Group by predicted cluster
        clusters = defaultdict(list)
        for true_label, pred_label in zip(true_labels, predicted_labels):
            clusters[pred_label].append(true_label)
        
        # Calculate purity: fraction of dominant class in each cluster
        total_correct = 0
        for cluster_members in clusters.values():
            most_common = max(set(cluster_members), key=cluster_members.count)
            total_correct += cluster_members.count(most_common)
        
        purity = total_correct / len(true_labels)
        return purity
    
    # Find optimal clustering with custom cost function
    clustering = find_optimal_clustering(
        tree_data=tree_data,
        cost_function=custom_cost,
        ground_truth=ground_truth,
        method='level',
        maximize=True  # We want to maximize purity
    )
    
    print(f"\nResults:")
    for i, (tree_path, pred, true) in enumerate(zip(tree_data.values(), clustering, ground_truth)):
        print(f"  {i}: {tree_path:<45} -> Cluster: {pred:<30} (True: {true})")
    
    score = custom_cost(ground_truth, clustering)
    print(f"\nPurity Score: {score:.4f}")


if __name__ == "__main__":
    small_example()
    custom_cost_function_example()
    
    print("\n\n")
    print("="*70)
    print("USAGE SUMMARY")
    print("="*70)
    print("""
The main function you need is:

    find_optimal_clustering(tree_data, cost_function, ground_truth, method='level', maximize=True)

Parameters:
  - tree_data: dict mapping IDs to hierarchical paths (e.g., {'0': 'A / B / C'})
  - cost_function: function(ground_truth, predicted) -> score
  - ground_truth: list of true cluster labels
  - method: 'level' (fast) or 'exhaustive' (slow but thorough)
  - maximize: True if higher cost is better, False if lower is better

Returns:
  - List of predicted cluster labels (one per datapoint)

The algorithm finds the optimal way to "cut" the tree to create clusters
that maximize/minimize your cost function while respecting the tree structure.
    """)

