"""
Compare the three clustering methods: level, dp, and exhaustive.
"""

from cluster_function import optimal_tree_clustering
from sklearn.metrics.cluster import adjusted_rand_score
import time

# Test with a subset of the data (to make exhaustive feasible)
tree_data_small = {
    '0': 'society / politics / military',
    '1': 'technology / cryptography / privacy',
    '2': 'society / conflict / terrorism',
    '3': 'technology / computing / hardware',
    '4': 'recreation / games / robotics',
    '5': 'religion / christianity / theology',
    '6': 'recreation / sports / hockey',
    '7': 'technology / computing / software',
    '8': 'technology / computing / operating systems',
    '9': 'health / nutrition / food sensitivity',
    '10': 'technology / computing / file formats',
    '11': 'recreation / electronics / sales',
    '12': 'society / history / indigenous cultures',
    '13': 'science / space / museums',
    '14': 'recreation / vehicles / motorcycles',
    '15': 'technology / computing / hardware',
    '16': 'recreation / sports / baseball',
    '17': 'technology / electronics / interference',
    '18': 'recreation / sports / hockey',
    '19': 'technology / computing / hardware',
}

ground_truth_small = [
    'talk.politics.misc', 'sci.crypt', 'talk.politics.mideast', 'comp.windows.x',
    'sci.electronics', 'soc.religion.christian', 'rec.sport.hockey',
    'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'sci.med',
    'comp.graphics', 'misc.forsale', 'talk.politics.guns', 'sci.space',
    'rec.motorcycles', 'sci.electronics', 'rec.sport.baseball', 'sci.electronics',
    'rec.sport.hockey', 'comp.sys.mac.hardware'
]

print("="*80)
print("METHOD COMPARISON: level vs dp vs exhaustive")
print("="*80)
print(f"\nDataset: {len(tree_data_small)} datapoints\n")

methods = [
    ('level', 'Level-wise cuts (fast)'),
    ('dp', 'Dynamic Programming (smart search)'),
    ('exhaustive', 'Exhaustive search (slow but complete)')
]

results = []

for method_name, method_desc in methods:
    print("-" * 80)
    print(f"Testing: {method_desc}")
    print("-" * 80)
    
    try:
        start = time.time()
        clustering, score, cut_nodes = optimal_tree_clustering(
            tree_data_small,
            adjusted_rand_score,
            ground_truth_small,
            method=method_name,
            maximize=True
        )
        elapsed = time.time() - start
        
        num_clusters = len(set(clustering))
        
        print(f"✓ Score: {score:.4f}")
        print(f"  Clusters: {num_clusters}")
        print(f"  Time: {elapsed:.6f} seconds")
        
        # Show cut nodes
        cut_paths = [node.get_path() for node in cut_nodes[:5]]
        print(f"  Cut at: {', '.join(cut_paths)}", end="")
        if len(cut_nodes) > 5:
            print(f" ... ({len(cut_nodes)} total)", end="")
        print()
        
        results.append({
            'method': method_name,
            'score': score,
            'clusters': num_clusters,
            'time': elapsed,
            'success': True
        })
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append({
            'method': method_name,
            'score': None,
            'clusters': None,
            'time': None,
            'success': False
        })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n{'Method':<15} {'Score':<10} {'Clusters':<10} {'Time (s)':<12} {'Status':<10}")
print("-" * 80)
for r in results:
    if r['success']:
        print(f"{r['method']:<15} {r['score']:<10.4f} {r['clusters']:<10} {r['time']:<12.6f} {'✓':<10}")
    else:
        print(f"{r['method']:<15} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'✗ Failed':<10}")

# Compare scores
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

successful = [r for r in results if r['success']]
if len(successful) > 1:
    scores = [r['score'] for r in successful]
    if len(set(scores)) == 1:
        print("✓ All methods found the SAME optimal solution!")
        print(f"  (Score: {scores[0]:.4f})")
    else:
        best = max(successful, key=lambda x: x['score'])
        print(f"⚠ Methods found DIFFERENT solutions:")
        for r in successful:
            marker = "★ BEST" if r['method'] == best['method'] else ""
            print(f"  {r['method']}: {r['score']:.4f} {marker}")
    
    # Compare times
    print(f"\n⏱️  Speed comparison:")
    times = sorted([(r['method'], r['time']) for r in successful], key=lambda x: x[1])
    fastest_time = times[0][1]
    for method, t in times:
        speedup = t / fastest_time if fastest_time > 0 else 1
        print(f"  {method:<15} {t:.6f}s  ({speedup:.1f}x)")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
For your 200-datapoint dataset:
  1. 'level' - RECOMMENDED: Fast (0.00006s), finds good solution
  2. 'dp'    - Alternative: Explores more cuts with depth limit
  3. 'exhaustive' - AVOID: Would take hours/days for 200 datapoints
""")

