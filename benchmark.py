import numpy as np
import os
import time
import json
from vector_db import VectorDB

import numpy as np
import os
import time
import json
import sys
from vector_db import VectorDB

def load_data(data_dir="data", dataset="syndata"):
    if dataset == "syndata":
        subdir_vec = "syndata-vectors"
        subdir_qry = "syndata-queries"
    elif dataset == "fmnist":
        subdir_vec = "fmnist-vectors"
        subdir_qry = "fmnist-queries"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    vectors_path = os.path.join(data_dir, subdir_vec, "vectors.npy")
    queries_path = os.path.join(data_dir, subdir_qry, "queries.npy")
    gt_path = os.path.join(data_dir, subdir_qry, "ground_truth_top100.json")
    
    if not os.path.exists(vectors_path):
        raise FileNotFoundError(f"Data not found at {vectors_path}. Run generate_{dataset}.py first.")
    
    vectors = np.load(vectors_path)
    queries = np.load(queries_path)
    
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)
        
    return vectors, queries, ground_truth

def compute_recall(results, gt_indices, k=10):
    """
    Compute Recall@K.
    recall = (intersection of results and gt) / k
    """
    # Extract indices from results
    result_indices = set([r[0] for r in results[:k]])
    # Extract top-k indices from ground truth (gt_indices is list of dicts)
    # The GT JSON structure is usually list of objects with "id"
    # But wait, create_syndata was doing list of objects? 
    # Let's check generate_syndata.py: yes, gt_entry.append({"id": ..., "score": ...})
    gt_set = set([item["id"] for item in gt_indices[:k]])
    
    intersection = result_indices.intersection(gt_set)
    return len(intersection) / float(len(gt_set))

def run_benchmark(dataset="syndata"):
    print(f"Loading data for {dataset}...")
    vectors, queries, ground_truth = load_data(dataset=dataset)
    num_queries_to_test = 10
    k = 10
    
    print(f"Loaded {len(vectors)} vectors, {len(queries)} queries.")
    
    # Initialize DB
    db = VectorDB()
    db.add_vectors(vectors)
    
    results_summary = []

    def benchmark_algo(name, setup_fn, search_fn, search_kwargs={}):
        print(f"\n--- Benchmarking {name} ---")
        
        # Build Index
        if setup_fn:
            setup_fn()
            
        latencies = []
        recalls = []
        
        for i in range(num_queries_to_test):
            q_start = time.time()
            results = search_fn(queries[i], k=k, **search_kwargs)
            lat = (time.time() - q_start) * 1000 # ms
            latencies.append(lat)
            
            # ground_truth keys are strings "0", "1"...
            gt = ground_truth[str(i)]
            recall = compute_recall(results, gt, k=k)
            recalls.append(recall)
            
        avg_lat = np.mean(latencies)
        avg_rec = np.mean(recalls)
        print(f"{name}: Avg Latency = {avg_lat:.2f} ms, Recall@{k} = {avg_rec:.2f}")
        results_summary.append({
            "Method": name,
            "Latency (ms)": f"{avg_lat:.2f}",
            "Recall": f"{avg_rec:.2f}"
        })

    # 1. Brute Force
    benchmark_algo("Brute Force", 
                   None, 
                   db.brute_force_search)

    # 2. IVF Index
    def setup_ivf():
        db.build_index("ivf", n_clusters=100)
    
    benchmark_algo("IVF (n_probes=1)", setup_ivf, db.search, {"n_probes": 1})
    benchmark_algo("IVF (n_probes=10)", None, db.search, {"n_probes": 10})

    # 3. LSH Index (Normal - Single Table)
    def setup_lsh_normal():
        # More bits for single table to avoid too many collisions
        db.build_index("lsh", nBITS=12, nTables=1)
        
    benchmark_algo("LSH (1 Table, 12 bits)", setup_lsh_normal, db.search)

    # 4. LSH Index (Multi-table)
    def setup_lsh_multi():
        # Fewer bits per table, but multiple tables
        db.build_index("lsh", nBITS=8, nTables=10)
        
    benchmark_algo("LSH (10 Tables, 8 bits)", setup_lsh_multi, db.search)

    # 5. HNSW
    def setup_hnsw():
        db.build_index("hnsw", M=16, ef_construction=100)
        
    benchmark_algo("HNSW (M=16, ef=100)", setup_hnsw, db.search)

    print(f"\n\n--- Final Summary ({dataset}) ---")
    print(f"{'Method':<25} | {'Latency (ms)':<15} | {'Recall':<10}")
    print("-" * 55)
    for res in results_summary:
        print(f"{res['Method']:<25} | {res['Latency (ms)']:<15} | {res['Recall']:<10}")

if __name__ == "__main__":
    import sys
    dataset = "syndata"
    # print("Enter dataset name (syndata or fmnist): ", end="")
    # dataset = input()
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    run_benchmark(dataset)
