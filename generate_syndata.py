import numpy as np
import os
import json

def generate_data(
    num_vectors=10000,
    dim=128,
    num_queries=100,
    output_dir="data"
):
    print(f"Generating {num_vectors} vectors of dimension {dim}...")
    
    # ensure directories exist
    vectors_dir = os.path.join(output_dir, "syndata-vectors")
    queries_dir = os.path.join(output_dir, "syndata-queries")
    os.makedirs(vectors_dir, exist_ok=True)
    os.makedirs(queries_dir, exist_ok=True)

    # Generate random vectors
    # We use float32 for standard vector ops
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    
    # Normalize vectors to unit length for cosine similarity
    # cosine_sim(a, b) = dot(a, b) / (|a| * |b|)
    # logic: if vectors are normalized, |a|=|b|=1, so cosine_sim = dot(a, b)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    # Save vectors
    np.save(os.path.join(vectors_dir, "vectors.npy"), vectors)
    print(f"Saved vectors to {vectors_dir}/vectors.npy")

    # Generate query vectors
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / q_norms
    
    # Save queries
    np.save(os.path.join(queries_dir, "queries.npy"), queries)
    print(f"Saved queries to {queries_dir}/queries.npy")
    
    # Generate Ground Truth for first 10 queries
    # Brute force top-100
    print("Generating ground truth for first 10 queries...")
    gt_indices = []
    k = 100
    
    # Compute dot product: queries (100, 128) @ vectors.T (128, 10000) -> (100, 10000)
    # We only need first 10
    subset_queries = queries[:10]
    sim_matrix = np.dot(subset_queries, vectors.T)
    
    gt_results = {}
    
    for i in range(10):
        # argsort gives indices of sorted elements, we want descending order
        # so we take negative or use [::-1]
        scores = sim_matrix[i]
        top_k_idx = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_idx]
        
        gt_entry = []
        for idx, score in zip(top_k_idx, top_k_scores):
            gt_entry.append({
                "id": int(idx),
                "score": float(score)
            })
        
        gt_results[i] = gt_entry

    gt_path = os.path.join(queries_dir, "ground_truth_top100.json")
    with open(gt_path, "w") as f:
        json.dump(gt_results, f, indent=2)
    print(f"Saved ground truth to {gt_path}")
    
if __name__ == "__main__":
    generate_data()
