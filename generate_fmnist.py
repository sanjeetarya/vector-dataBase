import numpy as np
import os
import requests
import gzip
import json

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"File {save_path} already exists.")
        return
    print(f"Downloading {url}...")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")

def load_mnist_images(path):
    """
    Parse MNIST idx3-ubyte file
    """
    with gzip.open(path, 'rb') as f:
        # Magic number, number of images, rows, cols
        # all big-endian 32-bit integers
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        print(f"Reading {num_images} images ({rows}x{cols})...")
        
        # Read data
        buffer = f.read(num_images * rows * cols)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, rows * cols)
        return data

def pca_reduce(data, target_dim=128):
    """
    Reduce dimensions using PCA (SVD).
    """
    print(f"Reducing dimensions from {data.shape[1]} to {target_dim} using PCA...")
    
    # Center data
    mean = np.mean(data, axis=0)
    centered = data - mean
    
    # Compute Covariance Matrix (D x D)
    # (784 x 784) is small enough for eigh
    cov = np.cov(centered, rowvar=False)
    
    # Eigendecomposition
    # eigh returns eigenvalues in ascending order
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    
    # Top-k eigenvectors (reverse order)
    top_vecs = eig_vecs[:, ::-1][:, :target_dim]
    
    # Project data
    reduced = np.dot(centered, top_vecs)
    
    # Whitening (optional, but good for cosine sim)
    # reduced /= np.sqrt(eig_vals[::-1][:target_dim])
    
    return reduced

def generate_fmnist(
    output_dir="data"
):
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz"
    }
    
    data_dir = os.path.join(output_dir, "raw_fmnist")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download
    for key, filename in files.items():
        download_file(base_url + filename, os.path.join(data_dir, filename))
        
    # Load Data
    train_images = load_mnist_images(os.path.join(data_dir, files["train_img"]))
    test_images = load_mnist_images(os.path.join(data_dir, files["test_img"]))
    
    # Combine for PCA training (better projection)
    all_images = np.concatenate([train_images, test_images], axis=0)
    
    # PCA
    all_reduced = pca_reduce(all_images, target_dim=128)
    
    # Split back
    train_reduced = all_reduced[:60000]
    test_reduced = all_reduced[60000:]
    
    # Select subset for vector DB
    # 10,000 vectors from test set (as requested in README) or train? 
    # USER prompt said: "10,000 vectors from Fashion-MNIST images... 100 query vectors from test set"
    # Actually usually you index train and query with test. 
    # But FMNIST test is exactly 10k. 
    # Let's index the first 10k of TRAIN set, and use 100 from TEST set as queries.
    # Or index the TEST set? User said "10,000 vectors... 100 query". Both 10k. 
    # Let's use TRAIN for base vectors (subset 10k) and TEST for queries (subset 100).
    
    vectors = train_reduced[:10000]
    queries = test_reduced[:100]
    
    # Normalize
    v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / v_norms
    
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / q_norms
    
    # Save
    out_vec_dir = os.path.join(output_dir, "fmnist-vectors")
    out_qry_dir = os.path.join(output_dir, "fmnist-queries")
    os.makedirs(out_vec_dir, exist_ok=True)
    os.makedirs(out_qry_dir, exist_ok=True)
    
    np.save(os.path.join(out_vec_dir, "vectors.npy"), vectors)
    np.save(os.path.join(out_qry_dir, "queries.npy"), queries)
    
    print(f"Saved {len(vectors)} vectors to {out_vec_dir}")
    print(f"Saved {len(queries)} queries to {out_qry_dir}")
    
    # Generate Ground Truth
    print("Generating ground truth...")
    gt_results = {}
    
    # Brute force
    sim_matrix = np.dot(queries[:10], vectors.T)
    k = 100
    
    for i in range(10):
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
        
    gt_path = os.path.join(out_qry_dir, "ground_truth_top100.json")
    with open(gt_path, "w") as f:
        json.dump(gt_results, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    generate_fmnist()
