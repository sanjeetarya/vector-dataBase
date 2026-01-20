# Vector Similarity Search Engine

## Problem

Implement an approximate nearest neighbor (ANN) system for high-dimensional vector similarity search.

## Dataset

Two datasets are available for evaluation:

### Synthetic Dataset (Default)
Located in `data/syndata-vectors/` and `data/syndata-queries/`:
- **10,000 vectors** of dimension 128 (normalized for cosine similarity)
- **100 query vectors** for testing
- **Ground truth** for first 10 queries (top-100 exact results) for evaluation

To generate, run:
```bash
python3 generate_syndata.py
```

### Fashion-MNIST Dataset (Optional)
Located in `data/fmnist-vectors/` and `data/fmnist-queries/`:
- **10,000 vectors** from Fashion-MNIST images (128-dimensional embeddings)
- **100 query vectors** from test set
- **Ground truth** for first 10 queries
- Includes class labels for additional evaluation metrics

To generate, run:
```bash
python3 generate_fmnist.py
```

Vectors are provided in both JSON and NumPy formats.

## Requirements

Your system should implement:

1. **Cosine Similarity**
   - Implement cosine similarity calculation
   - Support high-dimensional vectors

2. **Index Construction**
   - Build an efficient index structure for fast retrieval

3. **Query Top-K**
   - Implement query interface to find top-k nearest neighbors
   - Return results with similarity scores

4. **Comparison & Benchmarking**
   - Implement brute-force exact search for comparison
   - Compare ANN vs brute-force on accuracy and latency
   - Document performance tradeoffs

## Constraints

- Implement core algorithms yourself (don't use full ANN libraries like FAISS, Annoy, etc.)
- You may use basic data structures and math libraries

## Usage

### 1. Installation
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Data
You can use either synthetic data or the Fashion-MNIST dataset.

**Synthetic Data (Default):**
Generates 10,000 random 128-d vectors.
```bash
python3 generate_syndata.py
```

**Fashion-MNIST Data:**
Downloads real images, reduces them to 128 dimensions using PCA.
```bash
python3 generate_fmnist.py
```

### 3. Run Benchmark
Run the benchmark script and specify the dataset (`syndata` or `fmnist`).
```bash
python3 benchmark.py syndata
# OR
python3 benchmark.py fmnist
```

## Results

We implemented three key algorithms and compared them against a Brute Force baseline:
1.  **IVF (Inverted File Index)**: Clusters vectors using K-Means and searches only nearest clusters.
2.  **LSH (Locality Sensitive Hashing)**: Uses Random Hyperplane Projections (SimHash) with single and multi-table support.
3.  **HNSW (Hierarchical Navigable Small World)**: Graph-based approach for efficient navigation.

### Benchmark on Synthetic Data (10k vectors, 128-dim)

| Method | Configuration | Latency (ms) | Recall@10 | Notes |
|--------|--------------|--------------|-----------|-------|
| **Brute Force** | - | ~1.23 | **1.00** | Baseline. |
| **IVF** | n_probes=1 | ~0.14 | 0.06 | Very poor recall on random noise. |
| **IVF** | n_probes=10 | ~0.98 | 0.37 | improved recall but still low. |
| **LSH** | 10 Tables, 8 bits | ~0.17 | 0.21 | Hard to hash random noise preserving similarity. |
| **HNSW** | M=16, ef=100 | ~1.30 | **0.61** | Best performance on random data. |

### Benchmark on Fashion-MNIST (10k vectors, 128-dim)

| Method | Configuration | Latency (ms) | Recall@10 | Notes |
|--------|--------------|--------------|-----------|-------|
| **Brute Force** | - | ~0.89 | **1.00** | Highly optimized `numpy` dot product. Baseline for accuracy. |
| **IVF** | n_probes=1 | ~0.12 | 0.63 | **Fastest**. Good recall on real data (clusters well). |
| **IVF** | n_probes=10 | ~0.28 | **1.00** | Perfect recall with <1/3rd latency of Brute Force. |
| **LSH** | 10 Tables, 8 bits | ~0.54 | 0.89 | Good recall but slower due to multi-table overhead. |
| **HNSW** | M=16, ef=100 | ~0.32 | **1.00** | Excellent trade-off. Robust high recall. |

*Note: Latency is per query, averaged over 10 queries.*

### Synthetic Data vs Real Data
- **Synthetic Data**: Random high-dimensional noise is hard to cluster. IVF recall was low (~0.06 with n_probes=1) because specific clusters didn't capture nearest neighbors well.
- **Real Data (FMNIST)**: Real-world data lies on a lower-dimensional manifold. IVF and HNSW performed exceptionally well (100% recall), proving their effectiveness on structured data.

## Performance Tradeoffs

1.  **Latency vs Accuracy (Recall)**
    - **Brute Force**: Guarantees 100% recall but scales linearly $O(N)$. Slowest for large $N$.
    - **IVF**: Offers tunable trade-off via `n_probes`.
        - Low `n_probes` = Ultra fast, lower recall.
        - High `n_probes` = Approaches Brute Force speed and accuracy.
    - **HNSW**: Logarithmic scaling $O(\log N)$ makes it vastly superior for very large datasets ($N > 1M$), even if Python overhead masks this at $N=10k$.

2.  **Index Build Time**
    - **LSH**: Fastest construction ($O(N)$), just hashing.
    - **IVF**: Moderate construction ($O(N \cdot K)$), dominated by K-Means clustering.
    - **HNSW**: Slowest construction ($O(N \cdot \log N)$), requires inserting vectors one-by-one into the graph.

3.  **Memory Usage**
    - **LSH**: High memory for Multi-table (requires $L$ tables).
    - **IVF**: Low memory overhead (just centroids + reorganized list).
    - **HNSW**: Higher memory (stores graph edges for every node).
