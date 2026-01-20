# Vector Similarity Search Engine - Code & Algorithm Explanation

This document details the implementation of the Vector Similarity Search Engine. It covers the mathematical principles behind the algorithms and provides a function-by-function walkthrough of the codebase.

---

## 1. Core Concepts

### Vectors & Cosine Similarity
We represent data (e.g., images) as high-dimensional vectors. The similarity between two vectors $A$ and $B$ is often measured by the **Cosine Similarity**:
$$ \text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$
- **Optimization**: In our implementation, we **normalize** all vectors to unit length ($\|V\|=1$) during data generation.
- **Result**: Cosine similarity simplifies to the **Dot Product**: $A \cdot B$. This allows us to use fast matrix multiplication.

---

## 2. Algorithms "From Scratch"

### A. Brute Force (Exact Search)
- **Principle**: To find the nearest neighbors of a query $Q$, calculate the distance to **every** vector in the database and sort them.
- **Complexity**: $O(N \cdot D)$ where $N$ is number of vectors, $D$ is dimension.
- **Pros/Cons**: 100% accurate (Recall=1.0) but slow for massive datasets.

### B. IVF (Inverted File Index)
- **Principle**: Divide the vector space into $K$ clusters (Voronoi cells).
- **Training**: Use **K-Means Clustering** to find $K$ centroids.
- **Indexing**: Assign every vector to its nearest centroid. Store them in an "Inverted List" (Cluster ID $\to$ List of Vector IDs).
- **Search**:
    1.  Compare Query $Q$ to the $K$ centroids.
    2.  Select top `n_probes` nearest centroids.
    3.  Only search vectors inside those specific clusters.
- **Benefit**: Reduces search scope from $N$ to $\approx N \cdot (n\_probes / K)$.

### C. LSH (Locality Sensitive Hashing)
- **Principle**: Hash similar vectors to the same bucket with high probability.
- **Algorithm (Random Projection / SimHash)**:
    - Generate a random unit vector $r$ (hyperplane).
    - If vector $v$ is on "positive" side ($v \cdot r > 0$), bit is 1. Otherwise 0.
    - Repeat for $B$ bits to create a signature.
    - Vectors with same signature land in the same bucket.
- **Multi-Table**: A single hash table has many collisions or false negatives. We use $L$ independent hash tables. A match in **any** table is a candidate.

### D. HNSW (Hierarchical Navigable Small World)
- **Principle**: A multi-layer graph where efficient navigation happens.
- **Structure**:
    - **Bottom Layer (0)**: Contains all vectors. Connected as a "Small World" graph (short paths exist between nodes).
    - **Higher Layers**: Sparse versions of the graph. Acts as a "highway" to quickly traverse long distances across the vector space.
- **Search**:
    1.  Start at top layer. Greedy traverse to find closest node to $Q$.
    2.  Drop down to next layer, using that node as entry point.
    3.  Repeat until Layer 0.
    4.  Perform detailed search in Layer 0.

---

## 3. Code Walkthrough (`vector_db.py`)

### `VectorDB` Class
Wrapper class that manages data storage and index selection.
- `add_vectors(vectors)`: Stores the raw vector data numpy array.
- `build_index(index_type, **kwargs)`: Instantiates and builds the specific index (IVF, LSH, or HNSW).
- `search(query, **kwargs)`: Routes the query to the active index.

### `IVFIndex` Class
- `_kmeans_clustering()`: Custom implementation of Lloyd's algorithm for K-Means.
    - iteratively updates centroids by averaging assigned vectors.
- `build()`:
    1.  Runs K-Means to get centroids.
    2.  Populates `inverted_lists` buckets.
- `search()`:
    1.  Finds nearest centroids to query.
    2.  Concatenates vectors from those buckets into `candidate_vectors`.
    3.  Runs exact dot-product search on candidates.

### `LSHIndex` Class
- `build()`:
    - Creates `nTables` independent hash tables.
    - For each table, generates random matrix `projections` (shape $D \times bits$).
    - Computes hash: `hash = (vectors @ projections > 0)`.
    - Stores indices in a dictionary: `table[hash_int] = [indices]`.
- `search()`:
    - Hashes the query vector using the stored projections.
    - Looks up candidates in all tables.
    - Uses a `set` to deduplicate candidates.
    - Re-ranks candidates using exact dot product.

### `HNSWIndex` Class
- `__init__`: Sets parameters `M` (max edges) and `ef` (beam search width).
- `_get_level()`: Randomly assigns a max layer for a new node (exponentially decaying probability).
- `_search_layer(q, ep, ef, level)`:
    - **Greedy Best-First Search**.
    - Keeps a min-heap of `candidates` to explore and a max-heap of `found` nearest neighbors.
    - Explores neighbors of best candidate until distance stops improving.
- `insert(vector)`:
    - Finds entry point in top layer.
    - Greedily traverses down to node's assigned level.
    - From that level down to 0: finds `M` nearest neighbors and adds bidirectional edges.
    - **Heuristic**: If edges exceed `M`, keeps only the closest ones.
- `search(query)`:
    - Starts at top `entry_point`.
    - "Zooms in" by traversing down layers to find a good starting point for Layer 0.
    - Performs `ef_search` at Layer 0 to get final candidates.

---

## 4. Helper Scripts

- `generate_syndata.py`:
    - Uses `np.random.randn` to create standard normal distribution.
    - Normalizes rows using `linalg.norm` so they lie on the hypersphere.
    - Computes Ground Truth by running a full matrix multiplication (`queries @ vectors.T`) and sorting.

- `generate_fmnist.py`:
    - Downloads Fashion-MNIST (images of clothes).
    - **PCA (Principal Component Analysis)**:
        - Calculates covariance matrix of pixels.
        - Computes eigenvectors (principal directions).
        - Projects 784-dim images onto top 128 eigenvectors to create dense embeddings.
    - This creates "meaningful" vectors where spatial visual similarity $\approx$ cosine similarity.

- `benchmark.py`:
    - Loads data and truth.
    - Runs each algorithm loop:
        1.  `setup_fn()`: Builds index (timed).
        2.  `search_fn()`: Queries (timed).
        3.  `compute_recall()`: Measures intersection with Ground Truth.
    - Prints comparison table.
