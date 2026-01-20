import numpy as np
import time

import numpy as np
import time
import heapq
import random

class IVFIndex:
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.centroids = None
        self.inverted_lists = None
        self.assignments = None
        self.vectors = None
    
    def build(self, vectors):
        self.vectors = vectors
        print(f"Building IVF index with {self.n_clusters} clusters...")
        start_time = time.time()
        
        # 1. Cluster vectors
        self.centroids, self.assignments = self._kmeans_clustering(vectors, self.n_clusters)
        
        # 2. Build inverted lists
        self.inverted_lists = [[] for _ in range(self.n_clusters)]
        for idx, cluster_id in enumerate(self.assignments):
            self.inverted_lists[cluster_id].append(idx)
            
        print(f"IVF index built in {time.time() - start_time:.2f}s")

    def search(self, query_vector, k=5, n_probes=5):
        # 1. Find nearest centroids
        centroid_scores = np.dot(self.centroids, query_vector)
        top_probes = np.argsort(centroid_scores)[::-1][:n_probes]
        
        # 2. Gather candidate vectors
        candidate_indices = []
        for cluster_id in top_probes:
            candidate_indices.extend(self.inverted_lists[cluster_id])
            
        candidate_indices = np.array(candidate_indices)
        
        if len(candidate_indices) == 0:
            return []
            
        # 3. Exact search on candidates
        candidate_vectors = self.vectors[candidate_indices]
        scores = np.dot(candidate_vectors, query_vector)
        
        if k >= len(scores):
            local_top_k_idx = np.arange(len(scores))
        else:
            local_top_k_idx = np.argpartition(scores, -k)[-k:]
            
        top_k_scores = scores[local_top_k_idx]
        sorted_local_indices = np.argsort(top_k_scores)[::-1]
        
        results = []
        for i in sorted_local_indices:
            idx_in_candidates = local_top_k_idx[i]
            original_idx = candidate_indices[idx_in_candidates]
            results.append((int(original_idx), float(scores[idx_in_candidates])))
            
        return results

    def _kmeans_clustering(self, vectors, k_centroids, max_iters=20):
        n, d = vectors.shape
        random_indices = np.random.choice(n, k_centroids, replace=False)
        centroids = vectors[random_indices].copy()
        
        for _ in range(max_iters):
            sims = np.dot(vectors, centroids.T)
            assignments = np.argmax(sims, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k_centroids)
            
            for i in range(n):
                cluster_id = assignments[i]
                new_centroids[cluster_id] += vectors[i]
                counts[cluster_id] += 1
            
            for c in range(k_centroids):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                    norm = np.linalg.norm(new_centroids[c])
                    if norm > 0:
                        new_centroids[c] /= norm
            
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
            
        return centroids, assignments


class LSHIndex:
    def __init__(self, nBITS=16, nTables=1):
        """
        LSH with Random Hyperplanes (SimHash) for Cosine Similarity.
        nBITS: Number of bits per hash key (more bits = higher precision, fewer collisions).
        nTables: Number of hash tables (more tables = higher recall, more candidates).
        """
        self.nBITS = nBITS
        self.nTables = nTables
        self.tables = []
        self.projections = []
        self.vectors = None

    def build(self, vectors):
        self.vectors = vectors
        n, d = vectors.shape
        print(f"Building LSH index with {self.nTables} tables, {self.nBITS} bits...")
        start_time = time.time()
        
        self.tables = []
        self.projections = []
        
        for _ in range(self.nTables):
            # Random projection matrix for this table
            # vectors are normalized, so sign of dot product determines region
            projections = np.random.randn(d, self.nBITS)
            self.projections.append(projections)
            
            table = {}
            
            # Compute hashes for all vectors
            # (N, D) @ (D, Bits) -> (N, Bits)
            dots = np.dot(vectors, projections)
            bits = (dots > 0).astype(int)
            
            # Convert bits to integers for dict keys
            # Powers of 2: [1, 2, 4, ..., 2^(bits-1)]
            powers_of_two = (1 << np.arange(self.nBITS))
            hashes = np.dot(bits, powers_of_two)
            
            for idx, h_val in enumerate(hashes):
                if h_val not in table:
                    table[h_val] = []
                table[h_val].append(idx)
                
            self.tables.append(table)
            
        print(f"LSH index built in {time.time() - start_time:.2f}s")

    def search(self, query_vector, k=5):
        candidate_indices = set()
        
        for i in range(self.nTables):
            proj = self.projections[i]
            table = self.tables[i]
            
            # Compute query hash
            dots = np.dot(query_vector, proj)
            bits = (dots > 0).astype(int)
            powers_of_two = (1 << np.arange(self.nBITS))
            q_hash = np.dot(bits, powers_of_two)
            
            # Retrieve candidates
            if q_hash in table:
                candidate_indices.update(table[q_hash])
            
            # Optional: Multi-probe LSH could check Hamming distance neighbors here
            # But plain multi-table usually relies on multiple tables for coverage
            
        if not candidate_indices:
            return []
            
        candidate_indices = list(candidate_indices)
        candidate_vectors = self.vectors[candidate_indices]
        
        # Exact search on candidates
        scores = np.dot(candidate_vectors, query_vector)
        
        if k >= len(scores):
            local_top_k_idx = np.arange(len(scores))
        else:
            local_top_k_idx = np.argpartition(scores, -k)[-k:]
            
        top_k_scores = scores[local_top_k_idx]
        sorted_local_indices = np.argsort(top_k_scores)[::-1]
        
        results = []
        for i in sorted_local_indices:
            idx_in_candidates = local_top_k_idx[i]
            original_idx = candidate_indices[idx_in_candidates]
            results.append((int(original_idx), float(scores[idx_in_candidates])))
            
        return results


class HNSWIndex:
    def __init__(self, M=16, ef_construction=200, ef_search=50, m_L=None):
        """
        Hierarchical Navigable Small World Graph.
        M: Max number of connections per node per layer.
        ef_construction: Size of dynamic candidate list during construction.
        ef_search: Size of dynamic candidate list during search.
        m_L: Normalization factor for level generation (1/ln(M)).
        """
        self.M = M
        self.M0 = 2 * M  # Max connections for layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_L = m_L if m_L else 1.0 / np.log(M)
        
        self.vectors = None
        self.nodes = {} # id -> Node
        self.levels = [] # List of graph layers (dicts: id -> neighbors_list)
        self.entry_point = None
        self.max_level = -1
        
    def _get_level(self):
        return int(-np.log(random.random()) * self.m_L)
        
    def build(self, vectors):
        self.vectors = vectors
        self.nodes = {}
        self.levels = []
        self.entry_point = None
        self.max_level = -1
        
        n, d = vectors.shape
        print(f"Building HNSW index with M={self.M}, ef={self.ef_construction}...")
        start_time = time.time()
        
        for i in range(n):
            self._insert(i, vectors[i])
            if i % 1000 == 0 and i > 0:
                print(f"Inserted {i}/{n} vectors...")
                
        print(f"HNSW index built in {time.time() - start_time:.2f}s")
        
    def _dist(self, vec1, vec2):
        # 1.0 - dot (since we want distance, but we have normalized vectors so dot is similarity)
        # HNSW usually works with distances. Minimizing distance = Maximizing dot product.
        # We will use Negative Dot Product as "Distance" so smaller is better.
        return -np.dot(vec1, vec2)
        
    def _search_layer(self, q, ep, ef, level):
        """
        Greedy search in a layer.
        q: query vector
        ep: entry point (id)
        ef: size of candidate list
        level: layer level
        
        Returns: min-heap of (dist, id)
        """
        visited = {ep}
        # Candidates: min-heap of (dist, id)
        ep_dist = self._dist(self.vectors[ep], q)
        candidates = [(ep_dist, ep)] 
        heapq.heapify(candidates)
        
        # Best matches found so far: max-heap (to easily remove worst)
        # stores (-dist, id) because Python has min-heap
        found = [(-ep_dist, ep)]
        heapq.heapify(found)
        
        while candidates:
            curr_dist, curr_id = heapq.heappop(candidates)
            worst_found_dist = -found[0][0] # Positive distance
            
            if curr_dist > worst_found_dist:
                break
                
            for neighbor in self.levels[level].get(curr_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._dist(self.vectors[neighbor], q)
                    
                    worst_found_dist = -found[0][0]
                    if dist < worst_found_dist or len(found) < ef:
                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(found, (-dist, neighbor))
                        if len(found) > ef:
                            heapq.heappop(found)
                            
        # Return list of (dist, id), sorted by dist
        # found has (-dist, id)
        res = []
        for neg_dist, idx in found:
            res.append((-neg_dist, idx))
        res.sort()
        return res

    def _insert(self, idx, vector):
        level = self._get_level()
        
        # Initialize neighbors for this node up to its level
        # Dynamically grow levels list if needed
        while len(self.levels) <= level:
            self.levels.append({})
            
        for l in range(level + 1):
            self.levels[l][idx] = []
            
        curr_node = idx
        
        if self.entry_point is None:
            self.max_level = level
            self.entry_point = idx
            return
            
        ep = self.entry_point
        max_l = self.max_level
        
        # 1. Zoom in from max_level down to level + 1
        for l in range(max_l, level, -1):
            # Simplest greedy search: ef=1
            res = self._search_layer(vector, ep, 1, l)
            ep = res[0][1] # Closest one becomes next entry point
            
        # 2. Insert from level down to 0
        for l in range(min(level, max_l), -1, -1):
            # Search for nearest neighbors
            candidates = self._search_layer(vector, ep, self.ef_construction, l)
            
            # Select neighbors (simple heuristic: take top M)
            # candidates are sorted by dist
            neighbors = [c[1] for c in candidates[:self.M]]
            
            # Update bidirectional connections
            self.levels[l][idx] = neighbors
            for neighbor in neighbors:
                self.levels[l][neighbor].append(idx)
                # Prune if too many connections
                max_conn = self.M0 if l == 0 else self.M
                if len(self.levels[l][neighbor]) > max_conn:
                    # Keep closest
                    # Re-calculate distances for all neighbors of 'neighbor'
                    # ideally we'd cache this but for simplicity recompute
                    conns = self.levels[l][neighbor]
                    conn_dists = [(self._dist(self.vectors[n_idx], self.vectors[neighbor]), n_idx) for n_idx in conns]
                    conn_dists.sort()
                    self.levels[l][neighbor] = [c[1] for c in conn_dists[:max_conn]]
            
            # Update ep for next layer
            ep = candidates[0][1]
            
        if level > self.max_level:
            self.max_level = level
            self.entry_point = idx

    def search(self, query_vector, k=5):
        if self.entry_point is None:
            return []
            
        ep = self.entry_point
        max_l = self.max_level
        
        # 1. Zoom down to layer 0
        for l in range(max_l, 0, -1):
            res = self._search_layer(query_vector, ep, 1, l)
            ep = res[0][1]
            
        # 2. Search layer 0 with ef_search
        candidates = self._search_layer(query_vector, ep, self.ef_search, 0)
        
        # candidates are (dist, id) where dist is -cosine_sim
        # valid top-k
        top_k = candidates[:k]
        
        results = []
        for dist, idx in top_k:
            results.append((int(idx), -dist)) # Convert back to dot product
            
        return results


class VectorDB:
    def __init__(self):
        self.vectors = None
        self.index_type = "flat"
        self.index = None

    def add_vectors(self, vectors):
        self.vectors = vectors
        print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")

    def brute_force_search(self, query_vector, k=5):
        if self.vectors is None:
            raise ValueError("No vectors loaded")
        scores = np.dot(self.vectors, query_vector)
        if k >= len(scores):
            top_k_indices = np.arange(len(scores))
        else:
            top_k_indices = np.argpartition(scores, -k)[-k:]
        top_k_scores = scores[top_k_indices]
        sorted_indices_in_top_k = np.argsort(top_k_scores)[::-1]
        results = []
        for i in sorted_indices_in_top_k:
            idx = top_k_indices[i]
            results.append((int(idx), float(scores[idx])))
        return results

    def build_index(self, index_type="ivf", **kwargs):
        """
        Build index of specified type.
        index_type: 'ivf', 'lsh', 'hnsw'
        kwargs: parameters for the index (e.g. n_clusters, nBITS, M)
        """
        self.index_type = index_type
        if index_type == "ivf":
            self.index = IVFIndex(**kwargs)
        elif index_type == "lsh":
            self.index = LSHIndex(**kwargs)
        elif index_type == "hnsw":
            self.index = HNSWIndex(**kwargs)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        if self.vectors is not None:
            self.index.build(self.vectors)
        else:
            print("Warning: No vectors to build index. Load vectors first.")

    def search(self, query_vector, k=5, **kwargs):
        if self.index_type == "flat":
            return self.brute_force_search(query_vector, k)
        elif self.index:
            return self.index.search(query_vector, k, **kwargs)
        else:
            # Fallback
            return self.brute_force_search(query_vector, k)
