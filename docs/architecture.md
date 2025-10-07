# Architecture

Deep dive into GoFAISS implementation details, algorithms, and design decisions.

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│  (User code using GoFAISS API)                          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Public API Layer                        │
│  pkg/index/    - Index implementations                   │
│  pkg/vector/   - Vector operations                       │
│  pkg/search/   - High-level search API                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Core Components                         │
│  pkg/metric/   - Distance calculations                   │
│  pkg/storage/  - Persistence and serialization          │
│  internal/     - Internal utilities                      │
└─────────────────────────────────────────────────────────┘
```

### Package Structure

```
gofaiss/
├── pkg/                    # Public API
│   ├── metric/            # Distance metrics (L2, cosine, IP)
│   │   ├── l2.go
│   │   ├── cosine.go
│   │   └── inner_product.go
│   │
│   ├── index/             # Index implementations
│   │   ├── flat/          # Brute-force index
│   │   │   ├── index.go
│   │   │   └── search.go
│   │   │
│   │   ├── hnsw/          # HNSW implementation
│   │   │   ├── index.go
│   │   │   ├── graph.go
│   │   │   ├── search.go
│   │   │   └── build.go
│   │   │
│   │   ├── ivf/           # Inverted file index
│   │   │   ├── index.go
│   │   │   ├── clustering.go
│   │   │   └── search.go
│   │   │
│   │   ├── pq/            # Product quantization
│   │   │   ├── index.go
│   │   │   ├── codebook.go
│   │   │   └── quantize.go
│   │   │
│   │   └── ivfpq/         # IVF + PQ combined
│   │       ├── index.go
│   │       └── search.go
│   │
│   ├── vector/            # Vector utilities
│   │   ├── vector.go      # Vector type definition
│   │   ├── operations.go  # Vector math
│   │   └── generator.go   # Synthetic data
│   │
│   ├── storage/           # Persistence
│   │   ├── serializer.go  # Gob/JSON serialization
│   │   └── compression.go # Gzip support
│   │
│   └── search/            # High-level API
│       └── api.go         # Unified search interface
│
├── internal/              # Internal packages
│   ├── math/             # Low-level math operations
│   │   ├── simd.go       # SIMD optimizations (future)
│   │   └── distance.go   # Optimized distance functions
│   │
│   ├── storage/          # Internal storage utilities
│   │   └── buffer.go     # Memory buffers
│   │
│   └── utils/            # Helper functions
│       ├── heap.go       # Priority queue
│       └── sort.go       # Sorting utilities
│
├── cmd/                  # Command-line tools
│   ├── gofaiss-cli/     # CLI interface
│   │   └── main.go
│   │
│   └── gofaiss-server/  # HTTP server
│       └── main.go
│
├── benchmark/            # Benchmarking suite
│   ├── benchmark_comparison.go
│   ├── run_benchmark.sh
│   └── scripts/
│
└── examples/             # Usage examples
    ├── basic/
    ├── batch/
    └── persistence/
```

## Core Algorithms

### 1. Flat Index (Brute Force)

**Algorithm**: Linear scan with distance computation

```
function Search(query, k):
    distances = []
    for each vector in database:
        d = distance(query, vector)
        distances.append((d, vector.id))
    
    sort(distances)
    return top k results
```

**Complexity**:
- Time: O(n × d) where n=vectors, d=dimensions
- Space: O(n × d)

**Implementation Details**:
- Uses optimized distance functions
- Preallocates result buffers
- Sorts results using heap for k-selection

**Code Structure**:
```go
type FlatIndex struct {
    vectors    []vector.Vector
    dimension  int
    metric     string
    distanceFunc func([]float32, []float32) float32
}

func (idx *FlatIndex) Search(query []float32, k int) ([]SearchResult, error) {
    // Compute all distances
    distances := make([]SearchResult, len(idx.vectors))
    for i, v := range idx.vectors {
        distances[i] = SearchResult{
            ID:       v.ID,
            Distance: idx.distanceFunc(query, v.Data),
        }
    }
    
    // k-select using partial sort
    partialSort(distances, k)
    return distances[:k], nil
}
```

### 2. HNSW (Hierarchical Navigable Small World)

**Algorithm**: Multi-layer navigable graph

```
Build Phase:
    for each vector v:
        level = randomLevel()  // Exponential decay
        for l from level down to 0:
            neighbors = searchLayer(v, M, l)
            connect(v, neighbors, l)
            pruneEdges(v, M, l)

Search Phase:
    entry = topLayerNode
    for l from topLayer down to 0:
        entry = searchLayer(query, entry, efSearch, l)
    return k nearest from entry
```

**Complexity**:
- Build: O(n × log(n) × M × d)
- Search: O(log(n) × M × d)
- Space: O(n × M × log(n))

**Key Components**:

**1. Graph Structure**:
```go
type HNSWIndex struct {
    graphs     [][]Graph      // graphs[level][nodeID] = neighbors
    entryPoint int64
    maxLevel   int
    vectors    map[int64]vector.Vector
    
    // Parameters
    M              int  // Max connections per node
    EfConstruction int  // Search depth during build
    EfSearch       int  // Search depth during query
}

type Graph struct {
    Neighbors []int64  // IDs of connected nodes
}
```

**2. Level Assignment**:
```go
func (idx *HNSWIndex) randomLevel() int {
    // Exponential decay: P(level) = (1/ln(M)) ^ level
    mL := 1.0 / math.Log(float64(idx.M))
    r := rand.Float64()
    return int(-math.Log(r) * mL)
}
```

**3. Search Layer**:
```go
func (idx *HNSWIndex) searchLayer(query []float32, entry int64, 
                                   ef int, layer int) []Candidate {
    visited := make(map[int64]bool)
    candidates := NewMaxHeap(ef)  // Keep ef best candidates
    results := NewMinHeap(ef)
    
    candidates.Push(entry, distance(query, idx.vectors[entry]))
    visited[entry] = true
    
    for !candidates.Empty() {
        current := candidates.Pop()
        if current.distance > results.Top().distance {
            break  // No improvements possible
        }
        
        for _, neighborID := range idx.graphs[layer][current.id].Neighbors {
            if visited[neighborID] {
                continue
            }
            visited[neighborID] = true
            
            d := distance(query, idx.vectors[neighborID])
            if d < results.Top().distance || results.Size() < ef {
                candidates.Push(neighborID, d)
                results.Push(neighborID, d)
                if results.Size() > ef {
                    results.Pop()  // Remove worst
                }
            }
        }
    }
    
    return results.ToSortedSlice()
}
```

**4. Edge Pruning (Heuristic)**:
```go
func (idx *HNSWIndex) pruneEdges(nodeID int64, M int, layer int) {
    neighbors := idx.graphs[layer][nodeID].Neighbors
    if len(neighbors) <= M {
        return
    }
    
    // Keep M closest neighbors
    sort.Slice(neighbors, func(i, j int) bool {
        di := distance(idx.vectors[nodeID], idx.vectors[neighbors[i]])
        dj := distance(idx.vectors[nodeID], idx.vectors[neighbors[j]])
        return di < dj
    })
    
    idx.graphs[layer][nodeID].Neighbors = neighbors[:M]
}
```

**Design Decisions**:
- Used map for sparse graph storage (memory efficient)
- Separate graphs per layer for clarity
- Preallocate heaps to reduce allocations
- Lock-free reads, write locks only during insertion

### 3. IVF (Inverted File Index)

**Algorithm**: Clustering-based partitioning

```
Training Phase:
    centroids = kmeans(training_vectors, nlist)
    
Build Phase:
    for each vector v:
        nearest_cluster = argmin(distance(v, centroids))
        inverted_lists[nearest_cluster].append(v)

Search Phase:
    nearest_clusters = topK(query, centroids, nprobe)
    candidates = []
    for cluster in nearest_clusters:
        candidates.extend(inverted_lists[cluster])
    return topK(query, candidates, k)
```

**Complexity**:
- Train: O(iterations × nlist × training_size × d)
- Build: O(n × nlist × d)
- Search: O(nprobe × (n/nlist) × d)
- Space: O(n × d + nlist × d)

**Implementation**:
```go
type IVFIndex struct {
    centroids      [][]float32           // [nlist][dimension]
    invertedLists  [][]vector.Vector     // [nlist][vectors]
    nlist          int
    metric         string
    trained        bool
}

func (idx *IVFIndex) Train(vectors []vector.Vector) error {
    // K-means clustering
    idx.centroids = kmeans(vectors, idx.nlist, maxIterations)
    idx.trained = true
    return nil
}

func (idx *IVFIndex) Add(vectors []vector.Vector) error {
    if !idx.trained {
        return errors.New("index not trained")
    }
    
    // Assign each vector to nearest centroid
    for _, v := range vectors {
        clusterID := idx.findNearestCentroid(v.Data)
        idx.invertedLists[clusterID] = append(
            idx.invertedLists[clusterID], v)
    }
    return nil
}

func (idx *IVFIndex) Search(query []float32, k, nprobe int) ([]SearchResult, error) {
    // Find nprobe nearest clusters
    clusterDistances := make([]ClusterDistance, idx.nlist)
    for i, centroid := range idx.centroids {
        clusterDistances[i] = ClusterDistance{
            ID:       i,
            Distance: distance(query, centroid),
        }
    }
    sort.Slice(clusterDistances, ...)  // Sort by distance
    
    // Search in top nprobe clusters
    var candidates []SearchResult
    for i := 0; i < nprobe && i < len(clusterDistances); i++ {
        clusterID := clusterDistances[i].ID
        for _, v := range idx.invertedLists[clusterID] {
            candidates = append(candidates, SearchResult{
                ID:       v.ID,
                Distance: distance(query, v.Data),
            })
        }
    }
    
    // Return top k
    sort.Slice(candidates, ...)
    if len(candidates) > k {
        candidates = candidates[:k]
    }
    return candidates, nil
}
```

**K-means Implementation**:
```go
func kmeans(vectors []vector.Vector, k int, maxIter int) [][]float32 {
    // Initialize centroids (k-means++)
    centroids := kmeansppInit(vectors, k)
    
    for iter := 0; iter < maxIter; iter++ {
        // Assignment step
        assignments := make([]int, len(vectors))
        for i, v := range vectors {
            assignments[i] = findNearestCentroid(v.Data, centroids)
        }
        
        // Update step
        newCentroids := make([][]float32, k)
        counts := make([]int, k)
        
        for i, v := range vectors {
            cluster := assignments[i]
            if newCentroids[cluster] == nil {
                newCentroids[cluster] = make([]float32, len(v.Data))
            }
            vectorAdd(newCentroids[cluster], v.Data)
            counts[cluster]++
        }
        
        // Average
        for i := range newCentroids {
            if counts[i] > 0 {
                vectorScale(newCentroids[i], 1.0/float32(counts[i]))
            }
        }
        
        // Check convergence
        if centroidsConverged(centroids, newCentroids) {
            break
        }
        centroids = newCentroids
    }
    
    return centroids
}
```

### 4. Product Quantization (PQ)

**Algorithm**: Vector compression via subspace quantization

```
Training Phase:
    split vectors into M subvectors
    for each subspace m:
        codebook[m] = kmeans(subvectors[m], 256)  // 8-bit codes

Encoding Phase:
    for each vector v:
        split v into M subvectors
        for each subspace m:
            code[m] = argmin(distance(subvector[m], codebook[m]))
        store code (M bytes instead of M*d*4 bytes)

Search Phase:
    precompute distance tables for query
    for each encoded vector:
        distance = sum(distance_tables[m][code[m]] for m in M)
    return top k
```

**Complexity**:
- Train: O(M × iterations × 256 × (n/M) × (d/M))
- Encode: O(n × M × 256 × (d/M))
- Search: O(M × 256 × (d/M) + n × M)  // Much faster than O(n × d)
- Space: O(n × M) bytes vs O(n × d × 4) bytes

**Implementation**:
```go
type PQIndex struct {
    codebooks     [][][]float32  // [M][256][dsub]
    codes         [][]uint8      // [n][M]
    M             int            // Number of subquantizers
    dsub          int            // Dimension per subquantizer
    dimension     int
    vectorIDs     []int64
}

func (idx *PQIndex) Train(vectors []vector.Vector) error {
    idx.dsub = idx.dimension / idx.M
    idx.codebooks = make([][][]float32, idx.M)
    
    // Train each subquantizer independently
    for m := 0; m < idx.M; m++ {
        // Extract subvectors for this subspace
        subvectors := make([][]float32, len(vectors))
        for i, v := range vectors {
            start := m * idx.dsub
            end := start + idx.dsub
            subvectors[i] = v.Data[start:end]
        }
        
        // K-means with k=256 (8-bit codes)
        idx.codebooks[m] = kmeans(subvectors, 256)
    }
    
    return nil
}

func (idx *PQIndex) Add(vectors []vector.Vector) error {
    for _, v := range vectors {
        // Encode vector
        code := make([]uint8, idx.M)
        for m := 0; m < idx.M; m++ {
            start := m * idx.dsub
            end := start + idx.dsub
            subvector := v.Data[start:end]
            
            // Find nearest centroid in codebook
            code[m] = idx.quantize(subvector, m)
        }
        
        idx.codes = append(idx.codes, code)
        idx.vectorIDs = append(idx.vectorIDs, v.ID)
    }
    return nil
}

func (idx *PQIndex) Search(query []float32, k int) ([]SearchResult, error) {
    // Precompute distance tables
    distanceTables := make([][]float32, idx.M)
    for m := 0; m < idx.M; m++ {
        start := m * idx.dsub
        end := start + idx.dsub
        querySubvector := query[start:end]
        
        // Distance from query subvector to all centroids
        distanceTables[m] = make([]float32, 256)
        for j := 0; j < 256; j++ {
            distanceTables[m][j] = distance(
                querySubvector, 
                idx.codebooks[m][j],
            )
        }
    }
    
    // Compute distances using lookup
    results := make([]SearchResult, len(idx.codes))
    for i, code := range idx.codes {
        dist := float32(0)
        for m := 0; m < idx.M; m++ {
            dist += distanceTables