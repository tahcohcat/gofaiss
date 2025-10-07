# API Reference

Complete API documentation for GoFAISS.

## Table of Contents

- [Vector Package](#vector-package)
- [Metric Package](#metric-package)
- [Index Types](#index-types)
  - [Flat Index](#flat-index)
  - [HNSW Index](#hnsw-index)
  - [IVF Index](#ivf-index)
  - [PQ Index](#pq-index)
  - [IVFPQ Index](#ivfpq-index)
- [Storage Package](#storage-package)
- [Stats Package](#stats-package)

---

## Vector Package

`import "github.com/tahcohcat/gofaiss/pkg/vector"`

### Types

#### Vector

```go
type Vector struct {
    ID   int64     // Vector identifier
    Data []float32 // Vector data
    Norm float32   // Precomputed norm for cosine similarity
}
```

Represents a vector with associated ID and optional precomputed norm.

#### SearchResult

```go
type SearchResult struct {
    ID       int64   // Vector ID
    Distance float32 // Distance to query
}
```

Represents a search result with distance metric.

### Functions

#### GenerateRandom

```go
func GenerateRandom(n, dim int, seed int64) []Vector
```

Generates `n` random vectors of dimension `dim` using the given random seed.

**Parameters:**
- `n`: Number of vectors to generate
- `dim`: Dimension of each vector
- `seed`: Random seed for reproducibility

**Returns:** Slice of random vectors with sequential IDs starting from 0.

**Example:**
```go
vectors := vector.GenerateRandom(1000, 128, 42)
```

#### ValidateDimension

```go
func ValidateDimension(vs []Vector, dim int) error
```

Validates that all vectors have the expected dimension.

**Parameters:**
- `vs`: Slice of vectors to validate
- `dim`: Expected dimension

**Returns:** Error if any vector has incorrect dimension.

#### Copy

```go
func Copy(s []float32) []float32
```

Returns a deep copy of a float32 slice.

#### Add

```go
func Add(a, b []float32) []float32
```

Element-wise addition of two vectors.

#### Subtract

```go
func Subtract(a, b []float32) []float32
```

Element-wise subtraction (a - b).

#### Scale

```go
func Scale(v []float32, s float32) []float32
```

Multiplies vector by scalar.

#### Norm

```go
func Norm(v []float32) float32
```

Computes L2 norm (magnitude) of a vector.

**Example:**
```go
magnitude := vector.Norm(myVector)
```

#### Normalize

```go
func Normalize(v []float32) []float32
```

Returns normalized vector (unit length). Returns original if norm is zero.

#### NormalizeInPlace

```go
func NormalizeInPlace(v []float32)
```

Normalizes vector in-place to unit length.

**Example:**
```go
vector.NormalizeInPlace(myVector)
```

#### Centroid

```go
func Centroid(vectors [][]float32) []float32
```

Computes the mean (centroid) of a set of vectors.

---

## Metric Package

`import "github.com/tahcohcat/gofaiss/pkg/metric"`

### Types

#### Type

```go
type Type string

const (
    L2     Type = "l2"     // Euclidean distance
    Cosine Type = "cosine" // Cosine similarity
    Dot    Type = "dot"    // Inner product
)
```

Supported distance metrics.

#### Metric

```go
type Metric interface {
    Distance(a, b []float32) float32
    Name() string
}
```

Interface for distance computations.

### Functions

#### New

```go
func New(t Type) (Metric, error)
```

Creates a new metric instance.

**Parameters:**
- `t`: Metric type (L2, Cosine, or Dot)

**Returns:** Metric instance or error for unknown type.

**Example:**
```go
m, err := metric.New(metric.L2)
if err != nil {
    log.Fatal(err)
}
dist := m.Distance(vec1, vec2)
```

---

## Index Types

### Common Interface

All indexes implement:

```go
type Index interface {
    Add(vectors []Vector) error
    Search(query []float32, k int) ([]SearchResult, error)
    BatchSearch(queries [][]float32, k int) ([][]SearchResult, error)
    Dimension() int
    Stats() stats.Stats
}
```

---

## Flat Index

`import "github.com/tahcohcat/gofaiss/pkg/index/flat"`

Exact brute-force search index. Guarantees 100% recall.

### Types

```go
type Index struct {
    // Private fields
}
```

### Functions

#### New

```go
func New(dim int, metric string) (*Index, error)
```

Creates a new flat index.

**Parameters:**
- `dim`: Vector dimension (must be > 0)
- `metric`: Distance metric ("l2" or "cosine")

**Returns:** Index instance or error.

**Example:**
```go
idx, err := flat.New(128, "l2")
if err != nil {
    log.Fatal(err)
}
```

### Methods

#### Add

```go
func (idx *Index) Add(vs []Vector) error
```

Adds vectors to the index.

**Parameters:**
- `vs`: Vectors to add

**Returns:** Error if dimension mismatch or zero vector with cosine metric.

**Concurrency:** Thread-safe.

#### Search

```go
func (idx *Index) Search(q []float32, k int) ([]SearchResult, error)
```

Searches for k nearest neighbors.

**Parameters:**
- `q`: Query vector
- `k`: Number of neighbors to return

**Returns:** Sorted results (closest first) or error.

**Concurrency:** Thread-safe (read lock).

#### BatchSearch

```go
func (idx *Index) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error)
```

Performs batch search for multiple queries.

#### Dimension

```go
func (idx *Index) Dimension() int
```

Returns vector dimension.

#### GetVectors

```go
func (idx *Index) GetVectors() []Vector
```

Returns all stored vectors.

#### Stats

```go
func (idx *Index) Stats() stats.Stats
```

Returns index statistics.

#### Save/Load

```go
func (idx *Index) Save(w storage.Writer) error
func (idx *Index) Load(r storage.Reader) error
```

Serialization methods.

---

## HNSW Index

`import "github.com/tahcohcat/gofaiss/pkg/index/hnsw"`

Hierarchical Navigable Small World graph index. Fast approximate search with high recall.

### Types

#### Config

```go
type Config struct {
    Metric         string // Distance metric
    M              int    // Connections per layer
    EfConstruction int    // Build quality
    EfSearch       int    // Search quality
}
```

HNSW configuration parameters.

**Field Details:**
- `M`: Number of bi-directional links per node (typical: 16-64)
  - Higher = better recall, more memory
- `EfConstruction`: Candidate list size during construction (typical: 100-500)
  - Higher = better quality, slower build
- `EfSearch`: Candidate list size during search (typical: 50-500)
  - Can be adjusted at runtime

#### Node

```go
type Node struct {
    ID    int64
    Data  []float32
    Level int
    Edges [][]int64 // Edges per level
}
```

Internal graph node structure.

### Functions

#### New

```go
func New(dim int, metricType string, config Config) (*Index, error)
```

Creates a new HNSW index.

**Parameters:**
- `dim`: Vector dimension
- `metricType`: Distance metric ("l2", "cosine", or "dot")
- `config`: HNSW configuration

**Returns:** Index instance or error.

**Example:**
```go
config := hnsw.Config{
    M:              16,
    EfConstruction: 200,
    EfSearch:       50,
}
idx, err := hnsw.New(128, "l2", config)
```

#### DefaultConfig

```go
func DefaultConfig() Config
```

Returns default HNSW configuration.

**Returns:**
```go
Config{
    Metric:         "l2",
    M:              16,
    EfConstruction: 200,
    EfSearch:       200,
}
```

### Methods

#### Add

```go
func (idx *Index) Add(vectors []Vector) error
```

Adds vectors to the HNSW graph.

**Parameters:**
- `vectors`: Vectors to add

**Returns:** Error if dimension mismatch.

**Concurrency:** Thread-safe (write lock).

**Note:** Auto-generates IDs if Vector.ID is 0.

#### Search

```go
func (idx *Index) Search(query []float32, k int) ([]SearchResult, error)
```

Searches for k nearest neighbors using HNSW algorithm.

**Parameters:**
- `query`: Query vector
- `k`: Number of neighbors

**Returns:** Approximately k nearest neighbors, sorted by distance.

**Concurrency:** Thread-safe (read lock).

**Complexity:** O(log N) average case.

#### BatchSearch

```go
func (idx *Index) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error)
```

Batch search for multiple queries.

#### SetEfSearch

```go
func (idx *Index) SetEfSearch(ef int)
```

Adjusts search quality parameter at runtime.

**Parameters:**
- `ef`: New efSearch value (higher = better recall, slower)

**Example:**
```go
idx.SetEfSearch(100) // Increase search quality
```

#### Remove

```go
func (idx *Index) Remove(id int64) error
```

Removes a vector from the index.

**Parameters:**
- `id`: Vector ID to remove

**Returns:** Error if vector not found.

**Note:** Expensive operation, rebuilds connections.

#### Size

```go
func (idx *Index) Size() int
```

Returns number of vectors in index.

#### Dimension

```go
func (idx *Index) Dimension() int
```

Returns vector dimension.

#### Stats

```go
func (idx *Index) Stats() stats.Stats
```

Returns detailed statistics including:
- Total vectors
- Memory usage
- Max graph level
- Configuration parameters

**Example:**
```go
stats := idx.Stats()
fmt.Printf("Max level: %v\n", stats.ExtraInfo["maxLevel"])
```

---

## IVF Index

`import "github.com/tahcohcat/gofaiss/pkg/index/ivf"`

Inverted File index using k-means clustering. Balanced speed and memory.

### Types

#### Config

```go
type Config struct {
    Metric string // Distance metric
    Nlist  int    // Number of clusters
}
```

IVF configuration.

**Field Details:**
- `Nlist`: Number of Voronoi cells (typical: sqrt(N), range: [100, 65536])

### Functions

#### New

```go
func New(dim int, metricType string, config Config) (*Index, error)
```

Creates a new IVF index.

**Example:**
```go
config := ivf.Config{
    Metric: "l2",
    Nlist:  1000,
}
idx, err := ivf.New(128, "l2", config)
```

#### DefaultConfig

```go
func DefaultConfig(numVectors int) Config
```

Returns default configuration based on dataset size.

**Parameters:**
- `numVectors`: Expected number of vectors

**Returns:** Config with `nlist = sqrt(numVectors)`, clamped to [10, 65536].

### Methods

#### Train

```go
func (idx *Index) Train(vectors []Vector) error
```

Trains the index by clustering vectors into cells.

**Parameters:**
- `vectors`: Training vectors (need at least `nlist` vectors)

**Returns:** Error if insufficient training data.

**Required:** Must be called before `Add()`.

**Example:**
```go
err := idx.Train(trainingVectors)
if err != nil {
    log.Fatal(err)
}
```

#### IsTrained

```go
func (idx *Index) IsTrained() bool
```

Returns whether index has been trained.

#### Add

```go
func (idx *Index) Add(vectors []Vector) error
```

Adds vectors to the index (assigns to nearest cluster).

**Returns:** Error if index not trained.

#### Search

```go
func (idx *Index) Search(query []float32, k int) ([]SearchResult, error)
```

Searches using inverted file algorithm.

**Note:** Uses `nprobe` parameter to determine how many clusters to search.

#### SetNProbe

```go
func (idx *Index) SetNProbe(nprobe int)
```

Sets number of clusters to search.

**Parameters:**
- `nprobe`: Number of nearest clusters to search (1-nlist)
  - Higher = better recall, slower search
  - Typical: 1-20

**Example:**
```go
idx.SetNProbe(10) // Search 10 nearest clusters
```

#### BatchSearch

```go
func (idx *Index) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error)
```

Batch search for multiple queries.

#### Dimension

```go
func (idx *Index) Dimension() int
```

Returns vector dimension.

#### Stats

```go
func (idx *Index) Stats() stats.Stats
```

Returns statistics including cluster distribution.

---

## PQ Index

`import "github.com/tahcohcat/gofaiss/pkg/index/pq"`

Product Quantization index for memory-efficient vector storage.

### Types

#### Config

```go
type Config struct {
    M     int // Number of subquantizers
    Nbits int // Bits per subquantizer
}
```

PQ configuration.

**Field Details:**
- `M`: Number of subquantizers (must divide dimension evenly, typical: 8-64)
- `Nbits`: Bits per code (typical: 8, meaning 256 centroids)
  - Memory: ~M bytes per vector
  - Compression: ~(dimension * 4) / M

### Functions

#### NewIndex

```go
func NewIndex(dim int, cfg Config) (*Index, error)
```

Creates a new PQ index.

**Parameters:**
- `dim`: Vector dimension (must be divisible by M)
- `cfg`: PQ configuration

**Returns:** Index instance or error.

**Example:**
```go
config := pq.Config{
    M:     16,  // 16 subquantizers
    Nbits: 8,   // 256 centroids each
}
idx, err := pq.NewIndex(128, config)
```

### Methods

#### Train

```go
func (idx *Index) Train(vectors []Vector) error
```

Trains PQ codebooks using k-means on subspaces.

**Parameters:**
- `vectors`: Training vectors (need at least 2^Nbits vectors)

**Returns:** Error if insufficient training data or dimension mismatch.

**Required:** Must be called before `Add()`.

**Example:**
```go
trainingVectors := vector.GenerateRandom(5000, 128, 42)
err := idx.Train(trainingVectors)
```

#### Add

```go
func (idx *Index) Add(vectors []Vector) error
```

Adds vectors by encoding them with PQ.

**Returns:** Error if index not trained.

**Note:** Vectors are automatically compressed using learned codebooks.

#### Search

```go
func (idx *Index) Search(query []float32, k int) ([]SearchResult, error)
```

Searches using asymmetric distance computation.

**Note:** Uses full-precision query against compressed database.

#### BatchSearch

```go
func (idx *Index) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error)
```

Batch search for multiple queries.

#### Dimension

```go
func (idx *Index) Dimension() int
```

Returns vector dimension.

#### Stats

```go
func (idx *Index) Stats() stats.Stats
```

Returns statistics including compression ratio.

**Example:**
```go
stats := idx.Stats()
compressionRatio := float64(stats.TotalVectors * dim * 4) / (stats.MemoryUsageMB * 1024 * 1024)
fmt.Printf("Compression: %.1fx\n", compressionRatio)
```

---

## IVFPQ Index

`import "github.com/tahcohcat/gofaiss/pkg/index/ivfpq"`

Combines IVF and PQ for fast search with compression. Best for large-scale datasets.

### Types

#### Config

```go
type Config struct {
    Metric string // Distance metric
    Nlist  int    // Number of IVF clusters
    M      int    // PQ subquantizers
    Nbits  int    // PQ bits per code
}
```

IVFPQ configuration combining IVF and PQ parameters.

### Functions

#### New

```go
func New(dim int, metricType string, config Config) (*Index, error)
```

Creates a new IVFPQ index.

**Example:**
```go
config := ivfpq.Config{
    Metric: "l2",
    Nlist:  1000,  // IVF clusters
    M:      16,    // PQ subquantizers
    Nbits:  8,     // PQ bits
}
idx, err := ivfpq.New(128, "l2", config)
```

### Methods

Similar to IVF Index but with additional PQ compression:

- `Train()`: Trains both IVF clustering and PQ codebooks
- `Add()`: Assigns to clusters and compresses
- `Search()`: Fast approximate search with compression
- `SetNProbe()`: Adjusts search quality

---

## Storage Package

`import "github.com/tahcohcat/gofaiss/pkg/storage"`

Handles index persistence.

### Functions

#### SaveIndex

```go
func SaveIndex(idx interface{}, filename string, compress bool) error
```

Saves an index to a file.

**Parameters:**
- `idx`: Index to save (must implement Save method)
- `filename`: Output file path
- `compress`: Enable gzip compression

**Example:**
```go
err := storage.SaveIndex(idx, "index.faiss.gz", true)
```

#### LoadIndex

```go
func LoadIndex(idx interface{}, filename string, compress bool) error
```

Loads an index from a file.

**Parameters:**
- `idx`: Index to load into (must implement Load method)
- `filename`: Input file path
- `compress`: File is gzip compressed

**Example:**
```go
idx := &hnsw.Index{}
err := storage.LoadIndex(idx, "index.faiss.gz", true)
```

### Interfaces

#### Writer

```go
type Writer interface {
    Encode(v interface{}) error
}
```

Interface for encoding data.

#### Reader

```go
type Reader interface {
    Decode(v interface{}) error
}
```

Interface for decoding data.

---

## Stats Package

`import "github.com/tahcohcat/gofaiss/pkg/index/stats"`

Provides index statistics.

### Types

#### Stats

```go
type Stats struct {
    TotalVectors  int                    // Number of vectors
    Dimension     int                    // Vector dimension
    IndexType     string                 // Index type name
    MemoryUsageMB float64                // Estimated memory (MB)
    ExtraInfo     map[string]interface{} // Type-specific info
}
```

Index statistics structure.

**ExtraInfo Examples:**

HNSW:
```go
{
    "metric": "l2",
    "M": 16,
    "efConstruction": 200,
    "efSearch": 50,
    "maxLevel": 4,
}
```

IVF:
```go
{
    "metric": "l2",
    "nlist": 1000,
    "nprobe": 10,
    "clusterSizes": [...],
}
```

---

## Complete Usage Example

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tahcohcat/gofaiss/pkg/index/hnsw"
    "github.com/tahcohcat/gofaiss/pkg/metric"
    "github.com/tahcohcat/gofaiss/pkg/storage"
    "github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
    // Create index
    dim := 128
    config := hnsw.Config{
        M:              16,
        EfConstruction: 200,
        EfSearch:       50,
    }
    
    idx, err := hnsw.New(dim, "l2", config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Generate and add vectors
    vectors := vector.GenerateRandom(10000, dim, 42)
    if err := idx.Add(vectors); err != nil {
        log.Fatal(err)
    }
    
    // Search
    query := make([]float32, dim)
    results, err := idx.Search(query, 10)
    if err != nil {
        log.Fatal(err)
    }
    
    // Print results
    for i, r := range results {
        fmt.Printf("%d: ID=%d, Dist=%.4f\n", i+1, r.ID, r.Distance)
    }
    
    // Save index
    if err := storage.SaveIndex(idx, "index.faiss.gz", true); err != nil {
        log.Fatal(err)
    }
    
    // Statistics
    stats := idx.Stats()
    fmt.Printf("\nStats:\n")
    fmt.Printf("Vectors: %d\n", stats.TotalVectors)
    fmt.Printf("Memory: %.2f MB\n", stats.MemoryUsageMB)
}
```

---

## Error Handling

All methods that can fail return errors. Common error scenarios:

- **Dimension mismatch**: Query or vector dimension doesn't match index
- **Untrained index**: Adding to IVF/PQ/IVFPQ before training
- **Invalid configuration**: Negative dimensions, invalid metrics
- **Not found**: Removing non-existent vector
- **I/O errors**: File save/load failures

Always check errors:

```go
if err := idx.Add(vectors); err != nil {
    log.Fatalf("Add failed: %v", err)
}
```

---

## Concurrency

All index types are thread-safe:

- Multiple goroutines can call `Search()` concurrently
- `Add()` and other write operations use write locks
- Read operations use read locks for better performance

**Example:**
```go
// Safe concurrent search
var wg sync.WaitGroup
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(query []float32) {
        defer wg.Done()
        results, _ := idx.Search(query, 10)
        // Process results
    }(queries[i])
}
wg.Wait()
```
