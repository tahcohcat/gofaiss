# Getting Started

This guide will help you get started with GoFAISS, a pure Go implementation of Facebook's FAISS library for efficient similarity search and clustering of dense vectors.

## Installation

Install GoFAISS using `go get`:

```bash
go get github.com/tahcohcat/gofaiss
```

## Core Concepts

### Vectors

GoFAISS works with dense vectors represented as `[]float32`. Each vector can have an associated ID and is stored in the `vector.Vector` struct:

```go
type Vector struct {
    ID   int64
    Data []float32
    Norm float32 // Precomputed norm for cosine similarity
}
```

### Distance Metrics

GoFAISS supports three distance metrics:

- **L2 (Euclidean)**: `"l2"` - Standard Euclidean distance
- **Cosine Similarity**: `"cosine"` - Measures angular distance
- **Inner Product**: `"dot"` - Dot product similarity

### Index Types

GoFAISS provides several index types optimized for different use cases:

| Index Type | Speed | Memory | Accuracy | Use Case |
|------------|-------|--------|----------|----------|
| **Flat** | Slow | High | 100% | Exact search, small datasets |
| **HNSW** | Fast | High | ~98% | Fast approximate search |
| **IVF** | Medium | Medium | ~95% | Balanced speed/memory |
| **PQ** | Medium | Low | ~90% | Memory-constrained environments |
| **IVFPQ** | Fast | Low | ~93% | Large-scale with compression |

## Quick Start Examples

### 1. Flat Index (Exact Search)

Best for small datasets where you need 100% accuracy:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tahcohcat/gofaiss/pkg/index/flat"
    "github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
    // Create index with 128-dimensional vectors using L2 distance
    dim := 128
    idx, err := flat.New(dim, "l2")
    if err != nil {
        log.Fatal(err)
    }
    
    // Generate random vectors for testing
    vectors := vector.GenerateRandom(1000, dim, 42)
    
    // Add vectors to index
    if err := idx.Add(vectors); err != nil {
        log.Fatal(err)
    }
    
    // Search for 10 nearest neighbors
    query := make([]float32, dim)
    results, err := idx.Search(query, 10)
    if err != nil {
        log.Fatal(err)
    }
    
    // Print results
    for i, r := range results {
        fmt.Printf("%d: ID=%d, Distance=%.4f\n", i+1, r.ID, r.Distance)
    }
}
```

### 2. HNSW Index (Fast Approximate Search)

Best for fast similarity search with high accuracy:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tahcohcat/gofaiss/pkg/index/hnsw"
    "github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
    dim := 128
    
    // Configure HNSW parameters
    config := hnsw.Config{
        M:              16,  // Number of connections per layer
        EfConstruction: 200, // Quality during build (higher = better)
        EfSearch:       50,  // Quality during search (higher = better)
    }
    
    // Create HNSW index
    idx, err := hnsw.New(dim, "l2", config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Add vectors
    vectors := vector.GenerateRandom(10000, dim, 42)
    if err := idx.Add(vectors); err != nil {
        log.Fatal(err)
    }
    
    // Adjust search quality at runtime
    idx.SetEfSearch(100) // Higher = better accuracy, slower
    
    // Search
    query := vectors[0].Data
    results, err := idx.Search(query, 10)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Found %d neighbors\n", len(results))
}
```

### 3. Product Quantization (Memory Efficient)

Best for large datasets with memory constraints:

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/tahcohcat/gofaiss/pkg/index/pq"
    "github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
    dim := 128
    
    // Configure PQ
    config := pq.Config{
        M:     16, // Number of subquantizers (must divide dim evenly)
        Nbits: 8,  // Bits per code (8 = 256 centroids)
    }
    
    idx, err := pq.NewIndex(dim, config)
    if err != nil {
        log.Fatal(err)
    }
    
    // PQ requires training
    trainingVectors := vector.GenerateRandom(5000, dim, 42)
    if err := idx.Train(trainingVectors); err != nil {
        log.Fatal(err)
    }
    
    // Add vectors (automatically compressed)
    vectors := vector.GenerateRandom(100000, dim, 43)
    if err := idx.Add(vectors); err != nil {
        log.Fatal(err)
    }
    
    // Search
    query := vectors[0].Data
    results, err := idx.Search(query, 10)
    if err != nil {
        log.Fatal(err)
    }
    
    // Check compression stats
    stats := idx.Stats()
    fmt.Printf("Memory: %.2f MB\n", stats.MemoryUsageMB)
    fmt.Printf("Compression: ~%.1fx\n",
        float64(len(vectors)*dim*4)/(stats.MemoryUsageMB*1024*1024))
}
```

### 4. IVF Index (Inverted File)

Best for balanced speed and memory usage:

```go
package main

import (
    "log"
    
    "github.com/tahcohcat/gofaiss/pkg/index/ivf"
    "github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
    dim := 128
    numVectors := 100000
    
    // Configure IVF
    config := ivf.Config{
        Metric: "l2",
        Nlist:  1000, // Number of clusters (typically sqrt(N))
    }
    
    idx, err := ivf.New(dim, "l2", config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Train clustering
    trainingVectors := vector.GenerateRandom(10000, dim, 42)
    if err := idx.Train(trainingVectors); err != nil {
        log.Fatal(err)
    }
    
    // Add vectors
    vectors := vector.GenerateRandom(numVectors, dim, 43)
    if err := idx.Add(vectors); err != nil {
        log.Fatal(err)
    }
    
    // Set search parameters
    idx.SetNProbe(10) // Number of clusters to search (higher = better)
    
    // Search
    query := vectors[0].Data
    results, _ := idx.Search(query, 10)
    log.Printf("Found %d results\n", len(results))
}
```

## Batch Operations

All indexes support batch search for better performance:

```go
// Prepare multiple queries
queries := make([][]float32, 100)
for i := 0; i < 100; i++ {
    queries[i] = make([]float32, dim)
}

// Batch search
results, err := idx.BatchSearch(queries, 10)
if err != nil {
    log.Fatal(err)
}

// results is [][]vector.SearchResult
for i, queryResults := range results {
    fmt.Printf("Query %d: found %d neighbors\n", i, len(queryResults))
}
```

## Persistence

Save and load indexes to disk:

```go
import "github.com/tahcohcat/gofaiss/pkg/storage"

// Save with gzip compression
err := storage.SaveIndex(idx, "index.faiss.gz", true)
if err != nil {
    log.Fatal(err)
}

// Load from disk
loadedIdx := &flat.Index{} // Use appropriate type
err = storage.LoadIndex(loadedIdx, "index.faiss.gz", true)
if err != nil {
    log.Fatal(err)
}
```

## Index Statistics

Get information about your index:

```go
stats := idx.Stats()
fmt.Printf("Vectors: %d\n", stats.TotalVectors)
fmt.Printf("Dimension: %d\n", stats.Dimension)
fmt.Printf("Memory: %.2f MB\n", stats.MemoryUsageMB)
fmt.Printf("Index Type: %s\n", stats.IndexType)

// HNSW-specific info
if info, ok := stats.ExtraInfo["maxLevel"]; ok {
    fmt.Printf("Max Level: %v\n", info)
}
```

## Performance Tips

### Choosing an Index Type

1. **< 10K vectors**: Use `Flat` for exact results
2. **10K - 1M vectors**: Use `HNSW` for speed or `IVF` for balance
3. **> 1M vectors**: Use `IVFPQ` for memory efficiency
4. **Need exact search**: Use `Flat` regardless of size

### HNSW Tuning

- **M**: Higher = better recall, more memory (typical: 16-64)
- **efConstruction**: Higher = better quality, slower build (typical: 100-500)
- **efSearch**: Adjust at runtime for speed/accuracy tradeoff

### IVF Tuning

- **nlist**: Typically `sqrt(num_vectors)`, range: [100, 65536]
- **nprobe**: Higher = better recall, slower search (typical: 1-20)

### PQ Tuning

- **M**: Higher = better accuracy, more computation (typical: 8-64)
- **nbits**: Higher = better accuracy, more memory (typical: 8)

## Common Patterns

### Cosine Similarity Search

```go
idx, err := hnsw.New(dim, "cosine", hnsw.DefaultConfig())
```

### Custom Vector IDs

```go
vectors := []vector.Vector{
    {ID: 1001, Data: vec1},
    {ID: 1002, Data: vec2},
}
idx.Add(vectors)
```

### Removing Vectors (HNSW only)

```go
err := idx.Remove(vectorID)
```

### Generate Test Data

```go
// Random vectors with fixed seed for reproducibility
vectors := vector.GenerateRandom(1000, 128, 42)

// Normalize for cosine similarity
for i := range vectors {
    vector.NormalizeInPlace(vectors[i].Data)
}
```

## Error Handling

GoFAISS returns errors for:

- Dimension mismatches
- Untrained indexes (IVF, PQ, IVFPQ)
- Invalid configurations
- File I/O errors

Always check errors:

```go
if err := idx.Add(vectors); err != nil {
    log.Fatalf("Failed to add vectors: %v", err)
}
```

## Next Steps

- Read the [API Reference](api.md) for detailed documentation
- Check the [Architecture](architecture.md) guide to understand internals
- See [Benchmarks](benchmarks.md) for performance comparisons
- Review the [examples](https://github.com/tahcohcat/gofaiss/tree/main/example) directory

## Complete Example

Here's a complete working example combining multiple concepts:

```go
package main

import (
    "fmt"
    "log"
    "time"
    
    "github.com/tahcohcat/gofaiss/pkg/index/hnsw"
    "github.com/tahcohcat/gofaiss/pkg/storage"
    "github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
    dim := 128
    numVectors := 50000
    
    // Create index
    config := hnsw.DefaultConfig()
    idx, err := hnsw.New(dim, "l2", config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Generate and add vectors
    fmt.Println("Generating vectors...")
    vectors := vector.GenerateRandom(numVectors, dim, 42)
    
    fmt.Println("Building index...")
    start := time.Now()
    if err := idx.Add(vectors); err != nil {
        log.Fatal(err)
    }
    buildTime := time.Since(start)
    
    // Benchmark search
    fmt.Println("Benchmarking search...")
    queries := vector.GenerateRandom(100, dim, 43)
    
    start = time.Now()
    for _, q := range queries {
        _, err := idx.Search(q.Data, 10)
        if err != nil {
            log.Fatal(err)
        }
    }
    searchTime := time.Since(start)
    
    // Print stats
    stats := idx.Stats()
    fmt.Printf("\n=== Results ===\n")
    fmt.Printf("Vectors: %d\n", stats.TotalVectors)
    fmt.Printf("Build time: %v\n", buildTime)
    fmt.Printf("Search time: %v (%.2f ms/query)\n",
        searchTime, float64(searchTime.Milliseconds())/100.0)
    fmt.Printf("Memory: %.2f MB\n", stats.MemoryUsageMB)
    
    // Save index
    fmt.Println("\nSaving index...")
    if err := storage.SaveIndex(idx, "my_index.faiss.gz", true); err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Done!")
}
```

## 🍔🐼 Demo

* Run the included [FoodPanda](../examples/kaggle_foodpanda_reviews/demo.md) demo to see a real world example using the library