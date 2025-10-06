# GoFAISS - FAISS-like Vector Search in Go

A pure Go implementation of Facebook's FAISS (Facebook AI Similarity Search) library for efficient similarity search and clustering of dense vectors.

## Features

- **Multiple Index Types**
  - `FlatIndex`: Exact brute-force search
  - `IVFIndex`: Inverted File Index with clustering
  - `IVFPQIndex`: IVF + Product Quantization (memory efficient)
  - `HNSWIndex`: Hierarchical Navigable Small World graphs (fastest)

- **Distance Metrics**
  - L2 (Euclidean distance)
  - Cosine similarity
  - Inner product

- **Persistence**
  - Save/Load indexes with gzip compression
  - Gob and JSON serialization support

- **Benchmarking**
  - Built-in ANN benchmarking framework
  - Recall@K, QPS, latency measurements
  - Synthetic dataset generation

## Installation

```bash
go get github.com/tahcohcat/gofaiss
```

## Quick Start

```go
package main

import (
    "github.com/tahcohcat/gofaiss/pkg/index/hnsw"
    "github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
    // Create HNSW index
    idx, _ := hnsw.New(128, "l2", hnsw.Config{
        M:              16,
        EfConstruction: 200,
    })

    // Add vectors
    vectors := []vector.Vector{
        {ID: 1, Data: make([]float32, 128)},
        {ID: 2, Data: make([]float32, 128)},
    }
    idx.Add(vectors)

    // Search
    query := make([]float32, 128)
    results, _ := idx.Search(query, 10)
    
    // Save index
    idx.SaveToFile("index.faiss.gz")
}
```

## Package Structure

```
gofaiss/
├── pkg/
│   ├── metric/       # Distance metrics
│   ├── index/        # Index implementations
│   ├── vector/       # Vector utilities
│   ├── storage/      # Serialization
│   └── search/       # Search API
├── internal/
│   ├── math/         # Low-level math
│   ├── storage/      # Internal storage
│   └── utils/        # Helpers
├── cmd/
│   ├── gofaiss-cli/     # CLI tool
│   └── gofaiss-server/  # Server
└── examples/         # Examples
```

## Benchmarks

| Index Type | Build Time | QPS | Memory | Recall@10 |
|------------|-----------|-----|---------|-----------|
| Flat       | 0ms       | 1,000 | 100% | 100% |
| HNSW       | 1,200ms   | 15,000 | 150% | 98% |
| IVF+PQ     | 800ms     | 8,000 | 15% | 95% |

## Examples

See the [examples](./examples) directory for more detailed usage examples.

## License

MIT License

---
