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

## Documentation

- **[Getting Started](./docs/getting-started.md)** - Installation, quick start, and usage examples
- **[API Reference](./docs/api.md)** - Complete API documentation for all packages
- **[Architecture](./docs/architecture.md)** - Design and implementation details
- **[Benchmarks](./docs/benchmarks.md)** - Performance comparisons and metrics
- **[Roadmap](./docs/roadmap.md)** - Future plans and features

## Benchmarks

**Dataset: 100K vectors, 128 dimensions**

| Index Type | Build Time | QPS | Memory | Recall@10 |
|------------|-----------|-----|---------|-----------|
| Flat       | 2ms       | 36 | 48.8 MB | 100% |
| HNSW       | 18,061ms  | 24,087 | 97.7 MB | 99.96% |
| IVF        | 4,572ms   | 343 | 49.0 MB | 26.1%* |
| PQ         | 4,034ms   | 50 | 1.65 MB | 19.7% |
| IVFPQ      | 8,506ms   | 516 | 1.04 MB | 4.0%* |

*Recall can be improved by tuning `nprobe` parameter (see [benchmarks](./docs/benchmarks.md))

See [detailed benchmarks](./docs/benchmarks.md) for more results, scaling analysis, and tuning guidelines.

## Examples

See the [examples](./example) directory for more detailed usage examples.

## License

MIT License

---
