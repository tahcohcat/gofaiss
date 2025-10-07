# GoFAISS

A pure Go implementation of Facebook's FAISS (Facebook AI Similarity Search) library for efficient similarity search and clustering of dense vectors.

## Overview

GoFAISS provides high-performance vector similarity search with multiple indexing strategies, distance metrics, and compression techniques. Built entirely in Go with zero external dependencies (except for benchmarking comparisons), it's designed for production use cases requiring fast nearest neighbor search.

## Key Features

### Multiple Index Types

- **FlatIndex**: Exact brute-force search with 100% recall
- **HNSWIndex**: Hierarchical Navigable Small World graphs for fastest approximate search
- **IVFIndex**: Inverted File Index with k-means clustering
- **PQIndex**: Product Quantization for memory-efficient storage
- **IVFPQIndex**: Combined IVF + PQ for balanced performance

### Distance Metrics

- **L2 (Euclidean)**: Standard distance metric for embeddings
- **Cosine Similarity**: Normalized vector comparison
- **Inner Product**: Dot product similarity

### Production Features

- **Persistence**: Save/load indexes with gzip compression
- **Serialization**: Both Gob and JSON format support
- **Concurrency**: Thread-safe operations
- **Memory Efficiency**: Product quantization achieves 32x compression
- **Benchmarking**: Built-in ANN benchmarking framework

## Performance Highlights

Based on 100K vectors with 128 dimensions:

| Index Type | QPS | Recall@10 | Memory | Build Time |
|------------|-----|-----------|--------|------------|
| HNSW | 24,087 | 98%+ | 97.7 MB | 18s |
| IVFPQ | 516 | 95% | 1.0 MB | 8.5s |
| IVF | 343 | 95% | 49.0 MB | 4.6s |
| PQ | 50 | 92% | 1.7 MB | 4.0s |
| Flat | 36 | 100% | 48.8 MB | 0.002s |

**Key Takeaways**:
- HNSW provides the fastest queries (24K QPS) with excellent recall
- IVFPQ offers the best memory/performance tradeoff (98% compression)
- All indexes are production-ready with consistent performance

## Quick Start

```go
package main

import (
    "github.com/tahcohcat/gofaiss/pkg/index/hnsw"
    "github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
    // Create HNSW index for 128-dimensional vectors
    idx, _ := hnsw.New(128, "l2", hnsw.Config{
        M:              16,
        EfConstruction: 200,
        EfSearch:       50,
    })

    // Add vectors
    vectors := []vector.Vector{
        {ID: 1, Data: make([]float32, 128)},
        {ID: 2, Data: make([]float32, 128)},
        // ... more vectors
    }
    idx.Add(vectors)

    // Search for k nearest neighbors
    query := make([]float32, 128)
    results, _ := idx.Search(query, 10)
    
    for _, result := range results {
        println("ID:", result.ID, "Distance:", result.Distance)
    }

    // Persist index
    idx.SaveToFile("vectors.faiss.gz")
}
```

## Installation

```bash
go get github.com/tahcohcat/gofaiss
```

**Requirements**: Go 1.21 or later

## Use Cases

GoFAISS is ideal for:

- **Semantic Search**: Find similar documents, images, or other embeddings
- **Recommendation Systems**: Product recommendations based on user embeddings
- **Deduplication**: Identify near-duplicate content
- **Clustering**: Group similar vectors together
- **Anomaly Detection**: Find outliers in high-dimensional spaces
- **RAG Systems**: Retrieval-augmented generation for LLM applications

## Comparison with Alternatives

### vs Python FAISS

- **Pros**: No Python dependency, native Go integration, smaller binary size
- **Cons**: Fewer index types (no GPU support yet)
- **Use When**: Building Go services, deploying to environments without Python

### vs hnswlib-go

- **Pros**: More index types, quantization support, better documentation
- **Performance**: 20-25% faster queries in benchmarks
- **Use When**: Need memory efficiency or multiple index strategies

### vs Pure Vector Databases

- **Pros**: Embeddable library, no separate service, simpler deployment
- **Cons**: No distributed search, no query language
- **Use When**: Single-node applications, embedded systems, simplicity matters

## Architecture

GoFAISS follows a modular architecture:

```
gofaiss/
├── pkg/
│   ├── metric/       # Distance calculations (L2, cosine, IP)
│   ├── index/        # Index implementations
│   │   ├── flat/     # Brute force search
│   │   ├── hnsw/     # Graph-based ANN
│   │   ├── ivf/      # Inverted file index
│   │   ├── pq/       # Product quantization
│   │   └── ivfpq/    # Combined IVF+PQ
│   ├── vector/       # Vector utilities
│   ├── storage/      # Persistence layer
│   └── search/       # High-level search API
├── internal/
│   ├── math/         # Optimized math operations
│   └── utils/        # Helper functions
└── cmd/
    ├── gofaiss-cli/     # Command-line interface
    └── gofaiss-server/  # HTTP server
```

## Project Status

**Current Version**: 0.1.0 (Alpha)

GoFAISS is under active development. The core functionality is stable and production-ready, but the API may change before v1.0.

**Production Readiness**:
- Core search algorithms
-  Persistence and serialization
- Comprehensive benchmarks
- Thread safety

**TODO**
- API stability (pre-v1.0)
- Distributed search
- GPU acceleration

## Contributing

Contributions are welcome! Please see our [GitHub repository](https://github.com/tahcohcat/gofaiss) for:

- Bug reports and feature requests
- Pull requests
- Documentation improvements
- Benchmark results from your environment

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Inspired by [Facebook's FAISS](https://github.com/facebookresearch/faiss)
- HNSW algorithm from [Malkov & Yashunin (2016)](https://arxiv.org/abs/1603.09320)
- Product Quantization from [Jégou et al. (2011)](https://ieeexplore.ieee.org/document/5432202)

## Next Steps

- [Getting Started Guide](getting-started.md) - Detailed installation and usage
- [Architecture Overview](architecture.md) - Deep dive into implementation
- [Benchmarks](benchmarks.md) - Comprehensive performance analysis
- [API Documentation](https://pkg.go.dev/github.com/tahcohcat/gofaiss) - Full API reference
