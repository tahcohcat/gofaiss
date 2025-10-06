# GoFAISS - Complete Project Structure

A production-ready, FAISS-like vector similarity search library in pure Go.

##  Project Structure

```
gofaiss/
├── go.mod                          # Go module definition
├── README.md                       # Main documentation
├── PROJECT_STRUCTURE.md            # This file
│
├── cmd/                            # Command-line tools
│   ├── gofaiss-cli/               # CLI for benchmarking and demos
│   │   └── main.go
│   └── gofaiss-server/            # Optional REST/gRPC server (future)
│       └── main.go
│
├── pkg/                            # Public API packages
│   ├── vector/                    # Vector utilities
│   │   └── vector.go              # Vector types, operations, generation
│   │
│   ├── metric/                    # Distance metrics
│   │   ├── metric.go              # Metric interface & factory
│   │   ├── l2.go                  # L2 (Euclidean) distance
│   │   ├── cosine.go              # Cosine similarity
│   │   └── dot.go                 # Inner product
│   │
│   ├── index/                     # Index implementations
│   │   ├── index.go               # Index interface definitions
│   │   │
│   │   ├── flat/                  # Flat (brute-force) index
│   │   │   └── flat.go
│   │   │
│   │   ├── hnsw/                  # HNSW graph index
│   │   │   ├── hnsw.go           # Main implementation
│   │   │   ├── node.go           # Graph node structure
│   │   │   └── graph.go          # Graph operations
│   │   │
│   │   ├── ivf/                   # Inverted file index
│   │   │   └── ivf.go
│   │   │
│   │   ├── pq/                    # Product quantization
│   │   │   └── pq.go
│   │   │
│   │   └── ivfpq/                 # IVF + PQ combined
│   │       └── ivfpq.go
│   │
│   ├── storage/                   # Serialization
│   │   ├── serialize.go           # Generic serialization
│   │   ├── gob.go                # Gob format
│   │   └── json.go               # JSON format (future)
│   │
│   └── search/                    # Search API
│       └── search.go              # Unified search interface
│
├── internal/                      # Internal (non-exported) packages
│   ├── math/                      # Low-level math operations
│   │   ├── simd.go               # SIMD-optimized operations
│   │   └── distance.go           # Distance computation helpers
│   │
│   ├── storage/                   # Internal storage utilities
│   │   ├── mmap.go               # Memory-mapped files
│   │   └── buffer.go             # Buffer management
│   │
│   └── utils/                     # Helper utilities
│       ├── logging.go            # Logging utilities
│       ├── config.go             # Configuration helpers
│       └── rand.go               # Random number generation
│
├── examples/                      # Usage examples
│   ├── basic_flat.go             # Simple flat index example
│   ├── hnsw_search.go            # HNSW index example
│   ├── ivf_demo.go               # IVF index example
│   ├── pq_compression.go         # PQ compression example
│   └── benchmark.go              # Benchmarking example
│
├── test/                          # Integration tests
│   ├── integration_test.go
│   └── benchmark_test.go
│
└── docs/                          # Additional documentation
    ├── API.md                     # API documentation
    ├── PERFORMANCE.md             # Performance guide
    └── ALGORITHMS.md              # Algorithm explanations
```

##  Key Components

### Core Packages

1. **pkg/vector** - Vector operations and utilities
   - `Vector` struct with ID and data
   - `SearchResult` for query results
   - Vector operations (normalize, dot, add, etc.)
   - Random vector generation for testing

2. **pkg/metric** - Distance metrics
   - `Metric` interface for pluggable distance functions
   - L2 (Euclidean distance)
   - Cosine distance
   - Dot product (MIPS)

3. **pkg/index** - Index implementations
   - Common `Index` interface
   - `TrainableIndex` for indexes requiring training
   - `BatchIndex` for batch operations
   - `SerializableIndex` for persistence

### Index Types

| Index Type | Speed | Memory | Accuracy | Use Case |
|------------|-------|--------|----------|----------|
| **Flat** | Slow | High | 100% | Small datasets, baseline |
| **HNSW** | Very Fast | Medium | 95-99% | Real-time search |
| **IVF** | Fast | Medium | 90-95% | Large datasets |
| **PQ** | Fast | Very Low | 85-95% | Memory constrained |
| **IVF+PQ** | Fast | Low | 90-95% | Production (most common) |

##  Quick Start

### Installation

```bash
go get github.com/tahcohcat/gofaiss
```

### Basic Usage

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
    vectors := vector.GenerateRandom(10000, 128, 42)
    idx.Add(vectors)

    // Search
    query := vectors[0].Data
    results, _ := idx.Search(query, 10)
}
```

##  Benchmarks

Run benchmarks with the CLI:

```bash
# Benchmark HNSW
go run cmd/gofaiss-cli/main.go bench -type hnsw -vectors 100000 -dim 128

# Benchmark all types
for type in flat hnsw pq; do
    go run cmd/gofaiss-cli/main.go bench -type $type -vectors 10000
done
```

### Expected Performance (on modern CPU)

**Dataset: 100K vectors, 128D, L2 metric**

| Index | Build Time | QPS | Memory | Recall@10 |
|-------|-----------|-----|---------|-----------|
| Flat | 0ms | 1,200 | 49 MB | 100% |
| HNSW | 15s | 18,000 | 78 MB | 98% |
| PQ | 2s | 12,000 | 8 MB | 92% |
| IVF+PQ | 3s | 15,000 | 12 MB | 94% |

##  Configuration Guidelines

### HNSW Configuration

```go
// Small dataset (<10K vectors)
Config{M: 12, EfConstruction: 100, EfSearch: 50}

// Medium dataset (10K-100K)
Config{M: 16, EfConstruction: 200, EfSearch: 100}

// Large dataset (>100K)
Config{M: 32, EfConstruction: 400, EfSearch: 200}

// Quality vs Speed
// Higher M = better accuracy, more memory
// Higher EfConstruction = better build quality, slower build
// Higher EfSearch = better accuracy, slower search
```

### PQ Configuration

```go
// Standard compression (32x)
Config{M: 16, Nbits: 8}  // 128D -> 16 bytes

// High compression (64x)
Config{M: 32, Nbits: 8}  // 128D -> 32 bytes (overkill)

// Balanced
Config{M: 8, Nbits: 8}   // 128D -> 8 bytes
```

### IVF Configuration

```go
// Rule of thumb: nlist = sqrt(num_vectors)
numVectors := 1000000
nlist := int(math.Sqrt(float64(numVectors)))  // ~1000

// Search nprobe (clusters to search)
// nprobe=1:  fastest, lowest accuracy
// nprobe=10: balanced
// nprobe=nlist: exhaustive (= flat search)
```

##  Testing

```bash
# Run all tests
go test ./...

# Run with benchmarks
go test -bench=. ./...

# Run specific package
go test ./pkg/index/hnsw/

# With coverage
go test -cover ./...
```

##  Development Workflow

1. **Add new index type:**
   - Create package in `pkg/index/<name>/`
   - Implement `Index` interface
   - Add tests
   - Update documentation

2. **Add new metric:**
   - Add to `pkg/metric/`
   - Implement `Metric` interface
   - Add to factory in `metric.go`

3. **Optimize performance:**
   - Profile with `pprof`
   - Add SIMD operations in `internal/math/`
   - Benchmark before/after

##  Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file

## 🔗 References

- [FAISS Paper](https://arxiv.org/abs/1702.08734)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [Product Quantization Paper](https://hal.inria.fr/inria-00514462v2/document)

##  Contact

- GitHub Issues: For bugs and feature requests
- Discussions: For questions and community chat
