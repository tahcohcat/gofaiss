# 🧠gofaiss

> **A Go-native, dependency-free vector search engine inspired by FAISS.**  
> Lightweight, portable, and built entirely in Go — no C++, no Python, no headaches.

---

##  Overview

**gofaiss** brings vector similarity search to the Go ecosystem — natively.  
It supports multiple index types (Flat, IVF, IVFPQ, HNSW), flexible metrics (L2, cosine, inner product), and built-in persistence.

### Why Go-native?

Most vector libraries in Go rely on C++ bindings (FAISS, hnswlib).  
gofaiss avoids that — making it:
- **Portable** across systems
- **Easy to build and deploy**
- **Deterministic** and reproducible

---

## Key Features

- **No dependencies** — pure Go implementation
- **Multiple index types** — Flat, IVF, IVFPQ, HNSW
- **Serialization** — gob, JSON, gzip
- **Benchmarks included**
- **Pluggable metrics and indexes**

---

## Installation

```bash
go get github.com/tahcohcat/gofaiss


### Quick Example

```go
package main

import (
    "fmt"
    "github.com/tahcohcat/gofaiss/index"
    "github.com/tahcohcat/gofaiss/metric"
)

func main() {
    // Create a 4D flat index using L2 distance
    idx := index.NewFlat(4, metric.L2{})

    // Add some vectors
    vectors := [][]float32{
        {1, 2, 3, 4},
        {2, 3, 4, 5},
        {10, 11, 12, 13},
    }
    idx.Add(vectors)

    // Search for nearest neighbors
    query := []float32{1.5, 2.5, 3.5, 4.5}
    distances, ids := idx.Search(query, 2)

    fmt.Println("IDs:", ids)
    fmt.Println("Distances:", distances)
}
```

## Benchmarks

| Dataset             | Index   | k  | Time (ms) | Recall | Notes       |
| ------------------- | ------- | -- | --------- | ------ | ----------- |
| 10K vectors (128D)  | Flat    | 10 | 12.4      | 1.00   | Full scan   |
| 10K vectors (128D)  | IVF(16) | 10 | 3.2       | 0.95   | 4× faster   |
| 100K vectors (128D) | HNSW    | 10 | 1.8       | 0.92   | Scales well |

### Running local benchmarks

```bash
cd benchmark
make benchmark-quick
make benchmark-10k
make benchmkar-100k
```

### Running tests

```bash
go test ./... -v
```