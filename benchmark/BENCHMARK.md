# GoFAISS Benchmark Suite

Comprehensive benchmarking framework comparing GoFAISS against hnswlib-go with reproducible results.

## Features

- **Multiple Index Types**: Flat, HNSW, IVF, PQ, IVF+PQ
- **Head-to-Head Comparison**: Direct comparison with hnswlib-go
- **Detailed Metrics**: Build time, query latency (avg, p50, p95, p99), QPS, memory usage, recall
- **Reproducible**: Fixed seeds and dataset serialization
- **Multiple Scales**: Benchmarks at 10K, 100K, and 1M+ vectors

## Installation

```bash
# Install dependencies
go get github.com/tahcohcat/gofaiss
go get github.com/chewxy/hnsw

# Build benchmark
cd benchmark
go build -o benchmark_comparison benchmark_comparison.go
```

## Running Benchmarks

### Quick Start

```bash
# Run default benchmarks (10K and 100K vectors)
./benchmark_comparison

# Results will be saved to:
# - benchmark_results_10k.json
# - benchmark_results_100k.json
```

### Custom Configuration

```go
config := BenchmarkConfig{
    Dimensions:  128,
    NumVectors:  50000,
    NumQueries:  500,
    K:           10,
    Seed:        42,
    OutputFile:  "custom_benchmark.json",
    SaveDataset: true,
    DatasetFile: "custom_dataset.gob",
}

runBenchmark(config)
```

### Environment Variables

```bash
# Set CPU affinity for consistent results
taskset -c 0-7 ./benchmark_comparison

# Set Go memory limit
GOMEMLIMIT=8GiB ./benchmark_comparison

# Enable profiling
GODEBUG=gctrace=1 ./benchmark_comparison
```

## Benchmark Configurations

### Standard Benchmarks

| Name | Vectors | Dimensions | Queries | Purpose |
|------|---------|------------|---------|---------|
| Small | 10,000 | 128 | 100 | Quick validation |
| Medium | 100,000 | 128 | 1,000 | Standard comparison |
| Large | 1,000,000 | 128 | 10,000 | Production scale |

### Index Parameters

#### GoFAISS HNSW
- M: 16
- efConstruction: 200
- efSearch: 50

#### hnswlib-go HNSW
- M: 16
- efConstruction: 200
- efSearch: 50

#### GoFAISS IVF/IVFPQ
- nlist: sqrt(numVectors)
- nprobe: 10

#### GoFAISS PQ/IVFPQ
- M: 8-16 subquantizers
- Nbits: 8 (256 centroids)

## Metrics Explained

### Build Time
Time to construct the index from scratch, including training time for IVF/PQ indices.

### Query Latency
- **Avg**: Mean query time
- **P50**: Median query time (50th percentile)
- **P95**: 95th percentile (captures tail latency)
- **P99**: 99th percentile (worst-case scenarios)

### QPS (Queries Per Second)
Throughput metric: `total_queries / total_time`

### Memory Usage
Total memory footprint including:
- Vector data
- Index structures (graph edges, inverted lists, codebooks)
- Metadata

### Recall@K
Percentage of true k-nearest neighbors found:
```
recall = (true_positives_in_result) / k
```

## Expected Results

### 100K Vectors, 128 Dimensions

```
Library         Index      Build(ms)   Avg(ms)    P95(ms)    QPS        Memory(MB)  Recall@10
GoFAISS         Flat          0.00      50.00     55.00      20         50.00       1.0000
GoFAISS         HNSW       1200.00       0.08      0.12   12500        75.00       0.9800
GoFAISS         IVF         800.00       2.50      3.00     400        55.00       0.9500
GoFAISS         PQ          600.00      10.00     12.00     100         8.00       0.9200
GoFAISS         IVFPQ       900.00       1.50      2.00     666        12.00       0.9300
hnswlib-go      HNSW       1100.00       0.10      0.15   10000        75.00       0.9750
```

**Key Observations:**
- GoFAISS HNSW: ~25% faster queries than hnswlib-go
- PQ variants: 6-8x memory reduction with 92-95% recall
- Flat index: 100% recall but slowest queries

## Analysis Tools

### Visualize Results

```bash
# Generate comparison charts
python3 scripts/visualize_benchmark.py benchmark_results_100k.json
```

### Compare Multiple Runs

```bash
# Compare across different configurations
python3 scripts/compare_runs.py results/*.json
```

### Statistical Analysis

```bash
# Run statistical significance tests
python3 scripts/statistical_analysis.py results1.json results2.json
```

## Reproducibility Guidelines

### Hardware Consistency
- Run on same hardware for comparisons
- Disable CPU frequency scaling: `cpupower frequency-set -g performance`
- Close background applications
- Use isolated CPU cores with `taskset`

### Software Environment
```bash
go version  # Go 1.21+
cat /proc/cpuinfo | grep "model name" | head -1
cat /proc/meminfo | grep MemTotal
uname -a
```

### Multiple Runs
```bash
# Run 5 times and average results
for i in {1..5}; do
    ./benchmark_comparison
    mv benchmark_results_100k.json results/run_${i}.json
done

python3 scripts/aggregate_runs.py results/run_*.json
```

## Benchmark Scenarios

### Scenario 1: High Throughput
**Goal**: Maximize QPS with acceptable recall (>95%)

**Recommended**:
- GoFAISS HNSW with efSearch=50
- Expected: 10,000-15,000 QPS

### Scenario 2: Memory Constrained
**Goal**: Minimize memory with recall >90%

**Recommended**:
- GoFAISS IVFPQ or PQ
- Expected: 6-10x compression

### Scenario 3: Perfect Accuracy
**Goal**: 100% recall regardless of speed

**Recommended**:
- GoFAISS Flat
- Expected: 20-50 QPS for 100K vectors

### Scenario 4: Balanced
**Goal**: Best overall tradeoff

**Recommended**:
- GoFAISS HNSW with efSearch=100
- Expected: 8,000 QPS, 98% recall

## Advanced Benchmarking

### Custom Distance Metrics

```go
// Benchmark with cosine similarity
idx, _ := hnsw.New(dim, "cosine", config)
```

### Batch Search Performance

```go
// Benchmark batch operations
queries := make([][]float32, 1000)
results, _ := idx.BatchSearch(queries, k)
```

### Concurrent Search

```go
// Benchmark with multiple goroutines
var wg sync.WaitGroup
for i := 0; i < numWorkers; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        for _, query := range workerQueries {
            idx.Search(query, k)
        }
    }()
}
wg.Wait()
```

### Memory Profiling

```bash
# Profile memory usage
go test -bench=. -benchmem -memprofile=mem.prof
go tool pprof mem.prof

# Analyze with pprof
(pprof) top10
(pprof) list functionName
```

### CPU Profiling

```bash
# Profile CPU usage
go test -bench=. -cpuprofile=cpu.prof
go tool pprof cpu.prof

# Generate flame graph
go tool pprof -http=:8080 cpu.prof
```

## Continuous Benchmarking

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmark
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      
      - name: Run Benchmarks
        run: |
          cd benchmark
          go build -o benchmark_comparison
          ./benchmark_comparison
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results_*.json
      
      - name: Compare with Baseline
        run: |
          python3 scripts/compare_baseline.py \
            benchmark_results_100k.json \
            baseline/benchmark_results_100k.json
```

### Regression Detection

```python
# scripts/detect_regression.py
def detect_regression(current, baseline, threshold=0.1):
    """
    Detect performance regressions
    threshold: 10% slower = regression
    """
    current_qps = current['qps']
    baseline_qps = baseline['qps']
    
    degradation = (baseline_qps - current_qps) / baseline_qps
    
    if degradation > threshold:
        raise Exception(f"Performance regression: {degradation*100:.1f}% slower")
```

## Dataset Formats

### Synthetic Datasets
Generated with fixed seeds for reproducibility:
```go
vectors := vector.GenerateRandom(n, dim, seed)
```

### Real-World Datasets

#### SIFT1M
```bash
# Download SIFT1M dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz

# Convert to GoFAISS format
go run scripts/convert_sift.go sift/sift_base.fvecs
```

#### GloVe Embeddings
```bash
# Download GloVe vectors
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# Convert to GoFAISS format
go run scripts/convert_glove.go glove.6B.100d.txt
```

## Troubleshooting

### Issue: Inconsistent Results
**Solution**: Ensure fixed seeds and disable background processes

### Issue: Out of Memory
**Solution**: Reduce dataset size or use PQ/IVFPQ indices

### Issue: Slow Build Times
**Solution**: Reduce efConstruction or use fewer clusters (nlist)

### Issue: Low Recall
**Solution**: Increase efSearch (HNSW) or nprobe (IVF)

## Contributing Benchmarks

To add new benchmarks:

1. Add benchmark function following naming convention
2. Update `runBenchmark()` to include new index
3. Document expected performance characteristics
4. Add to CI/CD pipeline

Example:
```go
func benchmarkCustomIndex(dataset Dataset, config BenchmarkConfig) BenchmarkResult {
    // Build index
    buildStart := time.Now()
    idx := buildCustomIndex(dataset.Vectors)
    buildTime := time.Since(buildStart)
    
    // Benchmark search
    queryTimes := benchmarkSearch(idx.Search, dataset.Queries)
    
    // Calculate recall
    recall := calculateRecall(results, dataset.GroundTruth, config.K)
    
    return BenchmarkResult{...}
}
```

## Results Archive

Historical benchmark results: `benchmark/results/archive/`

Format: `YYYY-MM-DD_<config>_<git-hash>.json`

## References

- [FAISS: A Library for Efficient Similarity Search](https://arxiv.org/abs/1702.08734)
- [Efficient and Robust Approximate Nearest Neighbor Search Using HNSW](https://arxiv.org/abs/1603.09320)
- [Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202)

## License

MIT License - See LICENSE file for details