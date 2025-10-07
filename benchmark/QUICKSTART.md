# GoFAISS Benchmark Quick Start Guide

This guide will help you quickly set up and run reproducible benchmarks comparing GoFAISS against hnswlib-go.

## Prerequisites

```bash
# Go 1.21 or later
go version

# Python 3.8+ (for visualization)
python3 --version

# Optional: jq for JSON processing
sudo apt-get install jq  # Ubuntu/Debian
brew install jq          # macOS
```

## Installation

```bash
# Clone the repository
git clone https://github.com/tahcohcat/gofaiss
cd gofaiss/benchmark

# Install Go dependencies
go mod download

# Install Python dependencies
pip3 install matplotlib numpy

# Or use make
make install-deps
```

## Quick Start (3 minutes)

### 1. Run Single Benchmark

```bash
# Build and run benchmark once
make benchmark-quick
```

This will:
- Build the benchmark binary
- Run benchmarks on 10K and 100K vector datasets
- Save results to `benchmark_results_*.json`

### 2. View Results

```bash
# Generate visualizations
make visualize
```

This creates charts in `benchmark_results_100k_charts/`:
- `qps_comparison.png` - Query throughput comparison
- `recall_vs_qps.png` - Accuracy vs speed tradeoff
- `memory_usage.png` - Memory footprint by index type
- `latency_percentiles.png` - Latency distribution
- `results_table.md` - Detailed metrics table

## Full Benchmark Suite (10 minutes)

For statistically valid results with confidence intervals:

```bash
# Run 5 iterations with statistics
make benchmark-full
```

This will:
1. Run 5 benchmark iterations
2. Compute mean, std dev, and 95% confidence intervals
3. Generate aggregated results
4. Create visualizations
5. Produce a comprehensive report

Results are saved to `results/YYYYMMDD_HHMMSS/`

## Understanding the Results

### Key Metrics

**QPS (Queries Per Second)**: Higher is better
- Measures throughput
- GoFAISS HNSW typically: 10,000-15,000 QPS
- hnswlib-go HNSW typically: 8,000-12,000 QPS

**Recall@10**: Higher is better (1.0 = perfect)
- Percentage of true nearest neighbors found
- Flat index: 1.0 (100% accurate)
- HNSW: ~0.98 (98% accurate)
- IVF/PQ: 0.92-0.95 (92-95% accurate)

**Memory (MB)**: Lower is better
- Total memory footprint
- Flat: 100% of vector data
- HNSW: ~150% (includes graph)
- PQ: ~10-15% (compressed)

**Latency Percentiles**:
- P50 (median): Typical query time
- P95: 95% of queries complete by this time
- P99: Worst-case scenarios (tail latency)

### Example Output

```
Library         Index      Build(ms)   Avg(ms)    P95(ms)    QPS        Memory(MB)  Recall@10
GoFAISS         Flat          0.00      50.00     55.00      20         50.00       1.0000
GoFAISS         HNSW       1200.00       0.08      0.12   12500        75.00       0.9800
GoFAISS         IVF         800.00       2.50      3.00     400        55.00       0.9500
GoFAISS         PQ          600.00      10.00     12.00     100         8.00       0.9200
GoFAISS         IVFPQ       900.00       1.50      2.00     666        12.00       0.9300
hnswlib-go      HNSW       1100.00       0.10      0.15   10000        75.00       0.9750
```

**Key Observations**:
- GoFAISS HNSW: 25% faster queries than hnswlib-go
- Similar recall performance (~98%)
- PQ variants offer 6-8x memory savings
- IVFPQ provides best memory/accuracy tradeoff

## Common Workflows

### Workflow 1: Quick Comparison

```bash
# Single run comparison
make benchmark-quick visualize
```

Open `benchmark_results_100k_charts/results_table.md` to see the comparison.

### Workflow 2: Regression Testing

```bash
# Set baseline (first time)
make benchmark-quick set-baseline

# Later, after code changes
make benchmark-quick compare
```

If regressions are detected, you'll see:
```
✗ CRITICAL REGRESSIONS DETECTED!
• GoFAISS HNSW:
  - QPS: 10000 vs 12000 (-16.7%)
```

### Workflow 3: Validate Reproducibility

```bash
# Run validation test
make validate
```

This runs 3 consecutive benchmarks and reports variance. Look for:
- CV (Coefficient of Variation) < 0.05: Excellent
- CV < 0.10: Good
- CV > 0.10: High variability (run more iterations)

### Workflow 4: Performance Tuning

```bash
# Run with CPU performance mode
make benchmark-perf

# Profile CPU usage
make profile-cpu

# Profile memory usage
make profile-mem
```

## Customizing Benchmarks

### Modify Dataset Size

Edit `benchmark_comparison.go`:

```go
configs := []BenchmarkConfig{
    {
        Dimensions:  128,
        NumVectors:  50000,  // Change this
        NumQueries:  500,    // And this
        K:           10,
        Seed:        42,
        OutputFile:  "custom_results.json",
    },
}
```

### Adjust Index Parameters

```go
// HNSW configuration
hnswConfig := gofaiss_hnsw.Config{
    M:              32,    // More connections = better recall, more memory
    EfConstruction: 400,   // Higher = better quality, slower build
    EfSearch:       100,   // Higher = better recall, slower search
}

// IVF configuration
ivfConfig := ivf.Config{
    Nlist: 200,  // More clusters = faster search, may reduce recall
}

// Search with different nprobe
results, _ := idx.Search(query, k, 20)  // Probe more clusters
```

### Add Custom Index

```go
func benchmarkCustomIndex(dataset Dataset, config BenchmarkConfig) BenchmarkResult {
    // Implement your benchmark
    buildStart := time.Now()
    idx := buildYourIndex(dataset.Vectors)
    buildTime := time.Since(buildStart)
    
    queryTimes := benchmarkSearch(idx.Search, dataset.Queries)
    recall := calculateRecall(results, dataset.GroundTruth, config.K)
    
    return BenchmarkResult{
        Library:     "GoFAISS",
        IndexType:   "CustomIndex",
        BuildTimeMs: float64(buildTime.Milliseconds()),
        // ... fill in metrics
    }
}
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/benchmark.yml`:

```yaml
name: Performance Benchmark

on: [pull_request]

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
          make ci
      
      - name: Check for Regressions
        run: |
          cd benchmark
          python3 scripts/compare_baseline.py \
            ci-results/benchmark_results_100k.json \
            baseline/benchmark_results_100k.json
      
      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark/ci-results/
```

### GitLab CI

Create `.gitlab-ci.yml`:

```yaml
benchmark:
  stage: test
  script:
    - cd benchmark
    - make benchmark-quick
    - make compare
  artifacts:
    paths:
      - benchmark/benchmark_results_*.json
      - benchmark/*_charts/
    when: always
```

## Troubleshooting

### Issue: Inconsistent Results

**Symptom**: High variance between runs (CV > 0.10)

**Solutions**:
1. Disable CPU frequency scaling:
   ```bash
   sudo cpupower frequency-set -g performance
   ```

2. Close background applications

3. Pin to specific CPU cores:
   ```bash
   taskset -c 0-7 ./benchmark_comparison
   ```

4. Run more iterations:
   ```bash
   # Edit run_benchmark.sh
   NUM_RUNS=10  # Instead of 5
   ```

### Issue: Out of Memory

**Symptom**: Process killed during benchmark

**Solutions**:
1. Reduce dataset size:
   ```go
   NumVectors: 10000,  // Instead of 100000
   ```

2. Use memory-efficient indices (PQ, IVFPQ)

3. Increase system memory or swap

### Issue: Slow Benchmarks

**Symptom**: Benchmarks take too long

**Solutions**:
1. Reduce number of queries:
   ```go
   NumQueries: 100,  // Instead of 1000
   ```

2. Use smaller dataset for testing:
   ```bash
   make benchmark-10k  # Instead of benchmark-100k
   ```

3. Reduce iterations:
   ```bash
   make benchmark-quick  # Single run instead of 5
   ```

### Issue: Import Errors

**Symptom**: `cannot find package "github.com/chewxy/hnsw"`

**Solution**:
```bash
go get github.com/chewxy/hnsw
go mod tidy
```

## Best Practices

### 1. Consistent Environment
- Always run benchmarks on the same hardware
- Use CPU performance mode
- Close unnecessary applications
- Use fixed CPU affinity

### 2. Statistical Validity
- Run at least 5 iterations for production benchmarks
- Check CV (Coefficient of Variation) < 0.10
- Use 95% confidence intervals for comparisons

### 3. Baseline Management
- Set baseline after major releases
- Update baseline when hardware changes
- Keep historical baselines in `results/archive/`

### 4. Documentation
- Document system configuration in results
- Note any special conditions (CPU throttling, etc.)
- Archive results with git commit hash

### 5. Regression Detection
- Always compare against baseline before merging
- Investigate any regression > 10%
- Consider trade-offs (e.g., memory for speed)

## Advanced Topics

### Custom Distance Metrics

```go
// Benchmark with cosine similarity
idx, _ := hnsw.New(dim, "cosine", config)
```

### Batch Search Performance

```go
// Measure batch throughput
batchResults, _ := idx.BatchSearch(queries, k)
```

### Concurrent Search

```go
// Benchmark with multiple threads
var wg sync.WaitGroup
for i := 0; i < runtime.NumCPU(); i++ {
    wg.Add(1)
    go func(workerQueries [][]float32) {
        defer wg.Done()
        for _, q := range workerQueries {
            idx.Search(q, k)
        }
    }(queries[i*batchSize:(i+1)*batchSize])
}
wg.Wait()
```

### Real-World Datasets

Download and convert datasets like SIFT1M or GloVe:

```bash
# SIFT1M
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
go run scripts/convert_sift.go sift/sift_base.fvecs
```

## Help & Support

```bash
# Show all available commands
make help

# Get detailed benchmark documentation
cat BENCHMARK.md

# Report issues
# https://github.com/tahcohcat/gofaiss/issues
```

## Summary

You now have a complete, reproducible benchmark suite for GoFAISS! Key commands:

```bash
make benchmark-quick      # Quick single run
make benchmark-full       # Full statistical analysis
make visualize           # Generate charts
make compare             # Check for regressions
make validate            # Test reproducibility
```

Happy benchmarking! 