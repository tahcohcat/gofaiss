# GoFAISS Benchmark Results

A simple initial benchmarking run to give an indication of the capabilities of the library. 

**Test Configuration:**
- Vectors: 10,000
- Dimensions: 128
- Queries: 100
- Metric: L2 distance

## Performance Comparison Matrix

| Index Type | Build Time | Avg Query Time | QPS | Memory Usage | Compression Ratio |
|------------|------------|----------------|-----|--------------|-------------------|
| **Flat** | 1 ms | 2.25 ms | 444 | N/A | 1x (baseline) |
| **HNSW** | 1,694 ms | ~0 ms* | ∞* | 9.77 MB | ~1x |
| **PQ** | 1,647 ms | 1.55 ms | 643 | 0.28 MB | 32x |

*HNSW search time showing 0ms indicates timing granularity issue in benchmark

## Detailed Results

### Flat Index (Brute Force)
- **Build Time:** 1 ms (fastest)
- **Search Time:** 225 ms total
- **Avg Query Time:** 2.25 ms
- **Throughput:** 443.89 QPS
- **Memory:** Not reported
- **Characteristics:** Exact search, no compression, slow queries

### HNSW (Hierarchical Navigable Small World)
- **Build Time:** 1,694 ms (slowest)
- **Search Time:** 0 ms total (timing issue)
- **Avg Query Time:** 0 ms
- **Throughput:** ∞
- **Memory:** 9.77 MB
- **Config:** M=16, efConstruction=200, efSearch=50, maxLevel=3
- **Characteristics:** Fast approximate search, moderate build time, higher memory

### PQ (Product Quantization)
- **Build Time:** 1,647 ms
- **Search Time:** 155 ms total
- **Avg Query Time:** 1.55 ms (31% faster than flat)
- **Throughput:** 642.76 QPS (45% faster than flat)
- **Memory:** 0.28 MB (35x smaller than HNSW)
- **Config:** M=16, Nbits=8, Ksub=256, dsub=8
- **Compression Ratio:** 32x
- **Characteristics:** Compressed storage, approximate search, good balance

## Trade-off Analysis

### Choose **Flat** when:
- Exact results required
- Small dataset (<1K vectors)
- Build time is critical
- Memory is abundant

### Choose **HNSW** when:
- Query speed is paramount
- Can tolerate approximate results
- Memory available
- Can afford longer build time

### Choose **PQ** when:
- Memory constrained
- Need compression
- Can tolerate slight accuracy loss
- Good query speed still needed

## Recommendations

For **10K vectors @ 128 dims**:
1. **Production queries:** HNSW (needs timing fix to verify)
2. **Memory constrained:** PQ (28x smaller than HNSW)
3. **Exact search:** Flat (if 2.25ms acceptable)

## Notes

- HNSW timing appears incorrect (0ms) - suggests benchmark timing granularity issue
- PQ achieves 32x compression with only 31% speed improvement over exact search
- All indexes need to implement proper Stats() for consistent memory reporting
