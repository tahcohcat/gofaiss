package main

import (
	"fmt"
	"log"
	"time"

	"github.com/tahcohcat/gofaiss/pkg/index/flat"
	"github.com/tahcohcat/gofaiss/pkg/index/hnsw"
	"github.com/tahcohcat/gofaiss/pkg/index/ivf"
	"github.com/tahcohcat/gofaiss/pkg/index/ivfpq"
	"github.com/tahcohcat/gofaiss/pkg/index/pq"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// BenchmarkResult stores benchmark metrics
type BenchmarkResult struct {
	IndexType   string
	BuildTimeMs float64
	AvgQueryMs  float64
	QPS         float64
	MemoryMB    float64
	Recall      float64
	NumVectors  int
	Dimension   int
}

func main() {
	// Configuration
	dim := 128
	numVectors := 10000
	numQueries := 100
	k := 10

	fmt.Println("=== GoFAISS Benchmark ===")
	fmt.Printf("Dataset: %d vectors, %d dimensions\n", numVectors, dim)
	fmt.Printf("Queries: %d, k=%d\n\n", numQueries, k)

	// Generate data
	fmt.Println("Generating synthetic data...")
	vectors := vector.GenerateRandom(numVectors, dim, 42)
	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = vector.GenerateRandom(1, dim, int64(i+1000))[0].Data
	}

	// Compute ground truth using flat index
	fmt.Println("Computing ground truth...")
	groundTruth := computeGroundTruth(vectors, queries, k)

	// Run benchmarks
	results := []BenchmarkResult{
		benchmarkFlat(vectors, queries, k, groundTruth),
		benchmarkHNSW(vectors, queries, k, groundTruth),
		benchmarkIVF(vectors, queries, k, groundTruth),
		benchmarkPQ(vectors, queries, k, groundTruth),
		benchmarkIVFPQ(vectors, queries, k, groundTruth),
	}

	// Print results
	fmt.Println("\n=== Benchmark Results ===")
	printResultsTable(results)
}

func benchmarkFlat(vectors []vector.Vector, queries [][]float32, k int, groundTruth [][]int64) BenchmarkResult {
	fmt.Println("\nBenchmarking Flat Index...")

	// Build
	buildStart := time.Now()
	idx, _ := flat.New(len(vectors[0].Data), "l2")
	idx.Add(vectors)
	buildTime := time.Since(buildStart)

	// Search
	results := benchmarkSearch(idx, queries, k)

	// Calculate recall
	recall := calculateRecall(results, groundTruth, k)

	stats := idx.Stats()

	return BenchmarkResult{
		IndexType:   "Flat",
		BuildTimeMs: float64(buildTime.Milliseconds()),
		AvgQueryMs:  results.avgQueryMs,
		QPS:         results.qps,
		MemoryMB:    stats.MemoryUsageMB,
		Recall:      recall,
		NumVectors:  len(vectors),
		Dimension:   len(vectors[0].Data),
	}
}

func benchmarkHNSW(vectors []vector.Vector, queries [][]float32, k int, groundTruth [][]int64) BenchmarkResult {
	fmt.Println("\nBenchmarking HNSW Index...")

	buildStart := time.Now()
	idx, _ := hnsw.New(len(vectors[0].Data), "l2", hnsw.Config{
		M:              16,
		EfConstruction: 200,
		EfSearch:       50,
	})
	idx.Add(vectors)
	buildTime := time.Since(buildStart)

	results := benchmarkSearch(idx, queries, k)
	recall := calculateRecall(results.results, groundTruth, k)
	stats := idx.Stats()

	return BenchmarkResult{
		IndexType:   "HNSW",
		BuildTimeMs: float64(buildTime.Milliseconds()),
		AvgQueryMs:  results.avgQueryMs,
		QPS:         results.qps,
		MemoryMB:    stats.MemoryUsageMB,
		Recall:      recall,
		NumVectors:  len(vectors),
		Dimension:   len(vectors[0].Data),
	}
}

func benchmarkIVF(vectors []vector.Vector, queries [][]float32, k int, groundTruth [][]int64) BenchmarkResult {
	fmt.Println("\nBenchmarking IVF Index...")

	buildStart := time.Now()
	idx, _ := ivf.New(len(vectors[0].Data), "l2", ivf.Config{
		Metric: "l2",
		Nlist:  100,
	})

	// Train
	trainSize := len(vectors) / 2
	if trainSize > 5000 {
		trainSize = 5000
	}
	idx.Train(vectors[:trainSize])
	idx.Add(vectors)
	buildTime := time.Since(buildStart)

	results := benchmarkSearchIVF(idx, queries, k, 10)
	recall := calculateRecall(results.results, groundTruth, k)
	stats := idx.Stats()

	return BenchmarkResult{
		IndexType:   "IVF",
		BuildTimeMs: float64(buildTime.Milliseconds()),
		AvgQueryMs:  results.avgQueryMs,
		QPS:         results.qps,
		MemoryMB:    stats.MemoryUsageMB,
		Recall:      recall,
		NumVectors:  len(vectors),
		Dimension:   len(vectors[0].Data),
	}
}

func benchmarkPQ(vectors []vector.Vector, queries [][]float32, k int, groundTruth [][]int64) BenchmarkResult {
	fmt.Println("\nBenchmarking PQ Index...")

	buildStart := time.Now()
	idx, _ := pq.NewIndex(len(vectors[0].Data), pq.Config{
		M:     16,
		Nbits: 8,
	})

	trainSize := len(vectors) / 2
	if trainSize > 5000 {
		trainSize = 5000
	}
	idx.Train(vectors[:trainSize])
	idx.Add(vectors)
	buildTime := time.Since(buildStart)

	results := benchmarkSearch(idx, queries, k)
	recall := calculateRecall(results.results, groundTruth, k)
	stats := idx.Stats()

	return BenchmarkResult{
		IndexType:   "PQ",
		BuildTimeMs: float64(buildTime.Milliseconds()),
		AvgQueryMs:  results.avgQueryMs,
		QPS:         results.qps,
		MemoryMB:    stats.MemoryUsageMB,
		Recall:      recall,
		NumVectors:  len(vectors),
		Dimension:   len(vectors[0].Data),
	}
}

func benchmarkIVFPQ(vectors []vector.Vector, queries [][]float32, k int, groundTruth [][]int64) BenchmarkResult {
	fmt.Println("\nBenchmarking IVF+PQ Index...")

	buildStart := time.Now()
	idx, _ := ivfpq.New(len(vectors[0].Data), "l2", ivfpq.Config{
		Metric: "l2",
		Nlist:  100,
		M:      8,
		Nbits:  8,
	})

	trainSize := len(vectors) / 2
	if trainSize > 5000 {
		trainSize = 5000
	}
	idx.Train(vectors[:trainSize])
	idx.Add(vectors)
	buildTime := time.Since(buildStart)

	results := benchmarkSearchIVF(idx, queries, k, 10)
	recall := calculateRecall(results.results, groundTruth, k)
	stats := idx.Stats()

	return BenchmarkResult{
		IndexType:   "IVF+PQ",
		BuildTimeMs: float64(buildTime.Milliseconds()),
		AvgQueryMs:  results.avgQueryMs,
		QPS:         results.qps,
		MemoryMB:    stats.MemoryUsageMB,
		Recall:      recall,
		NumVectors:  len(vectors),
		Dimension:   len(vectors[0].Data),
	}
}

type searchResults struct {
	results    [][]vector.SearchResult
	avgQueryMs float64
	qps        float64
}

func benchmarkSearch(idx interface{}, queries [][]float32, k int) searchResults {
	// Warmup
	for i := 0; i < 10 && i < len(queries); i++ {
		search(idx, queries[i], k, 0)
	}

	// Benchmark
	searchStart := time.Now()
	results := make([][]vector.SearchResult, len(queries))
	for i, query := range queries {
		res, err := search(idx, query, k, 0)
		if err != nil {
			log.Fatal(err)
		}
		results[i] = res
	}
	searchTime := time.Since(searchStart)

	return searchResults{
		results:    results,
		avgQueryMs: float64(searchTime.Milliseconds()) / float64(len(queries)),
		qps:        float64(len(queries)) / searchTime.Seconds(),
	}
}

func benchmarkSearchIVF(idx interface{}, queries [][]float32, k int, nprobe int) searchResults {
	// Warmup
	for i := 0; i < 10 && i < len(queries); i++ {
		search(idx, queries[i], k, nprobe)
	}

	// Benchmark
	searchStart := time.Now()
	results := make([][]vector.SearchResult, len(queries))
	for i, query := range queries {
		res, err := search(idx, query, k, nprobe)
		if err != nil {
			log.Fatal(err)
		}
		results[i] = res
	}
	searchTime := time.Since(searchStart)

	return searchResults{
		results:    results,
		avgQueryMs: float64(searchTime.Milliseconds()) / float64(len(queries)),
		qps:        float64(len(queries)) / searchTime.Seconds(),
	}
}

func search(idx interface{}, query []float32, k int, nprobe int) ([]vector.SearchResult, error) {
	switch v := idx.(type) {
	case *flat.Index:
		return v.Search(query, k)
	case *hnsw.Index:
		return v.Search(query, k)
	case *pq.Index:
		return v.Search(query, k)
	case *ivf.Index:
		return v.Search(query, k, nprobe)
	case *ivfpq.Index:
		return v.Search(query, k, nprobe)
	default:
		return nil, fmt.Errorf("unsupported index type")
	}
}

func computeGroundTruth(vectors []vector.Vector, queries [][]float32, k int) [][]int64 {
	idx, _ := flat.New(len(vectors[0].Data), "l2")
	idx.Add(vectors)

	groundTruth := make([][]int64, len(queries))
	for i, query := range queries {
		results, _ := idx.Search(query, k)
		gt := make([]int64, len(results))
		for j, r := range results {
			gt[j] = r.ID
		}
		groundTruth[i] = gt
	}
	return groundTruth
}

func calculateRecall(results [][]vector.SearchResult, groundTruth [][]int64, k int) float64 {
	if len(results) == 0 || len(groundTruth) == 0 {
		return 0.0
	}

	totalRecall := 0.0
	for i, gt := range groundTruth {
		if i >= len(results) {
			break
		}

		gtSet := make(map[int64]bool)
		for j := 0; j < k && j < len(gt); j++ {
			gtSet[gt[j]] = true
		}

		matches := 0
		for j := 0; j < k && j < len(results[i]); j++ {
			if gtSet[results[i][j].ID] {
				matches++
			}
		}

		recall := float64(matches) / float64(min(k, len(gt)))
		totalRecall += recall
	}

	return totalRecall / float64(len(groundTruth))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func printResultsTable(results []BenchmarkResult) {
	fmt.Println("┌────────────┬────────────┬────────────┬────────────┬────────────┬──────────┐")
	fmt.Println("│ Index Type │ Build (ms) │ Query (ms) │    QPS     │ Memory(MB) │ Recall@10│")
	fmt.Println("├────────────┼────────────┼────────────┼────────────┼────────────┼──────────┤")

	for _, r := range results {
		fmt.Printf("│ %-10s │ %10.2f │ %10.4f │ %10.0f │ %10.2f │ %8.4f │\n",
			r.IndexType,
			r.BuildTimeMs,
			r.AvgQueryMs,
			r.QPS,
			r.MemoryMB,
			r.Recall,
		)
	}

	fmt.Println("└────────────┴────────────┴────────────┴────────────┴────────────┴──────────┘")

	// Find best performers
	fmt.Println("\n=== Best Performers ===")

	var fastestBuild, fastestQuery, bestRecall, lowestMemory BenchmarkResult
	fastestBuild = results[0]
	fastestQuery = results[0]
	bestRecall = results[0]
	lowestMemory = results[0]

	for _, r := range results[1:] {
		if r.BuildTimeMs < fastestBuild.BuildTimeMs {
			fastestBuild = r
		}
		if r.AvgQueryMs < fastestQuery.AvgQueryMs {
			fastestQuery = r
		}
		if r.Recall > bestRecall.Recall {
			bestRecall = r
		}
		if r.MemoryMB < lowestMemory.MemoryMB {
			lowestMemory = r
		}
	}

	fmt.Printf("Fastest Build:   %s (%.2f ms)\n", fastestBuild.IndexType, fastestBuild.BuildTimeMs)
	fmt.Printf("Fastest Query:   %s (%.4f ms, %.0f QPS)\n", fastestQuery.IndexType, fastestQuery.AvgQueryMs, fastestQuery.QPS)
	fmt.Printf("Best Recall:     %s (%.4f)\n", bestRecall.IndexType, bestRecall.Recall)
	fmt.Printf("Lowest Memory:   %s (%.2f MB)\n", lowestMemory.IndexType, lowestMemory.MemoryMB)
}
