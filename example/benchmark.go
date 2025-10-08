package example

// import (
// 	"encoding/json"
// 	"fmt"
// 	"log"
// 	"math/rand"
// 	"os"
// 	"strings"
// 	"time"

// 	"github.com/tahcohcat/gofaiss/pkg/index/flat"
// 	gofaiss_hnsw "github.com/tahcohcat/gofaiss/pkg/index/hnsw"
// 	"github.com/tahcohcat/gofaiss/pkg/index/ivf"
// 	"github.com/tahcohcat/gofaiss/pkg/index/ivfpq"
// 	"github.com/tahcohcat/gofaiss/pkg/index/pq"
// 	"github.com/tahcohcat/gofaiss/pkg/vector"
// )

// // BenchmarkConfig defines the benchmark parameters
// type BenchmarkConfig struct {
// 	Dimensions   int
// 	NumVectors   int
// 	NumQueries   int
// 	K            int
// 	Seed         int64
// 	OutputFile   string
// 	SaveDataset  bool
// 	DatasetFile  string
// }

// // BenchmarkResult stores comprehensive metrics
// type BenchmarkResult struct {
// 	Library         string
// 	IndexType       string
// 	BuildTimeMs     float64
// 	AvgQueryMs      float64
// 	P50QueryMs      float64
// 	P95QueryMs      float64
// 	P99QueryMs      float64
// 	QPS             float64
// 	MemoryMB        float64
// 	Recall          float64
// 	NumVectors      int
// 	Dimension       int
// 	IndexParams     map[string]interface{}
// 	Timestamp       time.Time
// }

// // Dataset holds benchmark data
// type Dataset struct {
// 	Vectors    []vector.Vector
// 	Queries    [][]float32
// 	GroundTruth [][]int64
// }

// func main() {
// 	configs := []BenchmarkConfig{
// 		{
// 			Dimensions:  128,
// 			NumVectors:  10000,
// 			NumQueries:  100,
// 			K:           10,
// 			Seed:        42,
// 			OutputFile:  "benchmark_results_10k.json",
// 			SaveDataset: true,
// 			DatasetFile: "dataset_10k.gob",
// 		},
// 		{
// 			Dimensions:  128,
// 			NumVectors:  100000,
// 			NumQueries:  1000,
// 			K:           10,
// 			Seed:        42,
// 			OutputFile:  "benchmark_results_100k.json",
// 			SaveDataset: true,
// 			DatasetFile: "dataset_100k.gob",
// 		},
// 	}

// 	for _, config := range configs {
// 		fmt.Printf("\n=== Running Benchmark: %d vectors, %d dimensions ===\n", 
// 			config.NumVectors, config.Dimensions)
// 		runBenchmark(config)
// 	}
// }

// func runBenchmark(config BenchmarkConfig) {
// 	// Generate or load dataset
// 	fmt.Println("Generating dataset...")
// 	dataset := generateDataset(config)

// 	if config.SaveDataset {
// 		fmt.Printf("Saving dataset to %s...\n", config.DatasetFile)
// 		saveDataset(dataset, config.DatasetFile)
// 	}

// 	// Compute ground truth
// 	fmt.Println("Computing ground truth...")
// 	dataset.GroundTruth = computeGroundTruth(dataset.Vectors, dataset.Queries, config.K)

// 	// Run all benchmarks
// 	results := []BenchmarkResult{}

// 	// GoFAISS - Flat
// 	fmt.Println("\nBenchmarking GoFAISS Flat...")
// 	results = append(results, benchmarkGoFAISSFlat(dataset, config))

// 	// GoFAISS - HNSW
// 	fmt.Println("Benchmarking GoFAISS HNSW...")
// 	results = append(results, benchmarkGoFAISSHNSW(dataset, config))

// 	// GoFAISS - IVF
// 	fmt.Println("Benchmarking GoFAISS IVF...")
// 	results = append(results, benchmarkGoFAISSIVF(dataset, config))

// 	// GoFAISS - PQ
// 	fmt.Println("Benchmarking GoFAISS PQ...")
// 	results = append(results, benchmarkGoFAISSPQ(dataset, config))

// 	// GoFAISS - IVFPQ
// 	fmt.Println("Benchmarking GoFAISS IVFPQ...")
// 	results = append(results, benchmarkGoFAISSIVFPQ(dataset, config))

// 	// Print results
// 	printResultsTable(results)

// 	// Save results
// 	saveResults(results, config.OutputFile)
// 	fmt.Printf("\nResults saved to %s\n", config.OutputFile)
// }

// func generateDataset(config BenchmarkConfig) Dataset {
// 	rand.Seed(config.Seed)
	
// 	vectors := vector.GenerateRandom(config.NumVectors, config.Dimensions, config.Seed)
	
// 	queries := make([][]float32, config.NumQueries)
// 	for i := 0; i < config.NumQueries; i++ {
// 		queries[i] = vector.GenerateRandom(1, config.Dimensions, config.Seed+int64(i+1000))[0].Data
// 	}

// 	return Dataset{
// 		Vectors: vectors,
// 		Queries: queries,
// 	}
// }

// func computeGroundTruth(vectors []vector.Vector, queries [][]float32, k int) [][]int64 {
// 	idx, _ := flat.New(len(vectors[0].Data), "l2")
// 	idx.Add(vectors)

// 	groundTruth := make([][]int64, len(queries))
// 	for i, query := range queries {
// 		results, _ := idx.Search(query, k)
// 		gt := make([]int64, len(results))
// 		for j, r := range results {
// 			gt[j] = r.ID
// 		}
// 		groundTruth[i] = gt
// 	}
// 	return groundTruth
// }

// func benchmarkGoFAISSFlat(dataset Dataset, config BenchmarkConfig) BenchmarkResult {
// 	// Build
// 	buildStart := time.Now()
// 	idx, _ := flat.New(config.Dimensions, "l2")
// 	idx.Add(dataset.Vectors)
// 	buildTime := time.Since(buildStart)

// 	// Search
// 	queryTimes := benchmarkSearch(func(query []float32) ([]vector.SearchResult, error) {
// 		return idx.Search(query, config.K)
// 	}, dataset.Queries)

// 	// Calculate recall
// 	results := make([][]vector.SearchResult, len(dataset.Queries))
// 	for i, query := range dataset.Queries {
// 		results[i], _ = idx.Search(query, config.K)
// 	}
// 	recall := calculateRecall(results, dataset.GroundTruth, config.K)

// 	stats := idx.Stats()

// 	return BenchmarkResult{
// 		Library:     "GoFAISS",
// 		IndexType:   "Flat",
// 		BuildTimeMs: float64(buildTime.Milliseconds()),
// 		AvgQueryMs:  queryTimes.avg,
// 		P50QueryMs:  queryTimes.p50,
// 		P95QueryMs:  queryTimes.p95,
// 		P99QueryMs:  queryTimes.p99,
// 		QPS:         queryTimes.qps,
// 		MemoryMB:    stats.MemoryUsageMB,
// 		Recall:      recall,
// 		NumVectors:  config.NumVectors,
// 		Dimension:   config.Dimensions,
// 		IndexParams: map[string]interface{}{
// 			"metric": "l2",
// 		},
// 		Timestamp: time.Now(),
// 	}
// }

// func benchmarkGoFAISSHNSW(dataset Dataset, config BenchmarkConfig) BenchmarkResult {
// 	hnswConfig := gofaiss_hnsw.Config{
// 		Metric:         "l2",
// 		M:              16,
// 		EfConstruction: 200,
// 		EfSearch:       50,
// 	}

// 	buildStart := time.Now()
// 	idx, _ := gofaiss_hnsw.New(config.Dimensions, "l2", hnswConfig)
// 	idx.Add(dataset.Vectors)
// 	buildTime := time.Since(buildStart)

// 	queryTimes := benchmarkSearch(func(query []float32) ([]vector.SearchResult, error) {
// 		return idx.Search(query, config.K)
// 	}, dataset.Queries)

// 	results := make([][]vector.SearchResult, len(dataset.Queries))
// 	for i, query := range dataset.Queries {
// 		results[i], _ = idx.Search(query, config.K)
// 	}
// 	recall := calculateRecall(results, dataset.GroundTruth, config.K)

// 	stats := idx.Stats()

// 	return BenchmarkResult{
// 		Library:     "GoFAISS",
// 		IndexType:   "HNSW",
// 		BuildTimeMs: float64(buildTime.Milliseconds()),
// 		AvgQueryMs:  queryTimes.avg,
// 		P50QueryMs:  queryTimes.p50,
// 		P95QueryMs:  queryTimes.p95,
// 		P99QueryMs:  queryTimes.p99,
// 		QPS:         queryTimes.qps,
// 		MemoryMB:    stats.MemoryUsageMB,
// 		Recall:      recall,
// 		NumVectors:  config.NumVectors,
// 		Dimension:   config.Dimensions,
// 		IndexParams: map[string]interface{}{
// 			"M":              hnswConfig.M,
// 			"efConstruction": hnswConfig.EfConstruction,
// 			"efSearch":       hnswConfig.EfSearch,
// 		},
// 		Timestamp: time.Now(),
// 	}
// }

// func benchmarkGoFAISSIVF(dataset Dataset, config BenchmarkConfig) BenchmarkResult {
// 	ivfConfig := ivf.DefaultConfig(config.NumVectors)

// 	buildStart := time.Now()
// 	idx, _ := ivf.New(config.Dimensions, "l2", ivfConfig)
// 	trainSize := config.NumVectors / 2
// 	if trainSize > 5000 {
// 		trainSize = 5000
// 	}
// 	idx.Train(dataset.Vectors[:trainSize])
// 	idx.Add(dataset.Vectors)
// 	buildTime := time.Since(buildStart)

// 	nprobe := 10

// 	queryTimes := benchmarkSearch(func(query []float32) ([]vector.SearchResult, error) {
// 		return idx.Search(query, config.K, nprobe)
// 	}, dataset.Queries)

// 	results := make([][]vector.SearchResult, len(dataset.Queries))
// 	for i, query := range dataset.Queries {
// 		results[i], _ = idx.Search(query, config.K, nprobe)
// 	}
// 	recall := calculateRecall(results, dataset.GroundTruth, config.K)

// 	stats := idx.Stats()

// 	return BenchmarkResult{
// 		Library:     "GoFAISS",
// 		IndexType:   "IVF",
// 		BuildTimeMs: float64(buildTime.Milliseconds()),
// 		AvgQueryMs:  queryTimes.avg,
// 		P50QueryMs:  queryTimes.p50,
// 		P95QueryMs:  queryTimes.p95,
// 		P99QueryMs:  queryTimes.p99,
// 		QPS:         queryTimes.qps,
// 		MemoryMB:    stats.MemoryUsageMB,
// 		Recall:      recall,
// 		NumVectors:  config.NumVectors,
// 		Dimension:   config.Dimensions,
// 		IndexParams: map[string]interface{}{
// 			"nlist":  ivfConfig.Nlist,
// 			"nprobe": nprobe,
// 		},
// 		Timestamp: time.Now(),
// 	}
// }

// func benchmarkGoFAISSPQ(dataset Dataset, config BenchmarkConfig) BenchmarkResult {
// 	pqConfig := pq.Config{
// 		M:     16,
// 		Nbits: 8,
// 	}

// 	buildStart := time.Now()
// 	idx, _ := pq.NewIndex(config.Dimensions, pqConfig)
// 	trainSize := config.NumVectors / 2
// 	if trainSize > 5000 {
// 		trainSize = 5000
// 	}
// 	idx.Train(dataset.Vectors[:trainSize])
// 	idx.Add(dataset.Vectors)
// 	buildTime := time.Since(buildStart)

// 	queryTimes := benchmarkSearch(func(query []float32) ([]vector.SearchResult, error) {
// 		return idx.Search(query, config.K)
// 	}, dataset.Queries)

// 	results := make([][]vector.SearchResult, len(dataset.Queries))
// 	for i, query := range dataset.Queries {
// 		results[i], _ = idx.Search(query, config.K)
// 	}
// 	recall := calculateRecall(results, dataset.GroundTruth, config.K)

// 	stats := idx.Stats()

// 	return BenchmarkResult{
// 		Library:     "GoFAISS",
// 		IndexType:   "PQ",
// 		BuildTimeMs: float64(buildTime.Milliseconds()),
// 		AvgQueryMs:  queryTimes.avg,
// 		P50QueryMs:  queryTimes.p50,
// 		P95QueryMs:  queryTimes.p95,
// 		P99QueryMs:  queryTimes.p99,
// 		QPS:         queryTimes.qps,
// 		MemoryMB:    stats.MemoryUsageMB,
// 		Recall:      recall,
// 		NumVectors:  config.NumVectors,
// 		Dimension:   config.Dimensions,
// 		IndexParams: map[string]interface{}{
// 			"M":     pqConfig.M,
// 			"Nbits": pqConfig.Nbits,
// 		},
// 		Timestamp: time.Now(),
// 	}
// }

// func benchmarkGoFAISSIVFPQ(dataset Dataset, config BenchmarkConfig) BenchmarkResult {
// 	ivfpqConfig := ivfpq.DefaultConfig(config.NumVectors, config.Dimensions)

// 	buildStart := time.Now()
// 	idx, _ := ivfpq.New(config.Dimensions, "l2", ivfpqConfig)
// 	trainSize := config.NumVectors / 2
// 	if trainSize > 5000 {
// 		trainSize = 5000
// 	}
// 	idx.Train(dataset.Vectors[:trainSize])
// 	idx.Add(dataset.Vectors)
// 	buildTime := time.Since(buildStart)

// 	nprobe := 10

// 	queryTimes := benchmarkSearch(func(query []float32) ([]vector.SearchResult, error) {
// 		return idx.Search(query, config.K, nprobe)
// 	}, dataset.Queries)

// 	results := make([][]vector.SearchResult, len(dataset.Queries))
// 	for i, query := range dataset.Queries {
// 		results[i], _ = idx.Search(query, config.K, nprobe)
// 	}
// 	recall := calculateRecall(results, dataset.GroundTruth, config.K)

// 	stats := idx.Stats()

// 	return BenchmarkResult{
// 		Library:     "GoFAISS",
// 		IndexType:   "IVFPQ",
// 		BuildTimeMs: float64(buildTime.Milliseconds()),
// 		AvgQueryMs:  queryTimes.avg,
// 		P50QueryMs:  queryTimes.p50,
// 		P95QueryMs:  queryTimes.p95,
// 		P99QueryMs:  queryTimes.p99,
// 		QPS:         queryTimes.qps,
// 		MemoryMB:    stats.MemoryUsageMB,
// 		Recall:      recall,
// 		NumVectors:  config.NumVectors,
// 		Dimension:   config.Dimensions,
// 		IndexParams: map[string]interface{}{
// 			"nlist":  ivfpqConfig.Nlist,
// 			"M":      ivfpqConfig.M,
// 			"Nbits":  ivfpqConfig.Nbits,
// 			"nprobe": nprobe,
// 		},
// 		Timestamp: time.Now(),
// 	}
// }

// type queryMetrics struct {
// 	avg float64
// 	p50 float64
// 	p95 float64
// 	p99 float64
// 	qps float64
// }

// func benchmarkSearch(searchFunc func([]float32) ([]vector.SearchResult, error), queries [][]float32) queryMetrics {
// 	// Warmup
// 	for i := 0; i < 10 && i < len(queries); i++ {
// 		searchFunc(queries[i])
// 	}

// 	// Benchmark
// 	times := make([]float64, len(queries))
// 	start := time.Now()
// 	for i, query := range queries {
// 		queryStart := time.Now()
// 		_, err := searchFunc(query)
// 		if err != nil {
// 			log.Printf("Search error: %v", err)
// 		}
// 		times[i] = float64(time.Since(queryStart).Microseconds()) / 1000.0 // ms
// 	}
// 	totalTime := time.Since(start)

// 	// Calculate metrics
// 	avg := 0.0
// 	for _, t := range times {
// 		avg += t
// 	}
// 	avg /= float64(len(times))

// 	// Sort for percentiles
// 	sortedTimes := make([]float64, len(times))
// 	copy(sortedTimes, times)
// 	for i := 0; i < len(sortedTimes); i++ {
// 		for j := i + 1; j < len(sortedTimes); j++ {
// 			if sortedTimes[i] > sortedTimes[j] {
// 				sortedTimes[i], sortedTimes[j] = sortedTimes[j], sortedTimes[i]
// 			}
// 		}
// 	}

// 	p50 := sortedTimes[len(sortedTimes)/2]
// 	p95 := sortedTimes[int(float64(len(sortedTimes))*0.95)]
// 	p99 := sortedTimes[int(float64(len(sortedTimes))*0.99)]
// 	qps := float64(len(queries)) / totalTime.Seconds()

// 	return queryMetrics{
// 		avg: avg,
// 		p50: p50,
// 		p95: p95,
// 		p99: p99,
// 		qps: qps,
// 	}
// }

// func calculateRecall(results [][]vector.SearchResult, groundTruth [][]int64, k int) float64 {
// 	if len(results) == 0 || len(groundTruth) == 0 {
// 		return 0.0
// 	}

// 	totalRecall := 0.0
// 	for i, gt := range groundTruth {
// 		if i >= len(results) {
// 			break
// 		}

// 		gtSet := make(map[int64]bool)
// 		for j := 0; j < k && j < len(gt); j++ {
// 			gtSet[gt[j]] = true
// 		}

// 		matches := 0
// 		for j := 0; j < k && j < len(results[i]); j++ {
// 			if gtSet[results[i][j].ID] {
// 				matches++
// 			}
// 		}

// 		recall := float64(matches) / float64(min(k, len(gt)))
// 		totalRecall += recall
// 	}

// 	return totalRecall / float64(len(groundTruth))
// }

// func min(a, b int) int {
// 	if a < b {
// 		return a
// 	}
// 	return b
// }

// func printResultsTable(results []BenchmarkResult) {
// 	fmt.Println("\n" + strings.Repeat("=", 130))
// 	fmt.Printf("%-15s %-10s %12s %12s %12s %12s %12s %12s %10s\n",
// 		"Library", "Index", "Build(ms)", "Avg(ms)", "P95(ms)", "QPS", "Memory(MB)", "Recall@10", "Vectors")
// 	fmt.Println(strings.Repeat("=", 130))

// 	for _, r := range results {
// 		fmt.Printf("%-15s %-10s %12.2f %12.4f %12.4f %12.0f %12.2f %10.4f %10d\n",
// 			r.Library,
// 			r.IndexType,
// 			r.BuildTimeMs,
// 			r.AvgQueryMs,
// 			r.P95QueryMs,
// 			r.QPS,
// 			r.MemoryMB,
// 			r.Recall,
// 			r.NumVectors,
// 		)
// 	}

// 	fmt.Println(strings.Repeat("=", 130))

// 	// Analysis
// 	fmt.Println("\n=== Performance Analysis ===")
	
// 	var goFAISSHNSW, hnswlibHNSW *BenchmarkResult
// 	for i := range results {
// 		if results[i].Library == "GoFAISS" && results[i].IndexType == "HNSW" {
// 			goFAISSHNSW = &results[i]
// 		}
// 		if results[i].Library == "hnswlib-go" && results[i].IndexType == "HNSW" {
// 			hnswlibHNSW = &results[i]
// 		}
// 	}

// 	if goFAISSHNSW != nil && hnswlibHNSW != nil {
// 		qpsRatio := goFAISSHNSW.QPS / hnswlibHNSW.QPS
// 		buildRatio := goFAISSHNSW.BuildTimeMs / hnswlibHNSW.BuildTimeMs
// 		recallDiff := goFAISSHNSW.Recall - hnswlibHNSW.Recall

// 		fmt.Printf("\nGoFAISS HNSW vs hnswlib-go HNSW:\n")
// 		fmt.Printf("  Query Speed: %.2fx (%.0f vs %.0f QPS)\n", qpsRatio, goFAISSHNSW.QPS, hnswlibHNSW.QPS)
// 		fmt.Printf("  Build Time: %.2fx (%.0f vs %.0f ms)\n", buildRatio, goFAISSHNSW.BuildTimeMs, hnswlibHNSW.BuildTimeMs)
// 		fmt.Printf("  Recall Difference: %+.4f (%.4f vs %.4f)\n", recallDiff, goFAISSHNSW.Recall, hnswlibHNSW.Recall)
// 	}

// 	// Find best performers
// 	var bestQPS, bestRecall, lowestMem *BenchmarkResult
// 	for i := range results {
// 		if bestQPS == nil || results[i].QPS > bestQPS.QPS {
// 			bestQPS = &results[i]
// 		}
// 		if bestRecall == nil || results[i].Recall > bestRecall.Recall {
// 			bestRecall = &results[i]
// 		}
// 		if lowestMem == nil || results[i].MemoryMB < lowestMem.MemoryMB {
// 			lowestMem = &results[i]
// 		}
// 	}

// 	fmt.Printf("\nBest Performers:\n")
// 	fmt.Printf("  Highest QPS: %s %s (%.0f QPS)\n", bestQPS.Library, bestQPS.IndexType, bestQPS.QPS)
// 	fmt.Printf("  Best Recall: %s %s (%.4f)\n", bestRecall.Library, bestRecall.IndexType, bestRecall.Recall)
// 	fmt.Printf("  Lowest Memory: %s %s (%.2f MB)\n", lowestMem.Library, lowestMem.IndexType, lowestMem.MemoryMB)
// }

// func saveResults(results []BenchmarkResult, filename string) {
// 	data, err := json.MarshalIndent(results, "", "  ")
// 	if err != nil {
// 		log.Printf("Failed to marshal results: %v", err)
// 		return
// 	}

// 	err = os.WriteFile(filename, data, 0644)
// 	if err != nil {
// 		log.Printf("Failed to write results: %v", err)
// 	}
// }

// func saveDataset(dataset Dataset, filename string) {
// 	// Implement dataset saving using gob or json
// 	// This is a placeholder
// 	log.Printf("Dataset saving not implemented yet")
// }