package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/tahcohcat/gofaiss/pkg/index/flat"
	"github.com/tahcohcat/gofaiss/pkg/index/hnsw"
	"github.com/tahcohcat/gofaiss/pkg/index/pq"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

func main() {
	// Define commands
	benchCmd := flag.NewFlagSet("bench", flag.ExitOnError)
	buildCmd := flag.NewFlagSet("build", flag.ExitOnError)
	searchCmd := flag.NewFlagSet("search", flag.ExitOnError)

	// Benchmark flags
	benchType := benchCmd.String("type", "hnsw", "Index type (flat, hnsw, pq)")
	benchVectors := benchCmd.Int("vectors", 10000, "Number of vectors")
	benchDim := benchCmd.Int("dim", 128, "Vector dimension")
	benchQueries := benchCmd.Int("queries", 100, "Number of queries")

	// Build flags
	buildType := buildCmd.String("type", "hnsw", "Index type")
	buildInput := buildCmd.String("input", "", "Input vectors file")
	buildOutput := buildCmd.String("output", "index.faiss", "Output index file")
	buildDim := buildCmd.Int("dim", 128, "Vector dimension")

	// Search flags
	searchIndex := searchCmd.String("index", "index.faiss", "Index file")
	searchQuery := searchCmd.String("query", "", "Query vector file")
	searchK := searchCmd.Int("k", 10, "Number of results")

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "bench":
		benchCmd.Parse(os.Args[2:])
		runBenchmark(*benchType, *benchVectors, *benchDim, *benchQueries)
	case "build":
		buildCmd.Parse(os.Args[2:])
		runBuild(*buildType, *buildInput, *buildOutput, *buildDim)
	case "search":
		searchCmd.Parse(os.Args[2:])
		runSearch(*searchIndex, *searchQuery, *searchK)
	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("GoFAISS CLI - Vector Similarity Search Tool")
	fmt.Println("\nUsage:")
	fmt.Println("  gofaiss-cli bench   - Run benchmarks")
	fmt.Println("  gofaiss-cli build   - Build an index")
	fmt.Println("  gofaiss-cli search  - Search an index")
	fmt.Println("\nExamples:")
	fmt.Println("  gofaiss-cli bench -type hnsw -vectors 10000 -dim 128")
	fmt.Println("  gofaiss-cli build -type hnsw -input vectors.bin -output index.faiss")
	fmt.Println("  gofaiss-cli search -index index.faiss -query query.bin -k 10")
}

func runBenchmark(indexType string, numVectors, dim, numQueries int) {
	fmt.Printf("Running benchmark: %s index, %d vectors, %d dimensions\n",
		indexType, numVectors, dim)

	// Generate synthetic data
	fmt.Println("Generating data...")
	vectors := vector.GenerateRandom(numVectors, dim, 42)
	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		queries[i] = vector.GenerateRandom(1, dim, int64(i+1000))[0].Data
	}

	var idx interface{}
	var buildTime time.Duration
	var err error

	// Build index
	fmt.Println("Building index...")
	buildStart := time.Now()

	switch indexType {
	case "flat":
		flatIdx, e := flat.New(dim, "l2")
		if e != nil {
			log.Fatal(e)
		}
		err = flatIdx.Add(vectors)
		idx = flatIdx

	case "hnsw":
		hnswIdx, e := hnsw.New(dim, "l2", hnsw.Config{
			M:              16,
			EfConstruction: 200,
			EfSearch:       50,
		})
		if e != nil {
			log.Fatal(e)
		}
		err = hnswIdx.Add(vectors)
		idx = hnswIdx

	case "pq":
		pqIdx, e := pq.NewIndex(dim, pq.Config{M: 16, Nbits: 8})
		if e != nil {
			log.Fatal(e)
		}
		trainSize := numVectors / 2
		if trainSize > 10000 {
			trainSize = 10000
		}
		if err = pqIdx.Train(vectors[:trainSize]); err != nil {
			log.Fatal(err)
		}
		err = pqIdx.Add(vectors)
		idx = pqIdx

	default:
		log.Fatalf("Unknown index type: %s", indexType)
	}

	if err != nil {
		log.Fatal(err)
	}
	buildTime = time.Since(buildStart)

	// Warmup
	fmt.Println("Warming up...")
	for i := 0; i < 10 && i < len(queries); i++ {
		searchOne(idx, queries[i], 10)
	}

	// Benchmark search
	fmt.Println("Benchmarking search...")
	searchStart := time.Now()
	for _, query := range queries {
		_, err := searchOne(idx, query, 10)
		if err != nil {
			log.Fatal(err)
		}
	}
	searchTime := time.Since(searchStart)

	// Results
	fmt.Println("\n=== Benchmark Results ===")
	fmt.Printf("Index Type: %s\n", indexType)
	fmt.Printf("Vectors: %d, Dimension: %d\n", numVectors, dim)
	fmt.Printf("Build Time: %.2f ms\n", float64(buildTime.Milliseconds()))
	fmt.Printf("Search Time: %.2f ms total\n", float64(searchTime.Milliseconds()))
	fmt.Printf("Avg Query Time: %.4f ms\n",
		float64(searchTime.Milliseconds())/float64(numQueries))
	fmt.Printf("Queries Per Second: %.2f\n",
		float64(numQueries)/searchTime.Seconds())

	// Memory stats
	printStats(idx)
}

func runBuild(indexType, inputFile, outputFile string, dim int) {
	fmt.Printf("Building %s index from %s...\n", indexType, inputFile)
	// Implementation would load vectors from file and build index
	fmt.Println("Build not fully implemented - use as template")
}

func runSearch(indexFile, queryFile string, k int) {
	fmt.Printf("Searching index %s with query %s...\n", indexFile, queryFile)
	// Implementation would load index and perform search
	fmt.Println("Search not fully implemented - use as template")
}

func searchOne(idx interface{}, query []float32, k int) ([]vector.SearchResult, error) {
	switch v := idx.(type) {
	case *flat.Index:
		return v.Search(query, k)
	case *hnsw.Index:
		return v.Search(query, k)
	case *pq.Index:
		return v.Search(query, k)
	default:
		return nil, fmt.Errorf("unsupported index type")
	}
}

func printStats(idx interface{}) {
	var totalVecs int
	var memoryMB float64
	var extraInfo map[string]interface{}

	switch v := idx.(type) {
	case *flat.Index:
		stats := v.Stats()
		totalVecs = stats.TotalVectors
		memoryMB = stats.MemoryUsageMB
		extraInfo = stats.ExtraInfo
	case *hnsw.Index:
		stats := v.Stats()
		totalVecs = stats.TotalVectors
		memoryMB = stats.MemoryUsageMB
		extraInfo = stats.ExtraInfo
	case *pq.Index:
		stats := v.Stats()
		totalVecs = stats.TotalVectors
		memoryMB = stats.MemoryUsageMB
		extraInfo = stats.ExtraInfo
	}

	fmt.Printf("\n=== Index Statistics ===\n")
	fmt.Printf("Total Vectors: %d\n", totalVecs)
	fmt.Printf("Memory Usage: %.2f MB\n", memoryMB)
	if len(extraInfo) > 0 {
		fmt.Printf("Extra Info: %v\n", extraInfo)
	}
}
