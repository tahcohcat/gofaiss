package example

// import (
// 	"fmt"
// 	"log"

// 	"github.com/tahcohcat/gofaiss/pkg/index/flat"
// 	"github.com/tahcohcat/gofaiss/pkg/index/hnsw"
// 	"github.com/tahcohcat/gofaiss/pkg/index/pq"
// 	"github.com/tahcohcat/gofaiss/pkg/storage"
// 	"github.com/tahcohcat/gofaiss/pkg/vector"
// )

// func main() {
// 	// Example 1: Flat Index (Exact Search)
// 	fmt.Println("=== Flat Index Example ===")
// 	flatExample()

// 	// Example 2: HNSW Index (Fast Approximate Search)
// 	fmt.Println("\n=== HNSW Index Example ===")
// 	hnswExample()

// 	// Example 3: PQ Index (Memory Efficient)
// 	fmt.Println("\n=== PQ Index Example ===")
// 	pqExample()
// }

// func flatExample() {
// 	// Create flat index
// 	dim := 128
// 	idx, err := flat.New(dim, "l2")
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	// Generate some vectors
// 	vectors := vector.GenerateRandom(1000, dim, 42)

// 	// Add vectors
// 	if err := idx.Add(vectors); err != nil {
// 		log.Fatal(err)
// 	}

// 	// Search
// 	query := vectors[0].Data // use first vector as query
// 	results, err := idx.Search(query, 10)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	fmt.Printf("Found %d results\n", len(results))
// 	for i, r := range results[:5] {
// 		fmt.Printf("  %d: ID=%d, Distance=%.4f\n", i+1, r.ID, r.Distance)
// 	}

// 	// Save index
// 	if err := storage.SaveIndex(idx, "flat_index.faiss.gz", true); err != nil {
// 		log.Printf("Save failed: %v\n", err)
// 	}

// 	// Stats
// 	stats := idx.Stats()
// 	fmt.Printf("Index size: %d vectors, %.2f MB\n", stats.TotalVectors, stats.MemoryUsageMB)
// }

// func hnswExample() {
// 	// Create HNSW index
// 	dim := 128
// 	config := hnsw.Config{
// 		Metric:         "l2",
// 		M:              16,
// 		EfConstruction: 200,
// 		EfSearch:       50,
// 	}

// 	idx, err := hnsw.New(dim, "l2", config)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	// Generate vectors
// 	vectors := vector.GenerateRandom(10000, dim, 42)

// 	// Add vectors
// 	fmt.Println("Building HNSW index...")
// 	if err := idx.Add(vectors); err != nil {
// 		log.Fatal(err)
// 	}

// 	// Adjust search quality
// 	idx.SetEfSearch(100) // higher = better accuracy, slower search

// 	// Search
// 	query := vectors[0].Data
// 	results, err := idx.Search(query, 10)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	fmt.Printf("Found %d results\n", len(results))
// 	for i, r := range results[:5] {
// 		fmt.Printf("  %d: ID=%d, Distance=%.4f\n", i+1, r.ID, r.Distance)
// 	}

// 	// Batch search
// 	queries := make([][]float32, 100)
// 	for i := 0; i < 100; i++ {
// 		queries[i] = vectors[i].Data
// 	}

// 	batchResults, err := idx.BatchSearch(queries, 10)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	fmt.Printf("Batch search completed: %d queries\n", len(batchResults))

// 	// Stats
// 	stats := idx.Stats()
// 	fmt.Printf("Index size: %d vectors, %.2f MB\n", stats.TotalVectors, stats.MemoryUsageMB)
// 	fmt.Printf("Max level: %v\n", stats.ExtraInfo["maxLevel"])
// }

// func pqExample() {
// 	// Create PQ index
// 	dim := 128
// 	config := pq.Config{
// 		M:     16, // 16 subquantizers
// 		Nbits: 8,  // 8 bits = 256 centroids per subquantizer
// 	}

// 	idx, err := pq.NewIndex(dim, config)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	// Generate vectors
// 	trainingVectors := vector.GenerateRandom(5000, dim, 42)
// 	vectors := vector.GenerateRandom(100000, dim, 43)

// 	// Train the index
// 	fmt.Println("Training PQ index...")
// 	if err := idx.Train(trainingVectors); err != nil {
// 		log.Fatal(err)
// 	}

// 	// Add vectors (they will be compressed)
// 	fmt.Println("Adding vectors...")
// 	if err := idx.Add(vectors); err != nil {
// 		log.Fatal(err)
// 	}

// 	// Search
// 	query := vectors[0].Data
// 	results, err := idx.Search(query, 10)
// 	if err != nil {
// 		log.Fatal(err)
// 	}

// 	fmt.Printf("Found %d results\n", len(results))
// 	for i, r := range results[:5] {
// 		fmt.Printf("  %d: ID=%d, Distance=%.4f\n", i+1, r.ID, r.Distance)
// 	}

// 	// Stats
// 	stats := idx.Stats()
// 	fmt.Printf("Index size: %d vectors, %.2f MB\n", stats.TotalVectors, stats.MemoryUsageMB)
// 	fmt.Printf("Compression ratio: ~%.1fx\n",
// 		float64(stats.TotalVectors*dim*4)/(stats.MemoryUsageMB*1024*1024))
// }

// // Example comparing different index types
// func comparisonExample() {
// 	dim := 128
// 	numVectors := 10000

// 	vectors := vector.GenerateRandom(numVectors, dim, 42)
// 	query := vectors[0].Data

// 	fmt.Println("=== Index Comparison ===")

// 	// Flat
// 	flatIdx, _ := flat.New(dim, "l2")
// 	flatIdx.Add(vectors)
// 	flatResults, _ := flatIdx.Search(query, 10)
// 	flatStats := flatIdx.Stats()
// 	fmt.Printf("Flat: %.2f MB, %d results\n",
// 		flatStats.MemoryUsageMB, len(flatResults))

// 	// HNSW
// 	hnswIdx, _ := hnsw.New(dim, "l2", hnsw.DefaultConfig())
// 	hnswIdx.Add(vectors)
// 	hnswResults, _ := hnswIdx.Search(query, 10)
// 	hnswStats := hnswIdx.Stats()
// 	fmt.Printf("HNSW: %.2f MB, %d results\n",
// 		hnswStats.MemoryUsageMB, len(hnswResults))

// 	// PQ
// 	pqIdx, _ := pq.NewIndex(dim, pq.Config{M: 16, Nbits: 8})
// 	pqIdx.Train(vectors[:5000])
// 	pqIdx.Add(vectors)
// 	pqResults, _ := pqIdx.Search(query, 10)
// 	pqStats := pqIdx.Stats()
// 	fmt.Printf("PQ: %.2f MB, %d results\n",
// 		pqStats.MemoryUsageMB, len(pqResults))
// }
