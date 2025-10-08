//go:build integration

package integration

import (
	"path/filepath"
	"testing"

	"github.com/tahcohcat/gofaiss/internal/testutil"
	"github.com/tahcohcat/gofaiss/pkg/index/flat"
	"github.com/tahcohcat/gofaiss/pkg/index/hnsw"
	"github.com/tahcohcat/gofaiss/pkg/index/ivf"
	"github.com/tahcohcat/gofaiss/pkg/index/ivfpq"
	"github.com/tahcohcat/gofaiss/pkg/index/pq"
	"github.com/tahcohcat/gofaiss/pkg/storage"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// TestFlatIndexEndToEnd tests complete workflow for flat index
func TestFlatIndexEndToEnd(t *testing.T) {
	dim := 128
	numVectors := 1000
	k := 10

	// Create index
	idx, err := flat.New(dim, "l2")
	testutil.AssertNoError(t, err, "creating flat index")

	// Generate and add vectors
	vectors := vector.GenerateRandom(numVectors, dim, 42)
	err = idx.Add(vectors)
	testutil.AssertNoError(t, err, "adding vectors")

	// Search
	query := vectors[0].Data
	results, err := idx.Search(query, k)
	testutil.AssertNoError(t, err, "searching")
	testutil.AssertValidSearchResults(t, results, "flat index search")

	// Save to file
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "flat_index.gob")
	err = storage.SaveToFile(idx, filename, storage.FormatGob, true)
	testutil.AssertNoError(t, err, "saving index")

	// Load from file
	loaded, err := flat.New(dim, "l2")
	testutil.AssertNoError(t, err, "creating new index for loading")
	err = storage.LoadFromFile(loaded, filename, storage.FormatGob, true)
	testutil.AssertNoError(t, err, "loading index")

	// Verify loaded index works
	results2, err := loaded.Search(query, k)
	testutil.AssertNoError(t, err, "searching loaded index")
	testutil.CompareSearchResults(t, results2, results, "comparing loaded vs original")
}

// TestHNSWIndexEndToEnd tests complete workflow for HNSW index
func TestHNSWIndexEndToEnd(t *testing.T) {
	dim := 128
	numVectors := 5000
	k := 20

	config := hnsw.DefaultConfig()
	config.M = 16
	config.EfConstruction = 200

	idx, err := hnsw.New(dim, "l2", config)
	testutil.AssertNoError(t, err, "creating HNSW index")

	// Add vectors
	vectors := vector.GenerateRandom(numVectors, dim, 42)
	err = idx.Add(vectors)
	testutil.AssertNoError(t, err, "adding vectors")

	// Test different ef_search values
	efSearchValues := []int{10, 50, 100, 200}
	for _, ef := range efSearchValues {
		t.Run("ef_search_"+string(rune(ef)), func(t *testing.T) {
			idx.SetEfSearch(ef)
			query := vectors[0].Data
			results, err := idx.Search(query, k)
			testutil.AssertNoError(t, err, "searching with ef="+string(rune(ef)))
			testutil.AssertValidSearchResults(t, results, "HNSW search results")
		})
	}

	// Test removal
	removeID := vectors[100].ID
	err = idx.Remove(removeID)
	testutil.AssertNoError(t, err, "removing vector")

	// Verify vector was removed
	query := vectors[100].Data
	results, err := idx.Search(query, k)
	testutil.AssertNoError(t, err, "searching after removal")
	for _, r := range results {
		if r.ID == removeID {
			t.Errorf("Removed vector %d still in search results", removeID)
		}
	}
}

// TestIVFIndexEndToEnd tests complete workflow for IVF index
func TestIVFIndexEndToEnd(t *testing.T) {
	dim := 128
	numTrain := 10000
	numAdd := 5000
	k := 10

	config := ivf.Config{
		Metric: "l2",
		Nlist:  100,
	}

	idx, err := ivf.New(dim, "l2", config)
	testutil.AssertNoError(t, err, "creating IVF index")

	// Train
	trainVectors := vector.GenerateRandom(numTrain, dim, 42)
	err = idx.Train(trainVectors)
	testutil.AssertNoError(t, err, "training IVF index")
	testutil.AssertTrue(t, idx.IsTrained(), "index should be trained")

	// Add vectors
	vectors := vector.GenerateRandom(numAdd, dim, 43)
	err = idx.Add(vectors)
	testutil.AssertNoError(t, err, "adding vectors")

	// Test different nprobe values
	nprobeValues := []int{1, 5, 10, 20}
	for _, nprobe := range nprobeValues {
		t.Run("nprobe_"+string(rune(nprobe)), func(t *testing.T) {
			query := vectors[0].Data
			results, err := idx.Search(query, k, nprobe)
			testutil.AssertNoError(t, err, "searching with nprobe="+string(rune(nprobe)))
			testutil.AssertValidSearchResults(t, results, "IVF search results")
		})
	}

	// Check list distribution
	listSizes := idx.GetListSizes()
	nonEmptyLists := 0
	for _, size := range listSizes {
		if size > 0 {
			nonEmptyLists++
		}
	}
	t.Logf("Non-empty lists: %d/%d", nonEmptyLists, len(listSizes))
}

// TestPQIndexEndToEnd tests complete workflow for PQ index
func TestPQIndexEndToEnd(t *testing.T) {
	dim := 128
	numTrain := 10000
	numAdd := 1000
	k := 10

	config := pq.Config{
		M:     8,
		Nbits: 8,
	}

	idx, err := pq.NewIndex(dim, config)
	testutil.AssertNoError(t, err, "creating PQ index")

	// Train
	trainVectors := vector.GenerateRandom(numTrain, dim, 42)
	err = idx.Train(trainVectors)
	testutil.AssertNoError(t, err, "training PQ index")
	testutil.AssertTrue(t, idx.IsTrained(), "index should be trained")

	// Add vectors
	vectors := vector.GenerateRandom(numAdd, dim, 43)
	err = idx.Add(vectors)
	testutil.AssertNoError(t, err, "adding vectors")

	// Search
	query := vectors[0].Data
	results, err := idx.Search(query, k)
	testutil.AssertNoError(t, err, "searching")
	testutil.AssertValidSearchResults(t, results, "PQ search results")

	// Check compression
	stats := idx.Stats()
	if compressionRatio, ok := stats.ExtraInfo["compressionRatio"].(float64); ok {
		t.Logf("Compression ratio: %.2fx", compressionRatio)
		if compressionRatio <= 1 {
			t.Errorf("Expected compression ratio > 1, got %.2f", compressionRatio)
		}
	}
}

// TestIVFPQIndexEndToEnd tests complete workflow for IVFPQ index
func TestIVFPQIndexEndToEnd(t *testing.T) {
	dim := 128
	numTrain := 10000
	numAdd := 5000
	k := 10

	config := ivfpq.Config{
		Metric: "l2",
		Nlist:  100,
		M:      8,
		Nbits:  8,
	}

	idx, err := ivfpq.New(dim, "l2", config)
	testutil.AssertNoError(t, err, "creating IVFPQ index")

	// Train
	trainVectors := vector.GenerateRandom(numTrain, dim, 42)
	err = idx.Train(trainVectors)
	testutil.AssertNoError(t, err, "training IVFPQ index")
	testutil.AssertTrue(t, idx.IsTrained(), "index should be trained")

	// Add vectors
	vectors := vector.GenerateRandom(numAdd, dim, 43)
	err = idx.Add(vectors)
	testutil.AssertNoError(t, err, "adding vectors")

	// Search with different nprobe
	nprobeValues := []int{1, 10, 50}
	for _, nprobe := range nprobeValues {
		t.Run("nprobe_"+string(rune(nprobe)), func(t *testing.T) {
			query := vectors[0].Data
			results, err := idx.Search(query, k, nprobe)
			testutil.AssertNoError(t, err, "searching")
			testutil.AssertValidSearchResults(t, results, "IVFPQ search results")
		})
	}

	// Test serialization
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "ivfpq_index.gob")
	err = storage.SaveToFile(idx, filename, storage.FormatGob, true)
	testutil.AssertNoError(t, err, "saving IVFPQ index")

	// Load and verify
	loaded, err := ivfpq.New(dim, "l2", config)
	testutil.AssertNoError(t, err, "creating new IVFPQ for loading")
	err = storage.LoadFromFile(loaded, filename, storage.FormatGob, true)
	testutil.AssertNoError(t, err, "loading IVFPQ index")

	query := vectors[0].Data
	results1, _ := idx.Search(query, k, 10)
	results2, _ := loaded.Search(query, k, 10)
	testutil.CompareSearchResults(t, results2, results1, "loaded vs original IVFPQ")
}

// TestCrossFormatSerialization tests saving in one format and loading in another
func TestCrossFormatSerialization(t *testing.T) {
	dim := 64
	numVectors := 100

	idx, err := flat.New(dim, "l2")
	testutil.AssertNoError(t, err, "creating index")

	vectors := vector.GenerateRandom(numVectors, dim, 42)
	err = idx.Add(vectors)
	testutil.AssertNoError(t, err, "adding vectors")

	tmpDir := t.TempDir()

	formats := []storage.Format{storage.FormatGob, storage.FormatJSON}

	for _, format := range formats {
		t.Run(string(format), func(t *testing.T) {
			filename := filepath.Join(tmpDir, "index_"+string(format))

			// Save
			err = storage.SaveToFile(idx, filename, format, false)
			testutil.AssertNoError(t, err, "saving with "+string(format))

			// Load
			loaded, _ := flat.New(dim, "l2")
			err = storage.LoadFromFile(loaded, filename, format, false)
			testutil.AssertNoError(t, err, "loading with "+string(format))

			// Verify
			query := vectors[0].Data
			results1, _ := idx.Search(query, 5)
			results2, _ := loaded.Search(query, 5)
			testutil.CompareSearchResults(t, results2, results1, "format "+string(format))
		})
	}
}

// TestLargeDataset tests with a larger dataset
func TestLargeDataset(t *testing.T) {
	testutil.SkipIfShort(t, "large dataset test")

	dim := 256
	numVectors := 50000
	k := 100

	t.Log("Testing HNSW with large dataset...")
	config := hnsw.DefaultConfig()
	idx, err := hnsw.New(dim, "l2", config)
	testutil.AssertNoError(t, err, "creating HNSW index")

	// Add vectors in batches
	batchSize := 10000
	for i := 0; i < numVectors; i += batchSize {
		end := i + batchSize
		if end > numVectors {
			end = numVectors
		}
		vectors := vector.GenerateRandom(end-i, dim, int64(i))
		err = idx.Add(vectors)
		testutil.AssertNoError(t, err, "adding batch")
		t.Logf("Added %d/%d vectors", end, numVectors)
	}

	// Search
	query := make([]float32, dim)
	for i := range query {
		query[i] = float32(i)
	}

	testutil.MeasureTime(t, "Large dataset search", func() {
		results, err := idx.Search(query, k)
		testutil.AssertNoError(t, err, "searching")
		testutil.AssertValidSearchResults(t, results, "large dataset search")
	})

	stats := idx.Stats()
	t.Logf("Index stats: %d vectors, %.2f MB", stats.TotalVectors, stats.MemoryUsageMB)
}

// TestConcurrentOperations tests thread-safe operations
func TestConcurrentOperations(t *testing.T) {
	dim := 128
	idx, err := hnsw.New(dim, "l2", hnsw.DefaultConfig())
	testutil.AssertNoError(t, err, "creating index")

	// Add initial vectors
	vectors := vector.GenerateRandom(1000, dim, 42)
	err = idx.Add(vectors)
	testutil.AssertNoError(t, err, "adding vectors")

	// Concurrent searches
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(id int) {
			query := vectors[id].Data
			_, err := idx.Search(query, 10)
			if err != nil {
				t.Errorf("Concurrent search %d failed: %v", id, err)
			}
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}
}
