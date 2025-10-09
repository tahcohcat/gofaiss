package pq

import (
	"testing"

	"github.com/tahcohcat/gofaiss/pkg/vector"
)

func TestPQBasic(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Train the index
	trainVectors := vector.GenerateRandom(1000, 8, 42)
	if err := idx.Train(trainVectors); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}

	if !idx.IsTrained() {
		t.Error("Index should be trained")
	}

	// Add vectors
	vectors := vector.GenerateRandom(100, 8, 43)
	if err := idx.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	// Search
	query := vectors[0].Data
	results, err := idx.Search(query, 10)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if len(results) != 10 {
		t.Errorf("Expected 10 results, got %d", len(results))
	}
}

func TestPQInvalidConfig(t *testing.T) {
	tests := []struct {
		name   string
		dim    int
		config Config
	}{
		{
			name:   "zero dimension",
			dim:    0,
			config: Config{M: 4, Nbits: 8},
		},
		{
			name:   "M not divisible",
			dim:    10,
			config: Config{M: 3, Nbits: 8},
		},
		{
			name:   "invalid nbits",
			dim:    8,
			config: Config{M: 4, Nbits: 0},
		},
		{
			name:   "nbits too large",
			dim:    8,
			config: Config{M: 4, Nbits: 17},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewIndex(tt.dim, tt.config)
			if err == nil {
				t.Error("Expected error but got none")
			}
		})
	}
}

func TestPQAddBeforeTrain(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	vectors := vector.GenerateRandom(10, 8, 42)
	err = idx.Add(vectors)
	if err == nil {
		t.Error("Expected error when adding before training")
	}
}

func TestPQSearchBeforeTrain(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	query := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	_, err = idx.Search(query, 10)
	if err == nil {
		t.Error("Expected error when searching before training")
	}
}

func TestPQInsufficientTrainingData(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	// Too few vectors for training (need at least Ksub = 256)
	trainVectors := vector.GenerateRandom(10, 8, 42)
	err = idx.Train(trainVectors)
	if err == nil {
		t.Error("Expected error with insufficient training data")
	}
}

func TestPQBatchSearch(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	trainVectors := vector.GenerateRandom(1000, 8, 42)
	if err := idx.Train(trainVectors); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}

	vectors := vector.GenerateRandom(100, 8, 43)
	if err := idx.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	queries := [][]float32{
		vectors[0].Data,
		vectors[1].Data,
	}

	results, err := idx.BatchSearch(queries, 5)
	if err != nil {
		t.Fatalf("Failed to batch search: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 result sets, got %d", len(results))
	}

	for i, res := range results {
		if len(res) != 5 {
			t.Errorf("Query %d: expected 5 results, got %d", i, len(res))
		}
	}
}

func TestPQDimensionMismatch(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	trainVectors := vector.GenerateRandom(1000, 8, 42)
	if err := idx.Train(trainVectors); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}

	wrongDimVector := []vector.Vector{
		{ID: 1, Data: []float32{1, 2, 3}}, // Wrong dimension
	}

	err = idx.Add(wrongDimVector)
	if err == nil {
		t.Error("Expected dimension mismatch error")
	}
}

func TestPQStats(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	trainVectors := vector.GenerateRandom(1000, 8, 42)
	if err := idx.Train(trainVectors); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}

	vectors := vector.GenerateRandom(100, 8, 43)
	if err := idx.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	stats := idx.Stats()

	if stats.TotalVectors != 100 {
		t.Errorf("Expected 100 vectors, got %d", stats.TotalVectors)
	}
	if stats.Dimension != 8 {
		t.Errorf("Expected dimension 8, got %d", stats.Dimension)
	}
	if stats.IndexType != "PQ" {
		t.Errorf("Expected index type PQ, got %s", stats.IndexType)
	}

	// Check compression ratio
	if extraInfo, ok := stats.ExtraInfo["compressionRatio"].(float64); ok {
		if extraInfo <= 1 {
			t.Errorf("Expected compression ratio > 1, got %f", extraInfo)
		}
	}
}

func TestPQEmptySearch(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	trainVectors := vector.GenerateRandom(1000, 8, 42)
	if err := idx.Train(trainVectors); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}

	// No vectors added
	query := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	results, err := idx.Search(query, 10)
	if err != nil {
		t.Fatalf("Failed to search: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("Expected 0 results from empty index, got %d", len(results))
	}
}

func TestPQSize(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(8, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	trainVectors := vector.GenerateRandom(1000, 8, 42)
	if err := idx.Train(trainVectors); err != nil {
		t.Fatalf("Failed to train: %v", err)
	}

	if idx.Size() != 0 {
		t.Errorf("Expected size 0, got %d", idx.Size())
	}

	vectors := vector.GenerateRandom(50, 8, 43)
	if err := idx.Add(vectors); err != nil {
		t.Fatalf("Failed to add vectors: %v", err)
	}

	if idx.Size() != 50 {
		t.Errorf("Expected size 50, got %d", idx.Size())
	}
}

func TestPQDimension(t *testing.T) {
	config := Config{M: 4, Nbits: 8}
	idx, err := NewIndex(16, config)
	if err != nil {
		t.Fatalf("Failed to create index: %v", err)
	}

	if idx.Dimension() != 16 {
		t.Errorf("Expected dimension 16, got %d", idx.Dimension())
	}
}

func TestPQDifferentNbits(t *testing.T) {
	tests := []struct {
		name  string
		nbits int
		dim   int
		M     int
	}{
		{"4-bit quantization", 4, 8, 4},
		{"8-bit quantization", 8, 8, 4},
		//{"16-bit quantization", 16, 8, 4}, // times out on GitHub Actions
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := Config{M: tt.M, Nbits: tt.nbits}
			idx, err := NewIndex(tt.dim, config)
			if err != nil {
				t.Fatalf("Failed to create index: %v", err)
			}

			// Ensure Ksub is calculated correctly
			expectedKsub := 1 << tt.nbits
			if idx.Ksub != expectedKsub {
				t.Errorf("Expected Ksub=%d, got %d", expectedKsub, idx.Ksub)
			}

			trainVectors := vector.GenerateRandom(expectedKsub*10, tt.dim, 42)
			if err := idx.Train(trainVectors); err != nil {
				t.Fatalf("Failed to train: %v", err)
			}
		})
	}
}

func BenchmarkPQTrain(b *testing.B) {
	config := Config{M: 8, Nbits: 8}
	vectors := vector.GenerateRandom(10000, 128, 42)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, _ := NewIndex(128, config)
		_ = idx.Train(vectors)
	}
}

func BenchmarkPQAdd(b *testing.B) {
	config := Config{M: 8, Nbits: 8}
	idx, _ := NewIndex(128, config)

	trainVectors := vector.GenerateRandom(10000, 128, 42)
	_ =idx.Train(trainVectors)

	vectors := vector.GenerateRandom(1000, 128, 43)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = idx.Add(vectors)
	}
}

func BenchmarkPQSearch(b *testing.B) {
	config := Config{M: 8, Nbits: 8}
	idx, _ := NewIndex(128, config)

	trainVectors := vector.GenerateRandom(10000, 128, 42)
	_ = idx.Train(trainVectors)

	vectors := vector.GenerateRandom(10000, 128, 43)
	_ = idx.Add(vectors)

	query := vectors[0].Data

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(query, 10)
	}
}

func BenchmarkPQBatchSearch(b *testing.B) {
	config := Config{M: 8, Nbits: 8}
	idx, _ := NewIndex(128, config)

	trainVectors := vector.GenerateRandom(10000, 128, 42)
	_ = idx.Train(trainVectors)

	vectors := vector.GenerateRandom(1000, 128, 43)
	_ = idx.Add(vectors)

	queries := make([][]float32, 10)
	for i := range queries {
		queries[i] = vectors[i].Data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.BatchSearch(queries, 10)
	}
}
