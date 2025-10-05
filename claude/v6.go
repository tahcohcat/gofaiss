// BatchSearch performs batch search for IVFPQIndex
func (idx *IVFPQIndex) BatchSearch(queries [][]float32, k int, nprobe int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k, nprobe)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// Serialization support using encoding/gob

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"io"
	"os"
)

// Serializable interface for index serialization
type Serializable interface {
	Save(w io.Writer) error
	Load(r io.Reader) error
}

// FlatIndex serialization

func (idx *FlatIndex) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encoder := gob.NewEncoder(w)
	
	// Encode metadata
	if err := encoder.Encode(idx.dim); err != nil {
		return err
	}
	if err := encoder.Encode(idx.metric); err != nil {
		return err
	}
	
	// Encode vectors
	if err := encoder.Encode(len(idx.vectors)); err != nil {
		return err
	}
	for _, v := range idx.vectors {
		if err := encoder.Encode(v); err != nil {
			return err
		}
	}
	
	return nil
}

func (idx *FlatIndex) Load(r io.Reader) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(r)
	
	// Decode metadata
	if err := decoder.Decode(&idx.dim); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.metric); err != nil {
		return err
	}
	
	// Decode vectors
	var count int
	if err := decoder.Decode(&count); err != nil {
		return err
	}
	
	idx.vectors = make([]Vector, count)
	for i := 0; i < count; i++ {
		if err := decoder.Decode(&idx.vectors[i]); err != nil {
			return err
		}
	}
	
	return nil
}

func (idx *FlatIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	
	gzw := gzip.NewWriter(f)
	defer gzw.Close()
	
	return idx.Save(gzw)
}

func (idx *FlatIndex) LoadFromFile(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()
	
	return idx.Load(gzr)
}

// HNSWIndex serialization

func (idx *HNSWIndex) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encoder := gob.NewEncoder(w)
	
	// Encode metadata
	metadata := []interface{}{
		idx.dim,
		idx.metric,
		idx.M,
		idx.efConstruction,
		idx.efSearch,
		idx.maxLevel,
		idx.entryPoint,
		idx.levelMult,
		idx.nextID,
	}
	
	for _, m := range metadata {
		if err := encoder.Encode(m); err != nil {
			return err
		}
	}
	
	// Encode nodes
	if err := encoder.Encode(len(idx.vectors)); err != nil {
		return err
	}
	
	for _, node := range idx.vectors {
		if err := encoder.Encode(node); err != nil {
			return err
		}
	}
	
	return nil
}

func (idx *HNSWIndex) Load(r io.Reader) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(r)
	
	// Decode metadata
	if err := decoder.Decode(&idx.dim); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.metric); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.M); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.efConstruction); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.efSearch); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.maxLevel); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.entryPoint); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.levelMult); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.nextID); err != nil {
		return err
	}
	
	// Decode nodes
	var count int
	if err := decoder.Decode(&count); err != nil {
		return err
	}
	
	idx.vectors = make(map[int64]*HNSWNode, count)
	for i := 0; i < count; i++ {
		var node HNSWNode
		if err := decoder.Decode(&node); err != nil {
			return err
		}
		idx.vectors[node.ID] = &node
	}
	
	return nil
}

func (idx *HNSWIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	
	gzw := gzip.NewWriter(f)
	defer gzw.Close()
	
	return idx.Save(gzw)
}

func (idx *HNSWIndex) LoadFromFile(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()
	
	return idx.Load(gzr)
}

// IVFPQIndex serialization

func (idx *IVFPQIndex) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encoder := gob.NewEncoder(w)
	
	// Encode metadata
	if err := encoder.Encode(idx.dim); err != nil {
		return err
	}
	if err := encoder.Encode(idx.metric); err != nil {
		return err
	}
	if err := encoder.Encode(idx.nlist); err != nil {
		return err
	}
	if err := encoder.Encode(idx.trained); err != nil {
		return err
	}
	
	// Encode PQ
	pqData := struct {
		Dim       int
		M         int
		Nbits     int
		Dsub      int
		Ksub      int
		Codebooks [][]Vector
		Trained   bool
	}{
		Dim:       idx.pq.dim,
		M:         idx.pq.M,
		Nbits:     idx.pq.nbits,
		Dsub:      idx.pq.dsub,
		Ksub:      idx.pq.ksub,
		Codebooks: idx.pq.codebooks,
		Trained:   idx.pq.trained,
	}
	if err := encoder.Encode(pqData); err != nil {
		return err
	}
	
	// Encode centroids
	if err := encoder.Encode(len(idx.centroids)); err != nil {
		return err
	}
	for _, c := range idx.centroids {
		if err := encoder.Encode(c); err != nil {
			return err
		}
	}
	
	// Encode inverted lists
	if err := encoder.Encode(len(idx.lists)); err != nil {
		return err
	}
	for _, list := range idx.lists {
		if err := encoder.Encode(len(list)); err != nil {
			return err
		}
		for _, v := range list {
			if err := encoder.Encode(v); err != nil {
				return err
			}
		}
	}
	
	return nil
}

func (idx *IVFPQIndex) Load(r io.Reader) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(r)
	
	// Decode metadata
	if err := decoder.Decode(&idx.dim); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.metric); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.nlist); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.trained); err != nil {
		return err
	}
	
	// Decode PQ
	var pqData struct {
		Dim       int
		M         int
		Nbits     int
		Dsub      int
		Ksub      int
		Codebooks [][]Vector
		Trained   bool
	}
	if err := decoder.Decode(&pqData); err != nil {
		return err
	}
	
	idx.pq = &ProductQuantizer{
		dim:       pqData.Dim,
		M:         pqData.M,
		nbits:     pqData.Nbits,
		dsub:      pqData.Dsub,
		ksub:      pqData.Ksub,
		codebooks: pqData.Codebooks,
		trained:   pqData.Trained,
	}
	
	// Decode centroids
	var centroidCount int
	if err := decoder.Decode(&centroidCount); err != nil {
		return err
	}
	idx.centroids = make([]Vector, centroidCount)
	for i := 0; i < centroidCount; i++ {
		if err := decoder.Decode(&idx.centroids[i]); err != nil {
			return err
		}
	}
	
	// Decode inverted lists
	var listCount int
	if err := decoder.Decode(&listCount); err != nil {
		return err
	}
	idx.lists = make([][]PQVector, listCount)
	for i := 0; i < listCount; i++ {
		var vecCount int
		if err := decoder.Decode(&vecCount); err != nil {
			return err
		}
		idx.lists[i] = make([]PQVector, vecCount)
		for j := 0; j < vecCount; j++ {
			if err := decoder.Decode(&idx.lists[i][j]); err != nil {
				return err
			}
		}
	}
	
	return nil
}

func (idx *IVFPQIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	
	gzw := gzip.NewWriter(f)
	defer gzw.Close()
	
	return idx.Save(gzw)
}

func (idx *IVFPQIndex) LoadFromFile(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()
	
	return idx.Load(gzr)
}

// PQIndex serialization

func (idx *PQIndex) Save(w io.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encoder := gob.NewEncoder(w)
	
	if err := encoder.Encode(idx.dim); err != nil {
		return err
	}
	
	// Encode PQ
	pqData := struct {
		Dim       int
		M         int
		Nbits     int
		Dsub      int
		Ksub      int
		Codebooks [][]Vector
		Trained   bool
	}{
		Dim:       idx.pq.dim,
		M:         idx.pq.M,
		Nbits:     idx.pq.nbits,
		Dsub:      idx.pq.dsub,
		Ksub:      idx.pq.ksub,
		Codebooks: idx.pq.codebooks,
		Trained:   idx.pq.trained,
	}
	if err := encoder.Encode(pqData); err != nil {
		return err
	}
	
	// Encode vectors
	if err := encoder.Encode(len(idx.vectors)); err != nil {
		return err
	}
	for _, v := range idx.vectors {
		if err := encoder.Encode(v); err != nil {
			return err
		}
	}
	
	return nil
}

func (idx *PQIndex) Load(r io.Reader) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(r)
	
	if err := decoder.Decode(&idx.dim); err != nil {
		return err
	}
	
	// Decode PQ
	var pqData struct {
		Dim       int
		M         int
		Nbits     int
		Dsub      int
		Ksub      int
		Codebooks [][]Vector
		Trained   bool
	}
	if err := decoder.Decode(&pqData); err != nil {
		return err
	}
	
	idx.pq = &ProductQuantizer{
		dim:       pqData.Dim,
		M:         pqData.M,
		nbits:     pqData.Nbits,
		dsub:      pqData.Dsub,
		ksub:      pqData.Ksub,
		codebooks: pqData.Codebooks,
		trained:   pqData.Trained,
	}
	
	// Decode vectors
	var count int
	if err := decoder.Decode(&count); err != nil {
		return err
	}
	idx.vectors = make([]PQVector, count)
	for i := 0; i < count; i++ {
		if err := decoder.Decode(&idx.vectors[i]); err != nil {
			return err
		}
	}
	
	return nil
}

func (idx *PQIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	
	gzw := gzip.NewWriter(f)
	defer gzw.Close()
	
	return idx.Save(gzw)
}

func (idx *PQIndex) LoadFromFile(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	
	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()
	
	return idx.Load(gzr)
}

// Utility functions for batch operations and serialization

// SaveIndexToFile is a generic function to save any index
func SaveIndexToFile(idx interface{}, filename string) error {
	switch v := idx.(type) {
	case *FlatIndex:
		return v.SaveToFile(filename)
	case *HNSWIndex:
		return v.SaveToFile(filename)
	case *IVFPQIndex:
		return v.SaveToFile(filename)
	case *PQIndex:
		return v.SaveToFile(filename)
	default:
		return fmt.Errorf("unsupported index type")
	}
}

// ParallelBatchSearch performs parallel batch search using goroutines
func ParallelBatchSearch(idx interface{}, queries [][]float32, k int, workers int) ([][]SearchResult, error) {
	if workers <= 0 {
		workers = 4 // default workers
	}

	results := make([][]SearchResult, len(queries))
	errors := make([]error, len(queries))
	
	// Create work queue
	type job struct {
		index int
		query []float32
	}
	
	jobs := make(chan job, len(queries))
	var wg sync.WaitGroup
	
	// Worker function
	worker := func() {
		defer wg.Done()
		for j := range jobs {
			var res []SearchResult
			var err error
			
			switch v := idx.(type) {
			case *FlatIndex:
				res, err = v.Search(j.query, k)
			case *HNSWIndex:
				res, err = v.Search(j.query, k)
			case *PQIndex:
				res, err = v.Search(j.query, k)
			case *IVFPQIndex:
				res, err = v.Search(j.query, k, 10) // default nprobe
			case *IVFIndex:
				res, err = v.Search(j.query, k, 10)
			default:
				err = fmt.Errorf("unsupported index type")
			}
			
			results[j.index] = res
			errors[j.index] = err
		}
	}
	
	// Start workers
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go worker()
	}
	
	// Send jobs
	for i, query := range queries {
		jobs <- job{index: i, query: query}
	}
	close(jobs)
	
	// Wait for completion
	wg.Wait()
	
	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}
	
	return results, nil
}

// BatchAdd adds multiple vectors efficiently with batching
func BatchAdd(idx interface{}, vectors []Vector, batchSize int) error {
	if batchSize <= 0 {
		batchSize = 1000
	}
	
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}
		
		batch := vectors[i:end]
		
		switch v := idx.(type) {
		case *FlatIndex:
			if err := v.Add(batch); err != nil {
				return err
			}
		case *HNSWIndex:
			if err := v.Add(batch); err != nil {
				return err
			}
		case *IVFPQIndex:
			if err := v.Add(batch); err != nil {
				return err
			}
		case *PQIndex:
			if err := v.Add(batch); err != nil {
				return err
			}
		case *IVFIndex:
			if err := v.Add(batch); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unsupported index type")
		}
	}
	
	return nil
}

// IndexStats provides statistics about an index
type IndexStats struct {
	TotalVectors  int
	Dimension     int
	IndexType     string
	MemoryUsageKB float64
	ExtraInfo     map[string]interface{}
}

// GetIndexStats returns statistics for any index type
func GetIndexStats(idx interface{}) IndexStats {
	stats := IndexStats{
		ExtraInfo: make(map[string]interface{}),
	}
	
	switch v := idx.(type) {
	case *FlatIndex:
		stats.TotalVectors = v.Size()
		stats.Dimension = v.dim
		stats.IndexType = "FlatIndex"
		stats.MemoryUsageKB = float64(stats.TotalVectors*stats.Dimension*4) / 1024.0
		stats.ExtraInfo["metric"] = v.metric
		
	case *HNSWIndex:
		stats.TotalVectors = v.Size()
		stats.Dimension = v.dim
		stats.IndexType = "HNSWIndex"
		// Estimate memory: vectors + graph edges
		vectorMem := stats.TotalVectors * stats.Dimension * 4
		edgeMem := stats.TotalVectors * v.M * 8 * (v.maxLevel + 1) // rough estimate
		stats.MemoryUsageKB = float64(vectorMem+edgeMem) / 1024.0
		stats.ExtraInfo["metric"] = v.metric
		stats.ExtraInfo["M"] = v.M
		stats.ExtraInfo["efConstruction"] = v.efConstruction
		stats.ExtraInfo["efSearch"] = v.efSearch
		stats.ExtraInfo["maxLevel"] = v.maxLevel
		
	case *IVFPQIndex:
		stats.TotalVectors = v.Size()
		stats.Dimension = v.dim
		stats.IndexType = "IVFPQIndex"
		// Memory: compressed vectors + centroids + codebooks
		compressedMem := stats.TotalVectors * v.pq.M * 2 // 2 bytes per code
		centroidMem := v.nlist * stats.Dimension * 4
		codebookMem := v.pq.M * v.pq.ksub * v.pq.dsub * 4
		stats.MemoryUsageKB = float64(compressedMem+centroidMem+codebookMem) / 1024.0
		stats.ExtraInfo["metric"] = v.metric
		stats.ExtraInfo["nlist"] = v.nlist
		stats.ExtraInfo["M"] = v.pq.M
		stats.ExtraInfo["nbits"] = v.pq.nbits
		stats.ExtraInfo["trained"] = v.trained
		stats.ExtraInfo["listSizes"] = v.GetListSizes()
		
	case *PQIndex:
		stats.TotalVectors = v.Size()
		stats.Dimension = v.dim
		stats.IndexType = "PQIndex"
		compressedMem := stats.TotalVectors * v.pq.M * 2
		codebookMem := v.pq.M * v.pq.ksub * v.pq.dsub * 4
		stats.MemoryUsageKB = float64(compressedMem+codebookMem) / 1024.0
		stats.ExtraInfo["M"] = v.pq.M
		stats.ExtraInfo["nbits"] = v.pq.nbits
		
	case *IVFIndex:
		stats.TotalVectors = v.Size()
		stats.Dimension = v.dim
		stats.IndexType = "IVFIndex"
		vectorMem := stats.TotalVectors * stats.Dimension * 4
		centroidMem := v.nlist * stats.Dimension * 4
		stats.MemoryUsageKB = float64(vectorMem+centroidMem) / 1024.0
		stats.ExtraInfo["metric"] = v.metric
		stats.ExtraInfo["nlist"] = v.nlist
		stats.ExtraInfo["trained"] = v.trained
	}
	
	return stats
}

// IndexBuilder provides a fluent API for building indexes
type IndexBuilder struct {
	indexType string
	dim       int
	metric    string
	params    map[string]interface{}
}

// NewIndexBuilder creates a new index builder
func NewIndexBuilder(dim int) *IndexBuilder {
	return &IndexBuilder{
		dim:    dim,
		metric: "l2",
		params: make(map[string]interface{}),
	}
}

// WithMetric sets the distance metric
func (b *IndexBuilder) WithMetric(metric string) *IndexBuilder {
	b.metric = metric
	return b
}

// WithType sets the index type
func (b *IndexBuilder) WithType(indexType string) *IndexBuilder {
	b.indexType = indexType
	return b
}

// WithParam sets a parameter
func (b *IndexBuilder) WithParam(key string, value interface{}) *IndexBuilder {
	b.params[key] = value
	return b
}

// Build creates the index
func (b *IndexBuilder) Build() (interface{}, error) {
	switch b.indexType {
	case "flat":
		return NewFlatIndex(b.dim, b.metric)
		
	case "hnsw":
		M := 16
		efConstruction := 200
		if v, ok := b.params["M"].(int); ok {
			M = v
		}
		if v, ok := b.params["efConstruction"].(int); ok {
			efConstruction = v
		}
		return NewHNSWIndex(b.dim, b.metric, M, efConstruction)
		
	case "ivfpq":
		nlist := 100
		M := 8
		nbits := 8
		if v, ok := b.params["nlist"].(int); ok {
			nlist = v
		}
		if v, ok := b.params["M"].(int); ok {
			M = v
		}
		if v, ok := b.params["nbits"].(int); ok {
			nbits = v
		}
		return NewIVFPQIndex(b.dim, b.metric, nlist, M, nbits)
		
	case "pq":
		M := 8
		nbits := 8
		if v, ok := b.params["M"].(int); ok {
			M = v
		}
		if v, ok := b.params["nbits"].(int); ok {
			nbits = v
		}
		return NewPQIndex(b.dim, M, nbits)
		
	case "ivf":
		nlist := 100
		if v, ok := b.params["nlist"].(int); ok {
			nlist = v
		}
		return NewIVFIndex(b.dim, b.metric, nlist)
		
	default:
		return nil, fmt.Errorf("unknown index type: %s", b.indexType)
	}
}package faiss

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// Vector represents a single vector with an ID
type Vector struct {
	ID     int64
	Data   []float32
	Norm   float32
}

// SearchResult represents a search result with distance
type SearchResult struct {
	ID       int64
	Distance float32
}

// Index interface defines the basic operations for vector indexing
type Index interface {
	Add(vectors []Vector) error
	Search(query []float32, k int) ([]SearchResult, error)
	Remove(id int64) error
	Size() int
}

// FlatIndex implements a simple flat (brute-force) index
type FlatIndex struct {
	dim     int
	metric  string // "l2" or "cosine"
	vectors []Vector
	mu      sync.RWMutex
}

// NewFlatIndex creates a new flat index
func NewFlatIndex(dim int, metric string) (*FlatIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	return &FlatIndex{
		dim:     dim,
		metric:  metric,
		vectors: make([]Vector, 0),
	}, nil
}

// Add adds vectors to the index
func (idx *FlatIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.dim, len(v.Data))
		}
		
		// Compute norm for cosine similarity
		if idx.metric == "cosine" {
			v.Norm = computeNorm(v.Data)
			if v.Norm == 0 {
				return fmt.Errorf("zero vector not allowed for cosine metric")
			}
		}
		
		idx.vectors = append(idx.vectors, v)
	}
	return nil
}

// Search finds the k nearest neighbors
func (idx *FlatIndex) Search(query []float32, k int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.dim, len(query))
	}

	if k <= 0 || k > len(idx.vectors) {
		k = len(idx.vectors)
	}

	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = computeNorm(query)
		if queryNorm == 0 {
			return nil, fmt.Errorf("zero query vector not allowed for cosine metric")
		}
	}

	results := make([]SearchResult, len(idx.vectors))
	for i, v := range idx.vectors {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(query, v.Data)
		} else {
			dist = cosineDistance(query, v.Data, queryNorm, v.Norm)
		}
		results[i] = SearchResult{ID: v.ID, Distance: dist}
	}

	// Sort by distance (ascending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

// Remove removes a vector by ID
func (idx *FlatIndex) Remove(id int64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for i, v := range idx.vectors {
		if v.ID == id {
			idx.vectors = append(idx.vectors[:i], idx.vectors[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("vector with id %d not found", id)
}

// Size returns the number of vectors in the index
func (idx *FlatIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// Distance computation functions

func l2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func cosineDistance(a, b []float32, normA, normB float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	similarity := dot / (normA * normB)
	// Clamp to [-1, 1] to handle floating point errors
	if similarity > 1 {
		similarity = 1
	} else if similarity < -1 {
		similarity = -1
	}
	return 1 - similarity // Convert to distance
}

func computeNorm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// IVFIndex implements an Inverted File index (more efficient for large datasets)
type IVFIndex struct {
	dim       int
	metric    string
	nlist     int // number of clusters
	centroids []Vector
	lists     [][]Vector
	mu        sync.RWMutex
	trained   bool
}

// NewIVFIndex creates a new IVF index
func NewIVFIndex(dim int, metric string, nlist int) (*IVFIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	if nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}
	
	return &IVFIndex{
		dim:    dim,
		metric: metric,
		nlist:  nlist,
		lists:  make([][]Vector, nlist),
	}, nil
}

// Train trains the index using k-means clustering
func (idx *IVFIndex) Train(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) < idx.nlist {
		return fmt.Errorf("need at least %d vectors to train with %d clusters", idx.nlist, idx.nlist)
	}

	// Simple k-means clustering (simplified version)
	idx.centroids = kMeans(vectors, idx.nlist, idx.metric, 10)
	idx.trained = true
	
	return nil
}

// Add adds vectors to the index (requires training first)
func (idx *IVFIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch")
		}
		
		// Find nearest centroid
		nearestList := idx.findNearestCentroid(v.Data)
		idx.lists[nearestList] = append(idx.lists[nearestList], v)
	}
	
	return nil
}

// Search searches for k nearest neighbors (probes nprobe lists)
func (idx *IVFIndex) Search(query []float32, k int, nprobe int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.trained {
		return nil, fmt.Errorf("index must be trained before searching")
	}

	if nprobe <= 0 || nprobe > idx.nlist {
		nprobe = idx.nlist
	}

	// Find nearest centroids to probe
	centroidDists := make([]SearchResult, len(idx.centroids))
	for i, c := range idx.centroids {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(query, c.Data)
		} else {
			queryNorm := computeNorm(query)
			dist = cosineDistance(query, c.Data, queryNorm, c.Norm)
		}
		centroidDists[i] = SearchResult{ID: int64(i), Distance: dist}
	}
	
	sort.Slice(centroidDists, func(i, j int) bool {
		return centroidDists[i].Distance < centroidDists[j].Distance
	})

	// Search in the closest nprobe lists
	var results []SearchResult
	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = computeNorm(query)
	}

	for i := 0; i < nprobe; i++ {
		listIdx := int(centroidDists[i].ID)
		for _, v := range idx.lists[listIdx] {
			var dist float32
			if idx.metric == "l2" {
				dist = l2Distance(query, v.Data)
			} else {
				dist = cosineDistance(query, v.Data, queryNorm, v.Norm)
			}
			results = append(results, SearchResult{ID: v.ID, Distance: dist})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

func (idx *IVFIndex) findNearestCentroid(v []float32) int {
	minDist := float32(math.Inf(1))
	minIdx := 0
	
	for i, c := range idx.centroids {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(v, c.Data)
		} else {
			vNorm := computeNorm(v)
			dist = cosineDistance(v, c.Data, vNorm, c.Norm)
		}
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	return minIdx
}

func (idx *IVFIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	
	total := 0
	for _, list := range idx.lists {
		total += len(list)
	}
	return total
}

// Simple k-means implementation
func kMeans(vectors []Vector, k int, metric string, maxIter int) []Vector {
	// Initialize centroids randomly
	centroids := make([]Vector, k)
	step := len(vectors) / k
	for i := 0; i < k; i++ {
		idx := i * step
		if idx >= len(vectors) {
			idx = len(vectors) - 1
		}
		centroids[i] = Vector{
			ID:   int64(i),
			Data: make([]float32, len(vectors[idx].Data)),
		}
		copy(centroids[i].Data, vectors[idx].Data)
	}

	// Iterate
	for iter := 0; iter < maxIter; iter++ {
		// Assign vectors to nearest centroid
		assignments := make([]int, len(vectors))
		for i, v := range vectors {
			minDist := float32(math.Inf(1))
			minIdx := 0
			for j, c := range centroids {
				dist := l2Distance(v.Data, c.Data)
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}
			assignments[i] = minIdx
		}

		// Update centroids
		for i := range centroids {
			sum := make([]float32, len(centroids[i].Data))
			count := 0
			for j, a := range assignments {
				if a == i {
					for d := range sum {
						sum[d] += vectors[j].Data[d]
					}
					count++
				}
			}
			if count > 0 {
				for d := range sum {
					centroids[i].Data[d] = sum[d] / float32(count)
				}
			}
		}
	}

	// Compute norms for cosine metric
	if metric == "cosine" {
		for i := range centroids {
			centroids[i].Norm = computeNorm(centroids[i].Data)
		}
	}

	return centroids
}

func (idx *IVFIndex) Remove(id int64) error {
	return fmt.Errorf("remove not implemented for IVF index")
}

// ProductQuantizer implements product quantization for vector compression
type ProductQuantizer struct {
	dim       int       // original dimension
	M         int       // number of subquantizers
	nbits     int       // bits per subquantizer (typically 8)
	dsub      int       // dimension of each subspace (dim/M)
	ksub      int       // number of centroids per subquantizer (2^nbits)
	codebooks [][]Vector // M codebooks, each with ksub centroids
	trained   bool
	mu        sync.RWMutex
}

// NewProductQuantizer creates a new product quantizer
func NewProductQuantizer(dim int, M int, nbits int) (*ProductQuantizer, error) {
	if dim%M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M %d", dim, M)
	}
	if nbits <= 0 || nbits > 16 {
		return nil, fmt.Errorf("nbits must be between 1 and 16")
	}

	dsub := dim / M
	ksub := 1 << nbits // 2^nbits

	return &ProductQuantizer{
		dim:       dim,
		M:         M,
		nbits:     nbits,
		dsub:      dsub,
		ksub:      ksub,
		codebooks: make([][]Vector, M),
	}, nil
}

// Train trains the product quantizer using k-means on subvectors
func (pq *ProductQuantizer) Train(vectors []Vector) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(vectors) < pq.ksub {
		return fmt.Errorf("need at least %d training vectors", pq.ksub)
	}

	// Train each subquantizer independently
	for m := 0; m < pq.M; m++ {
		start := m * pq.dsub
		end := start + pq.dsub

		// Extract subvectors
		subvectors := make([]Vector, len(vectors))
		for i, v := range vectors {
			subvectors[i] = Vector{
				ID:   v.ID,
				Data: v.Data[start:end],
			}
		}

		// Train codebook using k-means
		pq.codebooks[m] = kMeans(subvectors, pq.ksub, "l2", 20)
	}

	pq.trained = true
	return nil
}

// Encode encodes a vector into PQ codes
func (pq *ProductQuantizer) Encode(v []float32) ([]uint16, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return nil, fmt.Errorf("quantizer must be trained before encoding")
	}
	if len(v) != pq.dim {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	codes := make([]uint16, pq.M)

	for m := 0; m < pq.M; m++ {
		start := m * pq.dsub
		end := start + pq.dsub
		subvec := v[start:end]

		// Find nearest centroid in this subspace
		minDist := float32(math.Inf(1))
		minIdx := 0

		for k, centroid := range pq.codebooks[m] {
			dist := l2Distance(subvec, centroid.Data)
			if dist < minDist {
				minDist = dist
				minIdx = k
			}
		}

		codes[m] = uint16(minIdx)
	}

	return codes, nil
}

// Decode reconstructs an approximate vector from PQ codes
func (pq *ProductQuantizer) Decode(codes []uint16) ([]float32, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if len(codes) != pq.M {
		return nil, fmt.Errorf("codes length mismatch")
	}

	result := make([]float32, pq.dim)

	for m := 0; m < pq.M; m++ {
		code := codes[m]
		if int(code) >= len(pq.codebooks[m]) {
			return nil, fmt.Errorf("invalid code %d for subquantizer %d", code, m)
		}

		start := m * pq.dsub
		centroid := pq.codebooks[m][code].Data
		copy(result[start:start+pq.dsub], centroid)
	}

	return result, nil
}

// ComputeDistanceTable computes distance table for asymmetric distance computation
func (pq *ProductQuantizer) ComputeDistanceTable(query []float32) ([][]float32, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return nil, fmt.Errorf("quantizer must be trained")
	}

	table := make([][]float32, pq.M)

	for m := 0; m < pq.M; m++ {
		start := m * pq.dsub
		end := start + pq.dsub
		subquery := query[start:end]

		table[m] = make([]float32, pq.ksub)
		for k, centroid := range pq.codebooks[m] {
			table[m][k] = l2Distance(subquery, centroid.Data)
		}
	}

	return table, nil
}

// ComputeDistance computes distance using precomputed table
func ComputePQDistance(codes []uint16, table [][]float32) float32 {
	var sum float32
	for m, code := range codes {
		dist := table[m][code]
		sum += dist * dist
	}
	return float32(math.Sqrt(float64(sum)))
}

// PQIndex implements an index with product quantization
type PQIndex struct {
	dim       int
	pq        *ProductQuantizer
	vectors   []PQVector
	mu        sync.RWMutex
}

type PQVector struct {
	ID    int64
	Codes []uint16
}

// NewPQIndex creates a new PQ index
func NewPQIndex(dim int, M int, nbits int) (*PQIndex, error) {
	pq, err := NewProductQuantizer(dim, M, nbits)
	if err != nil {
		return nil, err
	}

	return &PQIndex{
		dim:     dim,
		pq:      pq,
		vectors: make([]PQVector, 0),
	}, nil
}

// Train trains the PQ index
func (idx *PQIndex) Train(vectors []Vector) error {
	return idx.pq.Train(vectors)
}

// Add adds vectors to the index (encodes them)
func (idx *PQIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vectors {
		codes, err := idx.pq.Encode(v.Data)
		if err != nil {
			return err
		}
		idx.vectors = append(idx.vectors, PQVector{
			ID:    v.ID,
			Codes: codes,
		})
	}

	return nil
}

// Search performs approximate nearest neighbor search
func (idx *PQIndex) Search(query []float32, k int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	// Compute distance table for asymmetric distance
	table, err := idx.pq.ComputeDistanceTable(query)
	if err != nil {
		return nil, err
	}

	results := make([]SearchResult, len(idx.vectors))
	for i, v := range idx.vectors {
		dist := ComputePQDistance(v.Codes, table)
		results[i] = SearchResult{ID: v.ID, Distance: dist}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

func (idx *PQIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// OptimizedProductQuantizer implements OPQ with rotation
type OptimizedProductQuantizer struct {
	pq             *ProductQuantizer
	rotation       [][]float32 // rotation matrix
	rotationInv    [][]float32 // inverse rotation matrix
	trained        bool
	mu             sync.RWMutex
}

// NewOptimizedProductQuantizer creates a new OPQ
func NewOptimizedProductQuantizer(dim int, M int, nbits int) (*OptimizedProductQuantizer, error) {
	pq, err := NewProductQuantizer(dim, M, nbits)
	if err != nil {
		return nil, err
	}

	return &OptimizedProductQuantizer{
		pq: pq,
	}, nil
}

// Train trains the OPQ (simplified: uses PCA-like rotation)
func (opq *OptimizedProductQuantizer) Train(vectors []Vector) error {
	opq.mu.Lock()
	defer opq.mu.Unlock()

	if len(vectors) < opq.pq.dim {
		return fmt.Errorf("need at least %d training vectors", opq.pq.dim)
	}

	// Compute rotation matrix using simplified approach (identity for now)
	// In a full implementation, this would use iterative optimization
	opq.rotation = makeIdentityMatrix(opq.pq.dim)
	opq.rotationInv = makeIdentityMatrix(opq.pq.dim)

	// Apply rotation and train PQ
	rotatedVectors := make([]Vector, len(vectors))
	for i, v := range vectors {
		rotated := matrixVectorMultiply(opq.rotation, v.Data)
		rotatedVectors[i] = Vector{
			ID:   v.ID,
			Data: rotated,
		}
	}

	err := opq.pq.Train(rotatedVectors)
	if err != nil {
		return err
	}

	opq.trained = true
	return nil
}

// Encode encodes a vector with rotation
func (opq *OptimizedProductQuantizer) Encode(v []float32) ([]uint16, error) {
	opq.mu.RLock()
	defer opq.mu.RUnlock()

	if !opq.trained {
		return nil, fmt.Errorf("OPQ must be trained before encoding")
	}

	rotated := matrixVectorMultiply(opq.rotation, v)
	return opq.pq.Encode(rotated)
}

// ComputeDistanceTable computes distance table with rotation
func (opq *OptimizedProductQuantizer) ComputeDistanceTable(query []float32) ([][]float32, error) {
	opq.mu.RLock()
	defer opq.mu.RUnlock()

	if !opq.trained {
		return nil, fmt.Errorf("OPQ must be trained")
	}

	rotated := matrixVectorMultiply(opq.rotation, query)
	return opq.pq.ComputeDistanceTable(rotated)
}

// OPQIndex implements an index with optimized product quantization
type OPQIndex struct {
	dim       int
	opq       *OptimizedProductQuantizer
	vectors   []PQVector
	mu        sync.RWMutex
}

// NewOPQIndex creates a new OPQ index
func NewOPQIndex(dim int, M int, nbits int) (*OPQIndex, error) {
	opq, err := NewOptimizedProductQuantizer(dim, M, nbits)
	if err != nil {
		return nil, err
	}

	return &OPQIndex{
		dim:     dim,
		opq:     opq,
		vectors: make([]PQVector, 0),
	}, nil
}

// Train trains the OPQ index
func (idx *OPQIndex) Train(vectors []Vector) error {
	return idx.opq.Train(vectors)
}

// Add adds vectors to the index
func (idx *OPQIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vectors {
		codes, err := idx.opq.Encode(v.Data)
		if err != nil {
			return err
		}
		idx.vectors = append(idx.vectors, PQVector{
			ID:    v.ID,
			Codes: codes,
		})
	}

	return nil
}

// Search performs approximate nearest neighbor search
func (idx *OPQIndex) Search(query []float32, k int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	table, err := idx.opq.ComputeDistanceTable(query)
	if err != nil {
		return nil, err
	}

	results := make([]SearchResult, len(idx.vectors))
	for i, v := range idx.vectors {
		dist := ComputePQDistance(v.Codes, table)
		results[i] = SearchResult{ID: v.ID, Distance: dist}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

func (idx *OPQIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// Helper functions for matrix operations

func makeIdentityMatrix(n int) [][]float32 {
	mat := make([][]float32, n)
	for i := range mat {
		mat[i] = make([]float32, n)
		mat[i][i] = 1.0
	}
	return mat
}

func matrixVectorMultiply(mat [][]float32, vec []float32) []float32 {
	n := len(mat)
	result := make([]float32, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			result[i] += mat[i][j] * vec[j]
		}
	}
	return result
}

// IVFPQIndex combines IVF with Product Quantization for memory-efficient indexing
type IVFPQIndex struct {
	dim       int
	metric    string
	nlist     int // number of clusters
	pq        *ProductQuantizer
	centroids []Vector
	lists     [][]PQVector // inverted lists store PQ codes
	mu        sync.RWMutex
	trained   bool
}

// NewIVFPQIndex creates a new IVF+PQ index
func NewIVFPQIndex(dim int, metric string, nlist int, M int, nbits int) (*IVFPQIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	if nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}

	pq, err := NewProductQuantizer(dim, M, nbits)
	if err != nil {
		return nil, err
	}

	return &IVFPQIndex{
		dim:    dim,
		metric: metric,
		nlist:  nlist,
		pq:     pq,
		lists:  make([][]PQVector, nlist),
	}, nil
}

// Train trains both the IVF clustering and PQ quantization
func (idx *IVFPQIndex) Train(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) < idx.nlist {
		return fmt.Errorf("need at least %d vectors to train with %d clusters", idx.nlist, idx.nlist)
	}

	// Step 1: Train IVF (coarse quantizer) using k-means
	idx.centroids = kMeans(vectors, idx.nlist, idx.metric, 10)

	// Step 2: Assign vectors to clusters and compute residuals
	residuals := make([]Vector, len(vectors))
	for i, v := range vectors {
		// Find nearest centroid
		nearestIdx := idx.findNearestCentroidUnsafe(v.Data)
		centroid := idx.centroids[nearestIdx].Data

		// Compute residual (difference from centroid)
		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}

		residuals[i] = Vector{
			ID:   v.ID,
			Data: residual,
		}
	}

	// Step 3: Train PQ on residuals
	err := idx.pq.Train(residuals)
	if err != nil {
		return err
	}

	idx.trained = true
	return nil
}

// Add adds vectors to the index (requires training first)
func (idx *IVFPQIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch")
		}

		// Find nearest centroid
		nearestList := idx.findNearestCentroidUnsafe(v.Data)
		centroid := idx.centroids[nearestList].Data

		// Compute residual
		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}

		// Encode residual with PQ
		codes, err := idx.pq.Encode(residual)
		if err != nil {
			return err
		}

		// Add to inverted list
		idx.lists[nearestList] = append(idx.lists[nearestList], PQVector{
			ID:    v.ID,
			Codes: codes,
		})
	}

	return nil
}

// Search searches for k nearest neighbors (probes nprobe lists)
func (idx *IVFPQIndex) Search(query []float32, k int, nprobe int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.trained {
		return nil, fmt.Errorf("index must be trained before searching")
	}

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	if nprobe <= 0 || nprobe > idx.nlist {
		nprobe = idx.nlist
	}

	// Find nearest centroids to probe
	centroidDists := make([]SearchResult, len(idx.centroids))
	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = computeNorm(query)
	}

	for i, c := range idx.centroids {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(query, c.Data)
		} else {
			dist = cosineDistance(query, c.Data, queryNorm, c.Norm)
		}
		centroidDists[i] = SearchResult{ID: int64(i), Distance: dist}
	}

	sort.Slice(centroidDists, func(i, j int) bool {
		return centroidDists[i].Distance < centroidDists[j].Distance
	})

	// Search in the closest nprobe lists
	var results []SearchResult

	for i := 0; i < nprobe; i++ {
		listIdx := int(centroidDists[i].ID)
		if len(idx.lists[listIdx]) == 0 {
			continue
		}

		centroid := idx.centroids[listIdx].Data

		// Compute residual query
		residualQuery := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residualQuery[d] = query[d] - centroid[d]
		}

		// Compute distance table for residuals
		table, err := idx.pq.ComputeDistanceTable(residualQuery)
		if err != nil {
			return nil, err
		}

		// Compute distances for all vectors in this list
		for _, v := range idx.lists[listIdx] {
			dist := ComputePQDistance(v.Codes, table)
			results = append(results, SearchResult{ID: v.ID, Distance: dist})
		}
	}

	// Sort results by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

// findNearestCentroidUnsafe finds nearest centroid without locking (internal use)
func (idx *IVFPQIndex) findNearestCentroidUnsafe(v []float32) int {
	minDist := float32(math.Inf(1))
	minIdx := 0

	for i, c := range idx.centroids {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(v, c.Data)
		} else {
			vNorm := computeNorm(v)
			dist = cosineDistance(v, c.Data, vNorm, c.Norm)
		}
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	return minIdx
}

// Size returns the total number of vectors in the index
func (idx *IVFPQIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	total := 0
	for _, list := range idx.lists {
		total += len(list)
	}
	return total
}

// GetListSizes returns the sizes of all inverted lists (useful for debugging)
func (idx *IVFPQIndex) GetListSizes() []int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	sizes := make([]int, len(idx.lists))
	for i, list := range idx.lists {
		sizes[i] = len(list)
	}
	return sizes
}

// IVFOPQIndex combines IVF with Optimized Product Quantization
type IVFOPQIndex struct {
	dim       int
	metric    string
	nlist     int
	opq       *OptimizedProductQuantizer
	centroids []Vector
	lists     [][]PQVector
	mu        sync.RWMutex
	trained   bool
}

// NewIVFOPQIndex creates a new IVF+OPQ index
func NewIVFOPQIndex(dim int, metric string, nlist int, M int, nbits int) (*IVFOPQIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	if nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}

	opq, err := NewOptimizedProductQuantizer(dim, M, nbits)
	if err != nil {
		return nil, err
	}

	return &IVFOPQIndex{
		dim:    dim,
		metric: metric,
		nlist:  nlist,
		opq:    opq,
		lists:  make([][]PQVector, nlist),
	}, nil
}

// Train trains both IVF and OPQ
func (idx *IVFOPQIndex) Train(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) < idx.nlist {
		return fmt.Errorf("need at least %d vectors to train", idx.nlist)
	}

	// Train IVF
	idx.centroids = kMeans(vectors, idx.nlist, idx.metric, 10)

	// Compute residuals
	residuals := make([]Vector, len(vectors))
	for i, v := range vectors {
		nearestIdx := idx.findNearestCentroidUnsafe(v.Data)
		centroid := idx.centroids[nearestIdx].Data

		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}

		residuals[i] = Vector{
			ID:   v.ID,
			Data: residual,
		}
	}

	// Train OPQ on residuals
	err := idx.opq.Train(residuals)
	if err != nil {
		return err
	}

	idx.trained = true
	return nil
}

// Add adds vectors to the index
func (idx *IVFOPQIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch")
		}

		nearestList := idx.findNearestCentroidUnsafe(v.Data)
		centroid := idx.centroids[nearestList].Data

		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}

		codes, err := idx.opq.Encode(residual)
		if err != nil {
			return err
		}

		idx.lists[nearestList] = append(idx.lists[nearestList], PQVector{
			ID:    v.ID,
			Codes: codes,
		})
	}

	return nil
}

// Search searches for k nearest neighbors
func (idx *IVFOPQIndex) Search(query []float32, k int, nprobe int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.trained {
		return nil, fmt.Errorf("index must be trained before searching")
	}

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	if nprobe <= 0 || nprobe > idx.nlist {
		nprobe = idx.nlist
	}

	// Find nearest centroids
	centroidDists := make([]SearchResult, len(idx.centroids))
	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = computeNorm(query)
	}

	for i, c := range idx.centroids {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(query, c.Data)
		} else {
			dist = cosineDistance(query, c.Data, queryNorm, c.Norm)
		}
		centroidDists[i] = SearchResult{ID: int64(i), Distance: dist}
	}

	sort.Slice(centroidDists, func(i, j int) bool {
		return centroidDists[i].Distance < centroidDists[j].Distance
	})

	var results []SearchResult

	for i := 0; i < nprobe; i++ {
		listIdx := int(centroidDists[i].ID)
		if len(idx.lists[listIdx]) == 0 {
			continue
		}

		centroid := idx.centroids[listIdx].Data

		residualQuery := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residualQuery[d] = query[d] - centroid[d]
		}

		table, err := idx.opq.ComputeDistanceTable(residualQuery)
		if err != nil {
			return nil, err
		}

		for _, v := range idx.lists[listIdx] {
			dist := ComputePQDistance(v.Codes, table)
			results = append(results, SearchResult{ID: v.ID, Distance: dist})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

func (idx *IVFOPQIndex) findNearestCentroidUnsafe(v []float32) int {
	minDist := float32(math.Inf(1))
	minIdx := 0

	for i, c := range idx.centroids {
		var dist float32
		if idx.metric == "l2" {
			dist = l2Distance(v, c.Data)
		} else {
			vNorm := computeNorm(v)
			dist = cosineDistance(v, c.Data, vNorm, c.Norm)
		}
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	return minIdx
}

func (idx *IVFOPQIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	total := 0
	for _, list := range idx.lists {
		total += len(list)
	}
	return total
}

// HNSWIndex implements Hierarchical Navigable Small World graph index
type HNSWIndex struct {
	dim            int
	metric         string
	M              int     // number of connections per layer
	efConstruction int     // size of dynamic candidate list during construction
	efSearch       int     // size of dynamic candidate list during search
	maxLevel       int     // maximum layer
	entryPoint     int64   // entry point node ID
	vectors        map[int64]*HNSWNode
	levelMult      float64 // level generation multiplier
	mu             sync.RWMutex
	nextID         int64
}

type HNSWNode struct {
	ID     int64
	Data   []float32
	Level  int
	Edges  [][]int64 // edges at each level
}

// NewHNSWIndex creates a new HNSW index
func NewHNSWIndex(dim int, metric string, M int, efConstruction int) (*HNSWIndex, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if metric != "l2" && metric != "cosine" {
		return nil, fmt.Errorf("metric must be 'l2' or 'cosine'")
	}
	if M <= 0 {
		return nil, fmt.Errorf("M must be positive")
	}
	if efConstruction <= 0 {
		return nil, fmt.Errorf("efConstruction must be positive")
	}

	return &HNSWIndex{
		dim:            dim,
		metric:         metric,
		M:              M,
		efConstruction: efConstruction,
		efSearch:       efConstruction,
		maxLevel:       0,
		entryPoint:     -1,
		vectors:        make(map[int64]*HNSWNode),
		levelMult:      1.0 / math.Log(float64(M)),
		nextID:         0,
	}, nil
}

// SetEfSearch sets the search-time ef parameter
func (idx *HNSWIndex) SetEfSearch(ef int) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.efSearch = ef
}

// Add adds a single vector to the index
func (idx *HNSWIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch")
		}

		// Use provided ID or generate new one
		id := v.ID
		if id == 0 {
			id = idx.nextID
			idx.nextID++
		}

		// Determine level for this node
		level := idx.randomLevel()
		if level > idx.maxLevel {
			idx.maxLevel = level
		}

		node := &HNSWNode{
			ID:    id,
			Data:  v.Data,
			Level: level,
			Edges: make([][]int64, level+1),
		}

		// Initialize edge lists
		for i := 0; i <= level; i++ {
			node.Edges[i] = make([]int64, 0)
		}

		// If this is the first node, set it as entry point
		if idx.entryPoint == -1 {
			idx.entryPoint = id
			idx.vectors[id] = node
			continue
		}

		// Find nearest neighbors and connect
		idx.insertNode(node)
		idx.vectors[id] = node
	}

	return nil
}

// randomLevel generates a random level for a new node
func (idx *HNSWIndex) randomLevel() int {
	r := -math.Log(float64(1) - (float64(hash64(idx.nextID)%1000000) / 1000000.0))
	level := int(r * idx.levelMult)
	if level > 16 {
		level = 16 // cap at 16 levels
	}
	return level
}

func hash64(x int64) uint64 {
	h := uint64(x)
	h ^= h >> 33
	h *= 0xff51afd7ed558ccd
	h ^= h >> 33
	h *= 0xc4ceb9fe1a85ec53
	h ^= h >> 33
	return h
}

// insertNode inserts a node into the graph
func (idx *HNSWIndex) insertNode(node *HNSWNode) {
	// Start from entry point at top layer
	curr := idx.entryPoint
	currDist := idx.distance(node.Data, idx.vectors[curr].Data)

	// Greedy search through upper layers
	for lc := idx.maxLevel; lc > node.Level; lc-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range idx.vectors[curr].Edges[lc] {
				d := idx.distance(node.Data, idx.vectors[neighborID].Data)
				if d < currDist {
					currDist = d
					curr = neighborID
					changed = true
				}
			}
		}
	}

	// Insert at each layer from node.Level down to 0
	for lc := node.Level; lc >= 0; lc-- {
		candidates := idx.searchLayer(node.Data, curr, idx.efConstruction, lc)

		// Select M nearest neighbors
		M := idx.M
		if lc == 0 {
			M *= 2 // more connections at layer 0
		}

		neighbors := idx.selectNeighbors(candidates, M)

		// Add bidirectional links
		for _, neighborID := range neighbors {
			node.Edges[lc] = append(node.Edges[lc], neighborID)
			idx.vectors[neighborID].Edges[lc] = append(idx.vectors[neighborID].Edges[lc], node.ID)

			// Prune neighbors if needed
			if len(idx.vectors[neighborID].Edges[lc]) > M {
				// Keep M best connections
				neighborNode := idx.vectors[neighborID]
				candList := make([]SearchResult, len(neighborNode.Edges[lc]))
				for i, nid := range neighborNode.Edges[lc] {
					d := idx.distance(neighborNode.Data, idx.vectors[nid].Data)
					candList[i] = SearchResult{ID: nid, Distance: d}
				}
				sort.Slice(candList, func(i, j int) bool {
					return candList[i].Distance < candList[j].Distance
				})
				neighborNode.Edges[lc] = make([]int64, M)
				for i := 0; i < M; i++ {
					neighborNode.Edges[lc][i] = candList[i].ID
				}
			}
		}

		if len(candidates) > 0 {
			curr = candidates[0].ID
		}
	}
}

// searchLayer performs search at a specific layer
func (idx *HNSWIndex) searchLayer(query []float32, entryPoint int64, ef int, layer int) []SearchResult {
	visited := make(map[int64]bool)
	candidates := make([]SearchResult, 0)
	result := make([]SearchResult, 0)

	d := idx.distance(query, idx.vectors[entryPoint].Data)
	candidates = append(candidates, SearchResult{ID: entryPoint, Distance: d})
	result = append(result, SearchResult{ID: entryPoint, Distance: d})
	visited[entryPoint] = true

	for len(candidates) > 0 {
		// Pop nearest candidate
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Distance < candidates[j].Distance
		})
		curr := candidates[0]
		candidates = candidates[1:]

		// Check if we should continue
		if len(result) > 0 {
			sort.Slice(result, func(i, j int) bool {
				return result[i].Distance > result[j].Distance
			})
			if curr.Distance > result[0].Distance {
				break
			}
		}

		// Explore neighbors
		node := idx.vectors[curr.ID]
		if layer < len(node.Edges) {
			for _, neighborID := range node.Edges[layer] {
				if !visited[neighborID] {
					visited[neighborID] = true
					d := idx.distance(query, idx.vectors[neighborID].Data)

					if len(result) < ef || d < result[0].Distance {
						candidates = append(candidates, SearchResult{ID: neighborID, Distance: d})
						result = append(result, SearchResult{ID: neighborID, Distance: d})

						// Keep only ef best results
						if len(result) > ef {
							sort.Slice(result, func(i, j int) bool {
								return result[i].Distance < result[j].Distance
							})
							result = result[:ef]
						}
					}
				}
			}
		}
	}

	return result
}

// selectNeighbors selects M best neighbors using heuristic
func (idx *HNSWIndex) selectNeighbors(candidates []SearchResult, M int) []int64 {
	if len(candidates) <= M {
		result := make([]int64, len(candidates))
		for i, c := range candidates {
			result[i] = c.ID
		}
		return result
	}

	// Sort by distance
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	result := make([]int64, M)
	for i := 0; i < M; i++ {
		result[i] = candidates[i].ID
	}
	return result
}

// Search performs k-nearest neighbor search
func (idx *HNSWIndex) Search(query []float32, k int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	if idx.entryPoint == -1 {
		return []SearchResult{}, nil
	}

	// Start from entry point at top layer
	curr := idx.entryPoint
	currDist := idx.distance(query, idx.vectors[curr].Data)

	// Greedy search through upper layers
	for lc := idx.maxLevel; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			node := idx.vectors[curr]
			if lc < len(node.Edges) {
				for _, neighborID := range node.Edges[lc] {
					d := idx.distance(query, idx.vectors[neighborID].Data)
					if d < currDist {
						currDist = d
						curr = neighborID
						changed = true
					}
				}
			}
		}
	}

	// Search at layer 0 with efSearch
	candidates := idx.searchLayer(query, curr, idx.efSearch, 0)

	// Return top k
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	if k < len(candidates) {
		return candidates[:k], nil
	}
	return candidates, nil
}

// BatchSearch performs batch search for multiple queries
func (idx *HNSWIndex) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

func (idx *HNSWIndex) distance(a, b []float32) float32 {
	if idx.metric == "l2" {
		return l2Distance(a, b)
	}
	normA := computeNorm(a)
	normB := computeNorm(b)
	return cosineDistance(a, b, normA, normB)
}

func (idx *HNSWIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

func (idx *HNSWIndex) Remove(id int64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	node, exists := idx.vectors[id]
	if !exists {
		return fmt.Errorf("node %d not found", id)
	}

	// Remove all edges pointing to this node
	for _, otherNode := range idx.vectors {
		for lc := 0; lc < len(otherNode.Edges); lc++ {
			filtered := make([]int64, 0)
			for _, nid := range otherNode.Edges[lc] {
				if nid != id {
					filtered = append(filtered, nid)
				}
			}
			otherNode.Edges[lc] = filtered
		}
	}

	// Update entry point if needed
	if idx.entryPoint == id {
		if len(idx.vectors) > 1 {
			// Find new entry point at highest level
			for _, n := range idx.vectors {
				if n.ID != id && n.Level == idx.maxLevel {
					idx.entryPoint = n.ID
					break
				}
			}
			// If no node at max level, decrease max level
			if idx.entryPoint == id {
				idx.maxLevel--
				for _, n := range idx.vectors {
					if n.ID != id && n.Level == idx.maxLevel {
						idx.entryPoint = n.ID
						break
					}
				}
			}
		} else {
			idx.entryPoint = -1
			idx.maxLevel = 0
		}
	}

	delete(idx.vectors, id)
	return nil
}

// Batch operations for all index types

// BatchSearch performs batch search for FlatIndex
func (idx *FlatIndex) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// BatchSearch performs batch search for IVFIndex
func (idx *IVFIndex) BatchSearch(queries [][]float32, k int, nprobe int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k, nprobe)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// BatchSearch performs batch search for PQIndex
func (idx *PQIndex) BatchSearch(queries [][]float32, k int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// BatchSearch performs batch search for IVFPQIndex
func (idx *IVFPQIndex) BatchSearch(queries [][]float32, k int, nprobe int) ([][]SearchResult, error) {
	results := make([][]SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k, nprobe)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}