package faiss

import (
	"bytes"
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"
)

// Vector represents a single vector with an ID
type Vector struct {
	ID   int64
	Data []float32
	Norm float32
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
	metric  string
	vectors []Vector
	mu      sync.RWMutex
}

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

func (idx *FlatIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.dim, len(v.Data))
		}
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

func (idx *FlatIndex) Search(query []float32, k int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encoder := gob.NewEncoder(gzw)
	if err := encoder.Encode(idx.dim); err != nil {
		return err
	}
	if err := encoder.Encode(idx.metric); err != nil {
		return err
	}
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

	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(gzr)
	if err := decoder.Decode(&idx.dim); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.metric); err != nil {
		return err
	}
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

func (idx *HNSWIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	gzw := gzip.NewWriter(f)
	defer gzw.Close()

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encoder := gob.NewEncoder(gzw)
	metadata := []interface{}{
		idx.dim, idx.metric, idx.M, idx.efConstruction,
		idx.efSearch, idx.maxLevel, idx.entryPoint,
		idx.levelMult, idx.nextID,
	}
	for _, m := range metadata {
		if err := encoder.Encode(m); err != nil {
			return err
		}
	}
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

	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(gzr)
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

func (idx *IVFPQIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	gzw := gzip.NewWriter(f)
	defer gzw.Close()

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encoder := gob.NewEncoder(gzw)
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

	if err := encoder.Encode(len(idx.centroids)); err != nil {
		return err
	}
	for _, c := range idx.centroids {
		if err := encoder.Encode(c); err != nil {
			return err
		}
	}

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

	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(gzr)
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

// ANN Benchmarking

type BenchmarkResult struct {
	IndexType        string
	NumVectors       int
	Dimension        int
	QueryCount       int
	K                int
	Recall           float64
	QueriesPerSecond float64
	AvgQueryTimeMs   float64
	BuildTimeMs      float64
	MemoryUsageMB    float64
	ExtraParams      map[string]interface{}
}

type BenchmarkConfig struct {
	Dataset        []Vector
	Queries        [][]float32
	GroundTruth    [][]int64
	K              int
	IndexType      string
	IndexParams    map[string]interface{}
	SearchParams   map[string]interface{}
	WarmupQueries  int
}

func RunBenchmark(config BenchmarkConfig) (*BenchmarkResult, error) {
	result := &BenchmarkResult{
		IndexType:   config.IndexType,
		NumVectors:  len(config.Dataset),
		Dimension:   len(config.Dataset[0].Data),
		QueryCount:  len(config.Queries),
		K:           config.K,
		ExtraParams: make(map[string]interface{}),
	}

	// Build index
	buildStart := time.Now()
	var idx interface{}
	var err error

	switch config.IndexType {
	case "flat":
		metric := "l2"
		if m, ok := config.IndexParams["metric"].(string); ok {
			metric = m
		}
		flatIdx, e := NewFlatIndex(result.Dimension, metric)
		if e != nil {
			return nil, e
		}
		err = flatIdx.Add(config.Dataset)
		idx = flatIdx

	case "hnsw":
		metric := "l2"
		M := 16
		efConstruction := 200
		if m, ok := config.IndexParams["metric"].(string); ok {
			metric = m
		}
		if m, ok := config.IndexParams["M"].(int); ok {
			M = m
		}
		if ef, ok := config.IndexParams["efConstruction"].(int); ok {
			efConstruction = ef
		}
		hnswIdx, e := NewHNSWIndex(result.Dimension, metric, M, efConstruction)
		if e != nil {
			return nil, e
		}
		err = hnswIdx.Add(config.Dataset)
		if efSearch, ok := config.SearchParams["efSearch"].(int); ok {
			hnswIdx.SetEfSearch(efSearch)
			result.ExtraParams["efSearch"] = efSearch
		}
		idx = hnswIdx
		result.ExtraParams["M"] = M
		result.ExtraParams["efConstruction"] = efConstruction

	case "ivfpq":
		metric := "l2"
		nlist := 100
		M := 8
		nbits := 8
		nprobe := 10
		if m, ok := config.IndexParams["metric"].(string); ok {
			metric = m
		}
		if n, ok := config.IndexParams["nlist"].(int); ok {
			nlist = n
		}
		if m, ok := config.IndexParams["M"].(int); ok {
			M = m
		}
		if nb, ok := config.IndexParams["nbits"].(int); ok {
			nbits = nb
		}
		if np, ok := config.SearchParams["nprobe"].(int); ok {
			nprobe = np
		}

		ivfpqIdx, e := NewIVFPQIndex(result.Dimension, metric, nlist, M, nbits)
		if e != nil {
			return nil, e
		}

		trainSize := len(config.Dataset)
		if trainSize > 10000 {
			trainSize = 10000
		}
		err = ivfpqIdx.Train(config.Dataset[:trainSize])
		if err != nil {
			return nil, err
		}
		err = ivfpqIdx.Add(config.Dataset)
		idx = ivfpqIdx

		result.ExtraParams["nlist"] = nlist
		result.ExtraParams["M"] = M
		result.ExtraParams["nbits"] = nbits
		result.ExtraParams["nprobe"] = nprobe

	default:
		return nil, fmt.Errorf("unknown index type: %s", config.IndexType)
	}

	if err != nil {
		return nil, err
	}

	buildTime := time.Since(buildStart)
	result.BuildTimeMs = float64(buildTime.Milliseconds())

	// Warmup
	for i := 0; i < config.WarmupQueries && i < len(config.Queries); i++ {
		searchIndex(idx, config.Queries[i], config.K, config.SearchParams)
	}

	// Run queries and measure time
	queryStart := time.Now()
	allResults := make([][]SearchResult, len(config.Queries))

	for i, query := range config.Queries {
		res, err := searchIndex(idx, query, config.K, config.SearchParams)
		if err != nil {
			return nil, err
		}
		allResults[i] = res
	}

	queryTime := time.Since(queryStart)
	result.AvgQueryTimeMs = float64(queryTime.Milliseconds()) / float64(len(config.Queries))
	result.QueriesPerSecond = float64(len(config.Queries)) / queryTime.Seconds()

	// Calculate recall
	if len(config.GroundTruth) > 0 {
		totalRecall := 0.0
		for i, gt := range config.GroundTruth {
			if i >= len(allResults) {
				break
			}
			recall := calculateRecall(allResults[i], gt, config.K)
			totalRecall += recall
		}
		result.Recall = totalRecall / float64(len(config.GroundTruth))
	}

	// Estimate memory usage
	result.MemoryUsageMB = estimateMemory(idx, result.NumVectors, result.Dimension)

	return result, nil
}

func searchIndex(idx interface{}, query []float32, k int, params map[string]interface{}) ([]SearchResult, error) {
	switch v := idx.(type) {
	case *FlatIndex:
		return v.Search(query, k)
	case *HNSWIndex:
		return v.Search(query, k)
	case *IVFPQIndex:
		nprobe := 10
		if np, ok := params["nprobe"].(int); ok {
			nprobe = np
		}
		return v.Search(query, k, nprobe)
	default:
		return nil, fmt.Errorf("unsupported index type")
	}
}

func calculateRecall(results []SearchResult, groundTruth []int64, k int) float64 {
	if len(results) == 0 || len(groundTruth) == 0 {
		return 0.0
	}

	gtSet := make(map[int64]bool)
	for i := 0; i < k && i < len(groundTruth); i++ {
		gtSet[groundTruth[i]] = true
	}

	matches := 0
	for i := 0; i < k && i < len(results); i++ {
		if gtSet[results[i].ID] {
			matches++
		}
	}

	return float64(matches) / float64(min(k, len(groundTruth)))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func estimateMemory(idx interface{}, numVectors, dim int) float64 {
	switch v := idx.(type) {
	case *FlatIndex:
		return float64(numVectors*dim*4) / (1024 * 1024)
	case *HNSWIndex:
		vectorMem := numVectors * dim * 4
		edgeMem := numVectors * v.M * 8 * 4
		return float64(vectorMem+edgeMem) / (1024 * 1024)
	case *IVFPQIndex:
		compressedMem := numVectors * v.pq.M * 2
		centroidMem := v.nlist * dim * 4
		codebookMem := v.pq.M * v.pq.ksub * v.pq.dsub * 4
		return float64(compressedMem+centroidMem+codebookMem) / (1024 * 1024)
	default:
		return 0.0
	}
}

// CompareBenchmarks runs multiple benchmarks and compares them
func CompareBenchmarks(configs []BenchmarkConfig) ([]*BenchmarkResult, error) {
	results := make([]*BenchmarkResult, len(configs))
	for i, config := range configs {
		result, err := RunBenchmark(config)
		if err != nil {
			return nil, fmt.Errorf("benchmark %d failed: %v", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// GenerateSyntheticDataset creates synthetic vectors for testing
func GenerateSyntheticDataset(numVectors, dim int, seed int64) []Vector {
	rng := rand.New(rand.NewSource(seed))
	vectors := make([]Vector, numVectors)
	for i := 0; i < numVectors; i++ {
		data := make([]float32, dim)
		for j := 0; j < dim; j++ {
			data[j] = rng.Float32()
		}
		vectors[i] = Vector{
			ID:   int64(i),
			Data: data,
		}
	}
	return vectors
}

// GenerateSyntheticQueries creates synthetic query vectors
func GenerateSyntheticQueries(numQueries, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		query := make([]float32, dim)
		for j := 0; j < dim; j++ {
			query[j] = rng.Float32()
		}
		queries[i] = query
	}
	return queries
}

// ComputeGroundTruth computes exact nearest neighbors using brute force
func ComputeGroundTruth(dataset []Vector, queries [][]float32, k int, metric string) ([][]int64, error) {
	groundTruth := make([][]int64, len(queries))

	for i, query := range queries {
		results := make([]SearchResult, len(dataset))
		queryNorm := float32(0)
		if metric == "cosine" {
			queryNorm = computeNorm(query)
		}

		for j, v := range dataset {
			var dist float32
			if metric == "l2" {
				dist = l2Distance(query, v.Data)
			} else {
				dist = cosineDistance(query, v.Data, queryNorm, v.Norm)
			}
			results[j] = SearchResult{ID: v.ID, Distance: dist}
		}

		sort.Slice(results, func(a, b int) bool {
			return results[a].Distance < results[b].Distance
		})

		gt := make([]int64, k)
		for j := 0; j < k && j < len(results); j++ {
			gt[j] = results[j].ID
		}
		groundTruth[i] = gt
	}

	return groundTruth, nil
}

// FormatBenchmarkResult formats a benchmark result for display
func FormatBenchmarkResult(result *BenchmarkResult) string {
	var buf bytes.Buffer
	buf.WriteString(fmt.Sprintf("Index Type: %s\n", result.IndexType))
	buf.WriteString(fmt.Sprintf("Vectors: %d, Dimension: %d\n", result.NumVectors, result.Dimension))
	buf.WriteString(fmt.Sprintf("Queries: %d, K: %d\n", result.QueryCount, result.K))
	buf.WriteString(fmt.Sprintf("Recall@%d: %.4f\n", result.K, result.Recall))
	buf.WriteString(fmt.Sprintf("QPS: %.2f\n", result.QueriesPerSecond))
	buf.WriteString(fmt.Sprintf("Avg Query Time: %.4f ms\n", result.AvgQueryTimeMs))
	buf.WriteString(fmt.Sprintf("Build Time: %.2f ms\n", result.BuildTimeMs))
	buf.WriteString(fmt.Sprintf("Memory Usage: %.2f MB\n", result.MemoryUsageMB))
	if len(result.ExtraParams) > 0 {
		buf.WriteString(fmt.Sprintf("Extra Params: %v\n", result.ExtraParams))
	}
	return buf.String()
}Lock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}
	if k <= 0 || k > len(idx.vectors) {
		k = len(idx.vectors)
	}

	queryNorm := float32(0)
	if idx.metric == "cosine" {
		queryNorm = computeNorm(query)
		if queryNorm == 0 {
			return nil, fmt.Errorf("zero query vector not allowed")
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

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

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

func (idx *FlatIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

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

// Distance functions
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
	if similarity > 1 {
		similarity = 1
	} else if similarity < -1 {
		similarity = -1
	}
	return 1 - similarity
}

func computeNorm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// ProductQuantizer implements product quantization
type ProductQuantizer struct {
	dim       int
	M         int
	nbits     int
	dsub      int
	ksub      int
	codebooks [][]Vector
	trained   bool
	mu        sync.RWMutex
}

func NewProductQuantizer(dim int, M int, nbits int) (*ProductQuantizer, error) {
	if dim%M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M %d", dim, M)
	}
	if nbits <= 0 || nbits > 16 {
		return nil, fmt.Errorf("nbits must be between 1 and 16")
	}

	dsub := dim / M
	ksub := 1 << nbits

	return &ProductQuantizer{
		dim:       dim,
		M:         M,
		nbits:     nbits,
		dsub:      dsub,
		ksub:      ksub,
		codebooks: make([][]Vector, M),
	}, nil
}

func (pq *ProductQuantizer) Train(vectors []Vector) error {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	if len(vectors) < pq.ksub {
		return fmt.Errorf("need at least %d training vectors", pq.ksub)
	}

	for m := 0; m < pq.M; m++ {
		start := m * pq.dsub
		end := start + pq.dsub

		subvectors := make([]Vector, len(vectors))
		for i, v := range vectors {
			subvectors[i] = Vector{
				ID:   v.ID,
				Data: v.Data[start:end],
			}
		}

		pq.codebooks[m] = kMeans(subvectors, pq.ksub, "l2", 20)
	}

	pq.trained = true
	return nil
}

func (pq *ProductQuantizer) Encode(v []float32) ([]uint16, error) {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	if !pq.trained {
		return nil, fmt.Errorf("quantizer must be trained")
	}
	if len(v) != pq.dim {
		return nil, fmt.Errorf("vector dimension mismatch")
	}

	codes := make([]uint16, pq.M)
	for m := 0; m < pq.M; m++ {
		start := m * pq.dsub
		end := start + pq.dsub
		subvec := v[start:end]

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

func ComputePQDistance(codes []uint16, table [][]float32) float32 {
	var sum float32
	for m, code := range codes {
		dist := table[m][code]
		sum += dist * dist
	}
	return float32(math.Sqrt(float64(sum)))
}

// PQVector stores quantized vector
type PQVector struct {
	ID    int64
	Codes []uint16
}

// IVFPQIndex combines IVF with PQ
type IVFPQIndex struct {
	dim       int
	metric    string
	nlist     int
	pq        *ProductQuantizer
	centroids []Vector
	lists     [][]PQVector
	mu        sync.RWMutex
	trained   bool
}

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

func (idx *IVFPQIndex) Train(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) < idx.nlist {
		return fmt.Errorf("need at least %d vectors to train", idx.nlist)
	}

	idx.centroids = kMeans(vectors, idx.nlist, idx.metric, 10)

	residuals := make([]Vector, len(vectors))
	for i, v := range vectors {
		nearestIdx := idx.findNearestCentroidUnsafe(v.Data)
		centroid := idx.centroids[nearestIdx].Data

		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}
		residuals[i] = Vector{ID: v.ID, Data: residual}
	}

	err := idx.pq.Train(residuals)
	if err != nil {
		return err
	}

	idx.trained = true
	return nil
}

func (idx *IVFPQIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained")
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

		codes, err := idx.pq.Encode(residual)
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

func (idx *IVFPQIndex) Search(query []float32, k int, nprobe int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.trained {
		return nil, fmt.Errorf("index must be trained")
	}
	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}
	if nprobe <= 0 || nprobe > idx.nlist {
		nprobe = idx.nlist
	}

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

		table, err := idx.pq.ComputeDistanceTable(residualQuery)
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

func (idx *IVFPQIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	total := 0
	for _, list := range idx.lists {
		total += len(list)
	}
	return total
}

func (idx *IVFPQIndex) Remove(id int64) error {
	return fmt.Errorf("remove not implemented for IVF+PQ")
}

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

// HNSWIndex implements HNSW
type HNSWIndex struct {
	dim            int
	metric         string
	M              int
	efConstruction int
	efSearch       int
	maxLevel       int
	entryPoint     int64
	vectors        map[int64]*HNSWNode
	levelMult      float64
	mu             sync.RWMutex
	nextID         int64
}

type HNSWNode struct {
	ID    int64
	Data  []float32
	Level int
	Edges [][]int64
}

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

func (idx *HNSWIndex) SetEfSearch(ef int) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.efSearch = ef
}

func (idx *HNSWIndex) Add(vectors []Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, v := range vectors {
		if len(v.Data) != idx.dim {
			return fmt.Errorf("vector dimension mismatch")
		}

		id := v.ID
		if id == 0 {
			id = idx.nextID
			idx.nextID++
		}

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

		for i := 0; i <= level; i++ {
			node.Edges[i] = make([]int64, 0)
		}

		if idx.entryPoint == -1 {
			idx.entryPoint = id
			idx.vectors[id] = node
			continue
		}

		idx.insertNode(node)
		idx.vectors[id] = node
	}
	return nil
}

func (idx *HNSWIndex) randomLevel() int {
	r := -math.Log(1.0 - float64(rand.Intn(1000000))/1000000.0)
	level := int(r * idx.levelMult)
	if level > 16 {
		level = 16
	}
	return level
}

func (idx *HNSWIndex) insertNode(node *HNSWNode) {
	curr := idx.entryPoint
	currDist := idx.distance(node.Data, idx.vectors[curr].Data)

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

	for lc := node.Level; lc >= 0; lc-- {
		candidates := idx.searchLayer(node.Data, curr, idx.efConstruction, lc)

		M := idx.M
		if lc == 0 {
			M *= 2
		}

		neighbors := idx.selectNeighbors(candidates, M)

		for _, neighborID := range neighbors {
			node.Edges[lc] = append(node.Edges[lc], neighborID)
			idx.vectors[neighborID].Edges[lc] = append(idx.vectors[neighborID].Edges[lc], node.ID)

			if len(idx.vectors[neighborID].Edges[lc]) > M {
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

func (idx *HNSWIndex) searchLayer(query []float32, entryPoint int64, ef int, layer int) []SearchResult {
	visited := make(map[int64]bool)
	candidates := make([]SearchResult, 0)
	result := make([]SearchResult, 0)

	d := idx.distance(query, idx.vectors[entryPoint].Data)
	candidates = append(candidates, SearchResult{ID: entryPoint, Distance: d})
	result = append(result, SearchResult{ID: entryPoint, Distance: d})
	visited[entryPoint] = true

	for len(candidates) > 0 {
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Distance < candidates[j].Distance
		})
		curr := candidates[0]
		candidates = candidates[1:]

		if len(result) > 0 {
			sort.Slice(result, func(i, j int) bool {
				return result[i].Distance > result[j].Distance
			})
			if curr.Distance > result[0].Distance {
				break
			}
		}

		node := idx.vectors[curr.ID]
		if layer < len(node.Edges) {
			for _, neighborID := range node.Edges[layer] {
				if !visited[neighborID] {
					visited[neighborID] = true
					d := idx.distance(query, idx.vectors[neighborID].Data)

					if len(result) < ef || d < result[0].Distance {
						candidates = append(candidates, SearchResult{ID: neighborID, Distance: d})
						result = append(result, SearchResult{ID: neighborID, Distance: d})

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

func (idx *HNSWIndex) selectNeighbors(candidates []SearchResult, M int) []int64 {
	if len(candidates) <= M {
		result := make([]int64, len(candidates))
		for i, c := range candidates {
			result[i] = c.ID
		}
		return result
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	result := make([]int64, M)
	for i := 0; i < M; i++ {
		result[i] = candidates[i].ID
	}
	return result
}

func (idx *HNSWIndex) Search(query []float32, k int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}
	if idx.entryPoint == -1 {
		return []SearchResult{}, nil
	}

	curr := idx.entryPoint
	currDist := idx.distance(query, idx.vectors[curr].Data)

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

	candidates := idx.searchLayer(query, curr, idx.efSearch, 0)

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})

	if k < len(candidates) {
		return candidates[:k], nil
	}
	return candidates, nil
}

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

	if idx.entryPoint == id {
		if len(idx.vectors) > 1 {
			for _, n := range idx.vectors {
				if n.ID != id && n.Level == idx.maxLevel {
					idx.entryPoint = n.ID
					break
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

// k-means clustering
func kMeans(vectors []Vector, k int, metric string, maxIter int) []Vector {
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

	for iter := 0; iter < maxIter; iter++ {
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

	if metric == "cosine" {
		for i := range centroids {
			centroids[i].Norm = computeNorm(centroids[i].Data)
		}
	}

	return centroids
}

// Serialization
func (idx *FlatIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	gzw := gzip.NewWriter(f)
	defer gzw.Close()

	idx.mu.R