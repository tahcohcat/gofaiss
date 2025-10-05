package faiss

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