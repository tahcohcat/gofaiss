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