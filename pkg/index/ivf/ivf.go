package ivf

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/tahcohcat/gofaiss/pkg/index/stats"
	"github.com/tahcohcat/gofaiss/pkg/metric"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Index implements Inverted File index
type Index struct {
	dim       int
	metric    metric.Metric
	nlist     int // number of clusters
	centroids []vector.Vector
	lists     [][]vector.Vector // inverted lists
	mu        sync.RWMutex
	trained   bool
}

// Config holds IVF configuration
type Config struct {
	Metric string
	Nlist  int // number of clusters/cells
}

// DefaultConfig returns default IVF configuration
func DefaultConfig(numVectors int) Config {
	// Rule of thumb: nlist = sqrt(numVectors)
	nlist := int(math.Sqrt(float64(numVectors)))
	if nlist < 10 {
		nlist = 10
	}
	if nlist > 65536 {
		nlist = 65536
	}
	return Config{
		Metric: "l2",
		Nlist:  nlist,
	}
}

// New creates a new IVF index
func New(dim int, metricType string, config Config) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if config.Nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}

	m, err := metric.New(metric.Type(metricType))
	if err != nil {
		return nil, err
	}

	return &Index{
		dim:    dim,
		metric: m,
		nlist:  config.Nlist,
		lists:  make([][]vector.Vector, config.Nlist),
	}, nil
}

// Train trains the IVF index using k-means clustering
func (idx *Index) Train(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) < idx.nlist {
		return fmt.Errorf("need at least %d vectors to train with %d clusters", idx.nlist, idx.nlist)
	}

	// Run k-means to find centroids
	idx.centroids = kMeans(vectors, idx.nlist, idx.metric, 10)
	idx.trained = true

	return nil
}

// IsTrained returns whether the index is trained
func (idx *Index) IsTrained() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.trained
}

// Add adds vectors to the index
func (idx *Index) Add(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained before adding vectors")
	}

	if err := vector.ValidateDimension(vectors, idx.dim); err != nil {
		return err
	}

	// Assign each vector to nearest centroid
	for _, v := range vectors {
		nearestList := idx.findNearestCentroid(v.Data)
		idx.lists[nearestList] = append(idx.lists[nearestList], v)
	}

	return nil
}

// Search finds k nearest neighbors, probing nprobe lists
func (idx *Index) Search(query []float32, k int, nprobe int) ([]vector.SearchResult, error) {
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
	centroidDists := make([]vector.SearchResult, len(idx.centroids))
	for i, c := range idx.centroids {
		dist := idx.metric.Distance(query, c.Data)
		centroidDists[i] = vector.SearchResult{ID: int64(i), Distance: dist}
	}

	sort.Slice(centroidDists, func(i, j int) bool {
		return centroidDists[i].Distance < centroidDists[j].Distance
	})

	// Search in closest nprobe lists
	var results []vector.SearchResult
	for i := 0; i < nprobe; i++ {
		listIdx := int(centroidDists[i].ID)
		for _, v := range idx.lists[listIdx] {
			dist := idx.metric.Distance(query, v.Data)
			results = append(results, vector.SearchResult{ID: v.ID, Distance: dist})
		}
	}

	// Sort by distance
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

// BatchSearch performs batch search
func (idx *Index) BatchSearch(queries [][]float32, k int, nprobe int) ([][]vector.SearchResult, error) {
	results := make([][]vector.SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k, nprobe)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// Remove is not supported for IVF
func (idx *Index) Remove(id int64) error {
	return fmt.Errorf("remove not supported for IVF index")
}

// Size returns total number of vectors
func (idx *Index) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	total := 0
	for _, list := range idx.lists {
		total += len(list)
	}
	return total
}

// Dimension returns vector dimension
func (idx *Index) Dimension() int {
	return idx.dim
}

// Stats returns index statistics
func (idx *Index) Stats() stats.Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	vectorMem := idx.Size() * idx.dim * 4
	centroidMem := idx.nlist * idx.dim * 4
	memoryMB := float64(vectorMem+centroidMem) / (1024 * 1024)

	listSizes := make([]int, idx.nlist)
	for i, list := range idx.lists {
		listSizes[i] = len(list)
	}

	return stats.Stats{
		TotalVectors:  idx.Size(),
		Dimension:     idx.dim,
		IndexType:     "IVF",
		MemoryUsageMB: memoryMB,
		ExtraInfo: map[string]interface{}{
			"metric":    idx.metric.Name(),
			"nlist":     idx.nlist,
			"trained":   idx.trained,
			"listSizes": listSizes,
		},
	}
}

// GetListSizes returns sizes of all inverted lists
func (idx *Index) GetListSizes() []int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	sizes := make([]int, len(idx.lists))
	for i, list := range idx.lists {
		sizes[i] = len(list)
	}
	return sizes
}

// Internal methods

func (idx *Index) findNearestCentroid(v []float32) int {
	minDist := float32(math.Inf(1))
	minIdx := 0

	for i, c := range idx.centroids {
		dist := idx.metric.Distance(v, c.Data)
		if dist < minDist {
			minDist = dist
			minIdx = i
		}
	}
	return minIdx
}

// kMeans performs k-means clustering
func kMeans(vectors []vector.Vector, k int, m metric.Metric, maxIter int) []vector.Vector {
	// Initialize centroids using k-means++
	centroids := make([]vector.Vector, k)
	step := len(vectors) / k
	for i := 0; i < k; i++ {
		idx := i * step
		if idx >= len(vectors) {
			idx = len(vectors) - 1
		}
		centroids[i] = vector.Vector{
			ID:   int64(i),
			Data: vector.Copy(vectors[idx].Data),
		}
	}

	// Iterate
	for iter := 0; iter < maxIter; iter++ {
		// Assign vectors to nearest centroid
		assignments := make([]int, len(vectors))
		changed := false

		for i, v := range vectors {
			minDist := float32(math.Inf(1))
			minIdx := 0
			for j, c := range centroids {
				dist := m.Distance(v.Data, c.Data)
				if dist < minDist {
					minDist = dist
					minIdx = j
				}
			}
			if assignments[i] != minIdx {
				changed = true
			}
			assignments[i] = minIdx
		}

		if !changed {
			break // converged
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

	return centroids
}
