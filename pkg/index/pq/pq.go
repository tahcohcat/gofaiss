package pq

import (
	"fmt"
	"math"
	"sort"
	"sync"

	internalMath "github.com/tahcohcat/gofaiss/internal/math"
	"github.com/tahcohcat/gofaiss/pkg/index/stats"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Config holds Product Quantization configuration
type Config struct {
	M     int // number of subquantizers (must divide dimension evenly)
	Nbits int // bits per subquantizer (typically 8)
}

// Index implements Product Quantization index
type Index struct {
	dim       int
	M         int   // number of subquantizers
	Nbits     int   // bits per code
	Ksub      int   // number of centroids per subquantizer (2^Nbits)
	dsub      int   // dimension of each subspace (dim/M)
	codebooks [][]float32 // [M][Ksub][dsub] flattened
	codes     [][]uint8   // compressed codes for each vector
	ids       []int64     // vector IDs
	mu        sync.RWMutex
	trained   bool
}

// NewIndex creates a new PQ index
func NewIndex(dim int, cfg Config) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if cfg.M <= 0 {
		return nil, fmt.Errorf("m must be positive")
	}
	if dim%cfg.M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M %d", dim, cfg.M)
	}
	if cfg.Nbits <= 0 || cfg.Nbits > 16 {
		return nil, fmt.Errorf("nbits must be in [1,16]")
	}

	Ksub := 1 << cfg.Nbits // 2^Nbits
	dsub := dim / cfg.M

	return &Index{
		dim:   dim,
		M:     cfg.M,
		Nbits: cfg.Nbits,
		Ksub:  Ksub,
		dsub:  dsub,
		codes: make([][]uint8, 0),
		ids:   make([]int64, 0),
	}, nil
}

// Train trains the PQ index using k-means on subspaces
func (idx *Index) Train(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) < idx.Ksub {
		return fmt.Errorf("need at least %d vectors for training", idx.Ksub)
	}

	if err := vector.ValidateDimension(vectors, idx.dim); err != nil {
		return err
	}

	// Train codebook for each subspace
	idx.codebooks = make([][]float32, idx.M)
	
	for m := 0; m < idx.M; m++ {
		// Extract subspace vectors
		subVectors := make([][]float32, len(vectors))
		start := m * idx.dsub
		end := start + idx.dsub
		
		for i, v := range vectors {
			subVectors[i] = v.Data[start:end]
		}
		
		// Run k-means to get Ksub centroids
		centroids := kMeansSubspace(subVectors, idx.Ksub, 10)
		
		// Flatten centroids into codebook
		idx.codebooks[m] = make([]float32, idx.Ksub*idx.dsub)
		for k := 0; k < idx.Ksub; k++ {
			copy(idx.codebooks[m][k*idx.dsub:(k+1)*idx.dsub], centroids[k])
		}
	}

	idx.trained = true
	return nil
}

// Add adds vectors to the index (must be trained first)
func (idx *Index) Add(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.trained {
		return fmt.Errorf("index must be trained before adding")
	}

	if err := vector.ValidateDimension(vectors, idx.dim); err != nil {
		return err
	}

	// Encode each vector
	for _, v := range vectors {
		code := idx.encode(v.Data)
		idx.codes = append(idx.codes, code)
		idx.ids = append(idx.ids, v.ID)
	}

	return nil
}

// Search finds k nearest neighbors using asymmetric distance
func (idx *Index) Search(query []float32, k int) ([]vector.SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if !idx.trained {
		return nil, fmt.Errorf("index not trained")
	}

	if len(query) != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch")
	}

	if len(idx.codes) == 0 {
		return []vector.SearchResult{}, nil
	}

	// Precompute distance tables for each subspace
	distTables := make([][]float32, idx.M)
	for m := 0; m < idx.M; m++ {
		start := m * idx.dsub
		end := start + idx.dsub
		querySubspace := query[start:end]
		
		distTables[m] = make([]float32, idx.Ksub)
		for ksub := 0; ksub < idx.Ksub; ksub++ {
			centroid := idx.codebooks[m][ksub*idx.dsub : (ksub+1)*idx.dsub]
			distTables[m][ksub] = internalMath.L2DistanceSquared(querySubspace, centroid)
		}
	}

	// Compute distances to all vectors
	results := make([]vector.SearchResult, len(idx.codes))
	for i, code := range idx.codes {
		dist := float32(0)
		for m := 0; m < idx.M; m++ {
			dist += distTables[m][code[m]]
		}
		results[i] = vector.SearchResult{
			ID:       idx.ids[i],
			Distance: float32(math.Sqrt(float64(dist))),
		}
	}

	// Sort and return top k
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	if k < len(results) {
		return results[:k], nil
	}
	return results, nil
}

// BatchSearch performs batch search
func (idx *Index) BatchSearch(queries [][]float32, k int) ([][]vector.SearchResult, error) {
	results := make([][]vector.SearchResult, len(queries))
	for i, query := range queries {
		res, err := idx.Search(query, k)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// Size returns number of vectors
func (idx *Index) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.codes)
}

// Dimension returns vector dimension
func (idx *Index) Dimension() int {
	return idx.dim
}


// Stats returns index statistics
func (idx *Index) Stats() stats.Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Compressed codes
	codeMem := len(idx.codes) * idx.M
	// Codebooks
	codebookMem := idx.M * idx.Ksub * idx.dsub * 4
	memoryMB := float64(codeMem+codebookMem) / (1024 * 1024)

	compressionRatio := float64(idx.dim*4) / float64(idx.M)

	return stats.Stats{
		TotalVectors:  len(idx.codes),
		Dimension:     idx.dim,
		IndexType:     "PQ",
		MemoryUsageMB: memoryMB,
		ExtraInfo: map[string]interface{}{
			"M":                idx.M,
			"Nbits":            idx.Nbits,
			"Ksub":             idx.Ksub,
			"dsub":             idx.dsub,
			"compressionRatio": compressionRatio,
			"trained":          idx.trained,
		},
	}
}

// IsTrained returns whether the index is trained
func (idx *Index) IsTrained() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.trained
}

// Internal methods

// encode encodes a vector into PQ codes
func (idx *Index) encode(v []float32) []uint8 {
	code := make([]uint8, idx.M)
	
	for m := 0; m < idx.M; m++ {
		start := m * idx.dsub
		end := start + idx.dsub
		subVector := v[start:end]
		
		// Find nearest centroid in this subspace
		minDist := float32(math.Inf(1))
		minIdx := 0
		
		for ksub := 0; ksub < idx.Ksub; ksub++ {
			centroid := idx.codebooks[m][ksub*idx.dsub : (ksub+1)*idx.dsub]
			dist := internalMath.L2DistanceSquared(subVector, centroid)
			if dist < minDist {
				minDist = dist
				minIdx = ksub
			}
		}
		
		code[m] = uint8(minIdx)
	}
	
	return code
}

// kMeansSubspace performs k-means clustering on subspace vectors
func kMeansSubspace(vectors [][]float32, k int, maxIter int) [][]float32 {
	if len(vectors) == 0 || k <= 0 {
		return nil
	}
	
	dim := len(vectors[0])
	
	// Initialize centroids using k-means++ style
	centroids := make([][]float32, k)
	step := len(vectors) / k
	for i := 0; i < k; i++ {
		idx := i * step
		if idx >= len(vectors) {
			idx = len(vectors) - 1
		}
		centroids[i] = make([]float32, dim)
		copy(centroids[i], vectors[idx])
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
				dist := internalMath.L2DistanceSquared(v, c)
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
			break
		}
		
		// Update centroids
		for i := range centroids {
			sum := make([]float32, dim)
			count := 0
			
			for j, a := range assignments {
				if a == i {
					for d := range sum {
						sum[d] += vectors[j][d]
					}
					count++
				}
			}
			
			if count > 0 {
				for d := range sum {
					centroids[i][d] = sum[d] / float32(count)
				}
			}
		}
	}
	
	return centroids
}