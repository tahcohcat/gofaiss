package ivfpq

import (
	"fmt"
	"math"
	"sort"
	"sync"

	internalMath "github.com/tahcohcat/gofaiss/internal/math"
	"github.com/tahcohcat/gofaiss/pkg/index/stats"
	"github.com/tahcohcat/gofaiss/pkg/metric"
	"github.com/tahcohcat/gofaiss/pkg/storage"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Config holds IVFPQ configuration
type Config struct {
	Metric string
	Nlist  int // number of IVF clusters
	M      int // number of PQ subquantizers
	Nbits  int // bits per PQ code
}

// DefaultConfig returns default IVFPQ configuration
func DefaultConfig(numVectors, dim int) Config {
	nlist := int(math.Sqrt(float64(numVectors)))
	if nlist < 10 {
		nlist = 10
	}
	if nlist > 65536 {
		nlist = 65536
	}

	// Default PQ settings
	m := 8
	if dim%m != 0 {
		// Find divisor close to 8
		for m = 8; m <= 32; m++ {
			if dim%m == 0 {
				break
			}
		}
		if dim%m != 0 {
			m = 4 // Fallback
		}
	}

	return Config{
		Metric: "l2",
		Nlist:  nlist,
		M:      m,
		Nbits:  8,
	}
}

// Index implements IVF+PQ (Inverted File with Product Quantization)
type Index struct {
	dim       int
	metric    metric.Metric
	nlist     int                  // number of IVF clusters
	M         int                  // number of PQ subquantizers
	Nbits     int                  // bits per PQ code
	Ksub      int                  // number of centroids per subquantizer (2^Nbits)
	dsub      int                  // dimension of each subspace (dim/M)
	centroids []vector.Vector      // IVF centroids
	codebooks [][]float32          // PQ codebooks [M][Ksub*dsub]
	lists     [][]CompressedVector // inverted lists with compressed vectors
	mu        sync.RWMutex
	trained   bool
}

// CompressedVector represents a PQ-encoded vector in an inverted list
type CompressedVector struct {
	ID   int64
	Code []uint8 // PQ code
}

// New creates a new IVFPQ index
func New(dim int, metricType string, config Config) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive")
	}
	if config.Nlist <= 0 {
		return nil, fmt.Errorf("nlist must be positive")
	}
	if config.M <= 0 {
		return nil, fmt.Errorf("M must be positive")
	}
	if dim%config.M != 0 {
		return nil, fmt.Errorf("dimension %d must be divisible by M %d", dim, config.M)
	}
	if config.Nbits <= 0 || config.Nbits > 16 {
		return nil, fmt.Errorf("nbits must be in [1,16]")
	}

	m, err := metric.New(metric.Type(metricType))
	if err != nil {
		return nil, err
	}

	Ksub := 1 << config.Nbits
	dsub := dim / config.M

	return &Index{
		dim:    dim,
		metric: m,
		nlist:  config.Nlist,
		M:      config.M,
		Nbits:  config.Nbits,
		Ksub:   Ksub,
		dsub:   dsub,
		lists:  make([][]CompressedVector, config.Nlist),
	}, nil
}

// Train trains the IVFPQ index (both IVF clustering and PQ codebooks)
func (idx *Index) Train(vectors []vector.Vector) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(vectors) < idx.nlist*10 {
		return fmt.Errorf("need at least %d vectors for training", idx.nlist*10)
	}

	if err := vector.ValidateDimension(vectors, idx.dim); err != nil {
		return err
	}

	// Step 1: Train IVF (k-means clustering)
	idx.centroids = kMeans(vectors, idx.nlist, idx.metric, 10)

	// Step 2: Assign vectors to nearest centroids for residual computation
	assignments := make([]int, len(vectors))
	for i, v := range vectors {
		assignments[i] = idx.findNearestCentroid(v.Data)
	}

	// Step 3: Compute residuals
	residuals := make([][]float32, len(vectors))
	for i, v := range vectors {
		centroid := idx.centroids[assignments[i]].Data
		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}
		residuals[i] = residual
	}

	// Step 4: Train PQ on residuals
	idx.codebooks = make([][]float32, idx.M)

	for m := 0; m < idx.M; m++ {
		// Extract subspace vectors from residuals
		subVectors := make([][]float32, len(residuals))
		start := m * idx.dsub
		end := start + idx.dsub

		for i, r := range residuals {
			subVectors[i] = r[start:end]
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

// IsTrained returns whether the index is trained
func (idx *Index) IsTrained() bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.trained
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

	// For each vector: assign to nearest centroid, compute residual, encode with PQ
	for _, v := range vectors {
		// Find nearest centroid
		listIdx := idx.findNearestCentroid(v.Data)

		// Compute residual
		centroid := idx.centroids[listIdx].Data
		residual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			residual[d] = v.Data[d] - centroid[d]
		}

		// Encode residual with PQ
		code := idx.encodeResidual(residual)

		// Add to inverted list
		idx.lists[listIdx] = append(idx.lists[listIdx], CompressedVector{
			ID:   v.ID,
			Code: code,
		})
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
		centroid := idx.centroids[listIdx].Data

		// Compute query residual
		queryResidual := make([]float32, idx.dim)
		for d := 0; d < idx.dim; d++ {
			queryResidual[d] = query[d] - centroid[d]
		}

		// Precompute distance tables for PQ
		distTables := idx.computeDistanceTables(queryResidual)

		// Compute asymmetric distances for all vectors in this list
		for _, cv := range idx.lists[listIdx] {
			dist := idx.asymmetricDistance(distTables, cv.Code)
			results = append(results, vector.SearchResult{
				ID:       cv.ID,
				Distance: dist,
			})
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

// Remove is not supported for IVFPQ
func (idx *Index) Remove(id int64) error {
	return fmt.Errorf("remove not supported for IVFPQ index")
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

	// Compressed codes (much smaller than original vectors)
	codeMem := idx.Size() * idx.M

	// Codebooks
	codebookMem := idx.M * idx.Ksub * idx.dsub * 4

	// Centroids
	centroidMem := idx.nlist * idx.dim * 4

	memoryMB := float64(codeMem+codebookMem+centroidMem) / (1024 * 1024)

	// Compression ratio: original size / compressed size
	originalSize := idx.Size() * idx.dim * 4
	compressedSize := codeMem
	compressionRatio := float64(originalSize) / float64(compressedSize)

	listSizes := make([]int, idx.nlist)
	for i, list := range idx.lists {
		listSizes[i] = len(list)
	}

	return stats.Stats{
		TotalVectors:  idx.Size(),
		Dimension:     idx.dim,
		IndexType:     "IVFPQ",
		MemoryUsageMB: memoryMB,
		ExtraInfo: map[string]interface{}{
			"metric":           idx.metric.Name(),
			"nlist":            idx.nlist,
			"M":                idx.M,
			"Nbits":            idx.Nbits,
			"Ksub":             idx.Ksub,
			"dsub":             idx.dsub,
			"compressionRatio": compressionRatio,
			"trained":          idx.trained,
			"listSizes":        listSizes,
		},
	}
}

// Save serializes the IVFPQ index
func (idx *Index) Save(w storage.Writer) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Write header
	if err := w.Encode(idx.dim); err != nil {
		return err
	}
	if err := w.Encode(idx.metric.Name()); err != nil {
		return err
	}
	if err := w.Encode(idx.nlist); err != nil {
		return err
	}
	if err := w.Encode(idx.M); err != nil {
		return err
	}
	if err := w.Encode(idx.Nbits); err != nil {
		return err
	}
	if err := w.Encode(idx.Ksub); err != nil {
		return err
	}
	if err := w.Encode(idx.dsub); err != nil {
		return err
	}
	if err := w.Encode(idx.trained); err != nil {
		return err
	}

	// Write centroids
	if err := w.Encode(idx.centroids); err != nil {
		return err
	}

	// Write codebooks
	if err := w.Encode(idx.codebooks); err != nil {
		return err
	}

	// Write inverted lists
	if err := w.Encode(idx.lists); err != nil {
		return err
	}

	return nil
}

// Load deserializes the IVFPQ index
func (idx *Index) Load(r storage.Reader) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Read header
	if err := r.Decode(&idx.dim); err != nil {
		return err
	}

	var metricName string
	if err := r.Decode(&metricName); err != nil {
		return err
	}
	m, err := metric.New(metric.Type(metricName))
	if err != nil {
		return err
	}
	idx.metric = m

	if err := r.Decode(&idx.nlist); err != nil {
		return err
	}
	if err := r.Decode(&idx.M); err != nil {
		return err
	}
	if err := r.Decode(&idx.Nbits); err != nil {
		return err
	}
	if err := r.Decode(&idx.Ksub); err != nil {
		return err
	}
	if err := r.Decode(&idx.dsub); err != nil {
		return err
	}
	if err := r.Decode(&idx.trained); err != nil {
		return err
	}

	// Read centroids
	if err := r.Decode(&idx.centroids); err != nil {
		return err
	}

	// Read codebooks
	if err := r.Decode(&idx.codebooks); err != nil {
		return err
	}

	// Read inverted lists
	if err := r.Decode(&idx.lists); err != nil {
		return err
	}

	return nil
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

func (idx *Index) encodeResidual(residual []float32) []uint8 {
	code := make([]uint8, idx.M)

	for m := 0; m < idx.M; m++ {
		start := m * idx.dsub
		end := start + idx.dsub
		subVector := residual[start:end]

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

func (idx *Index) computeDistanceTables(queryResidual []float32) [][]float32 {
	distTables := make([][]float32, idx.M)

	for m := 0; m < idx.M; m++ {
		start := m * idx.dsub
		end := start + idx.dsub
		querySubspace := queryResidual[start:end]

		distTables[m] = make([]float32, idx.Ksub)
		for ksub := 0; ksub < idx.Ksub; ksub++ {
			centroid := idx.codebooks[m][ksub*idx.dsub : (ksub+1)*idx.dsub]
			distTables[m][ksub] = internalMath.L2DistanceSquared(querySubspace, centroid)
		}
	}

	return distTables
}

func (idx *Index) asymmetricDistance(distTables [][]float32, code []uint8) float32 {
	dist := float32(0)
	for m := 0; m < idx.M; m++ {
		dist += distTables[m][code[m]]
	}
	return float32(math.Sqrt(float64(dist)))
}

// kMeans performs k-means clustering for IVF
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

// kMeansSubspace performs k-means clustering on subspace vectors
func kMeansSubspace(vectors [][]float32, k int, maxIter int) [][]float32 {
	if len(vectors) == 0 || k <= 0 {
		return nil
	}

	dim := len(vectors[0])

	// Initialize centroids
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
